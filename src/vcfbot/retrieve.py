"""Multi-query retrieval for broad, multi-facet questions.

A single dense query embedding sits in exactly ONE neighborhood of the vector
space. Questions that span several distinct things — e.g. "RAM/CPU/storage for
ALL VCF components" touches vCenter sizing, NSX form factors, host counts, SDDC
Manager, VCF Operations, each phrased differently and living far apart — cannot
be served by one embedding: it lands in whichever neighborhood phrases the query
most literally and fills every retrieved slot from there, starving the rest.
Raising TOP_K only deepens that one neighborhood. The fix is a fan-out: one
sub-query per relevant component, retrieved separately and round-robin merged
(index.query_multi) so every component gets representation.

The hard part is knowing WHICH components. Two modes, by question type:

- SPECIFIC question ("NSX Edge memory") → pick the components most similar by
  embedding cosine (index.select_components). Narrowing is correct, no LLM.

- BREADTH question ("requirements for ALL components") → narrowing is exactly
  wrong, and the corpus has no single component-sizing table. So we ANCHOR ON
  THE DOC'S OWN INVENTORY: retrieve the management-components inventory section
  (which the docs DO provide), then extract the component
  names from that grounded text and fan a sizing sub-query per component. The
  enumeration comes from the live doc, not a hardcoded list, so it tracks VCF
  versions; the extraction LLM call only copies names out of text in front of
  it (a task small models do reliably — unlike recall or reasoning). The TOC
  manifest is the fallback if inventory extraction comes up empty.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from .config import Settings
from .index import component_manifest
from .index import query as _query_single
from .index import query_multi, select_components
from .rerank import rerank_order

# A component is "in scope" for a SPECIFIC question if its cosine similarity is
# within this margin of the best-matching component. One clear winner → narrow
# (single query); several clustered → multi-component fan-out.
_SCOPE_MARGIN = 0.04

# Explicit breadth intent: the user wants every component, not the closest few.
# Stable English, not version-specific fact, so fine to encode as behavior.
_BREADTH_RE = re.compile(
    r"\b(all|every|each|full|complete|entire)\b[^.?!]{0,40}\bcomponents?\b"
    r"|\bcomponents?\b[^.?!]{0,40}\b(all|every|each|list)\b",
    re.IGNORECASE,
)

# Safety bound on a fan-out so a pathological extraction can't explode a request.
_BREADTH_CAP = 28

# Probes that ASSEMBLE the grounded inventory text. These name components only
# to STEER retrieval onto the inventory/diagram/licensing pages — the authority
# is the doc text those probes surface, and the component list is EXTRACTED from
# it, not from these strings. (Hardcoding the probe ≠ hardcoding the answer: if
# the doc's components change, the extraction changes; the probes just need to
# land near the right pages, which semantic search does robustly.)
_INVENTORY_PROBES = [
    "First VCF Instance management components the management appliances deployed in management domain",
    "all management components including SDDC Manager vCenter NSX VCF Operations VCF Automation",
    "management appliances diagram License Server VCF Operations Collector",
    "License Server Overview licensing management appliance",
    "NSX Edge node appliance management domain edge cluster",
    "VCF Installer appliance resource requirements deploy",
    "management domain ESX hosts vSAN cluster minimum number",
]

# Breadth questions span ~8 components, so the final cut must stay wide enough to
# carry every component's sizing into synthesis. top_k (≈6 on a default install)
# is right for a single-facet question but starves a breadth answer; rerank still
# reorders the full candidate pool, we just keep more of the top of it here.
_BREADTH_KEEP = 24

# Per-component classification sub-query template. The breadth fan-out already
# fans a SIZING sub-query per component; on its own that retrieves only sizing
# chunks, so the required-vs-optional framing never reaches the synthesis context
# and the model guesses from a sizing workbook's selectability — mislabeling
# essential components (e.g. VCF Operations) as optional. Adding a CLASSIFICATION
# sub-query per component pulls the docs' own "is this required / essential /
# always-deployed" text into the pool for the synthesis prompt to reason from.
#
# This is the same version-robust pattern as the sizing fan-out: the component
# NAMES come from the live doc-inventory extraction, not a hardcoded list, and it
# pins no page or table — so it tracks VCF versions and survives re-pagination.
def _classification_subquery(component: str) -> str:
    return (
        f"Is {component} a required, optional, essential, or always-deployed "
        f"management component of the VCF management domain?"
    )


_EXTRACT_SYSTEM = """/nothink
The text below is from VMware Cloud Foundation 9.1 documentation. List EVERY distinct VCF management component, appliance, or infrastructure element that is DEPLOYED and would need CPU/RAM/storage (e.g. SDDC Manager, vCenter, NSX Manager, NSX Edge, VCF Operations, VCF Automation, License Server, VCF Installer, ESX hosts/vSAN).
Output ONLY a JSON array of names as they appear in the text. Do not invent anything not present in the text."""


def _extract_inventory_components(settings: Settings) -> list[str]:
    """Anchor on the doc's component inventory and extract the deployed components.

    Returns [] on any failure so the caller can fall back to the TOC manifest.
    """
    try:
        res = query_multi(_INVENTORY_PROBES, settings, k_total=18)
        text = " ".join(res.get("documents", [[]])[0])[:6500]
        if not text.strip():
            return []
        client = OpenAI(base_url=settings.chat_base_url, api_key=settings.api_key)
        resp = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        span = re.search(r"\[.*\]", raw, re.DOTALL)
        if not span:
            return []
        data = json.loads(span.group(0))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in data:
        if not isinstance(x, str):
            continue
        name = x.strip()
        key = name.lower()
        if 1 < len(name) < 45 and key not in seen:
            seen.add(key)
            out.append(name)
    return out[:_BREADTH_CAP]


def _fan_out(question: str, components: list[str], settings: Settings, k_total: int | None = None):
    """Fan a sub-query per component (+ the original) and round-robin merge.

    k_total is raised so every component gets ~2 chunks of depth, capped to stay
    well under the chat model's context window.
    """
    subs = [question]
    for c in components:
        subs.append(f"{c} — {question}")
        subs.append(_classification_subquery(c))
    if k_total is None:
        k_total = min(30, max(settings.top_k, 2 * len(subs)))
    return query_multi(subs, settings, k_total=k_total)


def _cand_count(settings: Settings) -> int:
    """How many candidates to retrieve before the final cut: a larger pool to
    rerank when reranking is on, else just top_k."""
    return settings.rerank_top_n if settings.rerank_enabled else settings.top_k


def _select(result: dict, idxs: list[int]) -> dict:
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    return {
        "documents": [[docs[i] for i in idxs]],
        "metadatas": [[metas[i] if i < len(metas) else None for i in idxs]],
        "distances": [[dists[i] if i < len(dists) else 0.0 for i in idxs]],
    }


def _finish(
    question: str, result: dict, settings: Settings, keep: int | None = None
) -> dict:
    """Rerank the candidate pool (if enabled) and cut to `keep` (default top_k).

    When reranking is off this is a no-op: the pool was already sized by the
    caller (top_k for single/specific, the fan-out's own k_total for breadth),
    so trimming here would clobber the pre-rerank breadth behavior.
    """
    if not settings.rerank_enabled:
        return result
    docs = (result.get("documents") or [[]])[0]
    if len(docs) <= 1:
        return result
    order = rerank_order(question, docs, settings) or list(range(len(docs)))
    return _select(result, order[: keep or settings.top_k])


def retrieve(question: str, settings: Settings):
    """Single entry point for both the CLI and the server.

    Always returns a chroma-shaped result dict so callers don't branch. When
    RERANK_ENABLED, retrieve a larger candidate pool then rerank down to top_k.
    """
    cand = _cand_count(settings)

    if not settings.multi_query:
        return _finish(question, _query_single(question, settings, k=cand), settings)

    if _BREADTH_RE.search(question):
        # Enumerate from the doc's own inventory; fall back to the TOC manifest.
        components = _extract_inventory_components(settings)
        if not components:
            components = component_manifest(settings)[:_BREADTH_CAP]
        if not components:
            return _finish(question, _query_single(question, settings, k=cand), settings)
        kt = cand if settings.rerank_enabled else None
        return _finish(
            question,
            _fan_out(question, components, settings, k_total=kt),
            settings,
            keep=_BREADTH_KEEP,
        )

    # Specific question: let embedding similarity pick the relevant components.
    ranked = select_components(
        question, settings, top_n=max(1, settings.multi_query_max - 1)
    )
    if not ranked:
        return _finish(question, _query_single(question, settings, k=cand), settings)
    top_score = ranked[0][1]
    chosen = [label for label, score in ranked if score >= top_score - _SCOPE_MARGIN]
    if len(chosen) <= 1:
        return _finish(question, _query_single(question, settings, k=cand), settings)
    subs = [question] + [f"{label} — {question}" for label in chosen]
    return _finish(question, query_multi(subs, settings, k_total=cand), settings)
