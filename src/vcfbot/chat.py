"""Interactive RAG chat REPL backed by LM Studio + chromadb."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .config import Settings, load_settings
from .providers import stream_chat
from .retrieve import retrieve as vector_query

SYSTEM_PROMPT = """/nothink
You are vcfbot, a focused assistant for VMware Cloud Foundation documentation.

Rules:
- Answer ONLY from the provided context snippets. Quote or paraphrase the directly applicable wording.
- A snippet may contain multiple separate requirements (e.g. one for the main HCI cluster AND a different one for dedicated Storage Clusters). Read carefully and apply only the requirement that matches the user's question. Do NOT blend or average separate requirements.
- Match the user's SCENARIO. The docs cover greenfield (a NEW fleet / first deployment / initial bring-up), upgrade (moving an EXISTING environment to a new version), and expansion (adding an instance, domain, or cluster to an existing fleet) — and the SAME topic (e.g. "deploy management services", "prerequisites", "deploy a component") is documented SEPARATELY for each. Work out which scenario the question is about and apply ONLY the matching guidance: a step or prerequisite written for an upgrade is NOT the answer to a new-deployment question, and vice versa. A snippet that introduces itself with the other scenario's framing (e.g. "to continue with the upgrade…") does not answer a greenfield question. If the context only covers a different scenario than the one asked, say so plainly and give what greenfield/new-deployment guidance the context does contain, rather than presenting the wrong-scenario steps as the answer.
- When asked for the smallest / minimum / cheapest option and a snippet lists several sizes or counts (e.g. Small/Medium/Large, or a node count), pick the LOWEST the documentation permits for the user's context — never default to a middle size. If the smallest option is marked lab / proof-of-concept-only or "not for production", say so explicitly and give BOTH the absolute smallest size AND the smallest production-supported size, using the doc's own wording. Distinguish a size CEILING (a stated maximum or limit, e.g. "FT limits the appliance to medium") from a MINIMUM — never report a stated maximum as if it were the minimum.
- For a MINIMAL / smallest / simple deployment, separate REQUIRED components from OPTIONAL ones using the DOCUMENTATION'S OWN language — not a sizing tool's selectability. Treat a component as REQUIRED if the docs call it essential / core / mandatory / "always deployed", or list it among the components the initial / base management domain deploys. Treat a component as OPTIONAL only when the docs explicitly say it is optional / not required / can be omitted for that profile. A planning or sizing WORKBOOK letting you DESELECT a component is a capacity-modeling convenience — it does NOT make that component optional for a supported deployment. When sources conflict, an explicit "mandatory / required / essential" statement outranks a workbook toggle or a feature-capability description; do NOT downgrade a component to optional merely because it is described in terms of the "capabilities" or "services" it adds. State the minimum NODE COUNT per component (e.g. a single-node appliance where the simple/non-HA model permits one), not just the appliance size.
- Default profile: most deployments use the SIMPLE (non-HA) model with the smallest supported footprint — typically a minimal ~3-node management cluster. Unless the user explicitly asks about High Availability or a larger profile, assume that simple, small-footprint context and the smallest supported sizes / single-node counts the Simple model permits.
- If the context does not contain the answer, say so plainly. Do not invent specifics.
- Cite every factual claim inline with the marker shown above each snippet, e.g. [vmware-cloud-foundation-9-1 p.42].
- Prefer concise, structured answers. Use bullet points for steps and lists.
"""


@dataclass
class Retrieved:
    text: str
    source: str
    page_start: int
    page_end: int
    section: str
    distance: float


def _retrieve(question: str, settings: Settings) -> list[Retrieved]:
    res = vector_query(question, settings)
    out: list[Retrieved] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}  # chroma can hand back None metadata; never let .get crash a query
        out.append(
            Retrieved(
                text=doc,
                source=meta.get("source", "?"),
                page_start=int(meta.get("page_start", 0)),
                page_end=int(meta.get("page_end", 0)),
                section=meta.get("section", "") or "",
                distance=float(dist),
            )
        )
    return out


def _cite(r: Retrieved) -> str:
    if r.page_start == r.page_end:
        return f"[{r.source} p.{r.page_start}]"
    return f"[{r.source} p.{r.page_start}-{r.page_end}]"


def _build_context(hits: list[Retrieved]) -> str:
    blocks: list[str] = []
    for h in hits:
        header = _cite(h)
        if h.section:
            header += f"  ({h.section})"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def run_repl(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    console = Console()
    console.print(
        Panel.fit(
            f"vcfbot — chat backend: [cyan]{settings.lm_studio_url}[/]\n"
            f"chat model: [cyan]{settings.chat_model}[/]  embed model: [cyan]{settings.embed_model}[/]\n"
            f"type [bold]/quit[/] to exit, [bold]/sources[/] after an answer to inspect retrieved chunks.",
            title="ready",
        )
    )
    last_hits: list[Retrieved] = []
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            question = Prompt.ask("[bold green]you[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            return
        if question == "/sources":
            if not last_hits:
                console.print("[dim]no previous query yet[/]")
                continue
            for h in last_hits:
                console.print(
                    Panel(
                        h.text,
                        title=f"{_cite(h)}  dist={h.distance:.3f}  {h.section}",
                        border_style="blue",
                    )
                )
            continue

        hits = _retrieve(question, settings)
        last_hits = hits
        context = _build_context(hits)
        user_msg = (
            f"Context snippets (each prefixed with its citation marker):\n\n{context}\n\n"
            f"Question: {question}"
        )
        history.append({"role": "user", "content": user_msg})

        console.print("[bold magenta]vcfbot[/]:")
        collected: list[str] = []
        for delta in stream_chat(settings, history):
            collected.append(delta)
            console.print(delta, end="", soft_wrap=True)
        console.print()
        answer = "".join(collected)
        history.append({"role": "assistant", "content": answer})
        # render markdown view at end (a second pass; nice for tables/lists)
        if any(ch in answer for ch in ("|", "- ", "**", "#")):
            console.print(Markdown(answer))
