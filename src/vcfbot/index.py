"""Embed chunks via LM Studio and persist them in chromadb."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from openai import BadRequestError, InternalServerError

from .chunk import Chunk, _ENC, chunk_pages
from .config import Settings, load_settings
from .extract import extract_source

# Apollo's llamacpp-embed server is launched with `--batch-size 2048`; LM
# Studio's defaults are typically higher. We can't easily count tokens in the
# server's tokenizer (nomic-bert WordPiece) without adding `transformers` as a
# dep, and tiktoken's cl100k counts diverge meaningfully from WordPiece on
# this text — so we use cl100k as a coarse pre-batcher and rely on the
# adaptive retry below to handle overflow regardless of tokenizer mismatch.
_EMBED_BATCH_TOKEN_BUDGET = 500


def _client(settings: Settings) -> OpenAI:
    """Embeddings client. Uses EMBED_BASE_URL (falls back to LM_STUDIO_URL)."""
    return OpenAI(base_url=settings.embed_base_url, api_key=settings.api_key)


def _collection(settings: Settings):
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=settings.collection,
        metadata={"hnsw:space": "cosine"},
    )


# nomic-embed-text-v1.5 is trained to expect role-specific prefixes:
# `search_document:` on stored chunks, `search_query:` on retrieval-time queries.
# Without them, retrieval recall is ~2-4 MTEB points worse. Other embedders
# (bge, e5, openai) don't use this exact convention, so gate by model name.
def _uses_nomic_prefixes(model: str) -> bool:
    return "nomic-embed" in (model or "").lower()


def _doc_prefix(model: str) -> str:
    return "search_document: " if _uses_nomic_prefixes(model) else ""


def _query_prefix(model: str) -> str:
    return "search_query: " if _uses_nomic_prefixes(model) else ""


def _embed(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _is_overflow_error(exc: Exception) -> bool:
    """Detect tokenizer/batch overflow across llama.cpp's two error styles:
    - 500 InternalServerError "too large to process. increase ... batch size"
    - 400 BadRequestError "exceed_context_size_error" / "max context size"
    """
    msg = str(exc)
    return any(
        marker in msg
        for marker in (
            "too large to process",
            "exceed_context_size",
            "max context size",
            "batch size",
        )
    )


def _embed_adaptive(
    client: OpenAI, model: str, texts: list[str]
) -> list[list[float]]:
    """Embed `texts`, splitting on overflow until each sub-batch succeeds.

    On a single text that still overflows (chunk longer than the model's
    n_ctx_train), truncates the text in half repeatedly. The truncated
    chunk's embedding represents only the kept prefix — a minor quality
    regression for outliers, but vastly better than skipping the whole
    reindex.
    """
    try:
        return _embed(client, model, texts)
    except (InternalServerError, BadRequestError) as e:
        if not _is_overflow_error(e):
            raise
        if len(texts) > 1:
            mid = len(texts) // 2
            left = _embed_adaptive(client, model, texts[:mid])
            right = _embed_adaptive(client, model, texts[mid:])
            return left + right
        # Single text too big — truncate by half and retry.
        only = texts[0]
        if len(only) <= 200:
            raise
        return _embed_adaptive(client, model, [only[: len(only) // 2]])


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    settings = settings or load_settings()
    client = _client(settings)
    prefixed = _query_prefix(settings.embed_model) + text
    return _embed(client, settings.embed_model, [prefixed])[0]


def _batched(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _token_count(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _batched_by_tokens(
    chunks: list[Chunk], prefix: str, token_budget: int
) -> Iterable[list[Chunk]]:
    """Yield chunk batches whose total (prefix+text) token count stays under
    `token_budget`. Required because some embed servers (apollo's llama.cpp
    --batch-size 2048) reject requests with too many tokens.
    """
    batch: list[Chunk] = []
    tokens = 0
    for c in chunks:
        ct = _token_count(prefix + c.text)
        # A single oversized chunk goes on its own (server may still reject;
        # better to surface that than silently mis-batch).
        if ct >= token_budget:
            if batch:
                yield batch
                batch, tokens = [], 0
            yield [c]
            continue
        if batch and tokens + ct > token_budget:
            yield batch
            batch, tokens = [], 0
        batch.append(c)
        tokens += ct
    if batch:
        yield batch


def index_pdf(
    pdf_path: Path,
    settings: Settings | None = None,
    batch: int = 32,  # retained for backward compat; token-budget batching is used below
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Extract, chunk, embed, and upsert one PDF. Returns number of chunks indexed."""
    settings = settings or load_settings()
    log = on_progress or (lambda _msg: None)

    log(f"extracting pages from {pdf_path.name}")
    pages = extract_source(pdf_path, tables=settings.table_chunks)
    log(f"extracted {len(pages)} pages with text")

    chunks: list[Chunk] = chunk_pages(
        pages,
        source=pdf_path.stem,
        target_tokens=settings.target_tokens,
        overlap_tokens=settings.overlap_tokens,
    )
    log(f"produced {len(chunks)} chunks")
    if not chunks:
        return 0

    collection = _collection(settings)
    client = _client(settings)

    doc_prefix = _doc_prefix(settings.embed_model)
    done = 0
    for group in _batched_by_tokens(chunks, doc_prefix, _EMBED_BATCH_TOKEN_BUDGET):
        # Prefix is applied only when embedding; the stored `documents` keep
        # the original text so retrieval / display isn't polluted.
        inputs = [doc_prefix + c.text for c in group]
        embeddings = _embed_adaptive(client, settings.embed_model, inputs)
        collection.upsert(
            ids=[c.id for c in group],
            embeddings=embeddings,
            documents=[c.text for c in group],
            metadatas=[
                {
                    "source": c.source,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section": c.section or "",
                }
                for c in group
            ],
        )
        done += len(group)
        log(f"embedded {done}/{len(chunks)} chunks")
    return len(chunks)


def query(text: str, settings: Settings | None = None, k: int | None = None):
    settings = settings or load_settings()
    collection = _collection(settings)
    emb = embed_query(text, settings)
    return collection.query(
        query_embeddings=[emb],
        n_results=k or settings.top_k,
        include=["documents", "metadatas", "distances"],
    )


def query_multi(
    texts: list[str], settings: Settings | None = None, k_total: int | None = None
):
    """Retrieve for several sub-queries and round-robin merge the results.

    Each sub-query is embedded and run against chroma independently, then the
    per-query hit lists are interleaved (best-of-each, then second-best-of-each,
    ...) with dedup by chunk id. Interleaving — rather than a single global
    distance sort — is the whole point: a tightly-matching neighborhood (e.g.
    one component whose page states the spec verbatim) would otherwise dominate
    every slot and starve the other facets, which is exactly the failure
    single-query retrieval has on broad questions. Returns the same shape as
    `query()` (one merged result list) so callers are interchangeable.
    """
    settings = settings or load_settings()
    collection = _collection(settings)
    k_total = k_total or settings.top_k
    client = _client(settings)
    prefix = _query_prefix(settings.embed_model)
    embs = _embed_adaptive(client, settings.embed_model, [prefix + t for t in texts])
    # Pull a generous slice per sub-query so the merge has options to dedup against.
    per = max(k_total, 10)
    res = collection.query(
        query_embeddings=embs,
        n_results=per,
        include=["documents", "metadatas", "distances"],
    )
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    dists = res.get("distances") or []
    nq = len(texts)
    seen: set = set()
    out_docs: list = []
    out_metas: list = []
    out_dists: list = []
    for rank in range(per):
        for qi in range(nq):
            if len(out_docs) >= k_total:
                break
            if qi >= len(docs) or rank >= len(docs[qi]):
                continue
            key = (
                ids[qi][rank]
                if qi < len(ids) and rank < len(ids[qi])
                else docs[qi][rank]
            )
            if key in seen:
                continue
            seen.add(key)
            out_docs.append(docs[qi][rank])
            out_metas.append(metas[qi][rank])
            out_dists.append(dists[qi][rank])
        if len(out_docs) >= k_total:
            break
    return {
        "documents": [out_docs],
        "metadatas": [out_metas],
        "distances": [out_dists],
    }


def collection_count(settings: Settings | None = None) -> int:
    settings = settings or load_settings()
    return _collection(settings).count()


# Component areas are derived from the PDF outline (TOC) headings rather than
# hardcoded, so they self-update when the corpus is re-indexed for a new VCF
# version. DITA-OT docs name each component chapter "<Component> Detailed
# Design" or "<Component> Model" — harvesting those segments yields the live
# component manifest (e.g. License Server, VCF Automation, Private AI) without
# baking version-specific knowledge into code or prompts.
_MANIFEST_RE = re.compile(r"^(.*?)\s+(?:Detailed Design|Model)$")
# Memoized per (chroma_dir, collection) and invalidated when the chunk count
# changes — a full corpus scan is too costly to repeat every query, but it only
# needs recomputing after an index/update.
_MANIFEST_CACHE: dict[tuple[str, str], tuple[int, list[str]]] = {}


def component_manifest(
    settings: Settings | None = None, min_count: int = 5, cap: int = 60
) -> list[str]:
    """Return the corpus's component areas, most-documented first.

    Harvested live from section-path metadata; cached until the collection's
    chunk count changes. Returns [] if the collection is empty or unscannable,
    in which case callers should fall back to plain behavior.
    """
    settings = settings or load_settings()
    collection = _collection(settings)
    key = (str(settings.chroma_dir), settings.collection)
    count = collection.count()
    cached = _MANIFEST_CACHE.get(key)
    if cached and cached[0] == count:
        return cached[1]

    areas: dict[str, int] = {}
    offset = 0
    step = 5000
    while True:
        got = collection.get(include=["metadatas"], limit=step, offset=offset)
        metas = got.get("metadatas") or []
        if not metas:
            break
        for meta in metas:
            section = (meta or {}).get("section", "") or ""
            for seg in section.split("›"):
                m = _MANIFEST_RE.match(seg.strip())
                if not m:
                    continue
                name = m.group(1).strip()
                if 2 < len(name) < 40:
                    areas[name] = areas.get(name, 0) + 1
        offset += step

    candidates = [n for n, c in areas.items() if c >= min_count]
    # Collapse topology/availability variants to their base component, purely by
    # structure (no hardcoded qualifier list): processing shortest-first, drop
    # any label that contains an already-kept label as a whole-word phrase. So
    # "High Availability VCF Operations" and "Simple VCF Operations" fold into
    # "VCF Operations"; "Single-Rack vSAN ESA Storage" folds into "Storage".
    kept: list[str] = []
    for label in sorted(candidates, key=len):
        if not any(_contains_phrase(label, base) for base in kept):
            kept.append(label)
    items = sorted(kept, key=lambda n: areas[n], reverse=True)[:cap]
    _MANIFEST_CACHE[key] = (count, items)
    return items


def _contains_phrase(text: str, phrase: str) -> bool:
    """True if `phrase` appears in `text` as a whole-word contiguous phrase."""
    return re.search(rf"(?:^|\W){re.escape(phrase)}(?:\W|$)", text) is not None


# Component-label embeddings, cached alongside the manifest (same invalidation
# on chunk-count change). Embedding the labels lets us pick the components
# RELEVANT to a question by cosine similarity instead of asking the small chat
# model to filter a long menu — which it does unreliably. Labels are embedded
# as "documents" (the thing being matched); the question uses the query prefix.
_LABEL_EMB_CACHE: dict[tuple[str, str], tuple[int, list[str], list[list[float]]]] = {}


def _component_label_embeddings(
    settings: Settings,
) -> tuple[list[str], list[list[float]]]:
    collection = _collection(settings)
    key = (str(settings.chroma_dir), settings.collection)
    count = collection.count()
    cached = _LABEL_EMB_CACHE.get(key)
    if cached and cached[0] == count:
        return cached[1], cached[2]
    labels = component_manifest(settings)
    embs: list[list[float]] = []
    if labels:
        client = _client(settings)
        prefix = _doc_prefix(settings.embed_model)
        embs = _embed_adaptive(client, settings.embed_model, [prefix + l for l in labels])
    _LABEL_EMB_CACHE[key] = (count, labels, embs)
    return labels, embs


def select_components(
    question: str, settings: Settings | None = None, top_n: int = 8
) -> list[tuple[str, float]]:
    """Return the manifest components most similar to `question`, best first.

    Deterministic relevance selection by embedding cosine — no chat-model
    judgment involved. Returns [(label, score), ...]; empty if no manifest.
    """
    settings = settings or load_settings()
    labels, embs = _component_label_embeddings(settings)
    if not labels:
        return []
    import numpy as np

    q = np.asarray(embed_query(question, settings), dtype=float)
    m = np.asarray(embs, dtype=float)
    q = q / (float(np.linalg.norm(q)) or 1.0)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    order = np.argsort(-sims)[:top_n]
    return [(labels[int(i)], float(sims[int(i)])) for i in order]


def index_pdf_incremental(
    pdf_path: Path,
    settings: Settings | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Diff-aware index. Embed only chunks whose IDs aren't already in the
    collection; delete orphan chunks (existed previously but no longer
    produced by the current PDF). Same chromadb collection is used in-place,
    so the running server keeps serving throughout.

    Returns a dict with keys: total, added, removed, unchanged.
    """
    settings = settings or load_settings()
    log = on_progress or (lambda _msg: None)

    log(f"extracting pages from {pdf_path.name}")
    pages = extract_source(pdf_path, tables=settings.table_chunks)
    log(f"extracted {len(pages)} pages with text")

    chunks: list[Chunk] = chunk_pages(
        pages,
        source=pdf_path.stem,
        target_tokens=settings.target_tokens,
        overlap_tokens=settings.overlap_tokens,
    )
    log(f"produced {len(chunks)} chunks")
    if not chunks:
        return {"total": 0, "added": 0, "removed": 0, "unchanged": 0}

    collection = _collection(settings)

    # Existing IDs for this source. include=[] avoids hauling documents +
    # embeddings into memory — we only care about which IDs are present.
    existing = collection.get(where={"source": pdf_path.stem}, include=[])
    existing_ids = set(existing.get("ids") or [])
    new_ids = {c.id for c in chunks}

    to_embed: list[Chunk] = [c for c in chunks if c.id not in existing_ids]
    orphan_ids: list[str] = list(existing_ids - new_ids)
    to_refresh: list[Chunk] = [c for c in chunks if c.id in existing_ids]
    unchanged = len(to_refresh)

    log(
        f"diff: {unchanged} unchanged-text, "
        f"{len(to_embed)} new, "
        f"{len(orphan_ids)} orphan"
    )

    # Chunks whose text is unchanged but may have moved to a different page in
    # the new PDF. Push fresh page_start/page_end/section metadata in-place —
    # no embed call needed. Batched to avoid handing chroma a 30k-row update.
    if to_refresh:
        refresh_batch = 500
        for i in range(0, len(to_refresh), refresh_batch):
            group = to_refresh[i : i + refresh_batch]
            collection.update(
                ids=[c.id for c in group],
                metadatas=[
                    {
                        "source": c.source,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "section": c.section or "",
                    }
                    for c in group
                ],
            )
        log(f"refreshed metadata on {len(to_refresh)} chunks")

    if to_embed:
        client = _client(settings)
        doc_prefix = _doc_prefix(settings.embed_model)
        done = 0
        for group in _batched_by_tokens(to_embed, doc_prefix, _EMBED_BATCH_TOKEN_BUDGET):
            inputs = [doc_prefix + c.text for c in group]
            embeddings = _embed_adaptive(client, settings.embed_model, inputs)
            collection.upsert(
                ids=[c.id for c in group],
                embeddings=embeddings,
                documents=[c.text for c in group],
                metadatas=[
                    {
                        "source": c.source,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "section": c.section or "",
                    }
                    for c in group
                ],
            )
            done += len(group)
            log(f"embedded {done}/{len(to_embed)} new chunks")

    # Pull orphan metadata BEFORE deletion so we can summarize what was
    # removed (section path + page range). include=["metadatas"] only —
    # documents and embeddings would balloon memory on full transitions.
    orphan_meta: list[dict] = []
    if orphan_ids:
        got = collection.get(ids=orphan_ids, include=["metadatas"])
        orphan_meta = list(got.get("metadatas") or [])
        log(f"deleting {len(orphan_ids)} orphan chunks")
        collection.delete(ids=orphan_ids)

    diff_sections = _summarize_diff(to_embed, orphan_meta)

    return {
        "total": len(chunks),
        "added": len(to_embed),
        "removed": len(orphan_ids),
        "unchanged": unchanged,
        "diff_sections": diff_sections,
    }


def _summarize_diff(
    added: list[Chunk],
    removed_meta: list[dict],
) -> list[dict]:
    """Group added + removed chunks by section. One row per section with
    counts and the union page range. Sorted by total impact desc, then
    section name. Output is JSON-serializable for the changelog.
    """
    # section -> {"added": n, "removed": n, "pages": [lo, hi]}
    bucket: dict[str, dict] = {}

    def touch(section: str, page_start: int, page_end: int, key: str) -> None:
        section = section or "(unsectioned)"
        slot = bucket.setdefault(
            section, {"added": 0, "removed": 0, "pages": None}
        )
        slot[key] += 1
        pg = slot["pages"]
        lo = page_start if pg is None else min(pg[0], page_start)
        hi = page_end if pg is None else max(pg[1], page_end)
        slot["pages"] = [lo, hi]

    for c in added:
        touch(c.section or "", c.page_start, c.page_end, "added")
    for m in removed_meta:
        touch(
            str(m.get("section") or ""),
            int(m.get("page_start") or 0),
            int(m.get("page_end") or 0),
            "removed",
        )

    rows = [
        {"section": sec, **vals} for sec, vals in bucket.items()
    ]
    rows.sort(key=lambda r: (-(r["added"] + r["removed"]), r["section"]))
    return rows


def reset_collection(settings: Settings | None = None) -> None:
    """Delete the vector collection's contents in-place via chromadb's API.

    Used instead of `shutil.rmtree(chroma_dir)` so the running server's open
    SQLite handles stay valid. Filesystem deletion of the underlying chroma
    files triggers SQLITE_READONLY_DBMOVED (error 1032) on subsequent writes,
    even from a fresh client.
    """
    settings = settings or load_settings()
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(name=settings.collection)
    except (ValueError, Exception):  # noqa: BLE001 — chromadb raises various types
        pass
    client.get_or_create_collection(
        name=settings.collection,
        metadata={"hnsw:space": "cosine"},
    )
