"""Embed chunks via LM Studio and persist them in chromadb."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from openai import BadRequestError, InternalServerError

from .chunk import Chunk, _ENC, chunk_pages
from .config import Settings, load_settings
from .extract import extract_pages

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
    pages = extract_pages(pdf_path)
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


def collection_count(settings: Settings | None = None) -> int:
    settings = settings or load_settings()
    return _collection(settings).count()


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
    pages = extract_pages(pdf_path)
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
    unchanged = len(chunks) - len(to_embed)

    log(
        f"diff: {unchanged} unchanged, "
        f"{len(to_embed)} new, "
        f"{len(orphan_ids)} orphan"
    )

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

    if orphan_ids:
        log(f"deleting {len(orphan_ids)} orphan chunks")
        collection.delete(ids=orphan_ids)

    return {
        "total": len(chunks),
        "added": len(to_embed),
        "removed": len(orphan_ids),
        "unchanged": unchanged,
    }


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
