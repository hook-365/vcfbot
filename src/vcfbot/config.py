"""Runtime config sourced from environment with sensible LM Studio defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # Compatibility alias — the original single URL still works when chat and
    # embeddings live on the same OpenAI-compatible server (LM Studio).
    lm_studio_url: str
    # Apollo and other split deployments serve chat and embed from different
    # processes/ports; these are independently overridable. Both default to
    # lm_studio_url so single-endpoint setups keep working.
    chat_base_url: str
    embed_base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    pdf_dir: Path
    chroma_dir: Path
    collection: str
    top_k: int
    target_tokens: int
    overlap_tokens: int
    # Multi-query retrieval: decompose broad, multi-facet questions into
    # several focused sub-queries so each component gets its own neighborhood
    # in the vector space (a single embedding can only sit in one). See
    # retrieve.py. Falls back to plain single-query when off or when the
    # question is already single-faceted.
    multi_query: bool
    multi_query_max: int
    # Emit atomic table chunks (pymupdf find_tables) so structured facts —
    # the required-vs-optional classification, sizing tables — survive as one
    # retrievable unit instead of fragmenting across 220-token prose chunks.
    table_chunks: bool
    # Cross-encoder reranking: over-retrieve a larger candidate pool by embedding
    # cosine, then a reranker reads query+chunk together and reorders by true
    # relevance — surfaces the authoritative chunk (e.g. the classification table)
    # that bi-encoder dense retrieval buries. Query-time only; no re-index.
    rerank_enabled: bool
    rerank_base_url: str
    rerank_api_key: str
    rerank_model: str
    rerank_top_n: int  # candidate pool retrieved before reranking
    # Chat provider for the synthesis step. "local" (default) keeps everything on
    # the OpenAI-compatible llama.cpp server; "anthropic" routes ONLY chat
    # completion to Claude (embeddings stay local). See providers.py.
    chat_provider: str
    anthropic_api_key: str
    anthropic_model: str
    anthropic_max_tokens: int


def load_settings() -> Settings:
    base = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    return Settings(
        lm_studio_url=base,
        chat_base_url=os.getenv("CHAT_BASE_URL", base),
        embed_base_url=os.getenv("EMBED_BASE_URL", base),
        # LM Studio and llama.cpp ignore the key, but the openai SDK requires
        # a non-empty string.
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        chat_model=os.getenv("CHAT_MODEL", "local-model"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
        pdf_dir=Path(os.getenv("PDF_DIR", ROOT / "data" / "pdfs")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", ROOT / "data" / "chroma")),
        collection=os.getenv("CHROMA_COLLECTION", "vcfbot"),
        top_k=int(os.getenv("TOP_K", "6")),
        target_tokens=int(os.getenv("TARGET_TOKENS", "800")),
        overlap_tokens=int(os.getenv("OVERLAP_TOKENS", "100")),
        multi_query=os.getenv("MULTI_QUERY", "true").lower() not in ("0", "false", "no"),
        multi_query_max=int(os.getenv("MULTI_QUERY_MAX", "8")),
        table_chunks=os.getenv("TABLE_CHUNKS", "true").lower() not in ("0", "false", "no"),
        rerank_enabled=os.getenv("RERANK_ENABLED", "false").lower() in ("1", "true", "yes"),
        rerank_base_url=os.getenv("RERANK_BASE_URL", "https://api.voyageai.com/v1"),
        rerank_api_key=os.getenv("RERANK_API_KEY", os.getenv("VOYAGE_API_KEY", "")),
        rerank_model=os.getenv("RERANK_MODEL", "rerank-2.5-lite"),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "100")),
        chat_provider=os.getenv("CHAT_PROVIDER", "local").lower(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
        anthropic_max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192")),
    )
