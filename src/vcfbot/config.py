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
    )
