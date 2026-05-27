"""Token-aware chunking of extracted pages with citation-preserving metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken

from .extract import Page

# cl100k_base is fine as a generic token estimator across most models.
_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    id: str
    source: str  # filename stem (e.g., "vmware-cloud-foundation-9-1")
    text: str
    page_start: int
    page_end: int
    section: str | None


def _token_count(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _make_id(source: str, page_start: int, page_end: int, text: str) -> str:
    h = hashlib.sha1(f"{source}|{page_start}|{page_end}|{text}".encode()).hexdigest()
    return f"{source}-{page_start:04d}-{h[:10]}"


def chunk_pages(
    pages: list[Page],
    source: str,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Greedy pack paragraphs into ~target_tokens chunks with token-based overlap.

    Page span (page_start, page_end) is tracked so citations can point at the
    actual page range the chunk covers.
    """
    chunks: list[Chunk] = []
    buf_parts: list[str] = []
    buf_tokens = 0
    buf_page_start: int | None = None
    buf_page_end: int | None = None
    buf_section: str | None = None

    def flush() -> None:
        nonlocal buf_parts, buf_tokens, buf_page_start, buf_page_end, buf_section
        if not buf_parts or buf_page_start is None or buf_page_end is None:
            return
        text = "\n\n".join(buf_parts).strip()
        if not text:
            return
        chunks.append(
            Chunk(
                id=_make_id(source, buf_page_start, buf_page_end, text),
                source=source,
                text=text,
                page_start=buf_page_start,
                page_end=buf_page_end,
                section=buf_section,
            )
        )
        # Seed next buffer with the tail of the just-emitted chunk for overlap.
        # Reset buf_section so the next added paragraph re-initializes it from
        # the actual page that's contributing new content — otherwise every
        # chunk after the first inherits the very first page's section.
        if overlap_tokens > 0:
            tail_tokens = _ENC.encode(text, disallowed_special=())[-overlap_tokens:]
            tail = _ENC.decode(tail_tokens)
            buf_parts = [tail]
            buf_tokens = len(tail_tokens)
            buf_page_start = buf_page_end
        else:
            buf_parts = []
            buf_tokens = 0
            buf_page_start = None
            buf_page_end = None
        buf_section = None

    for page in pages:
        for para in _split_paragraphs(page.text):
            ptoks = _token_count(para)
            # Oversize paragraph: hard-split on token boundary.
            if ptoks > target_tokens:
                tokens = _ENC.encode(para, disallowed_special=())
                for start in range(0, len(tokens), target_tokens):
                    piece = _ENC.decode(tokens[start : start + target_tokens])
                    if buf_tokens + _token_count(piece) > target_tokens and buf_parts:
                        flush()
                    if buf_page_start is None:
                        buf_page_start = page.page_num
                    if buf_section is None:
                        buf_section = page.section
                    buf_page_end = page.page_num
                    buf_parts.append(piece)
                    buf_tokens += _token_count(piece)
                    if buf_tokens >= target_tokens:
                        flush()
                continue

            if buf_tokens + ptoks > target_tokens and buf_parts:
                flush()
            if buf_page_start is None:
                buf_page_start = page.page_num
            if buf_section is None:
                buf_section = page.section
            buf_page_end = page.page_num
            buf_parts.append(para)
            buf_tokens += ptoks

    flush()
    return chunks
