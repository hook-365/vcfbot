"""PDF -> per-page text with detected section heading via PyMuPDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Page:
    page_num: int  # 1-indexed
    text: str
    section: str | None  # nearest preceding heading detected on or before this page


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


SECTION_SEP = " › "
# Skip top-of-tree entries that don't add useful context to a citation.
_TOC_SKIP_TITLES = {"table of contents"}


def _build_section_map(doc: pymupdf.Document) -> dict[int, str]:
    """Map page_num -> hierarchical section path from the PDF outline.

    The outline (`doc.get_toc()`) gives a real, hand-authored chapter tree.
    This is far more reliable than font-size heuristics on DITA-OT PDFs,
    where the visual heading hierarchy doesn't survive the transform.
    """
    toc = doc.get_toc()
    if not toc:
        return {}

    # Stable order: by page, then by level so deeper levels follow their parent.
    toc = sorted(toc, key=lambda e: (e[2], e[0]))

    page_to_path: dict[int, str] = {}
    stack: list[tuple[int, str]] = []  # (level, title)
    cursor = 0

    for page_num in range(1, doc.page_count + 1):
        # Consume all TOC entries that begin on or before this page.
        while cursor < len(toc) and toc[cursor][2] <= page_num:
            level, title, _page = toc[cursor]
            title = (title or "").strip()
            cursor += 1
            if not title or title.lower() in _TOC_SKIP_TITLES:
                continue
            # Pop deeper-or-equal levels so we maintain a proper hierarchy.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        if stack:
            page_to_path[page_num] = SECTION_SEP.join(t for _, t in stack)
    return page_to_path


def extract_pages(pdf_path: Path) -> list[Page]:
    pages: list[Page] = []
    with pymupdf.open(pdf_path) as doc:
        section_map = _build_section_map(doc)
        for i, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            if not text:
                continue
            pages.append(Page(page_num=i, text=text, section=section_map.get(i)))
    return pages
