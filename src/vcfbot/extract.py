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
    atomic: bool = False  # if True, the chunker emits this whole, never packed/split
                          # (used for tables — a flattened table must survive as one unit)


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


# The VCF Planning & Preparation Workbook defers all per-appliance sizing to its
# "Static Reference Tables" sheet — the authoritative base sizes (vCPU/RAM/disk
# per appliance per size). We DON'T dump rows as text (the sheet is a
# side-by-side calculator; raw rows embed as noise and don't retrieve). Instead
# we parse the labeled blocks and emit ONE coherent natural-language Page per
# appliance, so a query like "smallest supported NSX Manager size" retrieves a
# tight, citable unit. The numbers come from the data (auto-update on re-fetch);
# only the block LAYOUT is assumed here, and it degrades gracefully if Broadcom
# restructures the sheet. (Deployment *totals* are a different question — that's
# the calculator in planner.py, not retrieval.)
_REF_SHEET = "Static Reference Tables"
_METRIC = {
    "cpu": "cpu", "cpus": "cpu", "vcpu": "cpu",
    "ram": "ram", "memory": "ram",
    "disk": "disk", "storage": "disk",
}
# Smallest-first, for the "smallest size" note and ordering.
_SIZE_ORDER = [
    "Extra Small", "Extra_Small", "Tiny", "Light", "Small", "Standard",
    "Medium", "Large", "Extra Large", "XLarge", "X-Large",
]


def _is_num(x: str) -> bool:
    return bool(re.match(r"^-?\d+(?:\.\d+)?$", x or ""))


def _clean_appl(name: str) -> str:
    name = re.sub(r"(?i)\s+(appliance|size|node)s?$", "", name).strip()
    name = name.replace("NSX-T", "NSX")
    return re.sub(r"\s+", " ", name)


def _clean_size(appl: str, size: str) -> str:
    # Drop a redundant appliance prefix on the size ("NSX Edge Small" -> "Small").
    size = re.sub(rf"(?i)^{re.escape(appl)}\s+", "", size).strip()
    return size.replace("Extra_Small", "Extra Small")


def _parse_reference_tables(seq: list[tuple[str, str]]) -> tuple[dict, dict]:
    """Parse (B, C) cell pairs into per-appliance sizing.

    Two block shapes: A) header "<appliance> <metric>" + "Value", then
    "<size> | <number>" rows; B) header "<appliance>" + "Value", then
    "CPU/RAM/Disk | <number>" rows (single-size appliances like SDDC Manager).
    """
    per: dict[str, dict[str, dict[str, str]]] = {}
    single: dict[str, dict[str, str]] = {}
    appl = metric = mode = None
    for b, c in seq:
        bl = b.lower()
        m = re.match(r"(.+?)\s+(cpu|cpus|vcpu|ram|memory|disk|storage)$", bl)
        if m and c.lower() == "value":
            appl, metric, mode = _clean_appl(b[: m.end(1)]), _METRIC[m.group(2)], "A"
            continue
        if (
            c.lower() == "value" and b and bl not in _METRIC
            and not re.search(r"(cpu|ram|memory|disk|storage)$", bl)
            and "size list" not in bl and "storage si" not in bl
        ):
            appl, metric, mode = _clean_appl(b), None, "B"
            continue
        if mode == "A" and appl and b and _is_num(c):
            if metric == "disk":
                size = _clean_size(appl, re.sub(r"Default$", "", b).strip())
                if size in per.get(appl, {}):  # reject storage-tier variants
                    per[appl][size]["disk"] = c
            else:
                per.setdefault(appl, {}).setdefault(_clean_size(appl, b), {})[metric] = c
        elif mode == "B" and appl and bl in _METRIC and _is_num(c):
            single.setdefault(appl, {})[_METRIC[bl]] = c
    return per, single


def _spec(d: dict[str, str]) -> str:
    parts = []
    if "cpu" in d:
        parts.append(f"{d['cpu']} vCPU")
    if "ram" in d:
        parts.append(f"{d['ram']} GB RAM")
    if "disk" in d:
        parts.append(f"{d['disk']} GB storage")
    return ", ".join(parts)


def extract_xlsx(xlsx_path: Path) -> list[Page]:
    """Emit one clean per-appliance sizing Page from the workbook's reference
    tables. `page_num` is just an ordinal; `section` names the sheet.
    """
    import openpyxl  # heavy-ish; only imported on the xlsx path

    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    if _REF_SHEET not in wb.sheetnames:
        return []
    seq: list[tuple[str, str]] = []
    for row in wb[_REF_SHEET].iter_rows(min_col=2, max_col=3, values_only=True):
        b = str(row[0]).strip() if row[0] is not None else ""
        c = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
        seq.append((b, c))
    per, single = _parse_reference_tables(seq)

    section = f"Planning and Preparation Workbook{SECTION_SEP}{_REF_SHEET}"
    pages: list[Page] = []
    n = 0
    for appl, d in single.items():
        spec = _spec(d)
        if not spec:
            continue
        n += 1
        pages.append(Page(n, f"{appl} appliance resource requirements: {spec}.", section, atomic=True))
    for appl, sizes in per.items():
        order = sorted(
            sizes, key=lambda s: _SIZE_ORDER.index(s) if s in _SIZE_ORDER else 99
        )
        items = [f"{s} ({_spec(sizes[s])})" for s in order if _spec(sizes[s])]
        if not items:
            continue
        smallest = next((s for s in order if _spec(sizes[s])), "")
        text = (
            f"{appl} appliance deployment sizes and per-size resource requirements "
            f"(from the VCF Planning & Preparation Workbook): "
            + "; ".join(items)
            + f". The smallest available {appl} size is {smallest}."
        )
        n += 1
        pages.append(Page(n, text, section, atomic=True))
    return pages


# ── Table-aware extraction ────────────────────────────────────────────────
# pymupdf get_text() linearizes tables into bullet-soup, so structured facts
# (the required-vs-optional management-components classification table, per-appliance
# sizing tables, config-maximums) fragment across the small 220-token chunks and
# no single chunk holds the complete spec. page.find_tables() recovers the
# row/column structure (and inline markers like "(optional)") on these DITA-OT
# PDFs. Each detected table is emitted as ONE atomic Page so the chunker keeps
# it whole. This is ADDITIVE — the normal per-page prose chunks are unchanged,
# so re-indexing only embeds the net-new table chunks (and orphans nothing).

_MD_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _table_to_text(table, section: str | None) -> str:
    """Render a found table as a clean, self-describing markdown block.

    Leads with the section path so the chunk is contextualized for retrieval,
    normalizes pymupdf's in-cell <br> artifacts, and drops the separator row.
    """
    try:
        md = table.to_markdown()
    except Exception:  # noqa: BLE001
        return ""
    lines: list[str] = []
    for ln in md.split("\n"):
        if _MD_SEP_RE.match(ln):
            continue
        ln = ln.replace("<br>", " ")
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln and ln.strip("| "):
            lines.append(ln)
    body = "\n".join(lines).strip()
    # Need real content: at least a header + 2 data rows worth.
    if body.count("\n") < 2 or len(body) < 60:
        return ""
    lead = f"Table — {section}" if section else "Table"
    return f"{lead}\n{body}"


def extract_tables(pdf_path: Path, only_pages: set[int] | None = None) -> list[Page]:
    """Emit one atomic Page per detected table. `only_pages` (1-indexed) limits
    extraction to specific pages — used for targeted/validation runs without a
    full-document table scan.
    """
    pages: list[Page] = []
    with pymupdf.open(pdf_path) as doc:
        section_map = _build_section_map(doc)
        for i, page in enumerate(doc, start=1):
            if only_pages is not None and i not in only_pages:
                continue
            try:
                found = page.find_tables()
            except Exception:  # noqa: BLE001
                continue
            for table in found.tables:
                text = _table_to_text(table, section_map.get(i))
                if text:
                    pages.append(
                        Page(page_num=i, text=text, section=section_map.get(i), atomic=True)
                    )
    return pages


def extract_source(path: Path, tables: bool = False) -> list[Page]:
    """Dispatch extraction by file type (PDF prose vs xlsx workbook). When
    `tables` is set, also emit atomic table chunks for PDFs (additive)."""
    if path.suffix.lower() == ".xlsx":
        return extract_xlsx(path)
    pages = extract_pages(path)
    if tables:
        pages += extract_tables(path)
    return pages
