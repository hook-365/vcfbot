"""Download source PDFs with Last-Modified-based caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .sources import SOURCES, Source

# Cloudflare in front of techdocs.broadcom.com 403s non-browser UAs even for
# public PDF downloads, so we identify as a normal browser.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
REFERER = "https://techdocs.broadcom.com/"


@dataclass
class FetchResult:
    name: str
    path: Path
    bytes: int
    sha256: str
    last_modified: str | None
    cached: bool


def _meta_path(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(".meta.json")


def _load_meta(pdf_path: Path) -> dict | None:
    mp = _meta_path(pdf_path)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def fetch_one(source: Source, out_dir: Path, force: bool = False) -> FetchResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{source.name}.pdf"
    prev_meta = None if force else _load_meta(pdf_path)

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": REFERER,
    }
    if prev_meta and prev_meta.get("last_modified") and pdf_path.exists():
        headers["If-Modified-Since"] = prev_meta["last_modified"]

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        with client.stream("GET", source.url, headers=headers) as resp:
            if resp.status_code == 304 and pdf_path.exists():
                return FetchResult(
                    name=source.name,
                    path=pdf_path,
                    bytes=pdf_path.stat().st_size,
                    sha256=prev_meta["sha256"],
                    last_modified=prev_meta.get("last_modified"),
                    cached=True,
                )
            resp.raise_for_status()
            hasher = hashlib.sha256()
            total = 0
            tmp_path = pdf_path.with_suffix(".pdf.part")
            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    fh.write(chunk)
                    hasher.update(chunk)
                    total += len(chunk)
            tmp_path.replace(pdf_path)
            last_modified = resp.headers.get("last-modified")

    sha = hasher.hexdigest()
    meta = {
        "name": source.name,
        "url": source.url,
        "product": source.product,
        "version": source.version,
        "last_modified": last_modified,
        "sha256": sha,
        "bytes": total,
    }
    _meta_path(pdf_path).write_text(json.dumps(meta, indent=2))
    return FetchResult(
        name=source.name,
        path=pdf_path,
        bytes=total,
        sha256=sha,
        last_modified=last_modified,
        cached=False,
    )


def fetch_all(out_dir: Path, force: bool = False) -> list[FetchResult]:
    return [fetch_one(s, out_dir, force=force) for s in SOURCES]
