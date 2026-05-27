"""Append-only changelog of corpus updates.

Each line in `data/changelog.jsonl` is a JSON object describing an event where
the upstream content actually changed and we re-indexed. "No change" daily
checks aren't recorded — they'd just be noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ChangelogEntry:
    ts: str                              # ISO8601 UTC
    source: str                          # e.g. "vmware-cloud-foundation-9-1"
    broadcom_last_modified: str | None   # raw Last-Modified header from Broadcom
    sha256: str                          # of the newly-downloaded PDF
    bytes: int                           # PDF size
    chunks_before: int                   # collection size before
    chunks_after: int                    # collection size after
    duration_sec: float                  # wall-clock seconds
    # Optional incremental-update detail (added 2026-05-27). Old entries
    # written before this field landed will leave these as None.
    chunks_added: int | None = None      # chunks newly embedded
    chunks_removed: int | None = None    # orphan chunks deleted


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_entry(path: Path, entry: ChangelogEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def read_entries(path: Path, limit: int | None = None) -> list[ChangelogEntry]:
    if not path.exists():
        return []
    entries: list[ChangelogEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(ChangelogEntry(**data))
            except (json.JSONDecodeError, TypeError):
                # Skip malformed lines rather than crashing the API
                continue
    entries.sort(key=lambda e: e.ts, reverse=True)
    if limit is not None:
        entries = entries[:limit]
    return entries
