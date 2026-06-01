"""vcfbot — local RAG over Broadcom/VMware techdocs PDFs."""

from __future__ import annotations

import json
import shutil
import time

import click
from rich.console import Console

from .changelog import ChangelogEntry, append_entry, now_iso
from .chat import run_repl
from .config import load_settings
from .fetch import fetch_all
from .index import collection_count, index_pdf, index_pdf_incremental, reset_collection

console = Console()


@click.group()
def main() -> None:
    """vcfbot CLI."""


@main.command()
@click.option("--force", is_flag=True, help="Re-download even if Last-Modified matches.")
def fetch(force: bool) -> None:
    """Download configured source PDFs to data/pdfs/."""
    settings = load_settings()
    for r in fetch_all(settings.pdf_dir, force=force):
        tag = "cached" if r.cached else "downloaded"
        size_mb = r.bytes / (1024 * 1024)
        console.print(f"[green]{tag}[/] {r.name} — {size_mb:.1f} MB — sha {r.sha256[:12]}")


@main.command(name="index")
@click.option("--only", default=None, help="Index only a specific PDF stem (e.g. vmware-cloud-foundation-9-1).")
def index_cmd(only: str | None) -> None:
    """Extract, chunk, embed, and store all PDFs in data/pdfs/ into chromadb."""
    settings = load_settings()
    pdfs = sorted(settings.pdf_dir.glob("*.pdf"))
    if only:
        pdfs = [p for p in pdfs if p.stem == only]
    if not pdfs:
        console.print("[yellow]no PDFs found — run `vcfbot fetch` first[/]")
        return
    total = 0
    for pdf in pdfs:
        console.print(f"[cyan]indexing[/] {pdf.name} …")
        n = index_pdf(pdf, settings, on_progress=lambda m: console.print(f"  [dim]{m}[/]"))
        console.print(f"  [green]{n} chunks[/] upserted")
        total += n
    console.print(
        f"[bold]done.[/] total chunks indexed this run: {total}; "
        f"collection size: {collection_count(settings)}"
    )


@main.command()
def chat() -> None:
    """Start the interactive RAG REPL."""
    run_repl()


@main.command()
def status() -> None:
    """Print current settings and index size."""
    settings = load_settings()
    console.print(settings)
    try:
        console.print(f"collection chunks: [cyan]{collection_count(settings)}[/]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]could not read collection: {e}[/]")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option("--port", default=8765, show_default=True, type=int, help="Bind port.")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev).")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the web UI + API server."""
    import uvicorn

    console.print(f"[bold]vcfbot[/] serving at [cyan]http://{host}:{port}/[/]")
    uvicorn.run("vcfbot.server:app", host=host, port=port, reload=reload, log_level="info")


@main.command()
@click.option("--force", is_flag=True,
              help="Bypass If-Modified-Since, wipe chroma, and full-rebuild from scratch.")
def update(force: bool) -> None:
    """Daily refresh: fetch upstream PDFs and, if changed, diff-update the index.

    Default (incremental):
      Embeds only chunks whose IDs aren't already in chromadb (chunk IDs are
      content-addressed, so identical text → identical ID). Deletes orphans
      whose IDs no longer appear in the current chunk set. A typo fix in
      upstream usually touches a few dozen chunks — not the whole corpus.

    --force:
      Re-downloads PDFs unconditionally, wipes the collection via
      reset_collection, then re-embeds everything. Use when EMBED_MODEL or
      chunking parameters changed.

    No-op exit (well under a second) when Broadcom hasn't republished.
    """
    settings = load_settings()

    # Snapshot pre-fetch metadata so we can detect actual content changes.
    pre_sha: dict[str, str] = {}
    for meta_path in settings.pdf_dir.glob("*.meta.json"):
        try:
            data = json.loads(meta_path.read_text())
            pre_sha[data["name"]] = data.get("sha256", "")
        except (OSError, json.JSONDecodeError, KeyError):
            continue

    console.print("[cyan]fetching[/] upstream PDFs …")
    results = fetch_all(settings.pdf_dir, force=force)

    changed = [r for r in results if force or r.sha256 != pre_sha.get(r.name, "")]
    if not changed and not force:
        console.print("[green]up to date[/] — no upstream changes detected.")
        return

    console.print(
        f"[yellow]content changed[/] for {len(changed)} source(s): "
        + ", ".join(c.name for c in changed)
    )

    chunks_before = collection_count(settings)
    started = time.time()

    total_added = 0
    total_removed = 0
    # Per-source diff_sections keyed by pdf.stem; written into each
    # ChangelogEntry below. Diff section detail is only available from
    # the incremental path — `--force` rebuilds skip it.
    diff_by_source: dict[str, list[dict]] = {}

    if force:
        # Hard rebuild — chromadb API reset (NOT rmtree; that breaks open
        # SQLite handles on the running server).
        console.print("[dim]--force: resetting chroma collection and re-embedding[/]")
        reset_collection(settings)
        for pdf in sorted(settings.pdf_dir.glob("*.pdf")):
            console.print(f"[cyan]reindexing[/] {pdf.name} …")
            n = index_pdf(pdf, settings, on_progress=lambda m: console.print(f"  [dim]{m}[/]"))
            total_added += n
    else:
        # Diff update — only embed new chunk IDs; delete orphans.
        for pdf in sorted(settings.pdf_dir.glob("*.pdf")):
            console.print(f"[cyan]diff-indexing[/] {pdf.name} …")
            stats = index_pdf_incremental(
                pdf, settings, on_progress=lambda m: console.print(f"  [dim]{m}[/]")
            )
            total_added += stats["added"]
            total_removed += stats["removed"]
            diff_by_source[pdf.stem] = stats.get("diff_sections") or []

    duration = time.time() - started
    chunks_after = collection_count(settings)

    changelog_path = settings.chroma_dir.parent / "changelog.jsonl"
    ts = now_iso()
    for r in changed:
        append_entry(
            changelog_path,
            ChangelogEntry(
                ts=ts,
                source=r.name,
                broadcom_last_modified=r.last_modified,
                sha256=r.sha256,
                bytes=r.bytes,
                chunks_before=chunks_before,
                chunks_after=chunks_after,
                duration_sec=round(duration, 1),
                chunks_added=total_added,
                chunks_removed=total_removed,
                diff_sections=diff_by_source.get(r.name) or None,
            ),
        )
    console.print(
        f"[bold green]done.[/] +{total_added} new, -{total_removed} removed "
        f"in {duration:.1f}s. total chunks: {chunks_before} → {chunks_after}"
    )
