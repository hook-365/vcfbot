# vcfbot · project notes for Claude

Local RAG chatbot over Broadcom / VMware Cloud Foundation techdocs PDFs.
Designed to be auditable: every answer cites the page it came from, every
source links to Broadcom's hosted PDF at that exact page.

## What this project is

- **Goal**: query enterprise infrastructure docs locally without sending
  content to a hosted LLM. Privacy and verifiability are the entire point —
  never silently introduce a cloud-side dependency.
- **Source corpus**: the full VCF 9.1 doc set (one monolithic 184 MB /
  9,158-page PDF) indexed at `TARGET_TOKENS=220` → ~36,873 chunks as the
  current baseline.

## Stack

- Python 3.12. Single package `src/vcfbot/`. uv for local dev, pip inside
  the container.
- httpx (fetch), pymupdf (extract), tiktoken (chunk-size estimation),
  chromadb (vector store), openai SDK (talks to any OpenAI-compatible
  endpoint), FastAPI + uvicorn (web), click (CLI), rich (terminal UX).
- Frontend: plain HTML + CSS + ES modules under `src/vcfbot/static/`.
  No build step. marked + DOMPurify from jsdelivr CDN.

### Inference topology

The app speaks OpenAI-compatible HTTP to whatever's serving chat
completions and embeddings. Common shapes:

- **Single server** (e.g. LM Studio): set `LM_STUDIO_URL`, both chat and
  embeddings hit the same base URL.
- **Split servers** (e.g. one llama.cpp instance for chat, another for
  embeddings): set `CHAT_BASE_URL` and `EMBED_BASE_URL` independently.
  Both fall back to `LM_STUDIO_URL` if unset.

The CLI/server picks the right client per operation — never assume one
shared URL.

## Layout

```
src/vcfbot/
  sources.py        # PDF registry — name, PDF url, web landing url
  fetch.py          # downloader with Last-Modified caching; sends a Chrome UA
  extract.py        # pymupdf → per-page text + TOC-based section paths
  chunk.py          # tiktoken-aware chunker, preserves page ranges per chunk
  index.py          # embeds + upserts into chromadb; token-aware batching + adaptive retry
  changelog.py      # append-only JSONL log of corpus updates
  chat.py           # terminal RAG REPL
  server.py         # FastAPI: /api/status, /api/chat (SSE), /api/changelog
  config.py         # env-driven settings (CHAT_BASE_URL, EMBED_BASE_URL, TARGET_TOKENS, …)
  __init__.py       # click CLI (fetch / index / update / chat / serve / status)
  __main__.py       # `python -m vcfbot …` entry point for `docker exec`
  static/
    index.html      # web UI; two dialogs (About, Changelog), composer, source cards
    styles.css      # design tokens + components
    app.js          # streaming SSE client, citation handling, theme + export
scripts/
  daily-update.sh   # cron target (runs `python -m vcfbot update` in the container)
data/
  pdfs/             # downloaded source PDFs (gitignored)
  chroma/           # chromadb persistent collection (gitignored)
  changelog.jsonl   # corpus-update log (gitignored)
Dockerfile          # python:3.12-slim + requirements.txt + COPY src/vcfbot
docker-compose.yml  # binds to 127.0.0.1:8129; env defaults for chat/embed URLs
requirements.txt    # pinned-min deps for container builds (no uv inside)
pyproject.toml      # uv-managed deps for local dev
```

## Commands

```sh
# Local dev (uv)
uv sync
uv run vcfbot fetch                               # download configured PDFs
uv run vcfbot index                               # extract + chunk + embed + upsert
uv run vcfbot update                              # daily refresh (no-op if unchanged)
uv run vcfbot status                              # settings + chunk count
uv run vcfbot chat                                # terminal REPL
uv run vcfbot serve --host 0.0.0.0 --port 8765    # web UI + API

# Deployed container
docker exec vcfbot python -m vcfbot <cmd>         # we don't pip-install in the
                                                  # container, so the console
                                                  # script entry isn't on PATH;
                                                  # use python -m vcfbot
bash scripts/daily-update.sh                      # on-demand refresh
```

## Critical gotchas

1. **techdocs.broadcom.com 403s non-browser UAs.** Cloudflare blocks the
   default `httpx`/`requests` UA. `fetch.py` sends a Chrome UA + Referer.
   Any new crawler code for that domain must mirror this.

2. **nomic-embed-text-v1.5 expects role prefixes.**
   - `search_document: <text>` when embedding chunks for storage
   - `search_query: <text>` when embedding queries at retrieval time
   Skipping costs 2–4 MTEB recall points. Gated by model-name substring
   (`"nomic-embed"`) in `index.py::_uses_nomic_prefixes`. Other embedders
   (bge, e5, OpenAI) use different conventions — extend the small
   `_*_prefix` family if adding support.

3. **nomic-embed has n_ctx_train = 2048.** The model literally wasn't
   trained on contexts beyond 2048 tokens. `--ctx-size 8192` in the
   llama.cpp embed server has NO effect — the model's context is fixed at
   2048. This is the hard ceiling on individual chunk size.

4. **VMware techdocs tokenize ~6–7× heavier in BERT WordPiece than in
   cl100k.** Dashed acronyms like `VCF-VSAN-REQD-CFG-001` split per
   subword in WordPiece but merge cleanly in cl100k BPE. Net effect: a
   chunk at 220 cl100k tokens may be up to ~2000 nomic tokens. Hence:
   - `TARGET_TOKENS` default is **220** (was 800) — keeps chunks safely
     under the 2048 nomic ceiling
   - Token-budget batching in `index.py::_batched_by_tokens` is just a
     coarse pre-batcher; the real safety net is
     `index.py::_embed_adaptive`, which catches both `InternalServerError`
     ("too large to process") and `BadRequestError`
     ("exceed_context_size_error"), halves the batch, and as last resort
     truncates a single oversized input by half until it fits.

5. **Don't `shutil.rmtree(data/chroma/)`** while the server holds open
   SQLite handles. Filesystem unlink triggers `SQLITE_READONLY_DBMOVED`
   (error 1032) on subsequent writes, even from a fresh chromadb client.
   Use `index.py::reset_collection` instead — it calls chromadb's
   `delete_collection` / `create_collection` API, which keeps the
   underlying SQLite db file intact.

6. **The llama.cpp embed server's `--batch-size` flag controls
   *total tokens per request*, not per-input.** Bumping it lets you
   send more inputs concurrently, but does NOT raise the model's
   2048-token ceiling for any individual input (see gotcha 3). 8192 is a
   reasonable default; lower values force more round-trips.

7. **Changing `EMBED_MODEL` invalidates the index.** Different model =
   different vector space. Use `vcfbot update --force` (which calls
   `reset_collection` then re-embeds) — do NOT rmtree.

8. **Chunk IDs are text-stable** — `_make_id` in `chunk.py` hashes only
   `(source, text)`, *not* page numbers. This is deliberate: Broadcom
   occasionally re-paginates the master PDF without changing content
   (TOC insertions, layout reflow), and including pages in the hash
   would cascade-invalidate thousands of unchanged chunks. Pages live
   in metadata; `index_pdf_incremental` refreshes them in-place via
   `collection.update(metadatas=...)` for chunks whose text matched.
   Re-running `vcfbot index` is idempotent. Changing chunking
   parameters (e.g. `TARGET_TOKENS`) still produces different text
   slices → different hashes → still invalidates everything (a forced
   rebuild is the only path). Changing the hash function itself
   (this file) is a one-time migration: deploy + `update --force` once.

9. **Section detection uses the PDF outline, not font sizes.** DITA-OT
   PDFs (which Broadcom uses) flatten visual hierarchy at render time, so
   font-size heuristics produced empty sections. We walk
   `pymupdf.Document.get_toc()` instead — VCF 9.1 has ~6,995 outline
   entries, up to 6 levels deep, joined with ` › ` for display.

10. **Master PDF is monolithic and we don't serve it.** Broadcom publishes
    the entire VCF 9.1 doc set as one 184 MB PDF. Every section's
    "Download PDF" link on techdocs points at the same file. Source
    cards' "open PDF" link goes to **Broadcom's CDN** directly with a
    `#page=N` URL fragment — saves us serving bytes and gets us Cloudflare
    Range request caching for free.

11. **No app-level auth.** The FastAPI app is unauthenticated. Gate it
    at a reverse proxy (NPMplus, Caddy, etc.) with an access list or
    basic-auth before exposing anywhere meaningful.

12. **Container runs as uid 1000 (`vcfbot` user), not root.** The
    Dockerfile creates that user and `USER`s into it before `CMD`, so
    bind-mounted `data/` files are host-user-owned. On a host whose
    primary user has a different uid, change the Dockerfile's `1000` to
    match, or override at runtime via compose `user:`.

## Daily-update mechanism

Cron entry on the deployment host:
```
0 4 * * *  /path/to/vcfbot/scripts/daily-update.sh
```

The script execs `python -m vcfbot update` inside the running container. That
command:

1. Snapshots current per-source `sha256` from `data/pdfs/*.meta.json`.
2. Runs `fetch` (conditional GET — usually returns 304 in <1s).
3. If a source's sha changed, runs a **diff-aware update** via
   `index.py::index_pdf_incremental`: extracts + chunks the new PDF, queries
   chroma for existing IDs scoped to this source, embeds only chunks whose
   IDs aren't already present, and deletes orphans (IDs in chroma that
   aren't in the new chunk set). The running server keeps serving the
   whole time. Appends a `ChangelogEntry` with `chunks_added` /
   `chunks_removed` to `data/changelog.jsonl`.
4. `/api/changelog` surfaces those entries; the Changelog dialog (header
   button next to About) renders the last 20 with the +/− breakdown.

`update --force` keeps the old behavior — `reset_collection` via the
chromadb API (NOT rmtree, see gotcha 5), then full re-embed. Use only
when `EMBED_MODEL` or chunking parameters changed (different model =
different vector space → all old vectors invalid).

Per-run logs land in `/backup/logs/vcfbot-update-YYYYMMDD-HHMMSS.log`
(adjust the LOG_DIR var at the top of the script to taste). Retention is
30 days, pruned in-script via `find -mtime +N -delete`. Failure
summaries also append to a single growing tail-friendly errors log next
to it.

The reset-and-rebuild path runs WITHOUT stopping the server. chromadb uses
SQLite WAL so concurrent reads during the rebuild are safe; readers may
briefly see a partially-populated collection mid-reindex (it's 4am, no
users, no real-world impact).

## Known issue: full-rebuild throughput

A FULL reindex of VCF 9.1 at `TARGET_TOKENS=220` produces ~36k chunks. Each
chunk's WordPiece-tokenized size sits close to nomic's 2048 ceiling, so the
embed server can only process them ~1–2 at a time. With our current serial
`_embed_adaptive` loop, a fresh rebuild on a CPU-only embed server takes
hours; on Apple Silicon Metal via LM Studio, ~25 minutes.

**This only matters when `update --force` is invoked or `EMBED_MODEL`
changes.** Daily updates via the default diff-aware path embed a few dozen
chunks on a typical upstream change — seconds, not hours.

Future work to speed up forced rebuilds:

- Switch `_embed_adaptive` to use the async OpenAI client and issue ~10
  concurrent requests via `asyncio.gather` with a semaphore. Expected
  ~10× throughput on the same hardware.
- Or wire in a cross-encoder reranker and chunk at larger sizes again,
  relying on rerank to compensate for noisier dense retrieval. More
  speculative.

## When working on this project

- Prefer **config-driven additions** over restructuring. New PDFs are new
  rows in `sources.py`. New env knobs go in `config.py`. New embedder
  support extends the `_*_prefix` family in `index.py`.
- Keep the **chat ↔ embed endpoint split** clean. They route through
  different OpenAI client instances in `index.py::_client` and
  `chat.py` / `server.py`. Don't conflate them.
- Don't rebuild the heading heuristic. If sections look off, fix the TOC
  walker in `extract.py::_build_section_map`.
- If you must re-index, do it in a tracked background process. It is now
  slow (see "Known issue: reindex throughput").
- If you find yourself reaching for `shutil.rmtree(chroma_dir)`, stop and
  use `reset_collection` instead (gotcha 5).
- Bump `docker-compose.yml`'s `TARGET_TOKENS` only with a coordinated
  reindex; chunks of different sizes can't coexist meaningfully in the
  same collection.

## Style preferences

- Match existing file conventions: type hints, dataclasses, click for CLI,
  rich for terminal output.
- Frontend stays single-file-per-concern (`index.html` / `styles.css` /
  `app.js`) with no build step. Don't add npm.
- Tokens/CSS variables in `styles.css` are the source of truth for design;
  add new components by extending the token system, not by inlining values.
- Two dialogs in the header — About (how it works) and Changelog (what
  changed). They share `.about__*` styles by reusing the same class on
  the changelog dialog, with only content differing.
