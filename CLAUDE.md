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
  endpoint), anthropic SDK (optional Claude chat provider — embeddings stay
  local), FastAPI + uvicorn (web), click (CLI), rich (terminal UX).
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

- **Anthropic chat provider** (opt-in): set `CHAT_PROVIDER=anthropic` +
  `ANTHROPIC_API_KEY` to route ONLY the synthesis step to Claude via the
  native `anthropic` SDK. Embeddings stay local — the chat/embed split is
  preserved. `providers.py` is the only chat-completion call site; both
  `chat.py` and `server.py` go through it. Default `CHAT_PROVIDER=local`.

## Layout

```
src/vcfbot/
  sources.py        # PDF registry; vcf_source() templates URLs from one version string
  fetch.py          # downloader with Last-Modified caching; sends a Chrome UA
  extract.py        # pymupdf → per-page text + TOC-based section paths; atomic
                    #   table chunks (find_tables); xlsx → one ATOMIC per-appliance fact
  chunk.py          # tiktoken-aware chunker, preserves page ranges; atomic pages
                    #   (tables, per-appliance facts) emit as one indivisible chunk
  index.py          # embed + upsert; query / query_multi; component_manifest +
                    #   select_components (corpus-derived, embedding-ranked components)
  retrieve.py       # multi-query retrieval (single entry point for chat + server):
                    #   embedding component-selection for specific Qs; doc-inventory
                    #   anchoring for "all components" Qs; per-component sizing AND
                    #   classification sub-queries; round-robin merge; rerank+cut
  rerank.py         # optional cross-encoder rerank (Voyage /rerank); query-time,
                    #   no re-index; falls back to embedding order on any failure
  planner.py        # drives the Planning Workbook's own sizing formulas (formulas
                    #   lib) as a calculator — `vcfbot plan` / /api/plan
  providers.py      # chat-completion providers: local (OpenAI-compat) | anthropic
  changelog.py      # append-only JSONL log of corpus updates
  chat.py           # terminal RAG REPL
  server.py         # FastAPI: /api/status, /api/chat (SSE), /api/changelog, /api/plan
  config.py         # env-driven settings (CHAT_PROVIDER, ANTHROPIC_*, MULTI_QUERY,
                    #   RERANK_*, TABLE_CHUNKS, …)
  __init__.py       # click CLI (fetch / index / update / chat / serve / status)
  __main__.py       # `python -m vcfbot …` entry point for `docker exec`
  static/
    index.html      # web UI; chat|planner view tabs; status rail (chat/embed/
                    #   chunks/top-k/multi-q/rerank); About + Changelog dialogs
    styles.css      # design tokens + components
    app.js          # streaming SSE client, citation handling, theme + export;
                    #   status rail population; planner module
scripts/
  daily-update.sh   # cron target (runs `python -m vcfbot update` in the container)
  eval.py           # accuracy regression harness — fixed engineer-question
                    #   battery vs /api/chat; behavioral smoke checks, no
                    #   version-specific facts (stdlib only)
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
python scripts/eval.py                            # accuracy regression battery
python scripts/eval.py --quick                    # flagship question only
```

## Retrieval & synthesis architecture

Retrieval is the solvable-locally half; synthesis quality is model-bound.

- **Multi-query retrieval** (`retrieve.py`, gated by `MULTI_QUERY`, default on).
  A single dense query embedding sits in ONE neighborhood, so broad
  multi-component questions land in one neighborhood and starve the rest.
  Two modes:
  - **Specific question** → `index.select_components` ranks the
    corpus-derived component manifest (`index.component_manifest`, harvested
    from `… Detailed Design` / `… Model` TOC headings, deduped via
    `_contains_phrase`) by embedding cosine to the question; components within
    `_SCOPE_MARGIN` of the top fan out one sub-query each. One clear winner →
    plain single-query fallback.
  - **Breadth question** ("all/every component", matched by `_BREADTH_RE`) →
    narrowing is wrong and there's no single sizing table, so it ANCHORS ON
    THE DOC'S OWN INVENTORY: `_extract_inventory_components` gathers the
    management-components inventory text (fixed probes) and has the chat model
    extract the component names from that grounded text (a reliable copy task),
    then for EACH extracted component fans out TWO sub-queries: a **sizing**
    one (`"{c} — {question}"`) AND a **classification** one
    (`_classification_subquery`: "Is {c} required/optional/essential/always-
    deployed?"). Without the classification sub-query the pool is all sizing
    chunks and the model guesses required-vs-optional from the Planning
    Workbook's *selectability*, mislabeling essential components (e.g. VCF
    Operations) as optional. The classification sub-query pulls the docs' own
    "is this required" wording into the pool. TOC manifest is the fallback.
  - All sub-queries (+ the original, always sub-query #1) are **round-robin
    merged** in `index.query_multi` — interleave, NOT global distance sort, so
    a dominant neighborhood can't crowd out the others.
- **Cross-encoder reranking** (`rerank.py`, gated by `RERANK_ENABLED`, default
  off; Voyage `/rerank` by default). Bi-encoder dense retrieval embeds query and
  chunk separately, so the authoritative chunk (e.g. a classification table in a
  *security* blueprint) can sink below topically-closer-but-wrong chunks. When
  on, `retrieve()` over-retrieves a larger pool (`RERANK_TOP_N`, default 100),
  then `_finish` reranks query+chunk TOGETHER and cuts to the final size:
  `TOP_K` for single/specific Qs, `_BREADTH_KEEP` (24) for breadth so every
  component's sizing + the classification chunk survive. Query-time only, no
  re-index; any failure (no key, network) falls back to embedding order — rerank
  is a precision booster, never a hard dependency.
- **Component enumeration is corpus-derived, never hardcoded** — it tracks VCF
  versions. Hardcoding component names is the version-rot bias to avoid; the
  inventory probes only STEER retrieval to the right pages, the names come from
  the doc text. (SDDC Manager has no `… Detailed Design`/`… Model` section, so
  it's not in the TOC manifest — only the inventory-anchor path surfaces it.)
- **Required-vs-optional is a SYNTHESIS rule, not a pinned page.** Don't probe
  for the specific page/table that states a classification — Broadcom re-paginates
  and those rot (see gotcha 8). Two version-robust layers instead: (1) the
  per-component classification sub-query above (names come from live extraction,
  the query is a *question* so it pins no page); (2) a generic SYSTEM_PROMPT rule
  that keys off the docs' OWN words — *essential / core / mandatory / "always
  deployed"* → required; a sizing workbook letting you DESELECT a component is a
  capacity-modeling convenience, NOT an optional-for-support signal; an explicit
  "mandatory" statement outranks a workbook toggle or a capability description.
  Verified both directions on the SAME rule: VCF Operations → required (docs say
  mandatory/always-deployed), VCF Automation → optional (docs describe it as a
  Day-N deploy). Omitting a genuinely-optional component from a minimal answer
  is correct, not a bug — never force it in (that's the bias we avoid).
- **The synthesis ceiling.** With reliable retrieval, the remaining failures
  (dropping components, conflating sizing numbers across chunks, mis-applying
  required-vs-optional / min-vs-ceiling) are the chat model's. A small local
  model (qwen3-8b) is shaky here; `CHAT_PROVIDER=anthropic` (Sonnet/Opus, with
  adaptive thinking) fixes it on the SAME retrieved context. The system prompt
  encodes the behavioral rules (min-vs-ceiling, required-vs-optional grounded in
  the docs' own words, scenario-match for greenfield/upgrade/expansion, cite
  every claim) — it lives in BOTH `chat.py` and `server.py`; keep them in sync
  (a `diff` of the two SYSTEM_PROMPT blocks should be empty). `scripts/eval.py`
  is the regression check after any prompt/retrieval/model change.
- **Open retrieval gap: greenfield procedural questions.** "Prerequisites/steps
  to deploy a NEW management domain" embeds nearer the UPGRADE-framed "Deploy
  VCF Management Services" pages, so retrieval under-serves the greenfield
  prereqs (which DO exist — `VCF-MS-REQD-FIRST-*` DNS entries, the VCF Installer
  wizard prereqs). The scenario-match prompt rule makes the model say so
  honestly rather than pass off upgrade steps as greenfield, but the real fix is
  retrieval-side query decomposition for non-component procedural questions (not
  built; `_BREADTH_RE`-style fan-out only covers "all components" today).
- **Retrieval depth is bound by the chat model's context window.** Local
  qwen3-8b on Apollo is 12288-ctx, capping `TOP_K` ~20 and the breadth fan-out.
  A cloud model (Sonnet/Opus, ~1M ctx) lifts that — depth could be raised when
  `CHAT_PROVIDER=anthropic`. If you make retrieval depth larger, make it
  provider-aware so the local path doesn't overflow.

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

13. **Shell env shadows `.env` in docker-compose.** Compose resolves
    `${VAR}` interpolation from the shell environment FIRST, then `.env`.
    A stale `export ANTHROPIC_API_KEY=…` in your shell rc will override the
    key in `.env` and silently feed the container the wrong key (symptom:
    `401 invalid x-api-key` even after fixing `.env`). Fix: keep secrets only
    in `.env` (don't export them), or recreate with
    `env -u ANTHROPIC_API_KEY docker compose up -d`.

14. **Source URLs are templated from one `version` string.** `sources.py`
    `vcf_source("9.1")` builds the PDF url, web url, and name from the version
    (techdocs embeds it 4×). A minor bump is one edit; a major bump overrides
    the `series_dir` / `series_umbrella` slugs. Don't hand-write the URLs —
    that's the version-rot trap. Per-section HTML deep-links are deliberately
    NOT used (their slugs change every version); citations point at the PDF
    `#page=N`, which is stable and derivable from our own metadata.

15. **Haiku 4.5 400s on adaptive thinking.** `thinking: {type: "adaptive"}` is
    supported on Opus 4.x and Sonnet 4.6 but NOT Haiku 4.5 — sending it returns
    a 400. `providers.py::_anthropic_extra` gates it by model-name substring
    (`opus`/`sonnet`) so Haiku stays usable as the cheap default
    (`ANTHROPIC_MODEL=claude-haiku-4-5`, ~$0.009/q). Add the attr ONLY for models
    that support it; don't send a fixed `budget_tokens` (deprecated on 4.x).

16. **Rerank is a THIRD external provider (Voyage), distinct from chat & embed.**
    Keep the split clean: embeddings → `index.py::_client` (local), chat →
    `providers.py` (local OR Anthropic), rerank → `rerank.py` (Voyage `/rerank`).
    Never cross-wire them (Voyage has no chat/embeddings here; Anthropic has no
    embeddings). `RERANK_API_KEY` falls back to `VOYAGE_API_KEY`. Rerank failures
    degrade silently to embedding order — it's a precision booster, not a hard
    dependency, so a missing/invalid key never breaks retrieval.

17. **Required-vs-optional must not be pinned to pages.** A natural fix for
    component-classification misses is to probe for the specific page/table that
    states it. DON'T — those rot on re-pagination (gotcha 8).
    The signal enters retrieval via a per-component classification *question*
    sub-query (doc-derived names, no page) and is resolved by a generic
    SYSTEM_PROMPT rule keyed off the docs' words. See "Retrieval & synthesis
    architecture". Same lesson generalizes: steer with semantics, never citations.

18. **Workbook per-appliance sizing facts are ATOMIC chunks** (`extract_xlsx`
    sets `Page.atomic=True`). The xlsx statements are short, so the 220-token
    chunker would otherwise repack several appliances into one chunk — recreating
    the dense multi-appliance grid the workbook split apart, which makes the chat
    model CONFLATE numbers (attribute one appliance's GB/vCPU to another). Atomic
    = one appliance per chunk = clean attribution. Verified: it eliminated qwen3's
    cross-appliance conflation on the flagship sizing question (3/3 runs) and
    helps every model. Changing the xlsx chunking is a workbook-only re-index
    (`index_pdf_incremental` on the `.xlsx`) — the 9k-page PDF is untouched.
    NOTE: the planner reads the workbook directly via the `formulas` engine, NOT
    via these chunks, so chunking changes never affect the calculator.

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
- Keep the **chat ↔ embed split** clean. Embeddings always go through
  `index.py::_client` (local OpenAI-compatible). Chat goes through
  `providers.py` (local OR Anthropic, per `CHAT_PROVIDER`). Don't conflate
  them — never route embeddings to Anthropic (it has no embeddings API), and
  don't add a chat client outside `providers.py`.
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
