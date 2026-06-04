"""FastAPI server: /api/status, /api/chat (SSE), static frontend."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dataclasses import asdict

from .changelog import read_entries as read_changelog
from .config import load_settings
from .index import collection_count
from .providers import stream_chat_async
from .retrieve import retrieve as vector_query
from .sources import SOURCE_BY_NAME

STATIC_DIR = Path(__file__).resolve().parent / "static"

SYSTEM_PROMPT = """/nothink
You are vcfbot, a focused assistant for VMware Cloud Foundation documentation.

Rules:
- Answer ONLY from the provided context snippets. Quote or paraphrase the directly applicable wording.
- A snippet may contain multiple separate requirements (e.g. one for the main HCI cluster AND a different one for dedicated Storage Clusters). Read carefully and apply only the requirement that matches the user's question. Do NOT blend or average separate requirements.
- Match the user's SCENARIO. The docs cover greenfield (a NEW fleet / first deployment / initial bring-up), upgrade (moving an EXISTING environment to a new version), and expansion (adding an instance, domain, or cluster to an existing fleet) — and the SAME topic (e.g. "deploy management services", "prerequisites", "deploy a component") is documented SEPARATELY for each. Work out which scenario the question is about and apply ONLY the matching guidance: a step or prerequisite written for an upgrade is NOT the answer to a new-deployment question, and vice versa. A snippet that introduces itself with the other scenario's framing (e.g. "to continue with the upgrade…") does not answer a greenfield question. If the context only covers a different scenario than the one asked, say so plainly and give what greenfield/new-deployment guidance the context does contain, rather than presenting the wrong-scenario steps as the answer.
- When asked for the smallest / minimum / cheapest option and a snippet lists several sizes or counts (e.g. Small/Medium/Large, or a node count), pick the LOWEST the documentation permits for the user's context — never default to a middle size. If the smallest option is marked lab / proof-of-concept-only or "not for production", say so explicitly and give BOTH the absolute smallest size AND the smallest production-supported size, using the doc's own wording. Distinguish a size CEILING (a stated maximum or limit, e.g. "FT limits the appliance to medium") from a MINIMUM — never report a stated maximum as if it were the minimum.
- For a MINIMAL / smallest / simple deployment, separate REQUIRED components from OPTIONAL ones using the DOCUMENTATION'S OWN language — not a sizing tool's selectability. Treat a component as REQUIRED if the docs call it essential / core / mandatory / "always deployed", or list it among the components the initial / base management domain deploys. Treat a component as OPTIONAL only when the docs explicitly say it is optional / not required / can be omitted for that profile. A planning or sizing WORKBOOK letting you DESELECT a component is a capacity-modeling convenience — it does NOT make that component optional for a supported deployment. When sources conflict, an explicit "mandatory / required / essential" statement outranks a workbook toggle or a feature-capability description; do NOT downgrade a component to optional merely because it is described in terms of the "capabilities" or "services" it adds. State the minimum NODE COUNT per component (e.g. a single-node appliance where the simple/non-HA model permits one), not just the appliance size.
- Default profile: most deployments use the SIMPLE (non-HA) model with the smallest supported footprint — typically a minimal ~3-node management cluster. Unless the user explicitly asks about High Availability or a larger profile, assume that simple, small-footprint context and the smallest supported sizes / single-node counts the Simple model permits.
- If the context does not contain the answer, say so plainly. Do not invent specifics.
- Cite every factual claim inline with the marker shown above each snippet, e.g. [vmware-cloud-foundation-9-1 p.42].
- Prefer concise, structured answers. Use bullet points for steps and lists.
"""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []


class PlanRequest(BaseModel):
    # Friendly input names from planner.INPUT_CELLS; missing keys fall back to
    # the smallest-sensible defaults.
    inputs: dict = {}
    # Optional workload domains: each {vcenter_size, nsx_model, nsx_size}. Their
    # management appliances (vCenter + NSX) add to the management-domain footprint.
    workload_domains: list = []


def _cite_tag(source: str, page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"[{source} p.{page_start}]"
    return f"[{source} p.{page_start}-{page_end}]"


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="vcfbot", docs_url=None, redoc_url=None)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Source PDFs link directly to Broadcom's CDN (see `pdf_url` in /api/chat).
    # We don't serve them ourselves — saves bandwidth and offloads delivery to
    # Cloudflare. PDFs on disk under `data/pdfs/` are still used for indexing.

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/changelog")
    async def changelog(limit: int = 50) -> JSONResponse:
        path = settings.chroma_dir.parent / "changelog.jsonl"
        entries = read_changelog(path, limit=limit)
        return JSONResponse({"entries": [asdict(e) for e in entries]})

    @app.get("/api/status")
    async def status() -> JSONResponse:
        settings = load_settings()
        try:
            count = collection_count(settings)
        except Exception:  # noqa: BLE001
            count = 0
        # Report the model that actually answers, given the active provider.
        active_chat_model = (
            settings.anthropic_model
            if settings.chat_provider == "anthropic"
            else settings.chat_model
        )
        return JSONResponse(
            {
                "chat_provider": settings.chat_provider,
                "chat_model": active_chat_model,
                "embed_model": settings.embed_model,
                "lm_studio_url": settings.lm_studio_url,
                "collection_size": count,
                # Retrieval/synthesis knobs surfaced in the header status rail.
                "top_k": settings.top_k,
                "multi_query": settings.multi_query,
                "multi_query_max": settings.multi_query_max,
                "rerank_enabled": settings.rerank_enabled,
                "rerank_model": settings.rerank_model if settings.rerank_enabled else None,
                "rerank_top_n": settings.rerank_top_n,
            }
        )

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        settings = load_settings()

        async def stream() -> AsyncIterator[bytes]:
            try:
                # 1) retrieve
                res = await asyncio.to_thread(vector_query, req.question, settings)
                docs  = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]

                hits: list[dict] = []
                context_blocks: list[str] = []
                for doc, meta, dist in zip(docs, metas, dists):
                    meta = meta or {}  # chroma can hand back None metadata; never let .get crash a query
                    page_start = int(meta.get("page_start", 0))
                    page_end   = int(meta.get("page_end", page_start))
                    source     = meta.get("source", "?")
                    section    = (meta.get("section") or "").strip()
                    src_meta   = SOURCE_BY_NAME.get(source)
                    # PDFs get a #page=N deep-link; the xlsx workbook has no
                    # pages, so link to the asset itself.
                    if not src_meta:
                        pdf_url = None
                    elif src_meta.kind == "xlsx":
                        pdf_url = src_meta.url
                    else:
                        pdf_url = f"{src_meta.url}#page={page_start}"
                    hits.append({
                        "source":     source,
                        "page_start": page_start,
                        "page_end":   page_end,
                        "section":    section,
                        "distance":   float(dist),
                        "text":       doc,
                        "pdf_url":    pdf_url,
                        "web_url":    src_meta.web_url if src_meta else None,
                    })
                    header = _cite_tag(source, page_start, page_end)
                    if section:
                        header += f"  ({section})"
                    context_blocks.append(f"{header}\n{doc}")

                yield _sse("sources", {"hits": hits})

                context = "\n\n---\n\n".join(context_blocks)
                messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in req.history:
                    if m.role in ("user", "assistant") and m.content:
                        messages.append({"role": m.role, "content": m.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Context snippets (each prefixed with its citation marker):\n\n"
                            f"{context}\n\nQuestion: {req.question}"
                        ),
                    }
                )

                # 2) stream completion (local llama.cpp or Anthropic, per CHAT_PROVIDER)
                answer_parts: list[str] = []
                async for delta in stream_chat_async(settings, messages):
                    answer_parts.append(delta)
                    yield _sse("token", {"text": delta})

                yield _sse("done", {"answer": "".join(answer_parts)})
            except Exception as exc:  # noqa: BLE001
                yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Sizing calculator (Planner tab) ────────────────────────────────────
    @app.on_event("startup")
    async def _warm_planner() -> None:
        # Kick the ~60s formula-graph compile in the background so the first
        # user calc is ~4s, not ~60s. No-op if the workbook isn't present.
        try:
            from .planner import warm

            warm()
        except Exception:  # noqa: BLE001
            pass

    @app.get("/api/plan/options")
    async def plan_options() -> JSONResponse:
        from .planner import is_ready, options

        return JSONResponse({**options(), "ready": is_ready()})

    @app.post("/api/plan")
    async def plan(req: PlanRequest) -> JSONResponse:
        from .planner import DEFAULTS, compute

        # Start from smallest-sensible defaults; user inputs override.
        merged = {**DEFAULTS, **(req.inputs or {})}
        try:
            # compute() compiles on first call (~60s) then ~4s; off the loop.
            result = await asyncio.to_thread(
                compute, merged, req.workload_domains or []
            )
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}"}, status_code=500
            )
        return JSONResponse(result.to_dict())

    return app


app = create_app()
