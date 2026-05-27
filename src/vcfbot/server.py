"""FastAPI server: /api/status, /api/chat (SSE), static frontend."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from dataclasses import asdict

from .changelog import read_entries as read_changelog
from .config import load_settings
from .index import collection_count, query as vector_query
from .sources import SOURCE_BY_NAME

STATIC_DIR = Path(__file__).resolve().parent / "static"

SYSTEM_PROMPT = """/nothink
You are vcfbot, a focused assistant for VMware Cloud Foundation documentation.

Rules:
- Answer ONLY from the provided context snippets. Quote or paraphrase the directly applicable wording.
- A snippet may contain multiple separate requirements (e.g. one for the main HCI cluster AND a different one for dedicated Storage Clusters). Read carefully and apply only the requirement that matches the user's question. Do NOT blend or average separate requirements.
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
        return JSONResponse(
            {
                "chat_model": settings.chat_model,
                "embed_model": settings.embed_model,
                "lm_studio_url": settings.lm_studio_url,
                "collection_size": count,
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
                    page_start = int(meta.get("page_start", 0))
                    page_end   = int(meta.get("page_end", page_start))
                    source     = meta.get("source", "?")
                    section    = (meta.get("section") or "").strip()
                    src_meta   = SOURCE_BY_NAME.get(source)
                    pdf_url = (
                        f"{src_meta.url}#page={page_start}" if src_meta else None
                    )
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

                # 2) stream completion
                client = AsyncOpenAI(base_url=settings.chat_base_url, api_key=settings.api_key)
                stream_resp = await client.chat.completions.create(
                    model=settings.chat_model,
                    messages=messages,
                    stream=True,
                    temperature=0.2,
                )
                answer_parts: list[str] = []
                async for event in stream_resp:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta.content
                    if delta:
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

    return app


app = create_app()
