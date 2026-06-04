"""Cross-encoder reranking via an API (Voyage by default).

Bi-encoder dense retrieval (nomic) embeds query and chunk separately, so a
chunk whose section framing differs from the query (e.g. the required-vs-optional
classification table living in a *security* blueprint) sinks below topically-
closer-but-wrong chunks. A cross-encoder reads query+chunk TOGETHER and reorders
by true relevance, floating the authoritative chunk to the top. This is
query-time only — no re-embed, no index change.

Default endpoint is Voyage's `/rerank` (rerank-2.5-lite: cheap, 200M-token free
tier, sub-200ms). The request shape is the common one (query, documents, model,
top_k) so any compatible reranker works by pointing RERANK_BASE_URL elsewhere.
Any failure (no key, network, bad response) returns None so the caller falls
back to the original embedding order — reranking is a precision booster, never a
hard dependency.
"""

from __future__ import annotations

import httpx

from .config import Settings


def rerank_order(query: str, documents: list[str], settings: Settings) -> list[int] | None:
    """Return document indices in reranked (best-first) order, or None on failure."""
    if not settings.rerank_api_key or not documents:
        return None
    try:
        resp = httpx.post(
            settings.rerank_base_url.rstrip("/") + "/rerank",
            headers={
                "Authorization": f"Bearer {settings.rerank_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "documents": documents,
                "model": settings.rerank_model,
                "top_k": len(documents),
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        order = [
            int(item["index"])
            for item in sorted(
                data, key=lambda d: d.get("relevance_score", 0.0), reverse=True
            )
            if isinstance(item.get("index"), int) and 0 <= item["index"] < len(documents)
        ]
        return order or None
    except Exception:  # noqa: BLE001 — reranking must never break retrieval
        return None
