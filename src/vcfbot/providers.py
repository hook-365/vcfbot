"""Chat-completion providers for the synthesis step.

Local (OpenAI-compatible llama.cpp) is the default; Anthropic is opt-in via
CHAT_PROVIDER=anthropic. **Embeddings are unaffected** — they always run through
the local OpenAI-compatible client in index.py. This module is the chat side of
the chat/embed split, never the embed side.

Both providers honor one streaming contract: given the OpenAI-style message list
the app already builds (`[{role:"system"}, {role:"user"/"assistant"}...]`), yield
the answer text in chunks. Callers don't branch on provider.

The Anthropic path uses the **native `anthropic` SDK** (not an OpenAI-compatible
shim): `system` is a separate top-level argument, messages carry only
user/assistant turns, there is no `temperature` on Opus 4.8/4.7, and adaptive
thinking is on (this is exactly the min-vs-ceiling / required-vs-optional
reasoning the local 8B was weak at). `text_stream` yields only answer text —
thinking blocks are excluded, so the SSE stream stays clean.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator

from .config import Settings


def _split_system(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split OpenAI-style messages into (anthropic_system_blocks, user/assistant turns).

    Concatenates any `system` messages, strips the qwen-only `/nothink` directive
    (meaningless to Claude), and wraps the system text in a cache-controlled block.
    The cache_control is future-proofing — it only engages once the system prompt
    exceeds Opus's ~4096-token minimum cacheable prefix; below that it silently
    no-ops (no write premium), and the per-query retrieved context isn't reusable
    across queries anyway.
    """
    system_parts: list[str] = []
    turns: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    system_text = "\n\n".join(p for p in system_parts if p).strip()
    if system_text.startswith("/nothink"):
        system_text = system_text[len("/nothink"):].lstrip()
    system_blocks = (
        [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        if system_text
        else []
    )
    return system_blocks, turns


def _anthropic_extra(model: str) -> dict:
    """Adaptive thinking is supported on Opus 4.x and Sonnet 4.6, but NOT on
    Haiku 4.5 (sending it 400s). Gate by model so Haiku works as a cheap option.
    """
    m = (model or "").lower()
    if "opus" in m or "sonnet" in m:
        return {"thinking": {"type": "adaptive"}}
    return {}


def stream_chat(settings: Settings, messages: list[dict]) -> Iterator[str]:
    """Synchronous streaming chat (CLI REPL). Yields answer-text chunks."""
    if settings.chat_provider == "anthropic":
        import anthropic

        system, turns = _split_system(messages)
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system=system,
            messages=turns,
            **_anthropic_extra(settings.anthropic_model),
        ) as stream:
            for text in stream.text_stream:
                yield text
        return

    from openai import OpenAI

    client = OpenAI(base_url=settings.chat_base_url, api_key=settings.api_key)
    stream = client.chat.completions.create(
        model=settings.chat_model, messages=messages, stream=True, temperature=0.2
    )
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta


async def stream_chat_async(
    settings: Settings, messages: list[dict]
) -> AsyncIterator[str]:
    """Asynchronous streaming chat (server SSE). Yields answer-text chunks."""
    if settings.chat_provider == "anthropic":
        import anthropic

        system, turns = _split_system(messages)
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        async with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system=system,
            messages=turns,
            **_anthropic_extra(settings.anthropic_model),
        ) as stream:
            async for text in stream.text_stream:
                yield text
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=settings.chat_base_url, api_key=settings.api_key)
    stream = await client.chat.completions.create(
        model=settings.chat_model, messages=messages, stream=True, temperature=0.2
    )
    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            yield delta
