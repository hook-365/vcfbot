"""Interactive RAG chat REPL backed by LM Studio + chromadb."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .config import Settings, load_settings
from .index import query as vector_query

SYSTEM_PROMPT = """/nothink
You are vcfbot, a focused assistant for VMware Cloud Foundation documentation.

Rules:
- Answer ONLY from the provided context snippets. Quote or paraphrase the directly applicable wording.
- A snippet may contain multiple separate requirements (e.g. one for the main HCI cluster AND a different one for dedicated Storage Clusters). Read carefully and apply only the requirement that matches the user's question. Do NOT blend or average separate requirements.
- If the context does not contain the answer, say so plainly. Do not invent specifics.
- Cite every factual claim inline with the marker shown above each snippet, e.g. [vmware-cloud-foundation-9-1 p.42].
- Prefer concise, structured answers. Use bullet points for steps and lists.
"""


@dataclass
class Retrieved:
    text: str
    source: str
    page_start: int
    page_end: int
    section: str
    distance: float


def _retrieve(question: str, settings: Settings) -> list[Retrieved]:
    res = vector_query(question, settings)
    out: list[Retrieved] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            Retrieved(
                text=doc,
                source=meta.get("source", "?"),
                page_start=int(meta.get("page_start", 0)),
                page_end=int(meta.get("page_end", 0)),
                section=meta.get("section", "") or "",
                distance=float(dist),
            )
        )
    return out


def _cite(r: Retrieved) -> str:
    if r.page_start == r.page_end:
        return f"[{r.source} p.{r.page_start}]"
    return f"[{r.source} p.{r.page_start}-{r.page_end}]"


def _build_context(hits: list[Retrieved]) -> str:
    blocks: list[str] = []
    for h in hits:
        header = _cite(h)
        if h.section:
            header += f"  ({h.section})"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def run_repl(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    console = Console()
    console.print(
        Panel.fit(
            f"vcfbot — chat backend: [cyan]{settings.lm_studio_url}[/]\n"
            f"chat model: [cyan]{settings.chat_model}[/]  embed model: [cyan]{settings.embed_model}[/]\n"
            f"type [bold]/quit[/] to exit, [bold]/sources[/] after an answer to inspect retrieved chunks.",
            title="ready",
        )
    )
    client = OpenAI(base_url=settings.chat_base_url, api_key=settings.api_key)
    last_hits: list[Retrieved] = []
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            question = Prompt.ask("[bold green]you[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            return
        if question == "/sources":
            if not last_hits:
                console.print("[dim]no previous query yet[/]")
                continue
            for h in last_hits:
                console.print(
                    Panel(
                        h.text,
                        title=f"{_cite(h)}  dist={h.distance:.3f}  {h.section}",
                        border_style="blue",
                    )
                )
            continue

        hits = _retrieve(question, settings)
        last_hits = hits
        context = _build_context(hits)
        user_msg = (
            f"Context snippets (each prefixed with its citation marker):\n\n{context}\n\n"
            f"Question: {question}"
        )
        history.append({"role": "user", "content": user_msg})

        console.print("[bold magenta]vcfbot[/]:")
        stream = client.chat.completions.create(
            model=settings.chat_model,
            messages=history,
            stream=True,
            temperature=0.2,
        )
        collected: list[str] = []
        for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                collected.append(delta)
                console.print(delta, end="", soft_wrap=True)
        console.print()
        answer = "".join(collected)
        history.append({"role": "assistant", "content": answer})
        # render markdown view at end (a second pass; nice for tables/lists)
        if any(ch in answer for ch in ("|", "- ", "**", "#")):
            console.print(Markdown(answer))
