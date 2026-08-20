"""Step 2 — the "G" in RAG.

Step 1 found the chunks. This step pastes them into the prompt and asks Claude
to answer using only those, citing them as [#1], [#2].

That is the entire trick. RAG is not an architecture, it is a prompt with
freshly-retrieved text in it. Everything hard about RAG is upstream: what you
chunked, how you embedded it, and whether the nearest neighbours were any good.

The flag worth playing with is `--no-use_retrieval`, which asks the same question
with no context at all. (Flyte turns a `bool` parameter into a click flag pair,
so it is `--no-use_retrieval`, not `--use_retrieval false`.)

    flyte run --local step2_rag_answer.py answer \
        --question "What does the code-mode tutorial teach?"
    flyte run --local step2_rag_answer.py answer \
        --question "What does the code-mode tutorial teach?" --no-use_retrieval

The second one is the point of the exercise. You get one of two things, and
which one you get is luck: a confident answer about the wrong "code mode"
entirely, or a hedge listing several things it might be. Neither is checkable,
and neither tells you which parts came from anywhere real. Retrieval is what
replaces both with an answer whose every claim points at a file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import flyte
import flyte.report

import llm
import render
from config import llm_env
from step0_index import index
from store import DEFAULT_EMBEDDING_MODEL, open_collection, retrieve

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = llm_env

GROUNDED_SYSTEM = """You answer questions using only the CONTEXT provided.

Rules:
- Cite the chunks you used as [#1], [#2], matching the numbers in the context.
- If the context does not contain the answer, say so plainly. Do not fill the
  gap from general knowledge, and do not apologise at length.
- Prefer quoting specifics (commands, file names, numbers) over paraphrasing.
- Be concise. Three or four sentences is usually enough."""

UNGROUNDED_SYSTEM = """Answer the question as best you can from what you already know.
Be concise. Three or four sentences is usually enough."""


def _context_block(hits) -> str:
    return "\n\n".join(
        f"[#{h.rank}] (source: {h.source})\n{h.text}" for h in hits
    )


@env.task(report=True)
async def answer(
    question: str = "What does the code-mode tutorial teach?",
    top_k: int = 4,
    use_retrieval: bool = True,
    source: str = "workshops",
    max_docs: int = 0,
    collection_name: str = "docs",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    store_backend: str = "chroma",
    chunking: str = "structural",
) -> str:
    """Answer a question, with or without retrieved context."""
    hits = []
    chunks_in_index = 0

    if use_retrieval:
        store_dir = await index(
            source=source, max_docs=max_docs,
            collection_name=collection_name, embedding_model=embedding_model,
            store_backend=store_backend, chunking=chunking,
        )
        collection = open_collection(
            str(Path(await store_dir.download())), collection_name, embedding_model,
            store_backend,
        )
        chunks_in_index = collection.count()
        hits = retrieve(collection, question, k=top_k, embedding_model=embedding_model)

        prompt = f"CONTEXT:\n{_context_block(hits)}\n\nQUESTION: {question}"
        system = GROUNDED_SYSTEM
    else:
        prompt = question
        system = UNGROUNDED_SYSTEM

    log.info(f'\nQuestion: "{question}"')
    if use_retrieval:
        log.info(f"Context:  {len(hits)} chunks, top similarity {hits[0].similarity:.3f}" if hits else "Context:  none")
    else:
        log.info("Context:  none — retrieval is off")
    log.info(f"Asking {llm.describe()}…")

    reply = llm.answer(system, prompt)
    log.info(f"\nAnswer: {reply}")

    top = hits[0].similarity if hits else 0.0
    if use_retrieval:
        heading, subtitle = "Grounded answer", question
        body = (
            render.stats(
                model=llm.describe(),
                chunks_in_index=chunks_in_index,
                context_chunks=len(hits),
                top_similarity=f"{top:.3f}",
            )
            + f"<div class='answer'>{reply}</div>"
            + "<h2>What the model was given</h2>"
            + render.note(
                "These chunks, and nothing else, were pasted into the prompt. "
                "Every [#N] in the answer above points at one of them — so you "
                "can open the source file and check it."
            )
            + render.chunk_cards(hits)
        )
    else:
        heading, subtitle = "Ungrounded answer", question
        body = (
            render.stats(model=llm.describe(), context_chunks=0, retrieval="off")
            + render.note(
                "<b>Retrieval was disabled.</b> The model answered from training "
                "data alone. Compare this with the grounded run: the interesting "
                "failure is not a refusal, it is a confident answer with nothing "
                "behind it."
            )
            + f"<div class='answer'>{reply}</div>"
        )

    await flyte.report.replace.aio(render.page(heading, subtitle, body))
    await flyte.report.flush.aio()
    return reply


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(answer)
    print(f"Answer run: {run.name}")
    print(f"  {run.url}")
