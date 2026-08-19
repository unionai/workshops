"""Step 1 — search the index. No model anywhere.

Before letting an LLM near this, look at what retrieval actually is: you embed
the question with the same encoder that embedded the documents, and you ask the
store for the nearest vectors. That is it. There is no reasoning, no
understanding, and nothing that can decline to answer.

Which is why the similarity numbers matter. Ask something the corpus covers and
the top hit sits high. Ask something it does not cover and you still get four
chunks back — just with low scores. Retrieval never returns nothing, and a RAG
system that ignores those scores will happily hand the model garbage.

    flyte run --local step1_retrieve.py search --question "How do I use GRPO?"
    flyte run --local step1_retrieve.py search --question "What is the capital of France?"

This step calls step 0 as a subtask. Step 0's tasks are cached, so the index is
built once and every run after that starts instantly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import flyte
import flyte.report

import render
from config import index_env
from step0_index import index
from store import DEFAULT_EMBEDDING_MODEL, open_collection, retrieve

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = index_env


@env.task(report=True)
async def search(
    question: str = "How do I fine-tune a model with GRPO?",
    top_k: int = 4,
    source: str = "workshops",
    max_docs: int = 0,
    collection_name: str = "docs",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    """Retrieve the top-k chunks for a question and report them."""
    chroma_dir = await index(
        source=source, max_docs=max_docs,
        collection_name=collection_name, embedding_model=embedding_model,
    )

    collection = open_collection(
        str(Path(await chroma_dir.download())), collection_name, embedding_model,
    )
    hits = retrieve(collection, question, k=top_k, embedding_model=embedding_model)

    for hit in hits:
        log.info(f"  #{hit.rank}  {hit.similarity:.3f}  {hit.source}")

    top = hits[0].similarity if hits else 0.0
    verdict = (
        "The corpus covers this well." if top >= 0.75
        else "Weak match — the corpus probably does not cover this." if top < 0.55
        else "Middling match. Read the chunks before trusting an answer built on them."
    )

    await flyte.report.replace.aio(render.page(
        "Retrieved chunks",
        question,
        render.stats(
            chunks_in_index=collection.count(),
            returned=len(hits),
            top_similarity=f"{top:.3f}",
            embedding_model=embedding_model,
        )
        + render.note(
            f"<b>No language model was called.</b> This is a nearest-neighbour "
            f"lookup over {collection.count()} vectors. {verdict}"
        )
        + render.chunk_cards(hits),
    ))
    await flyte.report.flush.aio()

    lines = [f"#{h.rank} ({h.similarity:.3f}) {h.source}" for h in hits]
    return f"top similarity {top:.3f}\n" + "\n".join(lines)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(search)
    print(f"Search run: {run.name}")
    print(f"  {run.url}")
