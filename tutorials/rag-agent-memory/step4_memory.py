"""Step 4 — the same store, pointed the other way.

Steps 0-3 built an index ahead of time from a corpus and then only ever read it.
Agentic memory is the same Chroma collection, the same encoder, the same
nearest-neighbour lookup — except the agent writes to it as it goes.

Each turn does four things:

    1. embed the message, retrieve the k most relevant memories
    2. answer, with those memories in the system prompt
    3. make a second, cheap model call that extracts durable facts from the
       exchange as JSON
    4. embed those facts and write them back

That is the whole mechanism. What makes it feel like memory is step 1 running
against everything step 4 has ever written.

    flyte run --local step4_memory.py converse

The default script is three messages: you introduce yourself, you mention a
constraint, then you ask what it knows. Turn 3 has no special handling — it just
retrieves what turns 1 and 2 wrote.

Memory comes back as a `flyte.io.Dir`, so it outlives the run. Pass it to
another `converse` and the agent picks up where it left off.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Optional

import flyte
import flyte.io
import flyte.report

import llm
import render
from config import llm_env
from store import DEFAULT_EMBEDDING_MODEL, embed, load_encoder, new_work_dir, open_collection, retrieve

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = llm_env

MEMORY_COLLECTION = "agent_memory"

# A near-duplicate of something already stored teaches the agent nothing and
# crowds out the top-k. Anything this similar to an existing memory is dropped.
DEDUPE_THRESHOLD = 0.95

DEMO_SCRIPT = [
    "Hey, I'm Sage. I run a hacknight and I'm building demos on Flyte.",
    "I only have about 20 minutes per demo, so keep things short and runnable.",
    "What do you know about me so far?",
]

CHAT_SYSTEM = """You are an assistant with a persistent memory of this user.

The MEMORIES block holds things you have learned about them in earlier turns.
Use them when they are relevant — refer to them naturally, the way a person
would, rather than announcing that you retrieved them. If the memories are
empty or irrelevant, just answer normally. Keep replies to a few sentences."""

EXTRACTION_SYSTEM = """You extract durable facts about the user from one exchange.

Return each fact as a short, self-contained sentence that will still make sense
months from now, read on its own with no surrounding conversation.

Include: stable preferences, constraints, decisions, roles, projects, and
identity. Exclude: questions the user asked, small talk, anything about the
assistant, and anything true only right now. If the user asked a question and
revealed nothing new about themselves, return an empty list — that is the
common case and it is fine."""

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Durable facts about the user. Empty if none.",
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def _remember(collection, encoder, facts: list[str], turn: int) -> tuple[list[str], list[str]]:
    """Write new facts, skipping ones the store effectively already holds."""
    written: list[str] = []
    skipped: list[str] = []

    for fact in facts:
        fact = fact.strip()
        if not fact:
            continue

        vector = embed(encoder, [fact])[0]
        # Backend-agnostic: `nearest` already normalizes to cosine similarity,
        # so this threshold means the same thing on Chroma and on Qdrant.
        nearest = collection.nearest(vector, 1)
        if nearest and nearest[0].similarity >= DEDUPE_THRESHOLD:
            skipped.append(fact)
            continue

        collection.add(
            ids=[f"mem-{collection.count()}-{abs(hash(fact)) % 10**8}"],
            texts=[fact],
            vectors=[vector],
            metadatas=[{"source": f"turn {turn}", "title": "memory"}],
        )
        written.append(fact)

    return written, skipped


def _turn_html(n: int, message: str, reply: str, recalled, written, skipped) -> str:
    def bullets(items, empty):
        if not items:
            return f"<p class='empty'>{empty}</p>"
        return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"

    recalled_html = (
        render.chunk_cards(recalled, max_chars=300) if recalled
        else "<p class='empty'>Nothing in memory yet.</p>"
    )
    skipped_html = (
        f"<p class='empty'>Skipped as duplicates: {len(skipped)}</p>" if skipped else ""
    )
    return (
        f"<h2>Turn {n}</h2>"
        f"<div class='note'><b>User:</b> {html.escape(message)}</div>"
        f"<div class='answer'>{html.escape(reply)}</div>"
        f"<p class='sub'><b>Recalled before answering</b></p>{recalled_html}"
        f"<p class='sub'><b>Written after answering</b></p>"
        f"{bullets(written, 'Nothing worth remembering from this turn.')}{skipped_html}"
    )


@env.task(report=True)
async def converse(
    messages: Optional[list[str]] = None,
    memory_dir: Optional[flyte.io.Dir] = None,
    top_k: int = 5,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    store_backend: str = "chroma",
) -> flyte.io.Dir:
    """Run a short conversation, reading and writing memory each turn.

    Returns the memory directory. Feed it back in to continue the conversation
    in a later run — that is what makes this memory rather than a chat history.
    """
    script = messages or DEMO_SCRIPT

    # Start from prior memory if we were handed some, otherwise a fresh store.
    persist_dir = new_work_dir(f"memory_{store_backend}_")
    if memory_dir is not None:
        downloaded = Path(await memory_dir.download())
        for item in downloaded.iterdir():
            target = persist_dir / item.name
            if item.is_dir():
                __import__("shutil").copytree(item, target)
            else:
                __import__("shutil").copy2(item, target)

    collection = open_collection(str(persist_dir), MEMORY_COLLECTION, embedding_model, store_backend)
    encoder = load_encoder(embedding_model)
    started_with = collection.count()
    log.info(f"Memory opened with {started_with} facts")

    sections: list[str] = []
    total_written = 0

    for n, message in enumerate(script, start=1):
        # Logged before the model call, so the transcript reads in the order it
        # actually happened rather than as answers with no questions.
        log.info(f"\nTurn {n}")
        log.info(f"  you:    {message}")

        recalled = retrieve(collection, message, k=top_k, embedding_model=embedding_model)
        memory_block = (
            "\n".join(f"- {h.text}" for h in recalled) if recalled else "(nothing yet)"
        )

        reply = llm.answer(
            CHAT_SYSTEM,
            f"MEMORIES:\n{memory_block}\n\nUSER: {message}",
        )

        extracted = llm.extract(
            EXTRACTION_SYSTEM,
            f"USER SAID: {message}\n\nASSISTANT REPLIED: {reply}",
            FACTS_SCHEMA,
        )
        written, skipped = _remember(collection, encoder, extracted.get("facts", []), n)
        total_written += len(written)

        log.info(f"  agent:  {reply}")
        log.info(
            f"  memory: recalled {len(recalled)}, wrote {len(written)}"
            + (f", skipped {len(skipped)} duplicate(s)" if skipped else "")
        )
        sections.append(_turn_html(n, message, reply, recalled, written, skipped))

    await flyte.report.replace.aio(render.page(
        "Agent memory",
        f"{len(script)} turns · {llm.describe()}",
        render.stats(
            started_with=started_with,
            facts_written=total_written,
            memories_now=collection.count(),
            top_k=top_k,
        )
        + render.note(
            "Same Chroma collection, same encoder, same nearest-neighbour lookup "
            "as steps 0-3. The only difference is that <b>the agent writes to it</b>. "
            "Turn 3 has no special handling — it recalls turn 1 because turn 1 put "
            "something in the store that turn 3's question is near."
        )
        + "".join(sections),
    ))
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(str(persist_dir))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(converse)
    print(f"Memory run: {run.name}")
    print(f"  {run.url}")
