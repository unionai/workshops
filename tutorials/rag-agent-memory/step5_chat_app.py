"""Step 5 — all three ideas in one interface.

Steps 1-4 each showed one piece in isolation. This puts them together: a chat
where every message retrieves from the doc index (step 1), gets answered with
that context (step 2), moves the star on a live projection (step 3), and writes
what it learns about you back into memory (step 4).

Two ways to run it.

Locally, no cluster needed — opens Gradio on http://localhost:7860:

    python step5_chat_app.py --local
    python step5_chat_app.py --local --share          # reachable from Colab

This reuses step 0's index rather than rebuilding it. `index` and its tasks are
cached on their arguments, so once you have run step 0 the app resolves the same
Chroma directory in under a second and embeds nothing. It only builds an index
if none exists yet.

One thing to watch: the cache key includes the arguments, so if you ran step 0
with `--source flyte-docs`, pass the same `--source` here or you will build a
second index from the default corpus.

On a cluster, as a deployed app with the index mounted from a step 0 run:

    flyte create secret ANTHROPIC_API_KEY -p flytesnacks -d development
    python step5_chat_app.py

The deployed version mounts the Chroma directory through
`flyte.app.RunOutput`, so the app pod never rebuilds the index — it downloads
the artifact step 0 already produced. Pin a specific run with
`INDEX_RUN=<run-name>`, or leave it unset to take the latest successful one.

One caveat on the deployed version: memory lives on pod-local disk, so it is
lost when the pod scales down. Step 4's `flyte.io.Dir` is the durable answer —
see the README for how to wire the two together.
"""

from __future__ import annotations

import os

import flyte
import flyte.app

import llm
import render
from config import image
from step2_rag_answer import GROUNDED_SYSTEM, _context_block
from step3_visualize import _figure
from step4_memory import CHAT_SYSTEM, EXTRACTION_SYSTEM, FACTS_SCHEMA, _remember
from store import DEFAULT_EMBEDDING_MODEL, detect_backend, embed, load_encoder, open_collection, retrieve

COLLECTION_NAME = "docs"
MEMORY_COLLECTION = "agent_memory"
DEFAULT_TOP_K = 4

# Task that produced the index, as "<environment name>.<task name>".
INDEX_TASK = "rag-index.index"
# Per-backend, because the store directory records which engine built it — a
# single shared path would make switching --store fail the backend guard.
LOCAL_MEMORY_ROOT = "/tmp/agent_memory"


# ── App environment ───────────────────────────────────────────────────────────

_pinned_run = os.environ.get("INDEX_RUN")
_index_output = (
    flyte.app.RunOutput(type="directory", run_name=_pinned_run)
    if _pinned_run
    else flyte.app.RunOutput(type="directory", task_name=INDEX_TASK)
)

env = flyte.app.AppEnvironment(
    name="rag-agent-memory-chat",
    image=image,
    # The pod embeds one short query at a time and proxies to the model API.
    # The UMAP fit at startup is the only burst of real work.
    resources=flyte.Resources(cpu=2, memory="6Gi"),
    port=7860,
    requires_auth=False,
    secrets=[flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY")],
    parameters=[
        flyte.app.Parameter(
            name="chroma_dir", value=_index_output, download=True, env_var="CHROMA_DIR",
        ),
        flyte.app.Parameter(name="collection_name", value=COLLECTION_NAME),
        flyte.app.Parameter(name="embedding_model", value=DEFAULT_EMBEDDING_MODEL),
        # Must match the backend the mounted index was built with, or the store's
        # backend guard rejects it on startup. Set STORE_BACKEND before deploying.
        flyte.app.Parameter(
            name="store_backend",
            value=os.environ.get("STORE_BACKEND", "auto"),
            env_var="STORE_BACKEND",
        ),
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)

state: dict = {}


# ── Startup: load the index, fit the projection once ──────────────────────────

def build_state(chroma_dir: str, collection_name: str, embedding_model: str,
                store_backend: str = "chroma") -> dict:
    """Everything expensive happens here, once, before the first message."""
    import numpy as np
    import umap

    print(f"[startup] loading encoder {embedding_model}", flush=True)
    encoder = load_encoder(embedding_model)

    # "auto" means: ask the index which engine built it. The mounted RunOutput
    # could have come from a chroma or a qdrant step 0 and the app cannot know.
    if store_backend == "auto":
        store_backend = detect_backend(chroma_dir)
    print(f"[startup] opening index at {chroma_dir} (backend: {store_backend})", flush=True)
    docs = open_collection(chroma_dir, collection_name, embedding_model, store_backend)
    records = docs.all_records(with_vectors=True)
    vectors = np.asarray([r.vector for r in records], dtype="float32")

    print(f"[startup] fitting UMAP on {len(vectors)} vectors", flush=True)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(vectors) - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(vectors)

    # The agent's own store, separate from the doc index — read AND write.
    memory_dir = f"{LOCAL_MEMORY_ROOT}_{store_backend}"
    os.makedirs(memory_dir, exist_ok=True)
    memory = open_collection(memory_dir, MEMORY_COLLECTION, embedding_model, store_backend)

    print(f"[startup] ready — {docs.count()} chunks, {memory.count()} memories", flush=True)
    return {
        "encoder": encoder,
        "docs": docs,
        "memory": memory,
        "reducer": reducer,
        "coords": coords,
        "sources": [r.source for r in records],
        "id_to_index": {r.id: i for i, r in enumerate(records)},
        "embedding_model": embedding_model,
    }


@env.on_startup
async def app_startup(
    chroma_dir, collection_name: str, embedding_model: str, store_backend: str = "chroma",
) -> None:
    """Materialize the mounted index, then build the app's state.

    `chroma_dir` is deliberately untyped: locally it arrives as a `str` path, but
    on a cluster the `RunOutput` parameter materializes as a `flyte.io.Dir`
    object. Handing that straight to `Path()` fails with
    `TypeError: expected str, bytes or os.PathLike object, not Dir`, so both
    shapes get normalized to a local path here.
    """
    if not isinstance(chroma_dir, (str, os.PathLike)):
        chroma_dir = await chroma_dir.download()
    state.update(build_state(str(chroma_dir), collection_name, embedding_model, store_backend))


# ── The UI ────────────────────────────────────────────────────────────────────

def build_ui():
    # Only third-party imports are deferred here. Everything from this tutorial
    # is imported at module scope on purpose: Flyte's code bundler traces
    # *module-level* imports to decide which local files ship to the pod, so an
    # import hidden inside a function is missing at runtime —
    #     ModuleNotFoundError: No module named 'llm'
    # which only ever shows up on a cluster, never with --local.
    import gradio as gr
    import numpy as np

    def empty_figure():
        return _figure(state["coords"], state["sources"], [], [], (np.nan, np.nan))

    def chunk_markdown(hits) -> str:
        if not hits:
            return "_Retrieval is off, or nothing was returned._"
        blocks = []
        for hit in hits:
            preview = hit.text[:400].replace("\n", " ")
            blocks.append(
                f"**#{hit.rank} · `{hit.source}`** — similarity {hit.similarity:.3f}\n\n"
                f"{preview}…"
            )
        return "\n\n---\n\n".join(blocks)

    def memory_markdown() -> str:
        memory = state["memory"]
        if memory.count() == 0:
            return "_Nothing remembered yet. Tell it something about yourself._"
        lines = [
            f"- {r.text}  \n  <sub>{r.source}</sub>"
            for r in memory.all_records()
        ]
        return f"**{memory.count()} memories**\n\n" + "\n".join(lines)

    def respond(message, history, use_retrieval, use_memory, top_k):
        history = history or []
        if not (message or "").strip():
            return history, empty_figure(), chunk_markdown([]), memory_markdown(), ""

        model = state["embedding_model"]
        hits = (
            retrieve(state["docs"], message, k=int(top_k), embedding_model=model)
            if use_retrieval else []
        )
        memories = (
            retrieve(state["memory"], message, k=5, embedding_model=model)
            if use_memory else []
        )

        parts = []
        if memories:
            parts.append("MEMORIES:\n" + "\n".join(f"- {m.text}" for m in memories))
        if hits:
            parts.append(f"CONTEXT:\n{_context_block(hits)}")
        parts.append(f"QUESTION: {message}")

        system = GROUNDED_SYSTEM if use_retrieval else CHAT_SYSTEM
        reply = llm.answer(system, "\n\n".join(parts))

        # Learn something about the user from the exchange.
        if use_memory:
            try:
                extracted = llm.extract(
                    EXTRACTION_SYSTEM,
                    f"USER SAID: {message}\n\nASSISTANT REPLIED: {reply}",
                    FACTS_SCHEMA,
                )
                _remember(
                    state["memory"], state["encoder"],
                    extracted.get("facts", []), len(history) // 2 + 1,
                )
            except Exception as exc:  # never let memory-writing break the chat
                print(f"[memory] extraction failed: {exc}", flush=True)

        # Move the star.
        if hits:
            query_vector = np.asarray(embed(state["encoder"], [message]), dtype="float32")
            query_xy = state["reducer"].transform(query_vector)[0]
            indices = [state["id_to_index"][h.id] for h in hits if h.id in state["id_to_index"]]
            shown = [h for h in hits if h.id in state["id_to_index"]]
            figure = _figure(
                state["coords"], state["sources"], indices, shown, query_xy, message,
            )
        else:
            figure = empty_figure()

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        return history, figure, chunk_markdown(hits), memory_markdown(), ""

    with gr.Blocks(title="RAG + agent memory", fill_height=True) as demo:
        gr.Markdown(
            "## RAG and agent memory on Flyte\n"
            "Every message retrieves from the document index, answers from those "
            "chunks, moves the star on the projection, and writes what it "
            "learns about you into a second, writable store."
        )
        with gr.Row():
            with gr.Column(scale=5):
                # Gradio 6 dropped the `type` argument — the messages format
                # ({"role": ..., "content": ...}) is the only one now.
                chatbot = gr.Chatbot(height=440, label="Chat")
                box = gr.Textbox(
                    placeholder="Ask about the corpus, or tell it something about yourself…",
                    show_label=False,
                )
                with gr.Row():
                    use_retrieval = gr.Checkbox(value=True, label="Use retrieval")
                    use_memory = gr.Checkbox(value=True, label="Use memory")
                    top_k = gr.Slider(1, 10, value=DEFAULT_TOP_K, step=1, label="Top-k")
            with gr.Column(scale=6):
                plot = gr.Plot(label="Embedding space", value=empty_figure())
                with gr.Tabs():
                    with gr.Tab("Retrieved chunks"):
                        chunks_md = gr.Markdown(chunk_markdown([]))
                    with gr.Tab("Memories"):
                        memory_md = gr.Markdown(memory_markdown())

        box.submit(
            respond,
            [box, chatbot, use_retrieval, use_memory, top_k],
            [chatbot, plot, chunks_md, memory_md, box],
        )

    return demo


@env.server
def app_server(chroma_dir, collection_name: str, embedding_model: str,
               store_backend: str = "chroma"):
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def _run_locally(chroma_dir: str | None, share: bool = False, source: str = "workshops",
                 store_backend: str = "chroma") -> None:
    """Launch the same UI on your laptop, reusing step 0's index if it exists.

    This does not rebuild anything you have already built. `index` and its three
    tasks are cached on their arguments, so if you ran step 0 earlier — which the
    notebook does, several cells up — this resolves to the same Chroma directory
    in well under a second and no embedding happens.

    The catch is that the cache key includes the arguments. Run step 0 with
    `--source flyte-docs` and then start the app with defaults, and you get a
    *second* index built from the default corpus rather than the one you were
    just looking at. Hence `--source` here, so the two can be kept in step.
    """
    import time

    if not chroma_dir:
        from step0_index import index

        print(f"Looking for an existing '{source}' index from step 0…")
        started = time.time()
        flyte.init()
        run = flyte.run(index, source=source, store_backend=store_backend)
        chroma_dir = run.outputs().o0.path
        took = time.time() - started

        if took < 5:
            print(f"Reused step 0's index ({took:.1f}s, nothing re-embedded): {chroma_dir}")
        else:
            print(f"Built a new index in {took:.0f}s: {chroma_dir}")

    state.update(build_state(chroma_dir, COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, store_backend))
    build_ui().launch(
        # Colab has no localhost you can reach, so --share is the way in there.
        server_name="0.0.0.0" if share else "127.0.0.1",
        server_port=7860,
        share=share,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="run on this machine, no cluster")
    parser.add_argument("--chroma-dir", default=None, help="reuse a step 0 output directory")
    parser.add_argument(
        "--source", default="workshops",
        help="corpus to reuse or build — must match what you ran step 0 with",
    )
    parser.add_argument(
        "--store", default="chroma", choices=["chroma", "qdrant"],
        help="vector store backend — must match what you ran step 0 with",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="expose a public Gradio link — needed to reach the UI from Colab",
    )
    args = parser.parse_args()

    if args.local:
        _run_locally(args.chroma_dir, share=args.share, source=args.source,
                     store_backend=args.store)
    else:
        flyte.init_from_config()
        app = flyte.serve(env)
        print(f"Chat app deployed: {app.url}")
