"""Step 5 — all three ideas in one interface.

Steps 1-4 each showed one piece in isolation. This puts them together: a chat
where every message retrieves from the doc index (step 1), gets answered with
that context (step 2), moves the star on a live projection (step 3), and writes
what it learns about you back into memory (step 4).

Two ways to run it.

Locally, no cluster needed — builds the index if you have not already, then
opens Gradio on http://localhost:7860:

    python step5_chat_app.py --local

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

from config import image
from store import DEFAULT_EMBEDDING_MODEL

COLLECTION_NAME = "docs"
MEMORY_COLLECTION = "agent_memory"
DEFAULT_TOP_K = 4

# Task that produced the index, as "<environment name>.<task name>".
INDEX_TASK = "rag-index.index"
LOCAL_MEMORY_DIR = "/tmp/agent_memory_chroma"


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
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)

state: dict = {}


# ── Startup: load the index, fit the projection once ──────────────────────────

def build_state(chroma_dir: str, collection_name: str, embedding_model: str) -> dict:
    """Everything expensive happens here, once, before the first message."""
    import numpy as np
    import umap

    from store import load_encoder, open_collection

    print(f"[startup] loading encoder {embedding_model}", flush=True)
    encoder = load_encoder(embedding_model)

    print(f"[startup] opening index at {chroma_dir}", flush=True)
    docs = open_collection(chroma_dir, collection_name, embedding_model)
    data = docs.get(include=["embeddings", "metadatas"])
    vectors = np.asarray(data["embeddings"], dtype="float32")

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
    os.makedirs(LOCAL_MEMORY_DIR, exist_ok=True)
    memory = open_collection(LOCAL_MEMORY_DIR, MEMORY_COLLECTION, embedding_model)

    print(f"[startup] ready — {docs.count()} chunks, {memory.count()} memories", flush=True)
    return {
        "encoder": encoder,
        "docs": docs,
        "memory": memory,
        "reducer": reducer,
        "coords": coords,
        "sources": [(m or {}).get("source", "unknown") for m in data["metadatas"]],
        "id_to_index": {cid: i for i, cid in enumerate(data["ids"])},
        "embedding_model": embedding_model,
    }


@env.on_startup
async def app_startup(chroma_dir: str, collection_name: str, embedding_model: str) -> None:
    state.update(build_state(chroma_dir, collection_name, embedding_model))


# ── The UI ────────────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr
    import numpy as np

    import llm
    import render
    from step3_visualize import _figure
    from step4_memory import CHAT_SYSTEM, EXTRACTION_SYSTEM, FACTS_SCHEMA, _remember
    from step2_rag_answer import GROUNDED_SYSTEM, _context_block
    from store import embed, retrieve

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
        stored = memory.get(include=["documents", "metadatas"])
        lines = [
            f"- {doc}  \n  <sub>{(meta or {}).get('source', '')}</sub>"
            for doc, meta in zip(stored["documents"], stored["metadatas"])
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
def app_server(chroma_dir: str, collection_name: str, embedding_model: str):
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def _run_locally(chroma_dir: str | None, share: bool = False) -> None:
    """Launch the same UI on your laptop, building the index first if needed."""
    if not chroma_dir:
        print("No --chroma-dir given; building the index locally (cached after the first run)…")
        from step0_index import index

        flyte.init()
        run = flyte.run(index)
        chroma_dir = run.outputs().o0.path
        print(f"Index at {chroma_dir}")

    state.update(build_state(chroma_dir, COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL))
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
        "--share", action="store_true",
        help="expose a public Gradio link — needed to reach the UI from Colab",
    )
    args = parser.parse_args()

    if args.local:
        _run_locally(args.chroma_dir, share=args.share)
    else:
        flyte.init_from_config()
        app = flyte.serve(env)
        print(f"Chat app deployed: {app.url}")
