"""Step 3 — look at the vector space.

Everything so far has been numbers. This step draws them.

Each chunk in the index is a 384-dimensional vector. UMAP squashes those down to
two dimensions so they fit on a screen, keeping neighbours near neighbours. The
result is a map of the corpus where distance means "similar" — and the clusters
you see are topics nobody labelled.

Then the question gets embedded and pushed through the *same* fitted projection,
so it lands as a gold star in the same space. The top-k chunks light up in rank
colors matching the cards below.

    flyte run --local step3_visualize.py visualize --question "How do I use GRPO?"
    flyte run --local step3_visualize.py visualize --question "brain tumor segmentation"
    flyte run --local step3_visualize.py visualize --question "What is the capital of France?"

Ask those three in a row and watch the star move to a different neighbourhood
each time. On the third one the projection still has to place it somewhere, so
look at what lights up around it instead: four unrelated chunks scoring ~0.47,
which is what "the corpus does not cover this" actually looks like.

No API key needed. This step never calls a model.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import flyte
import flyte.io
import flyte.report

import render
from config import index_env
from step0_index import index
from store import DEFAULT_EMBEDDING_MODEL, embed, load_encoder, open_collection, retrieve

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = index_env


# ──────────────────────────────────────────────────────────────────────────────
# Fit the projection once, cache it
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto")
async def fit_projection(
    chroma_dir: flyte.io.Dir,
    collection_name: str = "docs",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> flyte.io.Dir:
    """Fit UMAP over every vector in the collection and save the fitted reducer.

    Cached, and that matters for more than speed. If you refit per question the
    whole cloud reshuffles between runs and the demo becomes unreadable — you
    want the map to hold still while the star moves across it.

    The fitted reducer is saved too, because projecting the query means pushing
    it through *this* fit rather than a new one.
    """
    import joblib
    import numpy as np
    import umap

    collection = open_collection(
        str(Path(await chroma_dir.download())), collection_name, embedding_model,
    )
    data = collection.get(include=["embeddings", "metadatas"])
    vectors = np.asarray(data["embeddings"], dtype="float32")
    log.info(f"Fitting UMAP on {vectors.shape[0]} vectors of {vectors.shape[1]} dims")

    reducer = umap.UMAP(
        n_components=2,
        # Local-vs-global tradeoff. Lower gives tighter, more numerous clusters.
        n_neighbors=min(n_neighbors, max(2, vectors.shape[0] - 1)),
        min_dist=min_dist,
        # Matches the normalized embeddings the store holds.
        metric="cosine",
        # Deterministic, so the map looks the same every cold start.
        random_state=random_state,
    )
    coords = reducer.fit_transform(vectors)

    out_dir = Path(tempfile.mkdtemp(prefix="umap_"))
    np.save(out_dir / "coords.npy", coords)
    joblib.dump(reducer, out_dir / "reducer.joblib")
    (out_dir / "meta.json").write_text(json.dumps({
        "ids": data["ids"],
        "sources": [(m or {}).get("source", "unknown") for m in data["metadatas"]],
    }))

    log.info(f"Projection fitted and cached at {out_dir}")
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Draw it
# ──────────────────────────────────────────────────────────────────────────────

def _figure(coords, sources, hit_indices, hits, query_xy):
    """Corpus in gray, retrieved chunks in rank colors, query as a gold star."""
    import plotly.graph_objects as go

    highlighted = set(hit_indices)
    background = [i for i in range(len(coords)) if i not in highlighted]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[float(coords[i][0]) for i in background],
        y=[float(coords[i][1]) for i in background],
        mode="markers",
        name=f"corpus ({len(background)} chunks)",
        marker=dict(size=6, color=render.MUTED_COLOR, opacity=0.75),
        text=[sources[i] for i in background],
        hovertemplate="%{text}<extra></extra>",
    ))

    for hit, idx in zip(hits, hit_indices):
        color = render.RANK_COLORS[(hit.rank - 1) % len(render.RANK_COLORS)]
        preview = hit.text[:160].replace("\n", " ")
        fig.add_trace(go.Scatter(
            x=[float(coords[idx][0])],
            y=[float(coords[idx][1])],
            mode="markers",
            name=f"#{hit.rank}  {hit.similarity:.3f}",
            marker=dict(size=15, color=color,
                        line=dict(width=1.5, color="rgba(0,0,0,.45)")),
            text=[f"#{hit.rank} · {hit.source}<br>{preview}…"],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[float(query_xy[0])], y=[float(query_xy[1])],
        mode="markers", name="your question",
        marker=dict(size=22, color=render.QUERY_COLOR, symbol="star",
                    line=dict(width=1.5, color="rgba(0,0,0,.55)")),
        hovertemplate="your question<extra></extra>",
    ))

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


@env.task(report=True)
async def visualize(
    question: str = "How do I fine-tune a model with GRPO?",
    top_k: int = 4,
    source: str = "workshops",
    max_docs: int = 0,
    collection_name: str = "docs",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    """Project the corpus, place the question in it, and report the picture."""
    import joblib
    import numpy as np

    chroma_dir = await index(
        source=source, max_docs=max_docs,
        collection_name=collection_name, embedding_model=embedding_model,
    )
    projection_dir = await fit_projection(
        chroma_dir, collection_name=collection_name, embedding_model=embedding_model,
    )

    proj_path = Path(await projection_dir.download())
    coords = np.load(proj_path / "coords.npy")
    reducer = joblib.load(proj_path / "reducer.joblib")
    meta = json.loads((proj_path / "meta.json").read_text())
    id_to_index = {chunk_id: i for i, chunk_id in enumerate(meta["ids"])}

    collection = open_collection(
        str(Path(await chroma_dir.download())), collection_name, embedding_model,
    )
    hits = retrieve(collection, question, k=top_k, embedding_model=embedding_model)
    hit_indices = [id_to_index[h.id] for h in hits if h.id in id_to_index]
    hits = [h for h in hits if h.id in id_to_index]

    # The query goes through the *fitted* reducer, not a new one — that is what
    # puts it in the same coordinate system as the cloud.
    encoder = load_encoder(embedding_model)
    query_vector = np.asarray(embed(encoder, [question]), dtype="float32")
    query_xy = reducer.transform(query_vector)[0]

    fig = _figure(coords, meta["sources"], hit_indices, hits, query_xy)
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    top = hits[0].similarity if hits else 0.0
    await flyte.report.replace.aio(render.page(
        "Where the question landed",
        question,
        render.stats(
            chunks_plotted=len(coords),
            highlighted=len(hits),
            top_similarity=f"{top:.3f}",
            projection="UMAP 384d → 2d",
        )
        + chart_html
        + render.note(
            "Gray dots are every chunk in the index. Colored dots are the ones "
            "retrieved for this question, numbered to match the cards below. The "
            "gold star is the question itself, pushed through the same fitted "
            "projection.<br><br>"
            "<b>Read the neighbourhood, not the distance.</b> UMAP has to put an "
            "out-of-corpus question <i>somewhere</i>, and it will happily drop it "
            "next to whatever is least unlike it — so a lonely-looking star is not "
            "the tell. The tell is that the lit-up chunks have nothing to do with "
            "each other or with what you asked, and the similarity scores are low. "
            "The numbers are ground truth; the map only shows you which "
            "neighbourhood they came from."
        )
        + "<h2>The retrieved chunks</h2>"
        + render.chunk_cards(hits),
    ))
    await flyte.report.flush.aio()

    return f"plotted {len(coords)} chunks, top similarity {top:.3f}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(visualize)
    print(f"Visualize run: {run.name}")
    print(f"  {run.url}")
