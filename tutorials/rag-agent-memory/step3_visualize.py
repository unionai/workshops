"""Step 3 — look at the vector space.

Everything so far has been numbers. This step draws them.

Each chunk in the index is a 384-dimensional vector. UMAP squashes those down to
two dimensions so they fit on a screen, keeping neighbours near neighbours. The
result is a map of the corpus where distance means "similar" — and the clusters
you see are topics nobody labelled.

Then the question gets embedded and pushed through the *same* fitted projection,
so it lands as an orange star in the same space. The retrieved chunks light up as
numbered blue dots — darker means a better match — matching the cards below.

    flyte run --local step3_visualize.py visualize --question "How do I use GRPO?"
    flyte run --local step3_visualize.py visualize --question "brain tumor segmentation"
    flyte run --local step3_visualize.py visualize --question "Who won the 2022 FIFA World Cup?"

Ask those three in a row and watch the star move to a different neighbourhood
each time. On the third one the projection still has to place it somewhere, so
look at what lights up around it instead: four unrelated chunks scoring ~0.47,
which is what "the corpus does not cover this" actually looks like.

The terminal prints only a one-line summary — the chart lives in the HTML report.
From a shell, `python open_report.py` opens the newest one; in the notebook,
`show_latest()` renders it inline.

No API key needed. This step never calls a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import flyte
import flyte.io
import flyte.report

import render
from config import index_env
from step0_index import index
from store import DEFAULT_EMBEDDING_MODEL, embed, load_encoder, new_work_dir, open_collection, retrieve

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
    store_backend: str = "chroma",
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
        store_backend,
    )
    records = collection.all_records(with_vectors=True)
    vectors = np.asarray([r.vector for r in records], dtype="float32")
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

    out_dir = new_work_dir("umap_")
    np.save(out_dir / "coords.npy", coords)
    joblib.dump(reducer, out_dir / "reducer.joblib")
    (out_dir / "meta.json").write_text(json.dumps({
        "ids": [r.id for r in records],
        "sources": [r.source for r in records],
    }))

    log.info(f"Projection fitted and cached at {out_dir}")
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Draw it
# ──────────────────────────────────────────────────────────────────────────────

def _figure(coords, sources, hit_indices, hits, query_xy, question: str = ""):
    """Corpus in gray, retrieved chunks on a rank ramp, the question as a star.

    Rank is an ordered magnitude, so it gets a sequential single-hue ramp with
    the darkest step as rank 1. Adjacent steps of a ramp are necessarily close,
    so the rank number is drawn *inside* each marker — color never carries the
    rank on its own.
    """
    import plotly.graph_objects as go

    highlighted = set(hit_indices)
    background = [i for i in range(len(coords)) if i not in highlighted]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[float(coords[i][0]) for i in background],
        y=[float(coords[i][1]) for i in background],
        mode="markers",
        name=f"corpus ({len(background)} chunks)",
        marker=dict(size=5, color=render.MUTED_COLOR, opacity=0.55),
        text=[sources[i] for i in background],
        hovertemplate="%{text}<extra></extra>",
    ))

    for hit, idx in zip(hits, hit_indices):
        preview = hit.text[:160].replace("\n", " ")
        fig.add_trace(go.Scatter(
            x=[float(coords[idx][0])],
            y=[float(coords[idx][1])],
            # The number is the point of markers+text: two adjacent chunks from
            # one document have near-identical embeddings and land on top of
            # each other, so without labels you count three dots and never learn
            # the fourth was underneath.
            mode="markers+text",
            name=f"#{hit.rank}  ·  {hit.similarity:.3f}  ·  {hit.source.split('/')[-2] if '/' in hit.source else hit.source}",
            marker=dict(
                size=19, color=render.rank_color(hit.rank), opacity=0.92,
                line=dict(width=1.5, color="#fcfcfb"),  # surface ring
            ),
            text=[str(hit.rank)],
            textposition="middle center",
            textfont=dict(color=render.rank_text_color(hit.rank), size=11),
            hovertext=[f"#{hit.rank} · {hit.source}<br>similarity {hit.similarity:.3f}<br>{preview}…"],
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[float(query_xy[0])], y=[float(query_xy[1])],
        mode="markers",
        name="◆ your question",
        marker=dict(size=24, color=render.QUERY_COLOR, symbol="star",
                    line=dict(width=1.5, color="#fcfcfb")),
        hovertext=[question or "your question"],
        hovertemplate="<b>your question</b><br>%{hovertext}<extra></extra>",
    ))

    # Label the star on the plot itself, so the chart stands on its own if
    # someone screenshots it out of the report.
    if question:
        fig.add_annotation(
            x=float(query_xy[0]), y=float(query_xy[1]),
            text=f"<b>“{question[:60]}{'…' if len(question) > 60 else ''}”</b>",
            showarrow=True, arrowhead=0, arrowwidth=1.2,
            arrowcolor=render.QUERY_COLOR, ax=0, ay=-38,
            font=dict(size=12, color="#0b0b0b"),
            bgcolor="rgba(252,252,251,.92)", bordercolor=render.QUERY_COLOR,
            borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        legend=dict(orientation="v", x=1.01, y=1, font=dict(size=11),
                    title=dict(text="<b>retrieved</b>", font=dict(size=11))),
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
    store_backend: str = "chroma",
) -> str:
    """Project the corpus, place the question in it, and report the picture."""
    import joblib
    import numpy as np

    store_dir = await index(
        source=source, max_docs=max_docs,
        collection_name=collection_name, embedding_model=embedding_model,
        store_backend=store_backend,
    )
    projection_dir = await fit_projection(
        store_dir, collection_name=collection_name, embedding_model=embedding_model,
        store_backend=store_backend,
    )

    proj_path = Path(await projection_dir.download())
    coords = np.load(proj_path / "coords.npy")
    reducer = joblib.load(proj_path / "reducer.joblib")
    meta = json.loads((proj_path / "meta.json").read_text())
    id_to_index = {chunk_id: i for i, chunk_id in enumerate(meta["ids"])}

    collection = open_collection(
        str(Path(await store_dir.download())), collection_name, embedding_model,
        store_backend,
    )
    log.info(f'\nQuestion: "{question}"')

    hits = retrieve(collection, question, k=top_k, embedding_model=embedding_model)
    for hit in hits:
        log.info(f"  #{hit.rank}  {hit.similarity:.3f}  {hit.source}")
    hit_indices = [id_to_index[h.id] for h in hits if h.id in id_to_index]
    hits = [h for h in hits if h.id in id_to_index]

    # The query goes through the *fitted* reducer, not a new one — that is what
    # puts it in the same coordinate system as the cloud.
    encoder = load_encoder(embedding_model)
    query_vector = np.asarray(embed(encoder, [question]), dtype="float32")
    query_xy = reducer.transform(query_vector)[0]

    fig = _figure(coords, meta["sources"], hit_indices, hits, query_xy, question)
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
            "Gray dots are every chunk in the index. The <b>numbered blue dots</b> "
            "are the ones retrieved for this question — <b>darker means a better "
            "match</b>, and the number on each dot is its rank, matching the cards "
            "below. The orange star is the question itself, pushed through the same "
            "fitted projection.<br><br>"
            "Two dots sitting on top of each other is not a glitch: consecutive "
            "chunks from one document have nearly identical embeddings, so they "
            "land in nearly the same place. That is chunk size made visible.<br><br>"
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

    log.info(
        f"Plotted {len(coords)} chunks. Open the chart with: python open_report.py"
    )
    return f"plotted {len(coords)} chunks, top similarity {top:.3f}"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(visualize)
    print(f"Visualize run: {run.name}")
    print(f"  {run.url}")
