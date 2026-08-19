"""Chunking and Chroma, shared by every step.

Nothing Flyte-specific lives here — these are the plain functions the tasks call.
Keeping them out of the step files means step 4 can build a *writable* store with
the same three lines step 0 uses to build a read-only one, which is the whole
argument the tutorial is making.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# Same model at index time and query time. If these ever disagree your
# retrieval quietly turns to noise, so the collection records which model built
# it and `open_collection` refuses a mismatch out loud.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Recursive character splitter: paragraphs → lines → sentences → words → chars.

    Character-based, not token-based, so we don't drag in a tokenizer just to
    split. bge-small takes 512 tokens (~2000 English characters), so the default
    1200 leaves headroom and nothing gets silently truncated at encode time.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    for sep in ("\n\n", "\n", ". ", " ", ""):
        if sep == "":
            step = max(1, chunk_size - overlap)
            return [text[i:i + chunk_size] for i in range(0, len(text), step)]
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        out: list[str] = []
        buf = ""
        for part in parts:
            piece = part + sep
            if len(piece) > chunk_size:
                if buf.strip():
                    out.append(buf.strip())
                    buf = ""
                out.extend(split_text(part, chunk_size, overlap))
                continue
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf.strip():
                    out.append(buf.strip())
                buf = (buf[-overlap:] if overlap and buf else "") + piece
        if buf.strip():
            out.append(buf.strip())
        return [c for c in out if c.strip()]
    return [text]


# ── Embeddings ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def load_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load the sentence-transformer once per process.

    bge-small is ~130MB and runs fine on a CPU, which is why steps 0, 1 and 3
    need no API key and no GPU — they work in a free Colab runtime.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(encoder, texts: list[str]) -> list[list[float]]:
    """Encode to L2-normalized vectors, which is what cosine distance expects."""
    return encoder.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()


# ── The store ─────────────────────────────────────────────────────────────────

def open_collection(
    persist_dir: str,
    name: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
):
    """Open (or create) a Chroma collection on disk.

    A Chroma `PersistentClient` directory is a sqlite database plus parquet
    shards. That whole directory is what Flyte hands between steps as a
    `flyte.io.Dir` — one artifact, cached and versioned like any other task
    output.
    """
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"embedding_model": embedding_model, "hnsw:space": "cosine"},
    )

    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with and built_with != embedding_model:
        raise ValueError(
            f"Collection '{name}' was built with {built_with}, but you are querying "
            f"it with {embedding_model}. Vectors from different models are not "
            f"comparable — rebuild the index or pass the matching model."
        )
    return collection


@dataclass
class Hit:
    """One retrieved chunk."""

    rank: int
    id: str
    text: str
    source: str
    similarity: float  # cosine similarity: 1.0 is identical, ~0 is unrelated


def retrieve(collection, query: str, k: int = 4, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> list[Hit]:
    """Embed the query and return the k nearest chunks.

    This is the entirety of "retrieval" in retrieval-augmented generation. No
    model is involved — it is a nearest-neighbour lookup in a vector space, and
    step 1 exists to make that concrete before any LLM shows up.
    """
    if collection.count() == 0:
        return []

    encoder = load_encoder(embedding_model)
    query_vector = embed(encoder, [query])[0]

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[Hit] = []
    for i, (doc, meta, distance) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ):
        hits.append(
            Hit(
                rank=i + 1,
                id=result["ids"][0][i],
                text=doc,
                source=(meta or {}).get("source", "unknown"),
                # Chroma's cosine distance is 1 - cosine_similarity.
                similarity=1.0 - distance,
            )
        )
    return hits
