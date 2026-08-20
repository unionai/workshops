"""Chunking, embeddings, and the vector store, shared by every step.

Nothing Flyte-specific lives here — these are the plain functions the tasks call.
Keeping them out of the step files means step 4 can build a *writable* store with
the same three lines step 0 uses to build a read-only one, which is the whole
argument the tutorial is making.

Two backends live behind one interface:

    --store chroma   (default)  Chroma, a sqlite file on disk
    --store qdrant              Qdrant in embedded mode, also a directory

Both are file-backed and need no server and no API key, which is what keeps the
whole tutorial runnable in Colab. Qdrant *also* runs as a server (and as a
managed cloud) — that mode is deliberately not wired up here, because a task
that writes to a remote database has no `flyte.io.Dir` to hand the next step,
and the artifact chaining is half of what this tutorial is showing.

The interface below is five operations. That is genuinely all a RAG pipeline
asks of a vector database, and seeing how little it is tends to demystify the
category.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

BACKENDS = ("chroma", "qdrant")

# Written into every store directory so a later step can tell what built it.
_META_FILE = "_store_meta.json"


def new_work_dir(prefix: str) -> Path:
    """A fresh directory for a task output that will become a `flyte.io.Dir`.

    Deliberately *not* `tempfile.mkdtemp()` with the system default. Under
    `--local`, Flyte's cache stores the path to a task's output directory, and
    macOS purges /var/folders periodically and on reboot — so a step 0 you ran
    on Monday hands step 3 a path that no longer exists on Wednesday, and the
    cached run dies with FileNotFoundError instead of rebuilding.

    Keeping the outputs under the project makes the local cache survive a
    reboot. On a cluster this is irrelevant: `Dir.from_local` uploads to blob
    storage and the local path stops mattering the moment it does.

    Override the location with RAG_WORK_DIR.
    """
    root = Path(os.environ.get("RAG_WORK_DIR", Path.cwd() / ".rag_work"))
    try:
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    except OSError:
        # Read-only working directory (some pods) — the system temp is fine
        # there, because the output is uploaded rather than re-read from disk.
        return Path(tempfile.mkdtemp(prefix=prefix))

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
    """Load the sentence-transformer once per process, from cache when possible.

    bge-small is ~130MB and runs fine on a CPU, which is why steps 0, 1 and 3
    need no API key and no GPU — they work in a free Colab runtime.

    We try `local_files_only=True` first. By default sentence-transformers
    revalidates the model against the Hub on *every* load — around fifteen HEAD
    requests over two TCP connections — even when the weights are already on
    disk and nothing gets downloaded. That makes every run need working network,
    which is a bad bet on conference wifi and pure latency everywhere else.

    Falling back to the network on failure keeps the first run (and any new
    model) working exactly as before.
    """
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        # Not cached yet — fetch it. This is the only path that needs network.
        log.info(f"Downloading {model_name} (one time, ~130MB)…")
        return SentenceTransformer(model_name)


def embed(encoder, texts: list[str]) -> list[list[float]]:
    """Encode to L2-normalized vectors, which is what cosine distance expects."""
    return encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        # A tqdm bar per batch is unreadable once Flyte captures the output, and
        # step 0 already logs its own indexed-N-of-M progress.
        show_progress_bar=False,
    ).tolist()


# ── What a vector store has to do ─────────────────────────────────────────────

@dataclass
class Hit:
    """One retrieved chunk."""

    rank: int
    id: str
    text: str
    source: str
    similarity: float  # cosine similarity: 1.0 is identical, ~0 is unrelated


@dataclass
class Record:
    """One stored item, read back out in bulk."""

    id: str
    text: str
    source: str
    vector: list[float] | None = None


class VectorStore:
    """The five operations this tutorial needs from a vector database.

    Both backends normalize to the *same* meaning of `similarity`: cosine
    similarity, higher is better. That normalization is the single most
    important line in each subclass — see the comments in `nearest`. Chroma
    reports a distance and Qdrant reports a score, and they run in opposite
    directions, so a backend that got this backwards would silently return the
    *worst* matches with no error anywhere.
    """

    backend: str = "?"

    def count(self) -> int:
        raise NotImplementedError

    def add(self, ids: list[str], texts: list[str], vectors: list[list[float]],
            metadatas: list[dict]) -> None:
        raise NotImplementedError

    def nearest(self, vector: list[float], k: int) -> list[Hit]:
        raise NotImplementedError

    def all_records(self, with_vectors: bool = False) -> list[Record]:
        raise NotImplementedError


# ── Chroma ────────────────────────────────────────────────────────────────────

class ChromaStore(VectorStore):
    """Chroma's `PersistentClient`: a sqlite database plus parquet shards."""

    backend = "chroma"

    def __init__(self, persist_dir: str, name: str, embedding_model: str):
        import chromadb

        client = _cached_client(("chroma", persist_dir), lambda: chromadb.PersistentClient(path=persist_dir))
        self._c = client.get_or_create_collection(
            name=name,
            metadata={"embedding_model": embedding_model, "hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._c.count()

    def add(self, ids, texts, vectors, metadatas) -> None:
        self._c.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    def nearest(self, vector, k) -> list[Hit]:
        n = self.count()
        if n == 0:
            return []
        r = self._c.query(
            query_embeddings=[vector],
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(
                rank=i + 1,
                id=r["ids"][0][i],
                text=doc,
                source=(meta or {}).get("source", "unknown"),
                # Chroma reports cosine *distance*: 0 is identical, 2 is opposite.
                similarity=1.0 - dist,
            )
            for i, (doc, meta, dist) in enumerate(
                zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
            )
        ]

    def all_records(self, with_vectors: bool = False) -> list[Record]:
        include = ["documents", "metadatas"] + (["embeddings"] if with_vectors else [])
        d = self._c.get(include=include)
        vectors = d.get("embeddings") if with_vectors else None
        return [
            Record(
                id=d["ids"][i],
                text=(d.get("documents") or [None] * len(d["ids"]))[i] or "",
                source=((d.get("metadatas") or [{}] * len(d["ids"]))[i] or {}).get("source", "unknown"),
                vector=list(vectors[i]) if vectors is not None else None,
            )
            for i in range(len(d["ids"]))
        ]


# ── Qdrant, embedded ──────────────────────────────────────────────────────────

# Qdrant point ids must be unsigned ints or UUIDs, but our ids are strings like
# "tutorials/x.md::3". Hash them into a stable UUID and keep the real one in the
# payload, so re-indexing the same chunk overwrites rather than duplicates.
_QDRANT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(raw: str) -> str:
    return str(uuid.uuid5(_QDRANT_NAMESPACE, raw))


class QdrantStore(VectorStore):
    """Qdrant in embedded mode — a local directory, no server, no API key.

    `QdrantClient(path=...)` runs the engine in-process against files on disk,
    which is what lets this backend keep the same `flyte.io.Dir` artifact model
    as Chroma. The same client class talks to a real server via `url=`, but that
    is a different tutorial (see the module docstring).
    """

    backend = "qdrant"

    def __init__(self, persist_dir: str, name: str, embedding_model: str):
        from qdrant_client import QdrantClient

        # Embedded Qdrant takes an exclusive lock on its directory: a second
        # QdrantClient on the same path raises "already accessed by another
        # instance". Local Flyte runs several tasks in one process and steps 3
        # and 5 legitimately open the same store twice, so clients are cached
        # per directory rather than constructed per call.
        self._client = _cached_client(("qdrant", persist_dir), lambda: QdrantClient(path=persist_dir))
        self._name = name

    def _exists(self) -> bool:
        return self._client.collection_exists(self._name)

    def count(self) -> int:
        return self._client.count(self._name).count if self._exists() else 0

    def add(self, ids, texts, vectors, metadatas) -> None:
        from qdrant_client import models

        if not vectors:
            return
        if not self._exists():
            # Unlike Chroma, Qdrant wants the dimensionality up front.
            self._client.create_collection(
                self._name,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE,
                ),
            )
        self._client.upsert(self._name, points=[
            models.PointStruct(
                id=_point_id(raw_id),
                vector=vec,
                payload={"chunk_id": raw_id, "text": text, **(meta or {})},
            )
            for raw_id, text, vec, meta in zip(ids, texts, vectors, metadatas)
        ])

    def nearest(self, vector, k) -> list[Hit]:
        if self.count() == 0:
            return []
        points = self._client.query_points(self._name, query=vector, limit=k).points
        return [
            Hit(
                rank=i + 1,
                id=(p.payload or {}).get("chunk_id", str(p.id)),
                text=(p.payload or {}).get("text", ""),
                source=(p.payload or {}).get("source", "unknown"),
                # Qdrant reports a cosine *score*: 1.0 is identical, 0 unrelated.
                # Already the number we want — do NOT subtract it from 1.
                similarity=float(p.score),
            )
            for i, p in enumerate(points)
        ]

    def all_records(self, with_vectors: bool = False) -> list[Record]:
        if not self._exists():
            return []
        out: list[Record] = []
        offset = None
        while True:  # scroll paginates; Chroma's .get() hands back everything
            batch, offset = self._client.scroll(
                self._name, limit=256, offset=offset,
                with_payload=True, with_vectors=with_vectors,
            )
            for p in batch:
                payload = p.payload or {}
                out.append(Record(
                    id=payload.get("chunk_id", str(p.id)),
                    text=payload.get("text", ""),
                    source=payload.get("source", "unknown"),
                    vector=list(p.vector) if with_vectors and p.vector is not None else None,
                ))
            if offset is None:
                break
        return out


# ── Opening one ───────────────────────────────────────────────────────────────

_CLIENTS: dict[tuple[str, str], object] = {}


def _cached_client(key, build):
    """One client per (backend, directory), for the whole process.

    Closed via `atexit` rather than left to garbage collection: embedded
    Qdrant's `__del__` runs during interpreter teardown, by which point
    `sys.meta_path` is gone and it cannot import what it needs to close
    cleanly. The result is a harmless but alarming `Exception ignored in
    QdrantClient.__del__` traceback after an otherwise successful script.
    `atexit` runs early enough that close() just works.
    """
    resolved = (key[0], str(Path(key[1]).resolve()))
    if resolved not in _CLIENTS:
        client = build()
        _CLIENTS[resolved] = client
        closer = getattr(client, "close", None)
        if callable(closer):
            atexit.register(lambda: _quietly_close(closer))
    return _CLIENTS[resolved]


def _quietly_close(closer) -> None:
    try:
        closer()
    except Exception:  # nothing useful to do while the process is ending
        pass


def _check_meta(persist_dir: str, embedding_model: str, backend: str) -> None:
    """Refuse to open a store with the wrong encoder or the wrong engine.

    Both failures are silent-by-default disasters. Query vectors from a
    different model land nowhere near the indexed ones, so retrieval keeps
    "working" and returns noise; and pointing Qdrant at a Chroma directory finds
    an empty collection rather than erroring, so you get zero hits and no clue
    why.
    """
    path = Path(persist_dir) / _META_FILE
    if path.exists():
        meta = json.loads(path.read_text())
        if meta.get("embedding_model") not in (None, embedding_model):
            raise ValueError(
                f"This store was built with {meta['embedding_model']}, but you are "
                f"querying it with {embedding_model}. Vectors from different models "
                f"are not comparable — rebuild the index, or pass the matching model."
            )
        if meta.get("backend") not in (None, backend):
            raise ValueError(
                f"This store was built with --store {meta['backend']}, but you asked "
                f"for --store {backend}. Rebuild the index with the backend you want."
            )
    else:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"embedding_model": embedding_model, "backend": backend}))


def open_collection(
    persist_dir: str,
    name: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    backend: str = "chroma",
) -> VectorStore:
    """Open (or create) a collection, on whichever backend was asked for.

    Either way the result is a *directory*, which is what Flyte hands between
    steps as a `flyte.io.Dir` — one artifact, cached and versioned like any
    other task output.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown --store {backend!r}. Expected one of: {', '.join(BACKENDS)}.")

    _check_meta(persist_dir, embedding_model, backend)
    cls = ChromaStore if backend == "chroma" else QdrantStore
    return cls(persist_dir, name, embedding_model)


def retrieve(store: VectorStore, query: str, k: int = 4,
             embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> list[Hit]:
    """Embed the query and return the k nearest chunks.

    This is the entirety of "retrieval" in retrieval-augmented generation. No
    model is involved — it is a nearest-neighbour lookup in a vector space, and
    step 1 exists to make that concrete before any LLM shows up.

    Note how little of it depends on the database: embed, ask for the nearest k.
    Everything backend-specific lives in the store classes above.
    """
    if store.count() == 0:
        return []
    encoder = load_encoder(embedding_model)
    return store.nearest(embed(encoder, [query])[0], k)
