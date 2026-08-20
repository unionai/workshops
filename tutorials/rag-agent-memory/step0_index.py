"""Step 0 — build the index.

Three tasks: fetch documents, split them into chunks, embed the chunks into a
Chroma collection. The collection is returned as a `flyte.io.Dir`, so every
later step takes it as an input instead of rebuilding it.

No LLM here, and no API key. Run this before you have found your Anthropic key.

    flyte run --local step0_index.py index
    flyte run --local step0_index.py index --source flyte-docs
    flyte run --local step0_index.py index --source hf \
        --dataset_repo rag-datasets/rag-mini-wikipedia
    flyte run --local step0_index.py index --source local --local_path ./my-notes
"""

from __future__ import annotations

import json
import logging
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import flyte
import flyte.io
import flyte.report

from config import index_env
from store import DEFAULT_EMBEDDING_MODEL, embed, load_encoder, new_work_dir, open_collection, split_text

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = index_env

# Named corpora. Both are public GitHub repos pulled as a tarball rather than
# cloned, so the image needs no `git` and the task behaves the same locally and
# in a pod. `subdirs` are matched against the path *inside* the archive.
GITHUB_SOURCES = {
    # The default: this very repository's tutorial write-ups. ~900KB and roughly 50 documents after the length filter,
    # and you can open any answer's citation to check it. Also gives step 3's
    # projection real topical clusters — fine-tuning, biotech, agents, RL.
    "workshops": {
        "repo": "unionai/workshops",
        "ref": "main",
        "subdirs": ("tutorials/", "docs/"),
    },
    # Bigger and more reference-like: the Flyte OSS docs. ~8MB.
    "flyte-docs": {
        "repo": "flyteorg/flyte",
        "ref": "master",
        "subdirs": ("docs/", "rfc/"),
    },
}

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def _clean_markdown(text: str) -> str:
    """Drop YAML frontmatter. Everything else, including code blocks, stays —
    the code samples are usually the part someone is actually asking about."""
    return _FRONTMATTER.sub("", text).strip()


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — fetch documents into a single jsonl
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto", report=True)
async def fetch_docs(
    source: str = "workshops",
    dataset_repo: str = "rag-datasets/rag-mini-wikipedia",
    dataset_config: str = "text-corpus",
    dataset_split: str = "passages",
    text_column: str = "passage",
    local_path: str = "",
    max_docs: int = 0,
) -> flyte.io.Dir:
    """Write one `{id, text, source, title}` per line to docs.jsonl.

    Cached on the arguments, so re-running step 0 with the same corpus skips
    straight to chunking.
    """
    out_dir = new_work_dir("rag_docs_")
    docs_path = out_dir / "docs.jsonl"
    written = 0

    with docs_path.open("w") as out:
        for doc in _iter_docs(
            source, dataset_repo, dataset_config, dataset_split,
            text_column, local_path, max_docs,
        ):
            out.write(json.dumps(doc) + "\n")
            written += 1

    if written == 0:
        raise ValueError(f"No documents found for source={source!r}. Nothing to index.")

    log.info(f"Fetched {written} documents from {source}")
    await flyte.report.replace.aio(
        f"<h2>Fetched documents</h2>"
        f"<p><b>Source:</b> {source}</p>"
        f"<p><b>Documents:</b> {written}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


def _iter_docs(source, dataset_repo, dataset_config, dataset_split, text_column, local_path, max_docs):
    """Yield `{id, text, source, title}` dicts from whichever corpus was asked for."""
    if source in GITHUB_SOURCES:
        yield from _iter_github(GITHUB_SOURCES[source], max_docs)
    elif source == "hf":
        yield from _iter_hf(dataset_repo, dataset_config, dataset_split, text_column, max_docs)
    elif source == "local":
        yield from _iter_local(local_path, max_docs)
    else:
        raise ValueError(
            f"Unknown source {source!r}. "
            f"Expected one of: {', '.join(GITHUB_SOURCES)}, hf, local."
        )


def _iter_github(spec: dict, max_docs: int):
    url = f"https://codeload.github.com/{spec['repo']}/tar.gz/refs/heads/{spec['ref']}"
    log.info(f"Downloading {url}")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        urllib.request.urlretrieve(url, archive.name)
        with tarfile.open(archive.name, "r:gz") as tar:
            count = 0
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".md"):
                    continue
                # Strip the "<repo>-<ref>/" prefix GitHub adds to every entry.
                rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
                if not rel.startswith(spec["subdirs"]):
                    continue

                handle = tar.extractfile(member)
                if handle is None:
                    continue
                text = _clean_markdown(handle.read().decode("utf-8", errors="replace"))
                if len(text) < 200:  # stubs and redirect pages carry no signal
                    continue

                yield {
                    "id": rel,
                    "text": text,
                    "source": rel,
                    "title": _title_of(text, rel),
                }
                count += 1
                if max_docs and count >= max_docs:
                    return


def _iter_hf(dataset_repo, dataset_config, dataset_split, text_column, max_docs):
    from datasets import load_dataset

    log.info(f"Loading {dataset_repo} [{dataset_config or '-'}/{dataset_split}]")
    ds = (
        load_dataset(dataset_repo, dataset_config, split=dataset_split)
        if dataset_config
        else load_dataset(dataset_repo, split=dataset_split)
    )
    if max_docs and max_docs > 0:
        ds = ds.select(range(min(max_docs, len(ds))))

    for i, row in enumerate(ds):
        text = (row.get(text_column) or "").strip()
        if not text:
            continue
        doc_id = str(row.get("id", i))
        yield {"id": doc_id, "text": text, "source": f"{dataset_repo}#{doc_id}", "title": doc_id}


def _iter_local(local_path, max_docs):
    root = Path(local_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"--local_path {local_path!r} is not a directory")

    count = 0
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".rst"} or not path.is_file():
            continue
        text = _clean_markdown(path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            continue
        rel = str(path.relative_to(root))
        yield {"id": rel, "text": text, "source": rel, "title": _title_of(text, rel)}
        count += 1
        if max_docs and count >= max_docs:
            return


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — split documents into chunks
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto", report=True)
async def chunk_documents(
    docs_dir: flyte.io.Dir,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> flyte.io.Dir:
    """Split each document into overlapping chunks.

    Chunk size is the main knob you will want to play with. Too large and a
    chunk covers several topics, so its embedding averages them into mush. Too
    small and the chunk loses the context that made it meaningful.
    """
    in_path = Path(await docs_dir.download()) / "docs.jsonl"

    out_dir = new_work_dir("rag_chunks_")
    chunks_path = out_dir / "chunks.jsonl"
    n_docs = n_chunks = 0

    with in_path.open() as fin, chunks_path.open("w") as fout:
        for line in fin:
            doc = json.loads(line)
            n_docs += 1
            for j, chunk in enumerate(split_text(doc["text"], chunk_size, chunk_overlap)):
                fout.write(json.dumps({
                    "chunk_id": f"{doc['id']}::{j}",
                    "text": chunk,
                    "source": doc["source"],
                    "title": doc.get("title", ""),
                }) + "\n")
                n_chunks += 1

    per_doc = n_chunks / max(n_docs, 1)
    log.info(f"Chunked {n_docs} docs into {n_chunks} chunks ({per_doc:.1f} per doc)")
    await flyte.report.replace.aio(
        f"<h2>Chunked documents</h2>"
        f"<p><b>Documents:</b> {n_docs}</p>"
        f"<p><b>Chunks:</b> {n_chunks} ({per_doc:.1f} per document)</p>"
        f"<p><b>chunk_size:</b> {chunk_size} characters &middot; "
        f"<b>overlap:</b> {chunk_overlap}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — embed the chunks into a Chroma collection
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto", report=True)
async def embed_and_index(
    chunks_dir: flyte.io.Dir,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = "docs",
    batch_size: int = 64,
    store_backend: str = "chroma",
) -> flyte.io.Dir:
    """Turn every chunk into a vector and write the store's persist directory.

    `store_backend` is part of the cache key, so switching between chroma and
    qdrant builds a separate index rather than handing you the other one.
    """
    chunks_path = Path(await chunks_dir.download()) / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks_path.open()]
    log.info(f"Embedding {len(rows)} chunks with {embedding_model} into {store_backend}")

    encoder = load_encoder(embedding_model)
    persist_dir = new_work_dir(f"{store_backend}_")
    store = open_collection(str(persist_dir), collection_name, embedding_model, store_backend)

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        store.add(
            ids=[r["chunk_id"] for r in batch],
            texts=[r["text"] for r in batch],
            vectors=embed(encoder, [r["text"] for r in batch]),
            metadatas=[{"source": r["source"], "title": r["title"]} for r in batch],
        )
        log.info(f"  indexed {min(start + batch_size, len(rows))}/{len(rows)}")

    log.info(f"Collection '{collection_name}' holds {store.count()} chunks in {store_backend}")
    await flyte.report.replace.aio(
        f"<h2>Embedded and indexed</h2>"
        f"<p><b>Backend:</b> {store_backend}</p>"
        f"<p><b>Embedding model:</b> {embedding_model}</p>"
        f"<p><b>Collection:</b> {collection_name}</p>"
        f"<p><b>Chunks indexed:</b> {store.count()}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(persist_dir))


# ──────────────────────────────────────────────────────────────────────────────
# The pipeline
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def index(
    source: str = "workshops",
    dataset_repo: str = "rag-datasets/rag-mini-wikipedia",
    dataset_config: str = "text-corpus",
    dataset_split: str = "passages",
    text_column: str = "passage",
    local_path: str = "",
    max_docs: int = 0,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = "docs",
    store_backend: str = "chroma",
) -> flyte.io.Dir:
    """Fetch, chunk, embed. Returns the store directory the other steps read."""
    docs = await fetch_docs(
        source, dataset_repo, dataset_config, dataset_split,
        text_column, local_path, max_docs,
    )
    chunks = await chunk_documents(docs, chunk_size, chunk_overlap)
    store_dir = await embed_and_index(
        chunks, embedding_model, collection_name, store_backend=store_backend,
    )

    await flyte.report.replace.aio(
        "<h2>Index ready</h2>"
        f"<p>Corpus <code>{source}</code> is embedded into collection "
        f"<code>{collection_name}</code> on <code>{store_backend}</code>.</p>"
        "<p>Every later step rebuilds this by calling <code>index()</code>, and "
        "the tasks are cached — so nothing here runs twice.</p>"
    )
    await flyte.report.flush.aio()
    return store_dir


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(index)
    print(f"Index run: {run.name}")
    print(f"  {run.url}")
