# RAG and agentic memory: one vector store, pointed two directions

A hands-on tour of vector stores on Flyte 2. You build a document index, search it
without a model, answer questions from it with Claude, *look* at the embedding space,
and then turn the same machinery around so an agent writes to it instead of only
reading — which is all "agentic memory" actually is.

Everything runs three ways: in Colab with no cluster, on your laptop with
`flyte run --local`, or on a Flyte cluster where the steps become containers.

---

## The idea

A vector store does exactly one thing: it holds a pile of vectors and finds the ones
nearest a query vector. That's it. There is no reasoning in there, nothing that
understands your question, and nothing that can decline to answer.

Almost everything people call "RAG" or "agent memory" is that single operation with
different plumbing around it:

|  | Retrieval-augmented generation | Agentic memory |
|---|---|---|
| Who writes to the store | a pipeline, ahead of time | the agent, as it goes |
| What's in it | documents you chose | facts the agent noticed |
| When it's written | before any question is asked | after every turn |
| How it's read | **embed query → k nearest** | **embed query → k nearest** |

The bottom row is identical. That's the whole point of this tutorial, and it's why
these two things live in one place instead of two: steps 0–3 build a read-only index,
step 4 changes *who holds the pen*, and nothing else changes at all.

### Why this is worth your time

RAG has a reputation as a solved, boring thing you wire together from a framework.
The framework hides the two decisions that actually determine whether it works:

- **What went into the chunks.** Chunk too big and one vector has to represent four
  topics, so it averages them into mush. Chunk too small and you retrieve a sentence
  that made sense only in a paragraph you threw away.
- **Whether the neighbours were any good.** Retrieval *always* returns something.
  Ask a corpus about French geography when it only knows about ML tutorials and you
  still get your top 4 — just at similarity 0.47 instead of 0.79. A system that
  doesn't look at that number will hand the model garbage with total confidence.

Step 1 makes you stare at both of those before any LLM shows up, and step 3 draws
them.

---

## What you'll build

Six steps. Each is a standalone `flyte run`, and each calls the ones before it as
subtasks — those are cached, so only the first run pays for the index.

| Step | File | What it teaches |
|---|---|---|
| **0** | `step0_index.py` | Fetch → chunk → embed → a Chroma directory as a `flyte.io.Dir`. *No API key needed.* |
| **1** | `step1_retrieve.py` | Retrieval alone. Top-k with similarity scores, **no model anywhere**. *No API key needed.* |
| **2** | `step2_rag_answer.py` | Add Claude. Grounded answers with `[#N]` citations, and `--use_retrieval false` to see what ungrounded looks like. |
| **3** | `step3_visualize.py` | Project 384 dimensions onto a screen. Watch the query star move between clusters. *No API key needed.* |
| **4** | `step4_memory.py` | The same store, written by the agent. Retrieve → answer → extract facts → write back. |
| **5** | `step5_chat_app.py` | All of it in one Gradio UI — chat, live projection, memory panel. Runs locally or deploys to a cluster. |

Steps 0, 1 and 3 never call a model. You can get all the way to a working retrieval
demo, with a picture, before finding your API key — which matters when you're running
a room full of people through this.

---

## Setup

### Colab

Open the notebook and run the cells. The first one clones this repo and installs
everything; there's no cluster and nothing to configure.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/rag-agent-memory/rag-agent-memory-tutorial.ipynb)

### Locally

```bash
cd tutorials/rag-agent-memory
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

For steps 2, 4 and 5, put your key in a `.env` file next to `config.py`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`config.py` calls `load_dotenv()`, so every step picks it up automatically.

### On a cluster

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --project flytesnacks \
    --domain development \
    --builder remote

# Same project and domain as the runs, or the pod won't see it:
flyte create secret ANTHROPIC_API_KEY -p flytesnacks -d development
```

Then drop the `--local` from any command below. The first remote run builds the
image; the rest start warm.

### Using a different model

Everything goes through `llm.py`, so the provider is an environment variable rather
than an edit:

```bash
# Cheaper for a workshop
export LLM_MODEL=claude-haiku-4-5

# OpenAI
export LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini

# A local model — Ollama, vLLM, LM Studio, anything OpenAI-compatible
export LLM_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.1
export OPENAI_API_KEY=unused
```

The embedding model is separate and always local: `BAAI/bge-small-en-v1.5`, ~130MB,
CPU-only. Retrieval never calls an API.

---

## 0. Build the index

```bash
flyte run --local step0_index.py index
```

Three cached tasks — `fetch_docs` → `chunk_documents` → `embed_and_index` — and a
Chroma persist directory comes out the far end as a `flyte.io.Dir`.

The default corpus is **this repository's own tutorial write-ups**: 48 READMEs,
about 400 chunks, a few seconds to embed. It's a good corpus for a workshop because
you can check every answer by opening the file it cites, and because the tutorials
cover genuinely different domains, so step 3's projection has real clusters in it.

Other corpora, same command:

```bash
# The Flyte OSS docs — bigger, ~8MB download
flyte run --local step0_index.py index --source flyte-docs

# Any HuggingFace dataset with a text column
flyte run --local step0_index.py index --source hf \
    --dataset_repo rag-datasets/rag-mini-wikipedia \
    --dataset_config text-corpus --dataset_split passages --text_column passage

# Your own notes
flyte run --local step0_index.py index --source local --local_path ~/notes
```

Both GitHub sources are pulled as tarballs rather than cloned, so the image needs no
`git` and the task behaves identically on your laptop and in a pod.

**Knobs worth turning:** `--chunk_size` (default 1200 characters) and
`--chunk_overlap` (150). Try 400 and 3000 and re-run step 1 — the retrieved text
changes character completely.

## 1. Search it, with no model at all

```bash
flyte run --local step1_retrieve.py search --question "How do I fine-tune a model with GRPO?"
```

```
#1  0.785  tutorials/llm-fine-tuning-grpo-math/README.md
#2  0.784  tutorials/llm-fine-tuning-grpo-countdown/README.md
#3  0.767  tutorials/llm-fine-tuning-grpo-math/README.md
#4  0.763  tutorials/llm-fine-tuning-lora-qlora/README.md
```

Now ask it something the corpus has never heard of:

```bash
flyte run --local step1_retrieve.py search --question "What is the capital of France?"
```

```
top similarity 0.471
```

**You still get four chunks.** That is the single most important thing to understand
about retrieval, and it's why this step exists before the model does. Retrieval has
no concept of "I don't know" — the scores are the only signal you get, and if you
don't threshold on them, nothing downstream will.

## 2. Answer from the chunks

```bash
flyte run --local step2_rag_answer.py answer \
    --question "What does the code-mode tutorial teach?"
```

The retrieved chunks get pasted into the prompt with instructions to cite them as
`[#1]`, `[#2]` and to refuse when the context doesn't cover the question. That's the
whole of the "generation" half — RAG is not an architecture, it's a prompt with fresh
text in it.

Now run it again with retrieval switched off:

```bash
flyte run --local step2_rag_answer.py answer \
    --question "What does the code-mode tutorial teach?" --use_retrieval false
```

The interesting failure is not a refusal. The model produces something fluent and
confident about a tutorial it has never seen. Retrieval is what makes the difference
*checkable* — every `[#N]` points at a file you can open.

> **Note:** read the citations in the HTML report, not in the terminal. Flyte's
> console renders with Rich, which treats `[#1]` as markup and swallows it. The
> report has them.

## 3. Look at the space

```bash
flyte run --local step3_visualize.py visualize --question "How do I use GRPO?"
flyte run --local step3_visualize.py visualize --question "brain tumor segmentation"
flyte run --local step3_visualize.py visualize --question "What is the capital of France?"
```

Every chunk is a 384-dimensional vector. UMAP squashes that to two so it fits on a
screen, keeping neighbours near neighbours. The corpus becomes a map with clusters
nobody labelled — the fine-tuning tutorials in one region, the biotech ones in
another. Your question is embedded, pushed through the *same fitted* projection, and
lands as a gold star.

Run those three in a row and the star jumps between neighbourhoods.

**One honest caveat, which the report also states:** UMAP has to place an
out-of-corpus question *somewhere*, and it will happily drop it next to whatever is
least unlike it. A lonely-looking star is not the tell. The tell is that the
highlighted chunks have nothing to do with each other or with what you asked, and the
similarity scores are low. The numbers are ground truth; the map shows you which
neighbourhood they came from.

The projection is fitted once and cached — deliberately. Refit per question and the
entire cloud reshuffles between runs, which makes the demo unreadable. You want the
map to hold still while the star moves.

## 4. Turn the store around

```bash
flyte run --local step4_memory.py converse
```

Same Chroma, same encoder, same nearest-neighbour lookup. The difference is that the
agent writes.

Each turn: embed the message and retrieve relevant memories, answer with them in the
system prompt, then make a second cheap model call that extracts durable facts from
the exchange as JSON, embed those, and write them back.

```
Memory opened with 0 facts
Turn 1: recalled 0, wrote 3
Turn 2: recalled 3, wrote 2
Turn 3: recalled 5, wrote 0
  You're Sage, and you run a hacknight. Right now you're putting together Flyte
  demos for it — keeping them to about 20 minutes each with short, runnable
  examples rather than big sprawling ones.
```

Turn 3 has no special handling. It recalls turn 1 because turn 1 put something in the
store that turn 3's question is near.

Memory comes back as a `flyte.io.Dir`, so it outlives the run — feed it to another
`converse` and the agent picks up where it left off:

```bash
flyte run --local step4_memory.py converse \
    --memory_dir <path-printed-by-the-last-run> \
    --messages '["Remind me what my time limit is and who I am."]'
```

```
Memory opened with 5 facts
```

Two details in here are load-bearing:

- **Extraction is schema-constrained, not parsed out of prose.** `llm.extract()` uses
  the model's structured-output mode, so there is no regex fishing a `{...}` block out
  of a paragraph. If extraction returned malformed JSON, memory would silently stop
  being written and the agent would just seem forgetful.
- **Near-duplicates are dropped.** Anything within 0.95 cosine similarity of an
  existing memory is skipped, because five phrasings of "the user likes short demos"
  crowd out the top-k and teach the agent nothing.

## 5. Put it together

Locally, no cluster:

```bash
python step5_chat_app.py --local
# → http://localhost:7860
```

On a cluster, as a deployed app:

```bash
python step5_chat_app.py
```

Chat on the left, live projection on the right, tabs for retrieved chunks and current
memories. Every message does all four things at once: retrieves, answers, moves the
star, and writes what it learned about you.

The deployed version mounts the index through `flyte.app.RunOutput`, so the app pod
downloads the artifact step 0 already produced instead of rebuilding it. Pin a run
with `INDEX_RUN=<run-name>`, or leave it unset for the latest successful one.

---

## Design notes

**Why the two-environment split.** `config.py` defines `index_env` (no secret) and
`llm_env` (secret). Steps 0, 1 and 3 use the first. That isn't tidiness — it means a
workshop attendee who hasn't sorted out an API key can still get to a working
retrieval demo with a picture, which is most of the lesson.

**Why steps call each other as subtasks.** Every step invokes `index()` rather than
making you thread a directory path between commands. Flyte's cache makes that free
after the first run — you get self-contained commands *and* one build.

**Why `flyte.io.Dir` and not `File`.** Chroma's persist directory is a sqlite database
plus parquet shards. `Dir` snapshots the whole thing as one artifact, so it's cached,
versioned, and mountable into an app pod like any other task output.

**Why the embedding model is checked.** `open_collection` records which encoder built
a collection and refuses to open it with a different one. Vectors from two models
aren't comparable, and the failure mode without this check is silent: retrieval keeps
working and quietly returns nonsense.

**Why chunk metadata carries `source`.** Citations are only useful if they point
somewhere you can go look. Every chunk keeps the file it came from, which is what
makes `[#3]` checkable rather than decorative.

---

## Troubleshooting

**`ANTHROPIC_API_KEY is not set`** — steps 2, 4 and 5 need it. Locally: a `.env` file
next to `config.py`. On a cluster: `flyte create secret ANTHROPIC_API_KEY -p <project>
-d <domain>`, in the *same* project and domain you're running in. A secret created
without those flags lives at a different scope and the pod won't see it.

**Citations missing from the terminal output** — they're in the HTML report. Rich
eats `[#1]` as console markup.

**`Collection 'docs' was built with X, but you are querying it with Y`** — you changed
`--embedding_model` after building. Rebuild the index, or pass the model it was built
with.

**UMAP's first fit takes ~20 seconds** — that's numba compiling. It's cached
afterwards, and subsequent questions are instant.

**Everything is slow on the first run** — you're downloading bge-small (~130MB) and
building the index. Every run after that hits the cache.

**Gradio errors about `Chatbot`** — this pins `gradio>=6.0`. Version 6 removed the
`type` argument that version 5 required, so a venv with 5.x installed will fail.

---

## Where to take it next

- **Re-ranking.** Retrieve 20 with the bi-encoder, then re-score with a cross-encoder
  and keep 4. Usually the single biggest quality win available.
- **Threshold on similarity.** Right now every question gets an answer. Refuse below
  ~0.5 and the "capital of France" case fails honestly instead of being answered from
  four irrelevant chunks.
- **Multi-user memory.** Add an `entity_id` to each memory's metadata and filter on it
  at query and write time. One store, many users.
- **Memory decay.** A scheduled Flyte task that nightly drops memories nothing has
  retrieved in N days. Memory that only grows eventually stops being useful.
- **Durable app memory.** Step 5's memory lives on pod-local disk and dies with the
  pod. Wire it to step 4's `flyte.io.Dir`, or run Chroma as its own always-on app.
- **Swap the store.** `embed_and_index` is the only task that knows what Chroma is.
  Point it at pgvector, LanceDB or Turbopuffer and nothing else changes.
