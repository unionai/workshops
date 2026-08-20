# RAG and agentic memory: one vector store, pointed two directions

A hands-on tour of vector stores on Flyte 2. You build a document index, search it
without a model, answer questions from it with Claude, *look* at the embedding space,
and then turn the same machinery around so an agent writes to it instead of only
reading — which is all "agentic memory" actually is.

Everything runs three ways: in Colab with no cluster, on your laptop with
`flyte run --local`, or on a Flyte cluster where the steps become containers.

---

## What is RAG?

**The problem.** A language model only knows what was in its training data. It has
never seen your company's docs, your notes, or anything written after its cutoff.
Fine-tuning it on your documents is expensive and has to be redone every time they
change, and pasting *all* your documents into the prompt doesn't fit — and would cost
a fortune if it did.

**The trick.** Don't give the model everything. Before it answers, go find the three
or four most relevant paragraphs and paste *those* into the prompt. That's
retrieval-augmented generation: **retrieval** finds the paragraphs, **generation** is
the model answering with them in front of it.

It's less clever than it sounds, and that's the point — you can build the whole thing
out of parts you can inspect.

### The two phases

RAG happens at two different times, and confusing them is the usual source of
confusion.

**Ahead of time — build the index** (step 0, once):

```
documents  →  chunks  →  embeddings  →  vector store
  ~50 docs     ~550 pieces   ~550 vectors    Chroma
  READMEs     ~1200 chars   384 numbers     on disk
             each          each
```

**At question time — retrieve and answer** (steps 1 and 2, every question):

```
"How do I use GRPO?"
        ↓ embed the question the same way
   384 numbers
        ↓ find the nearest vectors in the store
   top 4 chunks  ──────────────────┐
        ↓                          │
   paste into the prompt           │  ← this is the whole "augmented" part
        ↓                          │
   model answers, citing them  ←───┘
```

### The three words you need

**Embedding** — a list of numbers representing a piece of text, produced by a small
model, arranged so that *similar text lands near similar text*. This tutorial uses
`bge-small`, which turns any text into 384 numbers. It runs on a CPU in milliseconds
and needs no API key. "Nearby" is measured as **cosine similarity**, roughly 0 to 1;
in practice a good match here scores ~0.75+ and an unrelated one ~0.45.

**Chunk** — documents are too big to embed usefully, so they're split into pieces of
about 1200 characters. One vector has to stand for one chunk, so a chunk covering four
topics averages them into mush. Chunk size is the most consequential knob in the whole
pipeline.

**Vector store** — the database that holds the vectors and answers "which of these
are nearest to this one?" Here it's Chroma, a sqlite file on disk. That's genuinely
all it does.

### What actually reaches the model

No framework magic — step 2 builds a string that looks like this and sends it:

```
CONTEXT:
[#1] (source: tutorials/code-mode-analysis/README.md)
A code-mode agent does something different. Given the same tools, it writes...

[#2] (source: tutorials/code-mode-analysis/README.md)
Flyte runs the generated program in Monty, a Rust-based Python interpreter...

QUESTION: What does the code-mode tutorial teach?
```

Plus a system prompt saying *answer only from the context, cite chunks as [#N], and
say so if the answer isn't there.* That is RAG, in full. Everything else in this
tutorial is about making the retrieval half good, because the generation half is just
this.

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

**So what is agentic memory, concretely?** An assistant that "remembers you" usually
isn't doing anything exotic. After each exchange it asks a model *"did the user reveal
anything durable about themselves?"*, gets back something like `["The user's name is
Sage.", "Sage prefers demos under 20 minutes."]`, embeds those sentences, and stores
them. Next turn, before answering, it embeds your new message and retrieves the
nearest stored facts — the same top-k lookup step 1 does — and pastes them into the
system prompt.

That's why the agent in step 4 can answer "what do you know about me?" on turn 3
without any special handling for that question. It isn't recalling the conversation;
it's retrieving sentences that turns 1 and 2 happened to write, because your question
landed near them in the same vector space.

### Why this is worth your time

RAG has a reputation as a solved, boring thing you wire together from a framework.
The framework hides the two decisions that actually determine whether it works:

- **What went into the chunks.** Chunk too big and one vector has to represent four
  topics, so it averages them into mush. Chunk too small and you retrieve a sentence
  that made sense only in a paragraph you threw away.
- **Whether the neighbours were any good.** Retrieval *always* returns something.
  Ask a corpus of ML tutorials who won the 2022 World Cup and you still get your
  top 4 — just at similarity 0.47 instead of 0.82. A system that
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
| **2** | `step2_rag_answer.py` | Add Claude. Grounded answers with `[#N]` citations, and `--no-use_retrieval` to see what ungrounded looks like. |
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

### Flyte config — three paths, pick one

Everything here works with **no config at all**; `flyte run --local` needs nothing. But
there are two upgrades worth knowing about.

**1. Nothing.** Clone, install, run. This is what Colab does.

**2. Local persistence** — still no cluster, but past runs get recorded to SQLite so you
can browse them instead of scrolling back through terminal output:

```bash
flyte create config --local-persistence
flyte start tui          # browse past runs
```

That writes a three-line `.flyte/config.yaml` next to the tutorial:

```yaml
image:
  builder: local
local:
  persistence: true
```

One side effect worth knowing: once a project `.flyte/` exists, **the run cache moves
from `~/.flyte/local-cache/` to `./.flyte/local-cache/`** — project-scoped rather than
global. Good behaviour, but it changes which directory you clear (see Troubleshooting).

**3. A cluster** — a local devbox, or a hosted endpoint. This is what buys you real
containers, the run graph, retries, and deployed apps (step 5).

```bash
# Local devbox — needs Docker, so not Colab.
flyte start devbox                    # --gpu to pass host GPUs through
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks --domain development \
    --builder local --insecure --local-persistence

# Or a hosted endpoint (demo access: https://union.ai/)
flyte create config \
    --endpoint <your-endpoint> \
    --project flytesnacks --domain development \
    --builder remote
```

Either way, create the secret in the **same** project and domain, then drop `--local`
from any command below:

```bash
flyte create secret ANTHROPIC_API_KEY -p flytesnacks -d development
flyte run step0_index.py index --store_backend qdrant
```

The first remote run builds the image; the rest start warm.

### Seeing the output

This matters from a terminal. `flyte run --local` prints a one-line summary — the
*actual* output of steps 1–4 (retrieved chunks with their scores, the answer and its
citations, the UMAP chart, the memories) goes into an HTML report on disk.

```bash
python open_report.py          # open the newest report in your browser
python open_report.py 2        # the one before it
python open_report.py --path   # just print the path
```

Or add `--tui` to any run for Flyte's live task UI.

In the notebook you don't need this — `show_latest()` renders the report inline
after each cell.

> A step that calls step 0 as a subtask writes several reports in one run, one per
> task. The newest is the step you actually invoked, which is what `open_report.py`
> shows by default.

### Using a different vector store

Every step takes `--store_backend`, and both backends run with no server and no API
key, so this works in Colab exactly as it does locally:

```bash
flyte run --local step0_index.py index                          # chroma (default)
flyte run --local step0_index.py index --store_backend qdrant   # qdrant, embedded
flyte run --local step1_retrieve.py search --store_backend qdrant --question "..."
python step5_chat_app.py --local --store qdrant
```

The two produce **identical rankings and identical scores** — same encoder, same
cosine similarity, so they should, and it's a useful check that the seam is honest:

| | #1 | #2 | #3 | #4 |
|---|---|---|---|---|
| chroma | 0.785 | 0.784 | 0.767 | 0.763 |
| qdrant | 0.785 | 0.784 | 0.767 | 0.763 |

`store_backend` is part of the cache key, so switching builds a separate index rather
than handing you the other one, and each store directory records which engine wrote it
— point the wrong backend at one and you get a clear error rather than zero results.

**Why embedded Qdrant and not Qdrant Cloud.** `QdrantClient(path=...)` runs the engine
in-process against a local directory, which is what keeps the `flyte.io.Dir` artifact
model intact — step 0 still returns a directory the later steps consume, caching still
works, and no credential enters the workshop. A hosted Qdrant would break all three:
a task writing to a remote database has no artifact to hand the next step, and Flyte's
cache would start lying (a cache hit skips re-indexing even if the database was wiped).
That's a different tutorial, not a flag.

### What it takes to add a third store

`store.py` defines a `VectorStore` with five operations — `count`, `add`, `nearest`,
`all_records`, plus opening one. That is genuinely everything a RAG pipeline asks of a
vector database, and seeing how small it is tends to demystify the category. To add
pgvector or LanceDB, write one class; no step file changes.

Two things bit us implementing Qdrant, and they're the two that will bite you:

- **Score direction.** Chroma returns cosine *distance* (0 = identical); Qdrant returns
  a *score* (1.0 = identical). Each backend normalizes to "similarity, higher is
  better" in `nearest()`. Get it backwards and retrieval silently returns the **worst**
  matches — no error, no exception, just quietly inverted results.
- **ID types.** Chroma takes arbitrary strings like `tutorials/x.md::3`; Qdrant needs
  ints or UUIDs. `QdrantStore` hashes the chunk id into a stable UUID and keeps the
  original in the payload, so re-indexing overwrites instead of duplicating.

One implementation detail worth knowing if you extend it: embedded Qdrant takes an
exclusive lock on its directory, and a second `QdrantClient` on the same path raises.
Local Flyte runs several tasks in one process and steps 3 and 5 legitimately open the
same store twice, so `store.py` caches one client per directory.

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

The default corpus is **this repository's own tutorial write-ups**: around 50 documents,
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
#1  0.815  tutorials/rag-agent-memory/README.md
#2  0.785  tutorials/llm-fine-tuning-grpo-math/README.md
#3  0.784  tutorials/llm-fine-tuning-grpo-countdown/README.md
#4  0.767  tutorials/llm-fine-tuning-grpo-math/README.md
```

(This tutorial's own README is in the corpus, and it mentions GRPO in its examples —
so it outranks the actual GRPO tutorials. A small, honest illustration of why
retrieval quality is hard: the *most similar* text is not always the *most useful*.)

Now ask it something the corpus has never heard of:

```bash
flyte run --local step1_retrieve.py search --question "Who won the 2022 FIFA World Cup?"
```

```
top similarity 0.469
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
    --question "What does the code-mode tutorial teach?" --no-use_retrieval
```

> **CLI note:** Flyte turns a `bool` task parameter into a click flag pair, so it's
> `--no-use_retrieval`. `--use_retrieval false` fails with
> `Got unexpected extra argument (false)`.

You'll get one of two things, and which one is luck. Sometimes a confident answer
about a completely different "code mode" — the agentic-coding-tool persona, or the
Cloudflare MCP pattern. Sometimes a hedge that lists several things it might be and
asks you to narrow it down.

The hedge looks like the model behaving well, and in a sense it is. But notice what
you *still* can't do with either answer: check it. There's nothing to open, no claim
tied to anything. The grounded run's `[#3]` points at a file on disk. That's the
difference retrieval buys — not confidence, *verifiability*.

> **Note:** read the citations in the HTML report, not in the terminal. Flyte's
> console renders with Rich, which treats `[#1]` as markup and swallows it. The
> report has them.

## 3. Look at the space

```bash
flyte run --local step3_visualize.py visualize --question "How do I use GRPO?"
flyte run --local step3_visualize.py visualize --question "brain tumor segmentation"
flyte run --local step3_visualize.py visualize --question "Who won the 2022 FIFA World Cup?"
```

Every chunk is a 384-dimensional vector. UMAP squashes that to two so it fits on a
screen, keeping neighbours near neighbours. The corpus becomes a map with clusters
nobody labelled — the fine-tuning tutorials in one region, the biotech ones in
another. Your question is embedded, pushed through the *same fitted* projection, and
lands as an orange star, labelled with the question itself.

**Reading the chart:**

| Mark | Meaning |
|---|---|
| Small gray dots | every chunk in the index |
| **Numbered blue dots** | the retrieved chunks — **darker = better match**, number = rank |
| Orange star | your question, projected into the same space |

Rank is an ordered quantity, so it gets a single-hue ramp rather than a rainbow.
(An earlier version of this used red→orange→yellow→green, which was a mistake: red
reads as "bad" and green as "good", so it said the exact opposite of what it meant —
red was the *best* match.) Because adjacent steps of a ramp are necessarily similar,
the rank number is drawn inside each dot, so color never carries the rank alone.

**Two dots on top of each other is not a glitch.** Consecutive chunks from the same
document have nearly identical embeddings, so they project to nearly the same point —
you'll often see three dots when four were retrieved. That's chunk size made visible,
and the legend always lists all of them.

Run those three in a row and the star jumps between neighbourhoods. The terminal only
prints `plotted 543 chunks, top similarity 0.721` — **the chart is in the report**:

```bash
python open_report.py
```

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

### How it decides what to store

**There is no entity recognition, no NER, and no knowledge graph.** It is one extra
model call per turn, with a prompt and a schema. That's the whole mechanism, and it's
worth knowing because "the agent remembers me" sounds like it must be more.

The prompt (`EXTRACTION_SYSTEM` in `step4_memory.py`):

> You extract durable facts about the user from one exchange. Return each fact as a
> short, self-contained sentence that will still make sense months from now, read on
> its own with no surrounding conversation.
>
> **Include:** stable preferences, constraints, decisions, roles, projects, identity.
> **Exclude:** questions the user asked, small talk, anything about the assistant, and
> anything true only right now. If the user asked a question and revealed nothing new
> about themselves, return an empty list — that is the common case and it is fine.

Three things make it hold together:

- **The output is schema-constrained**, not parsed out of prose. `llm.extract()` forces
  `{"facts": ["...", "..."]}` through the model's structured-output mode, so there's no
  regex fishing a `{...}` block out of a paragraph. If extraction returned malformed
  JSON, memory would silently stop being written and the agent would just seem
  forgetful.
- **"Self-contained sentence" is doing real work.** A memory is retrieved months later
  with none of its conversation around it, so `"yes, that one"` is useless. The prompt
  pushes for `"Sage prefers demos under 20 minutes"` instead.
- **Near-duplicates are dropped.** Anything within 0.95 cosine similarity of an
  existing memory is skipped, because five phrasings of "the user likes short demos"
  crowd out the top-k and teach the agent nothing.

### Where this approach breaks

Worth seeing, because it's the honest limit of the naive design. Contradict yourself:

```bash
flyte run --local step4_memory.py converse --messages '[
  "I am Sage and I always run my demos in Python.",
  "Actually, I switched everything over to Rust last month.",
  "What language do I use?"]'
```

Everything ends up in the store — nothing is updated, nothing is deleted:

```
[turn 1] The user's name is Sage.
[turn 1] Sage always runs demos in Python.
[turn 2] Switched their projects/demos over to Rust as of about a month ago
[turn 3] The user's primary programming language is Rust, having switched about a month ago
[turn 3] The user previously used Python for their projects and demos before moving them to Rust
```

The agent *does* answer correctly ("You've been on Rust for about a month now — you
moved your projects and demos over from Python"), but notice **why**: retrieval handed
the model both the old and new facts, and the model reasoned its way to the right
answer at read time. The memory itself is still contradictory. Ask a subtler question,
or let the store grow until the stale fact ranks above the fresh one, and that stops
working.

Three specific gaps, all fixable, none fixed here:

- **No entity resolution.** "Sage", "the user" and "their" become unrelated sentences.
  Nothing links them, so there is no notion of *who* a fact is about — which is also
  why this is single-user.
- **No conflict resolution.** Dedupe only catches near-*duplicates*. Two facts that
  *contradict* are not similar in vector space at all — they're about the same topic
  with opposite content — so 0.95 cosine sails right past them.
- **No usable timestamps.** Metadata records `turn 1`, `turn 2`, and the counter
  restarts at 1 on every run — so a fact from today and one from last month are
  indistinguishable, and you can't prefer the recent one.

Real systems handle this with an update/delete path (retrieve related memories first,
then ask the model whether the new fact supersedes one), entity IDs, and wall-clock
timestamps. That's the natural next thing to build here.

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

**`Got unexpected extra argument (false)`** — Flyte renders a `bool` task parameter
as a click flag pair, so boolean arguments take no value. Use `--no-use_retrieval`
(or `--use_retrieval` to force it on), not `--use_retrieval false`. Run any step with
`--help` to see the exact flags it generated.

**Citations missing from the terminal output** — they're in the HTML report. Rich
eats `[#1]` as console markup.

**`Collection 'docs' was built with X, but you are querying it with Y`** — you changed
`--embedding_model` after building. Rebuild the index, or pass the model it was built
with.

**UMAP's first fit takes ~20 seconds** — that's numba compiling. It's cached
afterwards, and subsequent questions are instant.

**`FileNotFoundError: .../umap_xxxx/coords.npy` on a cached run** — you shouldn't hit
this any more, but here's the cause. Under `--local`, Flyte's cache stores the *path*
to a task's output directory. If those directories go to the system temp, macOS
purges them periodically and on reboot, so a step 0 you ran on Monday hands step 3 a
path that no longer exists on Wednesday. Task outputs now go to `.rag_work/` in this
directory instead (override with `RAG_WORK_DIR`), which survives a reboot.

**Forcing a genuinely clean run.** Two different directories are involved and only one
of them is the cache:

```bash
rm -rf ~/.flyte/local-cache    # the cache, when there is NO project .flyte/
rm -rf .flyte/local-cache      # the cache, once a project .flyte/ exists
rm -rf .rag_work               # the task outputs those cache entries point at
rm -rf /tmp/flyte/metadata     # run metadata and HTML reports only
```

**The cache moves.** With no project config it lives in `~/.flyte/local-cache/`; the
moment you run `flyte create config` in this directory, it becomes
`./.flyte/local-cache/`. Clearing the global one while a project config exists looks
like it worked and changes nothing — the symptom is a step that "runs" instantly and
prints no task logs.

`/tmp/flyte/metadata` is *not* the cache — deleting it alone changes nothing, and
deleting `.rag_work` without clearing the cache is worse than useless: the cache still
returns a hit pointing at a directory you just removed. Clear `~/.flyte/local-cache`
first.

**Everything is slow on the first run** — you're downloading bge-small (~130MB) and
building the index. Every run after that hits the cache.

**`HTTP Request: HEAD https://huggingface.co/...` on every run** — you shouldn't see
this, but here's what it was. By default sentence-transformers revalidates the model
against the Hub every time you load it: ~15 `HEAD` requests over 2 TCP connections,
asking "is my cached copy still current?" Nothing is re-downloaded — but it means
every run needs working network, which is a bad bet on conference wifi.

`store.load_encoder()` fixes this properly by trying `local_files_only=True` first
and only falling back to the network if the model genuinely isn't cached. Measured at
the socket layer:

| | TCP connections to huggingface.co |
|---|---|
| Before | 2, every run |
| After, model cached | **0** |
| After, empty cache (first run) | 2, downloads 138MB, then 0 forever |

If you want to be certain nothing is reaching out, count it yourself:

```bash
python - <<'EOF'
import socket, sys; sys.path.insert(0, ".")
calls = []; orig = socket.create_connection
socket.create_connection = lambda a, *x, **k: (calls.append(a), orig(a, *x, **k))[1]
from store import load_encoder; load_encoder()
print("connections:", len(calls), calls)
EOF
```

`export HF_HUB_OFFLINE=1` is a belt-and-braces option that makes any Hub call a hard
error rather than a silent fallback — useful if you want the failure to be loud.

**Gradio errors about `Chatbot`** — this pins `gradio>=6.0`. Version 6 removed the
`type` argument that version 5 required, so a venv with 5.x installed will fail.

---

## What this tutorial deliberately doesn't do

Worth saying plainly, so nobody walks away thinking they've seen production RAG:

- **There is no evaluation.** Nothing here tells you whether retrieval is *good* —
  only what it returned. That's the biggest gap, and it's the first thing to fix if
  you take this anywhere real. See "measure it" below.
- **Retrieval is a single dense lookup.** One embedding model, one top-k, no
  re-ranking, no keyword matching, no query rewriting. This is the simplest thing
  that works, not the best thing.
- **Chunking is naive.** Fixed-size character splitting with overlap. It ignores
  document structure entirely — headings, code blocks and tables all get chopped
  mid-thought.
- **Memory is single-user and never forgets.** No `entity_id`, no decay, no conflict
  resolution when you contradict yourself.

You can see the consequences in the tutorial's own output: ask *"How do I use GRPO?"*
and `detr-object-detection` comes back at #2. A small corpus and a bare dense lookup
will do that. Everything below is how people fix it.

---

## The rest of the RAG landscape

A map, roughly in order of value-for-effort. Almost everything here changes only
`store.retrieve()` or `step0_index.py` — the rest of the pipeline doesn't care.

### Make retrieval better (start here)

- **Re-ranking.** Retrieve 20 with the fast bi-encoder, then re-score those 20 with a
  **cross-encoder** that reads query and chunk *together* rather than embedding them
  separately. Slower per candidate, far more accurate, and you only run it on 20.
  Usually the single biggest quality win available. *(Change: `store.retrieve()`.)*
- **Hybrid search.** Dense embeddings are bad at exact tokens — error codes, flag
  names, `--no-use_retrieval`. Keyword search (BM25) is great at those and bad at
  paraphrase. Run both, merge with Reciprocal Rank Fusion. The two failure modes
  barely overlap, which is why hybrid beats either alone.
- **Contextual retrieval.** Before embedding each chunk, prepend a sentence or two of
  LLM-generated context situating it in its document ("This is from the GRPO
  tutorial's section on reward functions"). Fixes the chunk-lost-its-context problem
  at its root. Costs one cheap model call per chunk at index time — and prompt
  caching makes that far less painful than it sounds.
- **Query rewriting.** The user's phrasing isn't always the best search string.
  Rewrite it, expand it into several queries and merge results, or use **HyDE** —
  have the model write a *hypothetical answer*, embed that, and search with it, on the
  theory that a fake answer looks more like a real answer than the question does.
- **Metadata filtering.** Store `source`, date, or type alongside each chunk (this
  tutorial already stores `source`) and filter *before* the vector search. Cheapest
  possible precision win when your corpus has obvious partitions.
- **Smarter chunking.** Split on document structure instead of character counts.
  **Sentence-window**: embed one sentence, return its neighbours. **Parent-document**:
  embed small chunks for precision, hand the model the whole parent section for
  context. Both decouple "what you match on" from "what you send," which is the
  insight.

### Give it a different shape

- **Hierarchical / tree (RAPTOR).** Recursively cluster chunks and summarize each
  cluster, building a tree from raw text up to whole-corpus summaries, then retrieve
  at whichever level fits. Flat top-k structurally *cannot* answer "what are the main
  themes across all 46 documents?" — no single chunk contains the answer. A tree can.
- **Graph RAG.** Extract entities and relationships into a knowledge graph, then
  traverse it. Built for multi-hop questions — "which tutorials use the same base
  model as the GRPO one?" — where the answer requires joining facts that never appear
  in the same chunk. Microsoft's GraphRAG adds community detection plus per-community
  summaries. Powerful, and much heavier to build and maintain.
- **Multi-vector / late interaction (ColBERT).** One vector per token instead of per
  chunk, matched at query time. More precise, considerably more storage.

### Change who's in control

- **Agentic RAG.** Everything above retrieves exactly once, before answering. Instead,
  give the model retrieval as a *tool*: let it decide whether to search at all, what to
  search for, read the results, and search again with a better query. Multi-hop
  questions and "I don't know enough yet" both fall out naturally. This is the closest
  thing here to what you already know how to build — it's tool-calling with
  `store.retrieve` as the tool.
- **Corrective / self-reflective RAG.** Grade the retrieved chunks *before* answering.
  If they're weak, re-query, fall back to web search, or refuse. Cheap version, using
  numbers this tutorial already computes: if top similarity < 0.5, don't answer.
- **Routing.** Several indexes, and a small classifier or model picks which to search.

### Measure it

None of the above means anything without this, and it's the gap that matters most:

- **Retrieval metrics** — build a small golden set of question → correct-chunk pairs
  (50 hand-written pairs is enough to be useful), then track **recall@k** (was the
  right chunk in the top k?) and **MRR** (how high was it?). This is the number that
  moves when you add re-ranking, and without it you're guessing.
- **Answer metrics** — **faithfulness** (is every claim supported by the retrieved
  context?) and **relevance** (did it answer the question?). Usually scored by an
  LLM judge; RAGAS is the common off-the-shelf option.
- **The cheap version** — 20 questions in a list, run them after every change, read
  the answers yourself. Unglamorous, and it catches most regressions.

### Know when not to reach for RAG

- **Small corpus?** If everything fits in context, just paste it in. RAG exists
  because your data doesn't fit — under that threshold it's pure added complexity and
  a new failure mode.
- **Structured data?** "How many tutorials use GPUs?" is a SQL query, not a similarity
  search. Vectors are for fuzzy meaning, not counting and filtering.
- **Want a different style or format?** That's fine-tuning. RAG changes what the model
  *knows*, not how it *writes*.

---

## Concrete next steps for this code

- **Threshold on similarity.** Every question currently gets an answer. Refuse below
  ~0.5 and the World Cup case fails honestly instead of being answered from
  four irrelevant chunks. Two lines, and it's the smallest real improvement here.
- **Add a cross-encoder re-ranker.** Retrieve 20 in `store.retrieve()`, re-score with
  `sentence-transformers`' `CrossEncoder`, keep 4. Biggest quality-per-line win.
- **Multi-user memory.** Add an `entity_id` to each memory's metadata and filter on it
  at query and write time. One store, many users.
- **Memory decay.** A scheduled Flyte task that nightly drops memories nothing has
  retrieved in N days. Memory that only grows stops being useful.
- **Durable app memory.** Step 5's memory lives on pod-local disk and dies with the
  pod. Wire it to step 4's `flyte.io.Dir`, or run Chroma as its own always-on app.
- **Swap the store.** `embed_and_index` is the only task that knows what Chroma is.
  Point it at pgvector, LanceDB or Turbopuffer and nothing else changes.
