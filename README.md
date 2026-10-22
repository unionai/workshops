
# Flyte & Union.ai Tutorials

Tutorials and examples for building AI agents, ML pipelines, and data workflows with [Flyte 2](https://flyte.org/).

---

## 👉 Tonight's workshop: Fine-tuning open models with agents

### **[tutorials/langgraph-model-factory-agent/](tutorials/langgraph-model-factory-agent/)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unionai/workshops/blob/factory/tutorials/langgraph-model-factory-agent/langgraph-model-factory-agent-tutorial.ipynb)
<!-- branch: switch blob/factory to blob/main once merged -->

*An ML engineer agent runs the model factory: evaluate, fine-tune, promote, deploy, and
retrain when the data drifts. LangGraph for the agent, Flyte 2 for everything it touches.*

A support agent in production routes tickets with an API call. An ML engineer agent gets
the request to replace that call with a model we own: it evaluates open candidates in
parallel, fine-tunes the ones worth it, promotes the winner, deploys it, and tests the live
app. Then the data drifts and the loop turns again. Every tool call is a durable Flyte
task, so a run that dies halfway resumes without retraining anything, and every eval and
fine-tune leaves a report you can open.

**Three ways to run it** ([Setup](tutorials/langgraph-model-factory-agent/#setup) has the details):

- **A Flyte devbox on your own machine (recommended: the full workshop).** Docker Desktop,
  Python 3.10+, `uv`; `flyte start devbox --apple-silicon` on an Apple Silicon Mac, plain
  `flyte start devbox` with an NVIDIA GPU or on CPU.
- **The workshop's Union cluster (`tryv2`), from any laptop.** One `flyte create config`
  command and a browser sign-in; the cluster's GPUs do the work.
- **The notebook on Google Colab (no GPU, nothing to install).** Click the badge. Slightly
  pared down: the core loop end to end, without the live app and kill-and-resume steps.

Bring a laptop. The devbox and Colab paths need a Claude or OpenAI API key; the cluster
already has one.

```bash
git clone https://github.com/unionai/workshops
cd workshops/tutorials/langgraph-model-factory-agent
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

flyte start devbox --apple-silicon && flyte create config --devbox      # a Mac: the devbox path
# or, the cluster path:
# flyte create config --endpoint tryv2.hosted.unionai.cloud --project factorysea --domain development --builder remote

flyte run model_factory.py model_factory --fine_tune_too                # step 1: the factory, you at the controls
```

---

## Featured

| Example | Description |
|---------|-------------|
| [Apple Silicon: fine-tune on your Mac's GPU](tutorials/apple-silicon-fine-tuning/) | `flyte start devbox --apple-silicon`, `gpu="AppleSilicon"`: LoRA on `mps` for text-to-SQL (5% → 62% exact match, 14 s of training), live loss chart, the adapter served on the same GPU; one env var moves it to a T4 |
| [Apple Silicon: serve an LLM on your Mac](tutorials/apple-silicon-serving/) | Qwen3-1.7B behind an OpenAI-compatible Flyte app on the Metal GPU (streaming, tool calls), benchmarked from a task, torch on `mps` vs MLX native (79 vs 277 tok/s), a chat UI that finds the app by name |
| [Apple Silicon: an agent whose model and tools run on your Mac](tutorials/apple-silicon-agent/) | A LangGraph SQL agent with Qwen3-4B served on the Mac's GPU as its brain and Flyte tasks as its tools: every query is a durable child action, 16 questions fanned out and scored (81%), swap the brain for Claude with one env var |
| [Model Factory Agent](tutorials/langgraph-model-factory-agent/) | Tonight's workshop. A LangGraph agent plays ML engineer on Flyte 2: parallel evals, fine-tunes, promotion, a deployed router app it tests, drift and the retrain; runs on a Union cluster, on a Flyte devbox on your Mac, or in Colab with no cluster at all |
| [Model Factory Agent + Grafana](tutorials/langgraph-grafana-agent/) | A LangGraph agent plays ML engineer: evaluates five candidates on T4s in parallel, fine-tunes the right ones (and discovers a 149M encoder beats the chat models), promotes it as a Union artifact, deploys it, tests the live app, and turns the factory again when the data changes; Grafana Agent Observability watches every turn |
| [SkyRL SQL Agent](tutorials/skyrl-sql-agent/) | Multi-turn RL against a real database — write one SkyRL-Gym environment, then drive it with no model, with a 0.5B, with Claude, or with SkyRL's distributed trainer; GRPO rollouts fan out as durable Flyte tasks |
| [RAG and Agentic Memory](tutorials/rag-agent-memory/) | One vector store, pointed two directions — build a RAG index, watch retrieval work with no model involved, visualize the embedding space, then let an agent write its own memories back into it |
| [Code Mode — NYC Taxi analyst](tutorials/code-mode-analysis/) | Claude writes one program, the Monty sandbox runs it, and its loops fan out into durable parallel tasks over real NYC taxi data |
| [LangGraph Research Pipeline](tutorials/langgraph_agent_research/) | Research agent pipeline — LangGraph orchestrates planning and quality gates, Flyte fans out parallel researcher tasks |
| [LangGraph ReAct Agent](tutorials/starter-examples/langgraph-react-agent/) | Build a ReAct agent with LangGraph + OpenAI on Flyte |
| [Stable Diffusion](tutorials/starter-examples/stable-diffusion/) | Generate images from text prompts with SDXL Turbo on GPU |
| [DuckDB ETL](tutorials/starter-examples/duckdb-etl/) | Extract and transform data with DuckDB SQL |
| [Fraud Detection with Feast](tutorials/fraud-detection-feast/) | Real-time fraud scoring with Feast feature store + XGBoost |

---

## Get Started

| Tutorial | What you'll learn |
|----------|-------------------|
| [Flyte Basics](tutorials/starter-examples/flyte-basics/) | Flyte 2 fundamentals — tasks, pipelines, error handling, `TaskEnvironment`, `ReusePolicy`, `map()` |
| [LangGraph ReAct Agent](tutorials/starter-examples/langgraph-react-agent/) | Build a ReAct agent with tools in a single file |
| [Stable Diffusion](tutorials/starter-examples/stable-diffusion/) | GPU inference with Flyte reports |
| [Image Classifier](tutorials/starter-examples/image-classifier/) | Fine-tune ResNet18 on HuggingFace dataset with PyTorch |
| [DuckDB ETL](tutorials/starter-examples/duckdb-etl/) | Data pipeline with DuckDB SQL and Flyte reports |
| [Snowflake ETL](tutorials/starter-examples/snowflake-etl/) | ETL pipeline with the Snowflake connector |
| [Flyte Local Dev](tutorials/starter-examples/flyte-local-dev/) | Local dev features — TUI, caching, reports, tracing, serving (no cluster needed) |
| [Fraud Detection with Feast](tutorials/fraud-detection-feast/) | Fraud scoring pipeline with Feast feature store, XGBoost, and real-time serving |

---

## Agents

| Tutorial | Description |
|----------|-------------|
| [Apple Silicon SQL Agent](tutorials/apple-silicon-agent/) | LangGraph ReAct agent on a model served from your Mac's GPU, tools as Flyte tasks, no API key; 16 scored questions, brain swappable to Claude or a cluster GPU |
| [Model Factory Agent + Grafana](tutorials/langgraph-grafana-agent/) | The agent runs the model factory: parallel T4 evals, fine-tunes, artifact promotion, a deployed router app it tests and can roll back, an artifact trigger that validates every promotion, a human approval gate, and a day-two retrain; observed in Grafana Agent Observability, with a crash-and-resume that trains nothing twice |
| [Code Mode — NYC Taxi analyst](tutorials/code-mode-analysis/) | The agent writes a *program* instead of calling tools one at a time. It runs in the Monty sandbox, and a loop in the generated code becomes a fan-out of durable, parallel query tasks over 3M+ real taxi trips |
| [RAG and Agentic Memory](tutorials/rag-agent-memory/) | One Chroma store, pointed two directions — build a document index, search it with no model, answer from it with citations, see the embedding space in 2D, then let an agent write its own memories back into it |
| [LangGraph Research Pipeline](tutorials/langgraph_agent_research/) | Research agent pipeline — LangGraph orchestrates planning and quality gates, Flyte fans out parallel researcher tasks via Tavily web search |
| [LangGraph ReAct Agent](tutorials/starter-examples/langgraph-react-agent/) | Single-file ReAct agent with LangGraph |
| [Planner Multi-Agent System](tutorials/multi-agent-workflows/tutorial_planner_agent.ipynb) | Scalable planner multi-agent system |
| [ReAct Multi-Agent System](tutorials/multi-agent-workflows/tutorial_react_agent.ipynb) | Adaptive ReAct multi-agent system |
| [Debate Multi-Agent System](tutorials/multi-agent-workflows/tutorial_debate_agent.ipynb) | Multi-agent debate pattern |
| [Manager Multi-Agent System](tutorials/multi-agent-workflows/tutorial_manager_agent.ipynb) | Manager-worker agent delegation |
| [Sequential Multi-Agent System](tutorials/multi-agent-workflows/tutorial_sequential_agent.ipynb) | Sequential agent pipeline |
| [Autoresearch](tutorials/autoresearch/) | Autoresearch-style self-healing agent on Flyte |

## MCP

| Tutorial | Description |
|----------|-------------|
| [MCP Recipe Assistant](tutorials/mcp/) | Build and deploy a recipe assistant MCP server on Union |

## LLM Fine-Tuning

| Tutorial | Description |
|----------|-------------|
| [Apple Silicon: LoRA on `mps`](tutorials/apple-silicon-fine-tuning/) | Fine-tune Qwen2.5-0.5B for text-to-SQL on your Mac's GPU from a Flyte devbox task, then serve the adapter; the same pipeline runs on a T4 with `ACCELERATOR=T4` |
| [LoRA / QLoRA / Full](tutorials/llm-fine-tuning-lora-qlora/) | Fine-tune an LLM on text-to-SQL with LoRA, QLoRA, or full fine-tuning — live training reports, FastAPI serving, Gradio UI |
| [Nemotron 3 Nano — LoRA / QLoRA](tutorials/nemotron-fine-tuning-lora-qlora/) | Fine-tune NVIDIA's Nemotron 3 Nano 4B for tool calling — LoRA on an L40S or QLoRA on a T4 from the same pipeline, discovered LoRA targets for the Mamba hybrid, adapter serving |
| [Nemotron 3 Nano — GRPO](tutorials/nemotron-fine-tuning-grpo/) | Reinforcement learning on Nemotron 3 Nano 4B — Countdown puzzles with a verifier as the only reward, thinking off, live solve-rate charts |
| [Nemotron — Healthcare: clinical notes](tutorials/nemotron-healthcare-clinical-notes/) | Fine-tune Nemotron 3 Nano 4B to turn clinical case notes into fixed-schema JSON records — field-level scoring, LoRA on an L40S or QLoRA on a T4 |
| [Nemotron — Finance: XBRL tagging](tutorials/nemotron-finance-xbrl-tagging/) | Fine-tune Nemotron 3 Nano 4B to tag amounts in SEC filings with US-GAAP XBRL concepts (FiNER-139) — exact tag-set and pair-F1 scoring, LoRA on an L40S or QLoRA on a T4 |
| [Nemotron — Legal: contract clauses](tutorials/nemotron-legal-contract-clauses/) | QLoRA an 8B model on a T4 — Llama-3.1-Nemotron-Nano-8B classifies contract clauses into the 100-way LEDGAR taxonomy, reasoning switched off via system prompt |
| [Nemotron — Biomed: chemical-protein relations](tutorials/nemotron-biomed-chemprot-relations/) | Fine-tune Nemotron 3 Nano 4B to classify chemical-protein relations in PubMed abstracts (ChemProt) — the extraction step behind drug-target knowledge bases, LoRA on an L40S or QLoRA on a T4 |
| [GRPO — Code Generation](tutorials/llm-fine-tuning-grpo-code/) | Teach a model to write Python with GRPO — reward = sandboxed test execution, MBPP dataset, live reward/pass-rate charts |
| [GRPO — Distributed](tutorials/llm-fine-tuning-grpo-distributed/) | Scale GRPO across the cluster — fan out sandboxed verification to a reusable pool, then disaggregate rollouts onto vLLM workers with LoRA weight sync |
| [GRPO](tutorials/llm-fine-tuning-grpo/) | GRPO fine-tuning on math/reasoning tasks |
| [DPO](tutorials/llm-fine-tuning-dpo/) | Direct Preference Optimization for alignment |
| [PPO](tutorials/llm-fine-tuning-ppo/) | Proximal Policy Optimization for RLHF |

## ML / AI

| Tutorial | Description |
|----------|-------------|
| [Fraud Detection with Feast](tutorials/fraud-detection-feast/) | Fraud scoring pipeline with Feast feature store, XGBoost, and real-time serving |
| [Stable Diffusion](tutorials/starter-examples/stable-diffusion/) | Image generation with SDXL Turbo |
| [Image Classifier](tutorials/starter-examples/image-classifier/) | Fine-tune ResNet18 on Beans dataset |
| [DETR Object Detection](tutorials/detr-object-detection/) | Fine-tune DETR for object detection with live mAP charts |

## Biotech / Life Sciences

| Tutorial | Description |
|----------|-------------|
| [Genomic Variant Effect Prediction](tutorials/genomic-variant-effect/) | Score DNA mutations with HuggingFace Carbon genomic foundation model — zero-shot pathogenicity prediction on BRCA2, TP53, KRAS, and more |
| [DNA Sequence Generation & Analysis](tutorials/genomic-dna-generation/) | Generate DNA with Carbon and compare to real genes — GC content, codon usage, ORFs, dinucleotide frequencies |
| [Gene Comparison Across Species](tutorials/genomic-gene-comparison/) | Compare homologous genes across 6 species with Carbon scoring, phylogenetic trees, and ESMFold 3D structure comparison |
| [Protein Sequence Analysis](tutorials/protein-sequence-analysis/) | Analyze protein properties, compute sequence similarity, run ESM-2 embeddings, and predict 3D structures with ESMFold |
| [Drug Molecule Screening](tutorials/drug-molecule-screening/) | Virtual drug screening — compute physicochemical properties, apply Lipinski's Rule of Five, rank candidates by drug-likeness |
| [Cell Microscopy Classification](tutorials/cell-microscopy-classification/) | Fine-tune a Vision Transformer (ViT) to classify blood cell types from microscopy images |

## Data

| Tutorial | Description |
|----------|-------------|
| [DuckDB ETL](tutorials/starter-examples/duckdb-etl/) | SQL-based data pipeline with DuckDB |
| [Snowflake ETL](tutorials/starter-examples/snowflake-etl/) | ETL with Snowflake connector |
| [Lance Streaming for Vision](tutorials/lance-streaming-vision/) | Convert a swarm of tiny per-sample image files (real CPPE-5 detection data) into one Lance dataset, benchmark per-file vs Lance streaming against real object storage, then train / evaluate / explore a Faster R-CNN streamed straight from S3 on a T4 |

---

## Setup

```bash
# Clone the repository
git clone https://github.com/unionai/workshops
cd workshops

# Navigate to any tutorial
cd tutorials/langgraph_agent_research

# Create virtual environment and install dependencies
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Flyte Cluster (for remote runs)

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

Don't have a cluster? Request access at [flyte.org](https://flyte.org/).

### Run examples

**Remote:**
```bash
uv run flyte run workflow.py research_pipeline --query "Compare quantum computing approaches"
```

**Local:**
```bash
uv run flyte run --local workflow.py research_pipeline --query "Compare quantum computing approaches"
```

**Local with TUI:**
```bash
uv run flyte run --local --tui workflow.py research_pipeline --query "Compare quantum computing approaches"
```

**Start TUI dashboard:**
```bash
uv run flyte start tui
```
