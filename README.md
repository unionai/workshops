
# Flyte & Union.ai Tutorials

Tutorials and examples for building AI agents, ML pipelines, and data workflows with [Flyte 2](https://flyte.org/).

---

## Featured

| Example | Description |
|---------|-------------|
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
| [Code Mode — NYC Taxi analyst](tutorials/code-mode-analysis/) | The agent writes a *program* instead of calling tools one at a time. It runs in the Monty sandbox, and a loop in the generated code becomes a fan-out of durable, parallel query tasks over 3M+ real taxi trips |
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
| [LoRA / QLoRA / Full](tutorials/llm-fine-tuning-lora-qlora/) | Fine-tune an LLM on text-to-SQL with LoRA, QLoRA, or full fine-tuning — live training reports, FastAPI serving, Gradio UI |
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
