# Research Agent Pipeline

A research agent pipeline that integrates LangGraph and Flyte as co-orchestrators. LangGraph controls the pipeline logic — planning, fan-out, quality gates, and iterative deepening. Flyte provides the compute — each researcher runs as a separate task with its own container, resources, and observability.

## Architecture

```
research_pipeline (LangGraph pipeline graph, inside a Flyte task)
  ├── plan → split query into sub-topics
  ├── research (Send fan-out → Flyte tasks)
  │     ├── research_topic("topic A")  ┐
  │     ├── research_topic("topic B")  ├── parallel Flyte tasks, each running a ReAct agent
  │     └── research_topic("topic C")  ┘
  ├── synthesize → combine into report
  ├── quality_check → score + identify gaps
  │     ├── gaps found → identify_gaps → Send fan-out → research again
  │     └── good enough → finalize
  └── finalize → final report
```

Each `research_topic` task runs a LangGraph ReAct agent that searches the web via [Tavily](https://tavily.com/) — an AI-optimized search API — and loops until it has enough information.

## Setup

```bash
cd tutorials/langgraph_agent_research

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Add your API keys to `.env`:

```
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
```

## 1. Run Locally

No cluster needed — everything runs in-process:

```bash
# With TUI
flyte run --local --tui workflow.py research_pipeline \
  --query "Compare quantum computing approaches: superconducting vs trapped ion"

# Without TUI
flyte run --local workflow.py research_pipeline \
  --query "What are the pros and cons of electric vehicles?" \
  --num_topics 2 --max_searches 1
```

Or run the Gradio app locally:

```bash
RUN_MODE=local python app.py
```

## 2. Run on a Cluster

### Start the devbox

Make sure the Flyte devbox is installed and running:

```bash
flyte start devbox
```

This starts a local k3s cluster in Docker with a UI at http://localhost:30080/v2.

### Configure Flyte

Point the SDK at your cluster (run once from the project directory):

```bash
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

This writes `.flyte/config.yaml` in your project root.

### Create secrets

The tasks need API keys to call OpenAI and Tavily. Create them once per project/domain:

```bash
flyte create secret OPENAI_API_KEY --project flytesnacks --domain development
flyte create secret TAVILY_API_KEY --project flytesnacks --domain development
```

### Register tasks

Register the task environment and build container images. This makes the tasks available on the cluster:

```bash
flyte deploy workflow.py env
```

### Run the pipeline remotely

```bash
flyte run workflow.py research_pipeline \
  --query "Compare quantum computing approaches" \
  --num_topics 2 --max_searches 2 --max_iterations 1
```

Or run the Gradio app locally against the remote cluster:

```bash
python app.py
```

### Deploy the Gradio app to the cluster

```bash
flyte deploy app.py serving_env
```

The app references pre-registered tasks via `remote.Task.get()` with `auto_version="latest"`, so it always picks up the latest version. When you update task code, re-run `flyte deploy workflow.py env` to register the new version.

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | required | Research question |
| `--num-topics` | 3 | Number of sub-topics to research in parallel |
| `--max-searches` | 2 | Max web searches per sub-topic |
| `--max-iterations` | 2 | Max quality gate iterations |

## Project Structure

```
langgraph_agent_research/
├── config.py           # Flyte environment, secrets, resources
├── graph.py            # LangGraph graphs — pipeline + ReAct subgraph
├── workflow.py         # Flyte tasks — research_topic + research_pipeline orchestrator
├── app.py              # Gradio UI — kicks off the pipeline as a Flyte task
├── requirements.txt
└── tools/
    └── search.py       # Tavily web search tool
```

## How It Works

- **`graph.py`** defines two LangGraph graphs:
  - `build_research_subgraph()` — ReAct agent loop (agent ↔ tools) for a single topic
  - `build_pipeline_graph()` — pipeline graph (plan → Send fan-out → synthesize → quality check → loop)
- **`workflow.py`** defines two Flyte tasks:
  - `research_topic` — runs the ReAct subgraph on one topic (the compute unit)
  - `research_pipeline` — runs the pipeline graph, passing `research_topic` as the compute backend

The pipeline graph accepts the Flyte task as a parameter. LangGraph's `Send` API fans out work to it. On a cluster, each `Send` becomes a separate container.

See the [blog post](blog.md) for the full walkthrough.