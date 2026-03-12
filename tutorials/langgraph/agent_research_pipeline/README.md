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

Each `research_topic` task runs a LangGraph ReAct agent that searches the web via [Tavily](https://tavily.com/) and loops until it has enough information.

## Setup

```bash
cd tutorials/langgraph/agent_research_pipeline

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Add your API keys to `.env`:

```
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
```

## Run

```bash
# Local with TUI
flyte run --local --tui workflow.py research_pipeline \
  --query "Compare quantum computing approaches: superconducting vs trapped ion"

# Local without TUI
flyte run --local workflow.py research_pipeline \
  --query "What are the pros and cons of electric vehicles?" \
  --num-topics 2 --max-searches 1

# Remote (on a Flyte cluster)
flyte run workflow.py research_pipeline \
  --query "Compare quantum computing approaches" \
  --num-topics 5 --max-searches 3 --max-iterations 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | required | Research question |
| `--num-topics` | 3 | Number of sub-topics to research in parallel |
| `--max-searches` | 2 | Max web searches per sub-topic |
| `--max-iterations` | 2 | Max quality gate iterations |

## Project Structure

```
agent_research_pipeline/
├── config.py           # Flyte environment, secrets, resources
├── graph.py            # LangGraph graphs — pipeline + ReAct subgraph
├── workflow.py         # Flyte tasks — research_topic + research_pipeline orchestrator
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