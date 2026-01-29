# LangGraph Multi-Agent Research Workflow

Parallel research agents powered by LangGraph + Flyte. Each agent uses tool calling to search the web autonomously, then a synthesizer combines all findings.

```
research_workflow (orchestrator)
  ├── plan_research → ["topic A", "topic B", "topic C"]
  ├── research_topic("topic A")  ┐
  ├── research_topic("topic B")  ├── parallel Flyte tasks
  ├── research_topic("topic C")  ┘
  └── synthesize_reports → final report
```

Each `research_topic` runs a LangGraph agent that decides when and what to search using Tavily as a bound tool.

## Setup

```bash
cd tutorials/langgraph

# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

Add your API keys to a `.env` file:

```
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
```

## Run the Agent

**Local:**

```bash
python -m workflow --local --query "Compare quantum computing approaches: superconducting vs trapped ion vs photonic"
```

**Remote (Flyte cluster):**

```bash
# Connect to cluster (one-time setup)
flyte create config \
    --endpoint tryv2.hosted.unionai.cloud \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks

# Store secrets
flyte create secret OPENAI_API_KEY
flyte create secret TAVILY_API_KEY

# Run
python -m workflow --query "Compare quantum computing approaches"
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--local` | off | Run locally instead of on Flyte cluster |
| `--query` | required | Research question |
| `--num-topics` | 3 | Number of sub-topics to research in parallel |
| `--max-searches` | 2 | Max web searches per sub-topic |

## Files

| File | Purpose |
|------|---------|
| `graph.py` | LangGraph agent with `@tool` web search, `ToolNode`, and `MessagesState` |
| `workflow.py` | Flyte tasks: planner, parallel researchers, synthesizer, orchestrator |
| `config.py` | Flyte `TaskEnvironment` and API key loading |
| `requirements.txt` | Dependencies |