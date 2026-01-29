# LangGraph Agent Examples

LangGraph agent workflows running on Flyte. Each example is a self-contained workflow with its own graph and Flyte tasks, sharing common tools and config.

## Examples

### Research Agent

Parallel research agents that fan out across sub-topics, each using tool calling to search the web autonomously, then a synthesizer combines all findings.

```
research_workflow (orchestrator)
  ├── plan_research → ["topic A", "topic B", "topic C"]
  ├── research_topic("topic A")  ┐
  ├── research_topic("topic B")  ├── parallel Flyte tasks
  ├── research_topic("topic C")  ┘
  └── synthesize_reports → final report
```

```bash
python -m agent_research.workflow --local --query "Compare quantum computing approaches: superconducting vs trapped ion vs photonic"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--local` | off | Run locally instead of on Flyte cluster |
| `--query` | required | Research question |
| `--num-topics` | 3 | Number of sub-topics to research in parallel |
| `--max-searches` | 2 | Max web searches per sub-topic |

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

### Remote (Flyte cluster)

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

# Run without --local flag
python -m agent_research.workflow --query "Compare quantum computing approaches"
```

## Project Structure

```
tutorials/langgraph/
├── config.py              # Shared Flyte env + API keys
├── requirements.txt       # Shared dependencies
├── tools/
│   ├── __init__.py
│   └── search.py          # web_search tool (reusable across agents)
├── agent_research/
│   ├── __init__.py
│   ├── graph.py           # LangGraph agent with tool calling
│   └── workflow.py        # Flyte tasks + CLI
└── agent_reflection/      # (coming soon)
```

Tools in `tools/` are shared across all examples. Each example folder has its own `graph.py` (LangGraph StateGraph) and `workflow.py` (Flyte tasks + CLI).
