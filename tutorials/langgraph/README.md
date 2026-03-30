# LangGraph Agent Examples

LangGraph agent workflows running on Flyte. Each example is a self-contained workflow with its own graph and Flyte tasks, sharing common tools and config.

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

---

## Research Agent

A multi-agent research workflow that takes a question, breaks it into sub-topics, researches each one in parallel using web search, and synthesizes everything into a final report.

### What we're building

```
research_workflow (orchestrator)
  ├── plan_research → ["topic A", "topic B", "topic C"]
  ├── research_topic("topic A")  ┐
  ├── research_topic("topic B")  ├── parallel Flyte tasks
  ├── research_topic("topic C")  ┘
  └── synthesize_reports → final report
```

Each box above is a **Flyte task** — a unit of work that Flyte tracks, logs, and can run locally or on a cluster. Inside each `research_topic` task, a **LangGraph agent** loops through search → reason → search cycles until it has enough information.

### How it works

There are two key files:

**`agent_research/graph.py`** — The LangGraph agent. Defines a simple loop: the LLM decides whether to call `web_search` (via Tavily) or return a final answer. Each search and routing decision is traced with `@flyte.trace` for observability.

```
agent → (needs more info?) → web_search → agent → ... → final answer
```

**`agent_research/workflow.py`** — The Flyte orchestration. Breaks the work into separate tasks so you can see each step in the TUI/UI:

1. `plan_research` — asks the LLM to split your query into sub-topics
2. `research_topic` — runs the LangGraph agent on one sub-topic (fanned out in parallel via `asyncio.gather`)
3. `synthesize_reports` — combines all sub-topic findings into a final report
4. `research_workflow` — the top-level orchestrator that wires them together

### Run it

All examples use `flyte run`, which auto-discovers task parameters and exposes them as CLI flags.

```bash
# Quick run — fewer topics and searches for a fast demo
flyte run --local agent_research/workflow.py research_workflow \
  --query "What are the pros and cons of electric vehicles?" \
  --num-topics 2 --max-searches 1

# Full run with TUI — shows live task tree, parallel execution, and reports
flyte run --local --tui agent_research/workflow.py research_workflow \
  --query "Compare quantum computing approaches: superconducting vs trapped ion vs photonic"

# Remote — runs on a Flyte cluster (see Remote Setup below)
flyte run agent_research/workflow.py research_workflow \
  --query "Compare quantum computing approaches"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | required | Research question |
| `--num-topics` | 3 | Number of sub-topics to research in parallel |
| `--max-searches` | 2 | Max web searches per sub-topic |

### What you'll see

**Without `--tui`** — log output in your terminal showing the planner splitting topics, each agent making tool calls, and the final synthesis.

**With `--tui`** — a Textual-based terminal UI that shows:
- A live task tree with status indicators for each Flyte task
- Parallel `research_topic` tasks running side by side
- Expandable HTML reports for each task (agent graph visualization, sub-topic reports, final synthesis)
- Traces for each `web_search` call and agent routing decision

### Try different queries

```bash
# Compare tools or technologies
flyte run --local --tui agent_research/workflow.py research_workflow \
  --query "Compare vector databases: Pinecone vs Weaviate vs Chroma"

# Research a company or product
flyte run --local --tui agent_research/workflow.py research_workflow \
  --query "What is Anthropic and how does Claude compare to other LLMs?"

# Explore a trend
flyte run --local --tui agent_research/workflow.py research_workflow \
  --query "What is the current state of AI agents in production?"

# Make a decision
flyte run --local --tui agent_research/workflow.py research_workflow \
  --query "Pros and cons of moving from REST to GraphQL"

# Just for fun
flyte run --local --tui agent_research/workflow.py research_workflow \
  --query "Research the best coffee brewing methods and the science behind them"
```

### Make it your own

A few things to try after the first run:

- **Change the model** — swap `MODEL = "gpt-4.1-nano"` in `workflow.py` to `gpt-4.1-mini` or `gpt-4.1` for higher quality results
- **Edit the system prompt** — the agent's behavior is defined in `graph.py` in the `system_prompt` string. Try making it more focused (e.g., "only search for academic sources") or giving it a persona
- **Add a tool** — create a new `@tool` function in `tools/` (see `tools/search.py` for the pattern), import it in `graph.py`, and add it to the `tools` list. The LLM will automatically discover it via tool calling
- **Adjust parallelism** — try `--num-topics 5` for broader research, or `--num-topics 1` to watch a single agent work step by step

### Serve it

The same workflow that runs via `flyte run` can be served as a Gradio web app. The app (`agent_research/app.py`) calls `research_workflow` as a Flyte task, so you get the same reports, tracing, and caching — just with a UI on top.

```bash
# Local app + local task — everything runs on your machine
RUN_MODE=local python -m agent_research.app

# Local app + remote task — Gradio runs locally, workflow runs on the cluster
python -m agent_research.app

# Deploy to cluster — Gradio app runs on the cluster too
flyte deploy agent_research/app.py serving_env
```

The app includes sliders for sub-topics and searches per topic, plus example queries to get started.

---

## Research Agent Pipeline

A research pipeline that integrates LangGraph and Flyte as co-orchestrators. LangGraph controls the pipeline logic — planning, dynamic fan-out via `Send`, quality gates, and iterative deepening. Flyte provides the compute — each researcher runs as a separate task.

### What we're building

```
research_pipeline (LangGraph pipeline graph)
  ├── plan → split query into sub-topics
  ├── research (Send fan-out → Flyte tasks)
  │     ├── research_topic("topic A")  ┐
  │     ├── research_topic("topic B")  ├── parallel Flyte tasks
  │     └── research_topic("topic C")  ┘
  ├── synthesize → combine into report
  ├── quality_check → score + identify gaps
  │     ├── gaps found → research again (new Flyte tasks)
  │     └── good enough → finalize
  └── finalize → final report
```

### How it works

**`agent_research_pipeline/graph.py`** — Two LangGraph graphs: a ReAct research subgraph (agent ↔ tools loop) and a pipeline graph (plan → Send fan-out → synthesize → quality check → loop). The pipeline graph accepts a Flyte task as a parameter — this is how LangGraph dispatches to Flyte compute.

**`agent_research_pipeline/workflow.py`** — Flyte tasks that the pipeline dispatches to. `research_topic` runs the ReAct agent on one sub-topic. `research_pipeline` builds the LangGraph pipeline and invokes it.

### Run it

```bash
# Local with TUI
flyte run --local --tui agent_research_pipeline/workflow.py research_pipeline \
  --query "Compare quantum computing approaches: superconducting vs trapped ion"

# Remote
flyte run agent_research_pipeline/workflow.py research_pipeline \
  --query "Compare quantum computing approaches" \
  --num-topics 5 --max-searches 3 --max-iterations 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | required | Research question |
| `--num-topics` | 3 | Number of sub-topics to research in parallel |
| `--max-searches` | 2 | Max web searches per sub-topic |
| `--max-iterations` | 2 | Max quality gate iterations |

---

## More Examples

### ReAct Agent

Reason → Act → Observe loop with math and string tools. The agent breaks down problems step-by-step, choosing the right tool at each stage.

```
react_agent
  └── agent →(tool calls?)→ tools → agent →(loop)→ END
       Tools: add, multiply, power, word_count, letter_count, reverse_string
```

```bash
flyte run --local --tui agent_react/workflow.py react_agent \
  --request "How many words are in 'the quick brown fox jumps over the lazy dog' multiplied by 5?"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--request` | required | Task for the agent to solve |
| `--max-steps` | 10 | Max reasoning steps |

### Reflection Agent

Generate → critique → refine loop. The agent writes a response, scores it, and iterates until quality meets the threshold or max iterations are reached.

```
reflection_agent
  └── generate → critique →(score < threshold?)→ generate →(loop)→ END
```

```bash
flyte run --local --tui agent_reflection/workflow.py reflection_agent \
  --request "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--request` | required | Task for the agent to accomplish |
| `--quality-threshold` | 8 | Min quality score (1-10) to stop refining |
| `--max-iterations` | 3 | Max refinement iterations |

---

## Remote Setup (Flyte cluster)

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

# Run on cluster (omit --local)
flyte run agent_research/workflow.py research_workflow --query "Compare quantum computing approaches"
```

## Project Structure

```
tutorials/langgraph/
├── config.py              # Shared Flyte env + API keys
├── requirements.txt       # Shared dependencies
├── tools/
│   ├── __init__.py
│   ├── search.py          # web_search tool (Tavily)
│   ├── math.py            # add, multiply, power
│   └── string.py          # word_count, letter_count, reverse_string
├── agent_research/
│   ├── __init__.py
│   ├── graph.py           # LangGraph agent — search loop with tool calling
│   ├── workflow.py        # Flyte tasks — plan, fan-out research, synthesize
│   └── app.py             # Gradio UI — serve the workflow as a web app
├── agent_research_pipeline/
│   ├── config.py          # Flyte environment, secrets, resources
│   ├── graph.py           # LangGraph graphs — pipeline + ReAct subgraph
│   ├── workflow.py        # Flyte tasks — research_topic + pipeline orchestrator
│   └── tools/
│       └── search.py      # Tavily web search tool
├── agent_react/
│   ├── __init__.py
│   ├── graph.py           # ReAct graph with math + string tools
│   └── workflow.py        # Single task + reasoning trace report
└── agent_reflection/
    ├── __init__.py
    ├── graph.py           # Reflection graph with generate + critique loop
    └── workflow.py        # Single task + iteration history report
```

Tools in `tools/` are shared across most examples. `agent_research_pipeline/` is fully self-contained with its own config, tools, and dependencies. Each example folder has its own `graph.py` (LangGraph agent) and `workflow.py` (Flyte tasks).