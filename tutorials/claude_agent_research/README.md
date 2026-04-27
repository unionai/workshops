# Claude Agent Research Pipeline

A research agent pipeline using **Claude's tool-use API** and **Flyte** for orchestration. Same architecture as the [LangGraph version](../langgraph_agent_research/), but replaces LangGraph + OpenAI with Claude's native tool-use loop.

## Architecture

```
research_pipeline (Flyte orchestrator task)
  ├── plan → Claude breaks query into sub-topics
  ├── research (parallel Flyte tasks)
  │     ├── research_topic("topic A")  ┐
  │     ├── research_topic("topic B")  ├── each runs a Claude ReAct agent
  │     └── research_topic("topic C")  ┘
  ├── synthesize (Flyte task) → Claude combines all reports
  ├── quality_check (Flyte task) → Claude scores + identifies gaps
  │     ├── gaps found → research gaps → synthesize → quality_check again
  │     └── score >= 8 or no gaps → finalize
  └── final report
```

Each `research_topic` task runs a **ReAct-style agent loop** using Claude's native [tool-use API](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview). Claude's tool-use protocol naturally implements the ReAct (Reason → Act → Observe) pattern — no agent framework needed:
1. **Reason** — Claude receives the topic and decides what to search
2. **Act** — Claude returns a `tool_use` block → we execute the Tavily web search
3. **Observe** — Search results are sent back as `tool_result` → Claude incorporates them
4. **Repeat** until Claude has enough info (`stop_reason == "end_turn"`) → returns research summary

## Setup

```bash
cd tutorials/claude_agent_research

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Add your API keys to `.env`:

```
ANTHROPIC_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
```

## Run

```bash
# Local with TUI
flyte run --local --tui workflow.py research_pipeline \
  --query "Compare quantum computing approaches: superconducting vs trapped ion"

# Local without TUI
flyte run --local workflow.py research_pipeline \
  --query "What are the latest advances in fusion energy?" \
  --num_topics 2 --max_searches 2

# Remote (on a Flyte cluster)
flyte run workflow.py research_pipeline \
  --query "Compare quantum computing approaches" \
  --num_topics 3 --max_searches 3 --max_iterations 2
```

For remote runs, create secrets on the cluster:

flyte start devbox

```bash
flyte create secret ANTHROPIC_API_KEY --project flytesnacks --domain development
flyte create secret TAVILY_API_KEY --project flytesnacks --domain development
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | required | Research question |
| `--num_topics` | 3 | Number of sub-topics to research in parallel |
| `--max_searches` | 3 | Max web searches per sub-topic agent |
| `--max_iterations` | 2 | Max quality gate iterations |

## Project Structure

```
claude_agent_research/
├── config.py        # Flyte environment, secrets, resources
├── models.py        # Pydantic data models (TopicReport, QualityResult, PipelineResult)
├── agent.py         # Claude agent — ReAct loop, planning, synthesis, quality eval
├── workflow.py      # Flyte tasks + pipeline orchestrator
└── requirements.txt
```

## How It Works

- **`models.py`** — Pydantic models for structured data flow between tasks:
  - `TopicReport` — topic + research report from a single agent run
  - `QualityResult` — score + identified gaps from quality evaluation
  - `PipelineResult` — final output with report, sub-reports, score, and iteration count
- **`agent.py`** — Claude agent logic (no Flyte dependencies except `@flyte.trace` on search):
  - `run_research_agent()` — ReAct loop: prompt → Claude → tool calls → execute → repeat
  - `plan_topics()` — Claude breaks query into sub-topics
  - `synthesize_reports()` — Claude combines research into unified report
  - `evaluate_quality()` — Claude scores report and identifies gaps
- **`workflow.py`** — Flyte tasks (each visible in the UI while running):
  - `research_topic` → `TopicReport` — runs the Claude ReAct agent on one topic
  - `synthesize` — combines `list[TopicReport]` into a unified synthesis
  - `quality_check` → `QualityResult` — scores the report and identifies gaps
  - `research_pipeline` → `PipelineResult` — orchestrates plan → fan-out → synthesize → quality loop

Flyte natively serializes Pydantic models between tasks, so there's no manual JSON wrangling — just typed data flowing through the pipeline.

## Switching Models

Each Flyte task can use a different Claude model. The `MODEL` variable in `workflow.py` controls which model all tasks use — currently set to `claude-haiku-4-5` for speed. You can assign different models per step, e.g. Haiku for fast research loops and Sonnet for higher-quality synthesis:

```python
# workflow.py
RESEARCH_MODEL = "claude-haiku-4-5"     # fast, good enough for search loops
SYNTHESIS_MODEL = "claude-sonnet-4-6"   # higher quality for final reports
```
