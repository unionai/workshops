# Research Agent Workshop

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/unionai/workshops?quickstart=1)

A multi-agent research workflow built with **LangGraph** and **Flyte**. Ask a question, and parallel agents search the web, research sub-topics, and synthesize a final report.

```
research_workflow (orchestrator)
  ├── plan_research → ["topic A", "topic B", "topic C"]
  ├── research_topic("topic A")  ┐
  ├── research_topic("topic B")  ├── parallel Flyte tasks
  ├── research_topic("topic C")  ┘
  └── synthesize_reports → final report
```

Each `research_topic` runs a **LangGraph agent** that loops through search → reason → search cycles using Tavily web search. The agent in `graph.py` is a hand-built ReAct loop using `StateGraph` — not LangGraph's prebuilt `create_react_agent` — so you can see exactly how the agent → tools → agent cycle works under the hood.

## Project Structure

```
agent_research/
├── README.md           # You are here
├── config.py           # Flyte environment + API keys
├── graph.py            # LangGraph agent — search loop with tool calling
├── workflow.py         # Flyte tasks — plan, fan-out research, synthesize
├── app.py              # Gradio UI — serve the workflow as a web app
└── tools/
    └── search.py       # Web search tool (Tavily)
```

---

## Setup

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
cd tutorials/langgraph/agent_research
uv pip install -r requirements.txt
uv pip install keyrings.alt 

```bash
cd agent_research

# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r ../requirements.txt
```

Add your API keys to a `.env` file:

```
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
```

---

## Step 1: Run the Agent Locally

Run the full research workflow from the command line. Flyte executes each task locally on your machine.

```bash
# Quick run — 2 sub-topics, 1 search each
flyte run --local workflow.py research_workflow \
  --query "What are the pros and cons of electric vehicles?" \
  --num_topics 2 --max_searches 1
```

Add `--tui` to see a live terminal UI with task tree, parallel execution, and reports:

```bash
flyte run --local --tui workflow.py research_workflow \
  --query "Compare quantum computing approaches: superconducting vs trapped ion vs photonic"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | required | Research question |
| `--num_topics` | 3 | Number of sub-topics to research in parallel |
| `--max_searches` | 2 | Max web searches per sub-topic |

### What you'll see

- **Without `--tui`**: Log output showing the planner splitting topics, each agent making tool calls, and the final synthesis.
- **With `--tui`**: A Textual-based terminal UI showing a live task tree, parallel tasks running side by side, and expandable HTML reports.

### Try different queries

```bash
flyte run --local --tui workflow.py research_workflow \
  --query "Compare vector databases: Pinecone vs Weaviate vs Chroma"

flyte run --local --tui workflow.py research_workflow \
  --query "What is the current state of AI agents in production?"

flyte run --local --tui workflow.py research_workflow \
  --query "Research the best coffee brewing methods and the science behind them"
```

---

## Step 2: Serve the App Locally

Launch a Gradio web UI that kicks off the same workflow as a local Flyte task.

```bash
RUN_MODE=local python app.py
```

Open http://localhost:7860 in your browser. You get the same research workflow with a UI — sliders for sub-topics and searches, plus example queries.

---

## Step 3: Deploy to a Flyte Cluster

### One-time cluster setup

```bash
# Connect to the cluster
flyte create config \
    --endpoint tryv2.hosted.unionai.cloud \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks

# Store your API keys as secrets
flyte create secret SAGE_OPENAI_API_KEY
flyte create secret SAGE_TAVILY_API_KEY
```

### Run the workflow remotely

Same command as Step 1, just drop the `--local` flag. The workflow runs on the cluster instead of your machine.

```bash
flyte run workflow.py research_workflow \
  --query "Compare quantum computing approaches"
```

### Deploy the Gradio app remotely

Deploy the web UI to the cluster so it's always available — no local machine needed.

```bash
flyte deploy app.py serving_env
```

You can also run the Gradio app locally but have it kick off tasks on the remote cluster:

```bash
python app.py
```

---

## Make It Your Own

- **Change the model** — swap `MODEL = "gpt-4.1-nano"` in `workflow.py` to `gpt-4.1-mini` or `gpt-4.1` for higher quality
- **Edit the system prompt** — the agent's behavior is in `graph.py` in the `system_prompt` string
- **Add a tool** — create a new `@tool` function in `tools/` (see `tools/search.py` for the pattern), import it in `graph.py`, and add it to the `tools` list
- **Adjust parallelism** — try `--num_topics 5` for broader research, or `--num_topics 1` to watch a single agent step by step