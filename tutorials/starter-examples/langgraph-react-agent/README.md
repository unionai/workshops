# LangGraph ReAct Agent

A ReAct agent using LangGraph's prebuilt `create_react_agent` with math tools, running on Flyte.

## What it does

- Creates a ReAct (Reason + Act) agent with OpenAI and LangGraph
- Defines simple math tools (`add`, `multiply`) with `@flyte.trace` for observability
- The agent reasons about which tool to call, observes results, and loops until it has an answer

## Setup

```bash
cd tutorials/starter-examples/langgraph-react-agent

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

## Flyte Cluster (for remote runs)

To run remotely, configure your Flyte cluster endpoint:

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

Don't have a cluster? Request access at [flyte.org](https://flyte.org/).

## Run

**Remote:**
```bash
uv run flyte run langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"
```

**Local:**
```bash
uv run flyte run --local langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"
```

## Requirements

- `OPENAI_API_KEY` secret configured on your Flyte cluster