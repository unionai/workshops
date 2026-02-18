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

**Local with TUI:**

The TUI gives you a live interactive terminal dashboard showing each task's status, logs, and outputs as the run progresses.

```bash
uv pip install textual

uv run flyte run --local --tui langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"
```

Browse past local runs in TUI:
```bash
uv run flyte start tui
```

## Fetch remote run output

After a remote run completes, you can fetch the output with `flyte.remote`:

```python
import flyte
from flyte.remote import Run

flyte.init_from_config()

runs = list(Run.listall(task_name="langgraph_env.agent", sort_by=("created_at", "desc"), limit=1))
run = Run.get(runs[0].name)
print(f"Phase: {run.phase}")
if run.phase.name == "SUCCEEDED":
    print(run.outputs())
```

## Requirements

- `OPENAI_API_KEY` secret configured on your Flyte cluster