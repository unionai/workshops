# Flyte Local Dev Features

Everything you get from `pip install flyte` without a cluster — TUI, caching, reports, tracing, and local serving.

## Examples

| Script | Feature | What it does |
|--------|---------|-------------|
| `cached_agent.py` | Caching + TUI | ReAct agent with cached LLM calls |
| `cached_ml_pipeline.py` | Caching + Reports + TUI | ML pipeline with cached steps and evaluation report |
| `agent_with_report.py` | Reports | Agent that logs its reasoning trace as an HTML report |
| `serve_model.py` | Local Serving | Train a model and serve predictions via FastAPI |

## Setup

```bash
cd tutorials/starter-examples/flyte-local-dev

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

Set your OpenAI API key (for agent examples):

```bash
export OPENAI_API_KEY=your-key
# or create a .env file with OPENAI_API_KEY=your-key
```

## Run

### Caching + TUI

```bash
# First run — calls OpenAI
flyte run --local --tui cached_agent.py agent --request "What is 12 * 7 plus 3?"

# Second run — cache hit, returns instantly
flyte run --local --tui cached_agent.py agent --request "What is 12 * 7 plus 3?"
```

### ML Pipeline with Reports

```bash
# Run with TUI — load_data and split_data get cached
flyte run --local --tui cached_ml_pipeline.py pipeline --n_neighbors 3

# Change hyperparameters — cached steps are skipped
flyte run --local --tui cached_ml_pipeline.py pipeline --n_neighbors 5
```

### Agent with Report

```bash
flyte run --local agent_with_report.py agent --request "What is 12 * 7 plus 3?"
```

Open the `report.html` from the output path in your browser to see the agent trace.

### Local Serving

```bash
python serve_model.py
```

Then hit the endpoint:

```bash
curl "http://localhost:8080/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2"
```

### Browse Past Runs

```bash
flyte start tui
```

## Key Concepts

- **`cache="auto"`** — Cache task outputs in local SQLite, skip recomputation on same inputs
- **`report=True`** — Generate HTML reports from tasks, saved alongside output
- **`@flyte.trace`** — Sub-task observability, shows as child nodes in the TUI
- **`--tui`** — Interactive terminal dashboard for local runs
- **`flyte start tui`** — Browse past local runs
- **`flyte.with_servecontext(mode="local").serve()`** — Serve a FastAPI app locally
