# Flyte Local Dev Features

Everything you get from `pip install flyte` without a cluster — TUI, caching, reports, tracing, and local serving.

## Examples

| Script | Feature | What it does |
|--------|---------|-------------|
| `research_agent.py` | Caching + Reports + TUI | Research agent with DuckDuckGo search and calculator tools |
| `cached_ml_pipeline.py` | Caching + Reports + TUI | PyTorch MNIST pipeline with training curves and hyperparameter report |
| `serve_model.py` | Local Serving | Serve MNIST digit predictions via FastAPI |

## Setup

```bash
cd tutorials/starter-examples/flyte-local-dev

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

The included `.flyte/config.yaml` enables local run persistence so `flyte start tui` can browse past runs.

Set your OpenAI API key (for agent examples):

```bash
export OPENAI_API_KEY=your-key
# or create a .env file with OPENAI_API_KEY=your-key
```

## Run

### Research Agent

```bash
# First run — searches web, calls OpenAI, generates reasoning trace report
flyte run --local --tui research_agent.py agent --request "What is the population of France and what is 10% of it?"

# Same question — cache hit, returns instantly
flyte run --local --tui research_agent.py agent --request "What is the population of France and what is 10% of it?"

# Different question — fresh run
flyte run --local --tui research_agent.py agent --request "What is the GDP of Japan in USD?"
```

Open the `report.html` from the output path to see the full reasoning trace.

### ML Pipeline with Reports

```bash
# Run with TUI — load_data is cached after first run
flyte run --local --tui cached_ml_pipeline.py pipeline --epochs 5 --lr 0.001

# Change hyperparameters — data download is still cached
flyte run --local --tui cached_ml_pipeline.py pipeline --epochs 10 --lr 0.0005 --batch_size 128
```

Open the `report.html` from the output path to see training curves, hyperparameters, and test results.

### Local Serving

```bash
# Train the model first
flyte run --local cached_ml_pipeline.py pipeline --epochs 5 --lr 0.001

# Serve predictions
python serve_model.py
```

Then hit the endpoint:

```bash
curl "http://localhost:8080/predict?index=42"
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
