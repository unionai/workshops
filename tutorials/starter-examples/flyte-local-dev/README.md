# Flyte: Local Dev to Production

Build, test, and iterate locally — then deploy the same code to a Flyte cluster with GPUs. No rewrites.

## What's Here

| Script | What it does |
|--------|-------------|
| `ml_pipeline.py` | Train ResNet18 on MNIST with caching, retries, HTML reports, and TUI |
| `serve_model.py` | Serve predictions via FastAPI — locally or on a cluster |
| `agent_research.py` | LangGraph agent with DuckDuckGo search, caching, tracing, and reports |
| `agent_app.py` | Gradio UI that kicks off the agent as a Flyte task |
| `wandb_ml_pipeline.py` | ML pipeline with W&B experiment tracking via the Flyte W&B plugin |

## Setup

```bash
cd tutorials/starter-examples/flyte-local-dev

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

Set your OpenAI API key (for agent example):

```bash
export OPENAI_API_KEY=your-key
# or create a .env file with OPENAI_API_KEY=your-key
```

---

## Local Development

Everything runs on your machine — no cluster, no Docker.

### Train with TUI

```bash
# First run — downloads data, trains model, generates HTML report
flyte run --local --tui ml_pipeline.py pipeline --epochs 5 --lr 0.001

# Auto-open the report in your browser
flyte run --local --tui ml_pipeline.py pipeline --epochs 5 --lr 0.001 --open_report

# Change hyperparameters — data download is cached, only training re-runs
flyte run --local --tui ml_pipeline.py pipeline --epochs 10 --lr 0.0005 --batch_size 128
```

### Serve Locally

```bash
# Train first (saves model.pt)
flyte run --local ml_pipeline.py pipeline --epochs 5 --lr 0.001

# Serve predictions
python serve_model.py

# Test it
curl "http://localhost:8080/predict?index=42"
```

### Browse Past Runs

```bash
flyte start tui
```

### Research Agent (CLI)

```bash
flyte run --local --tui agent_research.py agent --request "What is the population of France and what is 10% of it?"
```

### Research Agent (Gradio UI)

```bash
# Local app + remote task (default) — kicks off agent on the cluster
python agent_app.py

# Fully local (no cluster needed)
RUN_MODE=local python agent_app.py
```

Open the printed URL in your browser, type a question, and the app kicks off the agent as a Flyte task. You get a clickable link to watch it execute on the platform. Set `RUN_MODE=local` for fully offline development.

### ML Pipeline with W&B Tracking

```bash
# Set your W&B API key
export WANDB_API_KEY=your-key
# or add WANDB_API_KEY=your-key to your .env file

# Train with W&B experiment tracking
flyte run --local --tui wandb_ml_pipeline.py pipeline --epochs 5 --lr 0.001
```

Same pipeline as `ml_pipeline.py` but every metric is logged to Weights & Biases. The `@wandb_init` decorator on the parent task creates a W&B run, and child tasks automatically share it — all training and evaluation metrics end up in one run.

---

## Deploy to Production

The same code runs on a remote Flyte cluster — swap `--local` for cluster execution.

### Train on the Cluster (with GPUs)

```bash
flyte run ml_pipeline.py pipeline --epochs 5 --lr 0.001
```

The `TaskEnvironment` already defines the image, resources, and GPU — Flyte builds the container and schedules the work.

### Deploy the Model as an API

```bash
flyte deploy serve_model.py serving_env
```

The `RunOutput` parameter automatically resolves the trained model from the latest pipeline run — no manual file paths. The same `lifespan` that loads `model.pt` locally now loads the model from Flyte's artifact store.

```bash
curl "https://your-app.apps.your-cluster.cloud/predict?index=42"
```

### Deploy the Agent UI

```bash
# Create the secret for the OpenAI API key
flyte create secret SAGE_OPENAI_API_KEY <your-key>

flyte deploy agent_app.py serving_env
```

The Gradio app runs on the cluster and kicks off the agent as a Flyte task. Each query gets a tracked run with full observability on the platform.

### Train with W&B on the Cluster

```bash
# Create the secret for the W&B API key
flyte create secret wandb_api_key <your-key>

flyte run wandb_ml_pipeline.py pipeline --epochs 5 --lr 0.001
```

Every metric is tracked in W&B and every task is tracked on the Flyte platform — full observability across both systems.

### Serve for Development (Remote)

```bash
flyte serve serve_model.py serving_env
```

Like `deploy` but designed for iteration — lets you override parameters dynamically.

---

## Key Concepts

| Feature | Local | Remote |
|---------|-------|--------|
| **Run pipeline** | `flyte run --local` | `flyte run` |
| **TUI** | `--tui` flag | Dashboard in UI |
| **Caching** | `cache="auto"` — local SQLite | `cache="auto"` — cluster cache |
| **Retries** | `retries=N` — with exponential backoff | `retries=N` — Flyte managed retries |
| **Reports** | `report=True` — local HTML file | `report=True` — in Flyte UI |
| **Serve** | `python serve_model.py` | `flyte deploy serve_model.py serving_env` |
| **Agent UI** | `python agent_app.py` | `flyte deploy agent_app.py serving_env` |
| **Model loading** | Falls back to `model.pt` on disk | `RunOutput` resolves from pipeline |
| **Secrets** | `.env` file / `load_dotenv()` | `flyte create secret` / `flyte.Secret` |
| **W&B plugin** | `@wandb_init` + `WANDB_API_KEY` env var | `@wandb_init` + `flyte.Secret` |
| **Compute** | Your CPU/GPU | `Resources(cpu=2, memory="4Gi", gpu=1)` |