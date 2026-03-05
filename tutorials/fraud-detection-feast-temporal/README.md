# Fraud Detection with Feast + Temporal

A fraud detection ML pipeline that uses **Feast** as a feature store and **Temporal** for orchestration. This is a high-fidelity port of the [Flyte version](../fraud-detection-feast/) to the [Temporal Python SDK](https://docs.temporal.io/develop/python/).

Uses the Sparkov simulated credit card transactions dataset — real merchant categories, locations, amounts, and user profiles.

```
fraud-detection-workflow
  ├── prepare_data           → download dataset, engineer features
  ├── train_model            → XGBoost classifier ──────────────→ model.joblib
  └── materialize_features   → feast apply + materialize ───────→ feast_artifacts/
       (run in parallel)

fraud-scorer (app)
  ├── loads model + feast artifacts from pipeline output
  └── GET /score?user_id=42&amt=500&category=grocery_pos&merch_lat=40.7&merch_long=-74.0
        ├── amt, category, location   ← from the request (current transaction)
        ├── spending profile, home   ← from Feast online store (user history)
        ├── z-score, distance        ← derived at scoring time
        └── → combined features → model score → fraud probability
```

## Project Structure

```
fraud-detection-feast-temporal/
├── README.md           # You are here
├── requirements.txt   # Dependencies
├── shared.py           # Feature definitions, haversine (shared by activities + app)
├── activities.py      # Temporal activities: prepare_data, train_model, materialize_features
├── workflows.py        # FraudDetectionWorkflow (orchestrates activities)
├── worker.py           # Temporal worker process
├── run_workflow.py     # Start the pipeline workflow
├── app.py              # FastAPI scoring app (consumes pipeline artifacts)
├── demo.py             # Gradio UI for interactive fraud scoring
└── prep.py             # Standalone data prep (optional, for local dev)
```

## Temporal Concepts

| Flyte | Temporal |
|-------|----------|
| `@env.task` | `@activity.defn` |
| Flyte workflow (async orchestration) | `@workflow.defn` class with `@workflow.run` |
| `flyte.io.Dir` / `flyte.io.File` | Paths (strings) — artifacts on shared filesystem |
| `asyncio.gather(task_a, task_b)` | `asyncio.gather(execute_activity(...), execute_activity(...))` |
| Task environment (image, resources) | Worker runs activities in process/thread pool |

## Setup

```bash
cd tutorials/fraud-detection-feast-temporal

# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

# Install dependencies
uv pip install -r requirements.txt

# macOS only: XGBoost needs OpenMP
brew install libomp
```

## Step 1: Start Temporal Server

Temporal requires a server. For local development:

```bash
# Install Temporal CLI: https://docs.temporal.io/cli
temporal server start-dev
```

Keep this running in a terminal.

## Step 2: Run the Worker

The worker executes activities and workflows:

```bash
python worker.py
```

Keep this running in a second terminal.

## Step 3: Run the Pipeline

Start the workflow (in a third terminal):

```bash
python run_workflow.py
```

### What happens

1. **`prepare_data`** — Downloads the Sparkov credit card fraud dataset (~500K transactions), engineers features (amount log, category encoding, distance, user aggregates), saves as parquets to `artifacts/<run_id>/`.

2. **`train_model`** and **`materialize_features`** run **in parallel** — both depend only on the prepared data:
   - **train_model** — Joins transaction + user features, computes derived features (amount z-score, distance from home), trains XGBoost with `scale_pos_weight` for class imbalance. Outputs `model.joblib`.
   - **materialize_features** — Creates a Feast feature store, materializes user spending profiles to a SQLite online store. Outputs `feast_artifacts/`.

3. **`copy_artifacts`** — Copies model and Feast dir to `artifacts/latest/` for serving. The run script also copies to `./model.joblib` and `./feast_artifacts` for local app testing.

## Step 4: Run the Scoring App

```bash
python app.py
```

Test with different transactions:

```bash
# Normal grocery purchase
curl "http://localhost:8080/score?user_id=42&amt=25.00&category=grocery_pos&merch_lat=33.9&merch_long=-80.3&hour=14&day_of_week=2"

# Suspiciously large purchase at a far-away merchant, late night
curl "http://localhost:8080/score?user_id=42&amt=9999.99&category=shopping_net&merch_lat=48.8&merch_long=2.3&hour=23&day_of_week=3"
```

## Step 5: Gradio Demo UI

```bash
# Against local app (run python app.py first)
python demo.py

# Against remote scoring API
API_URL=https://your-scoring-endpoint python demo.py
```

## Comparison with Flyte

| Aspect | Flyte | Temporal |
|--------|-------|----------|
| Orchestration | Flyte server (local/remote) | Temporal server |
| Task definition | `@env.task` on async function | `@activity.defn` on async function |
| Parallel execution | `asyncio.gather(task_a(), task_b())` | `asyncio.gather(execute_activity(...), ...)` |
| Data passing | `flyte.io.Dir` / `flyte.io.File` (remote fetch) | Paths on shared filesystem (or S3 for distributed) |
| Caching | `cache="auto"` on tasks | Not built-in; use activity ids / external cache |
| Reporting | `flyte.report.replace.aio(html)` | Activity logging, or external observability |
| Serving | `flyte deploy app.py` + RunOutput | Standalone FastAPI; artifacts via env vars |

## Key Concepts

### Why a Feature Store?

Without Feast, you'd compute features in your training script and re-implement the same logic in your serving code. Feast solves:
- **Training-serving skew** — same features everywhere
- **Data leakage** — point-in-time correct joins for training
- **Low-latency serving** — pre-materialized features in an online store

### Feature Engineering

The model uses three types of features:

| Type | Features | Source |
|------|----------|--------|
| **Transaction** | amt, category, merchant location | Request |
| **User profile** | txn_count, mean_amt, std_amt, home_lat/long, age | Feast |
| **Derived** | amt_zscore, amt_ratio, distance_from_home, hour, day_of_week | Computed |

## Production Considerations

- **Shared storage**: For multi-worker deployments, use S3 or a shared volume so activities can read/write artifacts across workers.
- **Activity timeouts**: Current values (10 min for prepare/train, 5 min for Feast) suit local runs; tune for your environment.
- **Retries**: Temporal activities support retry policies; add `retry_policy` to `execute_activity` for resilient runs.
