# Fraud Detection with Feast + Flyte

A fraud detection ML pipeline that uses **Feast** as a feature store and **Flyte** for orchestration. Uses the Sparkov simulated credit card transactions dataset — real merchant categories, locations, amounts, and user profiles.

<a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/fraud-detection-feast/ml-fraud-tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


```
fraud_detection_pipeline (orchestrator)
  ├── prepare_data           → download dataset, engineer features
  ├── train_model            → XGBoost classifier ──────────────→ model.joblib
  └── materialize_features   → feast apply + materialize ───────→ feast_artifacts/

fraud-scorer (app)
  ├── loads model + feast artifacts from pipeline (RunOutput)
  └── GET /score?user_id=42&amt=500&category=grocery_pos&merch_lat=40.7&merch_long=-74.0
        ├── amt, category, location   ← from the request (current transaction)
        ├── spending profile, home    ← from Feast online store (user history)
        ├── z-score, distance         ← derived at scoring time
        └── → combined features → model score → fraud probability
```

## Project Structure

```
fraud-detection-feast/
├── README.md               # You are here
├── requirements.txt        # Dependencies
├── config.py               # Flyte environment config
├── prep.py                 # Standalone data prep (optional, for local dev)
├── workflow.py             # Flyte tasks: prepare, train, materialize
├── report_helpers.py       # Styled HTML report components
├── app.py                  # FastAPI scoring app (consumes pipeline artifacts)
└── dashboard.py            # Gradio dashboard for interactive fraud scoring
```

---

## Setup

```bash
cd tutorials/fraud-detection-feast

# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# macOS only: XGBoost needs OpenMP
brew install libomp
```

---

## Step 1: Run the Pipeline Locally

The pipeline downloads the dataset, engineers features, trains an XGBoost model, and materializes user profiles to a Feast online store — all as Flyte tasks.

```bash
flyte run --local workflow.py fraud_detection_pipeline
```

Add `--tui` for a live terminal UI:

```bash
flyte run --local --tui workflow.py fraud_detection_pipeline
```

### What happens

1. **`prepare_data`** — Downloads the Sparkov credit card fraud dataset (~500K transactions), engineers features (amount log, category encoding, distance, user aggregates), saves as parquets.

2. **`train_model`** — Joins transaction + user features, computes derived features (amount z-score, distance from home), trains XGBoost with `scale_pos_weight` for class imbalance. Reports AUC-ROC, confusion matrix, and feature importance. Outputs `model.joblib`.

3. **`materialize_features`** — Creates a Feast feature store, materializes user spending profiles (avg amount, txn count, home location, age) to a SQLite online store. Outputs `feast_artifacts/`.

Steps 2 and 3 run **in parallel** since they both depend only on step 1.

---

## Step 2: Run the Scoring App Locally

The pipeline saves `model.joblib` and `feast_artifacts/` to the working directory, so the app can pick them up directly.

```bash
python app.py
```

Test with different transactions (pass `hour` and `day_of_week` to control the transaction time):

```bash
# Normal grocery purchase, local merchant, afternoon
curl "http://localhost:8080/score?user_id=42&amt=25.00&category=grocery_pos&merch_lat=33.9&merch_long=-80.3&hour=14&day_of_week=2"

# Suspiciously large purchase at a far-away merchant, late night
curl "http://localhost:8080/score?user_id=42&amt=9999.99&category=shopping_net&merch_lat=48.8&merch_long=2.3&hour=23&day_of_week=3"
```

```json
{
  "user_id": 42,
  "transaction": {"amt": 9999.99, "category": "shopping_net", "hour": 23, "day_of_week": 3},
  "fraud_prediction": "Fraud",
  "fraud_probability": 0.8,
  "model_probability": 0.5658,
  "risk_level": "HIGH",
  "signals": [
    "Rule override: z-score 49.1 + distance 3923mi → min 80%",
    "Amount $9999.99 is 49.1 std devs above user avg ($61.63)",
    "Transaction 3923 miles from user's usual area",
    "High-value transaction (>$1,000)"
  ],
  "scoring": {"amt_zscore": 49.13, "distance_from_home": 3923.0, "rule_applied": true},
  "user_profile": {"txn_count": 1188, "mean_amt": 61.63, "std_amt": 202.33, "age": 38}
}
```

The scoring flow mirrors a real payment system:
- **From the request**: transaction amount, merchant category, merchant location, time
- **From Feast**: user's spending history, home location, age
- **Computed at scoring time**: amount z-score, distance from home
- **Rule overrides**: business rules catch extreme cases the model can't extrapolate to
- **Signals**: human-readable flags explaining why a transaction looks suspicious

---

## Step 3: Deploy Remotely (on Flyte cluster)

### Run the pipeline remotely

```bash
flyte run workflow.py fraud_detection_pipeline
```

### Deploy the scoring app

The app uses `RunOutput` to automatically pull the latest model and Feast artifacts from the pipeline:

```bash
# Deploy with latest pipeline artifacts
flyte deploy app.py serving_env

# Pin to a specific pipeline run
flyte deploy app.py serving_env -- --run-name <run_name>
```

Every time you retrain (re-run the pipeline), redeploying the app picks up the new model automatically. Pass `--run-name` to pin to a specific version (useful for rollbacks or A/B testing).

### Test the remote endpoint

```bash
curl "https://<app-url>/score?user_id=42&amt=500.00&category=grocery_pos&merch_lat=40.71&merch_long=-74.01&hour=22&day_of_week=3"
```

---

## Step 4: Fraud Dashboard

An interactive dashboard that calls the scoring API. Works locally and deployed.

### Run locally

```bash
# Against local app (run `python app.py` in another terminal first)
python dashboard.py

# Against remote scoring API
API_URL=https://<app-url> python dashboard.py
```

### Deploy to Flyte

The dashboard uses `AppEndpoint` to auto-discover the scoring app URL by name — no hardcoded URLs needed:

```bash
# Auto-discover the fraud-scorer app endpoint
flyte deploy dashboard.py dashboard_env

# Or point at a specific scoring URL
flyte deploy dashboard.py dashboard_env -- --api-url https://<app-url>
```

Under the hood, `AppEndpoint(app_name="fraud-scorer")` resolves the deployed scoring app's endpoint automatically. Pass `--api-url` to override (e.g., if you have multiple scoring app versions deployed).

The dashboard uses a uvicorn factory pattern to avoid Gradio pickle issues — Flyte pickles a bare FastAPI app, but uvicorn builds the full Gradio app fresh on the worker.

Opens a browser UI where you can adjust user ID, amount, category, and merchant location — then see the fraud prediction, risk signals, and user profile in real time.

---

## Key Concepts

### Why a Feature Store?

Without Feast, you'd compute features in your training script and re-implement the same logic in your serving code. Feast solves:
- **Training-serving skew** — same features everywhere
- **Data leakage** — point-in-time correct joins for training
- **Low-latency serving** — pre-materialized features in an online store

In this example, the scoring endpoint combines two sources:
- **Request-time features** — transaction amount, category, merchant location
- **Feast features** — user's spending profile, home location, age

This is exactly how production fraud systems work: you can't pre-compute the current transaction's details, but you need the user's profile to contextualize it.

### The RunOutput Pattern

The app doesn't bundle data or models — it pulls them from the latest pipeline run:

```python
Parameter(
    name="model",
    value=RunOutput(task_name="fraud-detection-env.fraud_detection_pipeline", type="file", getter=(0,)),
    download=True,
    env_var="MODEL_PATH",
)
```

This means every time you retrain (re-run the pipeline), deploying the app picks up the new model automatically.

### XGBoost Configuration

The model uses [XGBoost](https://xgboost.readthedocs.io/) — a gradient-boosted tree algorithm that's a go-to for tabular fraud detection. The key parameters:

| Parameter | Value | Why |
|-----------|-------|-----|
| `n_estimators` | 300 | More trees = better separation between fraud/legit |
| `max_depth` | 6 | Deep enough to learn complex patterns without overfitting |
| `scale_pos_weight` | auto | Compensates for class imbalance (~0.6% fraud) by upweighting fraud samples |
| `min_child_weight` | 5 | Prevents splits on tiny groups — reduces false positives |
| `gamma` | 1 | Regularization that prunes weak splits — further reduces false positives |

The `min_child_weight` and `gamma` parameters are the key tuning knobs for fraud precision. Without them, the model over-triggers on edge cases (33% precision). With them, precision jumps to ~62% while recall stays above 90%.

Tree-based models like XGBoost can't extrapolate beyond their training data ranges, so the scoring app adds **rule overrides** for extreme cases (e.g., z-score > 10 + distance > 500mi). This is standard practice in production fraud systems.

### Understanding the Metrics

Fraud detection is an imbalanced classification problem — only ~0.6% of transactions are fraud. The standard metrics tell different parts of the story:

| Metric | What it measures | Fraud detection context |
|--------|-----------------|----------------------|
| **Precision** | Of all transactions flagged as fraud, how many actually were? | Low precision = too many false alarms, frustrating legitimate customers |
| **Recall** | Of all actual fraud, how much did we catch? | Low recall = fraud slipping through undetected |
| **AUC-ROC** | Overall ability to distinguish fraud from legit across all thresholds | High AUC = the model has learned meaningful patterns |
| **False Positives** | Legit transactions incorrectly flagged as fraud | Each one is a blocked card or a phone call to a confused customer |

There's always a **precision-recall tradeoff** — catching more fraud (higher recall) means casting a wider net, which flags more legitimate transactions too (lower precision). The right balance depends on your business:

- **Banks** tend to favor recall — missing fraud is expensive (chargebacks, liability, trust)
- **E-commerce** may favor precision — blocking legitimate purchases costs revenue
- **Rule overrides** in the scoring app act as a safety net for cases the model can't handle

### Feature Engineering

The model uses three types of features:

| Type | Features | Source |
|------|----------|--------|
| **Transaction** | amt, category, merchant location | Request |
| **User profile** | txn_count, mean_amt, std_amt, home_lat/long, age | Feast |
| **Derived** | amt_zscore, amt_ratio, distance_from_home, hour, day_of_week | Computed |

The **derived features** are the key — they answer "how unusual is this transaction *for this user*?" A $500 grocery purchase is normal for one user but suspicious for another.

---

## Make It Your Own

- **Try different models** — swap XGBoost for `RandomForestClassifier` or `LightGBM` in `workflow.py`
- **Add features** — add new fields like `city_pop` or merchant-level aggregates
- **Tune the threshold** — adjust the probability cutoffs in `app.py` for risk levels
- **Switch to Redis** — change the online store from SQLite to Redis for production latency
- **Add on-demand features** — compute features at request time (e.g., velocity checks)
