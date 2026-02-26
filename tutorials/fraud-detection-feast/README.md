# Fraud Detection with Feast + Flyte

A fraud detection ML pipeline that uses **Feast** as a feature store and **Flyte** for orchestration. Uses the Sparkov simulated credit card transactions dataset — real merchant categories, locations, amounts, and user profiles.

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
fraud-detection/
├── README.md               # You are here
├── requirements.txt        # Dependencies
├── config.py               # Flyte environment config
├── prep.py                 # Standalone data prep (optional, for local dev)
├── workflow.py             # Flyte tasks: prepare, train, materialize
└── app.py                  # FastAPI scoring app (consumes pipeline artifacts)
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

Test with different transactions:

```bash
# Normal grocery purchase, local merchant
curl "http://localhost:8080/score?user_id=42&amt=25.00&category=grocery_pos&merch_lat=33.9&merch_long=-80.3"

# Suspiciously large purchase at a far-away merchant
curl "http://localhost:8080/score?user_id=42&amt=9999.99&category=shopping_net&merch_lat=48.8&merch_long=2.3"
```

```json
{
  "user_id": 42,
  "transaction": {"amt": 9999.99, "category": "shopping_net", "hour": 23, "day_of_week": 3},
  "fraud_prediction": "Fraud",
  "fraud_probability": 0.5658,
  "risk_level": "HIGH",
  "signals": [
    "Amount $9999.99 is 49.1 std devs above user avg ($61.63)",
    "Transaction 3923 miles from user's usual area",
    "High-value transaction (>$1,000)"
  ],
  "user_profile": {"txn_count": 1188, "mean_amt": 61.63, "std_amt": 202.33, "age": 38}
}
```

The scoring flow mirrors a real payment system:
- **From the request**: transaction amount, merchant category, merchant location
- **From Feast**: user's spending history, home location, age
- **Computed at scoring time**: amount z-score, distance from home, time features
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
flyte deploy app.py serving_env
```

Every time you retrain (re-run the pipeline), redeploying the app picks up the new model automatically.

### Test the remote endpoint

```bash
curl "https://<app-url>/score?user_id=42&amt=500.00&category=grocery_pos&merch_lat=40.71&merch_long=-74.01"
```

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
