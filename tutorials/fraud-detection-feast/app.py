"""Fraud scoring app — loads model and Feast artifacts from the pipeline.

Simulates a real payment fraud detection flow:
  - Transaction details (amount, category, merchant location) come from the request
  - User profile (spending history, home location, age) comes from Feast online store
  - Derived features (z-score, distance from home) are computed at scoring time
  - All features are combined and scored by the trained XGBoost model

Remote: flyte deploy app.py serving_env
Local:  python app.py  (requires running the pipeline first)
"""

import math
import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

import flyte
from flyte.app import Parameter, RunOutput
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH_ENV = "MODEL_PATH"
FEAST_DIR_ENV = "FEAST_DIR"

# ------------------------------------------------------------------
# Features fetched from Feast online store (user spending profile)
# ------------------------------------------------------------------
FEAST_FEATURES = [
    "user_stats:txn_count",
    "user_stats:mean_amt",
    "user_stats:std_amt",
    "user_stats:max_amt",
    "user_stats:home_lat",
    "user_stats:home_long",
    "user_stats:age",
]


def haversine(lat1, lon1, lat2, lon2):
    """Compute distance in miles between two (lat, lon) points."""
    R = 3959
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and Feast feature store on startup."""
    from feast import FeatureStore

    # Load model
    model_path = os.environ.get(MODEL_PATH_ENV, "model.joblib")
    logger.info(f"Loading model from {model_path}")
    artifacts = joblib.load(model_path)
    app.state.model = artifacts["model"]
    app.state.feature_cols = artifacts["feature_cols"]
    app.state.category_mapping = artifacts["category_mapping"]

    # Load Feast feature store
    feast_dir = os.environ.get(FEAST_DIR_ENV, "feast_artifacts")
    logger.info(f"Loading Feast store from {feast_dir}")
    app.state.store = FeatureStore(repo_path=feast_dir)

    logger.info("Model and feature store loaded.")
    logger.info(f"Categories: {list(app.state.category_mapping.keys())}")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Fraud Scorer", lifespan=lifespan)

# The task name follows the pattern: "{env_name}.{function_name}"
PIPELINE_TASK = "fraud-detection-env.fraud_detection_pipeline"

serving_env = FastAPIAppEnvironment(
    name="fraud-scorer",
    app=app,
    description="Real-time fraud scoring using Feast online features",
    parameters=[
        # Model file — output[0] from fraud_detection_pipeline
        Parameter(
            name="model",
            value=RunOutput(task_name=PIPELINE_TASK, type="file", getter=(0,)),
            download=True,
            env_var=MODEL_PATH_ENV,
        ),
        # Feast artifacts directory — output[1] from fraud_detection_pipeline
        Parameter(
            name="feast_store",
            value=RunOutput(task_name=PIPELINE_TASK, type="directory", getter=(1,)),
            download=True,
            env_var=FEAST_DIR_ENV,
        ),
    ],
    image=flyte.Image.from_debian_base().with_pip_packages(
        "fastapi", "uvicorn", "joblib",
        "feast", "scikit-learn", "xgboost",
        "pandas", "pyarrow",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    requires_auth=False,
)


@app.get("/score")
async def score(
    user_id: int,
    amt: float,
    category: str = "grocery_pos",
    merch_lat: float = 40.0,
    merch_long: float = -74.0,
    hour: int | None = None,
    day_of_week: int | None = None,
) -> dict:
    """Score a transaction for fraud risk.

    Simulates a real payment flow:
    - Transaction details come from the request (amount, category, merchant location)
    - User profile comes from Feast online store (spending history, home location)
    - Derived features are computed at scoring time (z-score, distance)
    """
    now = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Transaction features (from the incoming request)
    # ------------------------------------------------------------------
    amt_log = float(np.log1p(amt))
    cat_mapping = app.state.category_mapping
    category_encoded = cat_mapping.get(category, 0)
    hour = hour if hour is not None else now.hour
    day_of_week = day_of_week if day_of_week is not None else now.weekday()

    # ------------------------------------------------------------------
    # User profile features (from Feast online store)
    # ------------------------------------------------------------------
    online_features = app.state.store.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=[{"user_id": user_id}],
    ).to_df()
    online_features = online_features.fillna(0)

    mean_amt = float(online_features["mean_amt"].iloc[0])
    std_amt = float(online_features["std_amt"].iloc[0])
    max_amt = float(online_features["max_amt"].iloc[0])
    home_lat = float(online_features["home_lat"].iloc[0])
    home_long = float(online_features["home_long"].iloc[0])
    age = int(online_features["age"].iloc[0])
    txn_count = int(online_features["txn_count"].iloc[0])

    # ------------------------------------------------------------------
    # Derived features (computed at scoring time)
    # ------------------------------------------------------------------
    amt_zscore = (amt - mean_amt) / max(std_amt, 1)
    amt_ratio = amt / max(mean_amt, 1)
    distance = haversine(home_lat, home_long, merch_lat, merch_long)

    # ------------------------------------------------------------------
    # Build feature vector (must match training order)
    # ------------------------------------------------------------------
    feature_row = {
        # Transaction features
        "amt": amt,
        "amt_log": amt_log,
        "category_encoded": category_encoded,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        # User profile
        "txn_count": txn_count,
        "mean_amt": mean_amt,
        "std_amt": std_amt,
        "max_amt": max_amt,
        "home_lat": home_lat,
        "home_long": home_long,
        "age": age,
        # Derived
        "amt_zscore": amt_zscore,
        "amt_ratio": amt_ratio,
        "distance_from_home": distance,
        "hour": hour,
        "day_of_week": day_of_week,
    }

    X = pd.DataFrame([feature_row])[app.state.feature_cols].values
    model_probability = float(app.state.model.predict_proba(X)[0, 1])

    # ------------------------------------------------------------------
    # Rule overrides — catch obvious fraud the model can't extrapolate to
    # Tree-based models can't extrapolate beyond training data ranges,
    # so production systems combine ML scores with business rules.
    # ------------------------------------------------------------------
    probability = model_probability
    rule_applied = None
    if amt_zscore > 10 and distance > 500:
        probability = max(probability, 0.8)
        rule_applied = f"Rule override: z-score {amt_zscore:.1f} + distance {distance:.0f}mi → min 80%"
    elif amt_zscore > 10:
        probability = max(probability, 0.5)
        rule_applied = f"Rule override: z-score {amt_zscore:.1f} → min 50%"
    elif distance > 1000:
        probability = max(probability, 0.3)
        rule_applied = f"Rule override: distance {distance:.0f}mi → min 30%"

    prediction = 1 if probability > 0.5 else 0

    # ------------------------------------------------------------------
    # Explainability signals
    # ------------------------------------------------------------------
    signals = []
    if rule_applied:
        signals.append(rule_applied)
    if mean_amt > 0 and amt > mean_amt + 2 * std_amt:
        signals.append(f"Amount ${amt:.2f} is {amt_zscore:.1f} std devs above user avg (${mean_amt:.2f})")
    if distance > 100:
        signals.append(f"Transaction {distance:.0f} miles from user's usual area")
    if 0 <= hour <= 5:
        signals.append(f"Transaction at unusual hour ({hour}:00 UTC)")
    if amt > 1000:
        signals.append("High-value transaction (>$1,000)")

    return {
        "user_id": user_id,
        "transaction": {
            "amt": amt,
            "category": category,
            "hour": hour,
            "day_of_week": day_of_week,
        },
        "fraud_prediction": "Fraud" if prediction else "Legit",
        "fraud_probability": round(probability, 4),
        "model_probability": round(model_probability, 4),
        "risk_level": "HIGH" if probability > 0.5 else "MEDIUM" if probability > 0.1 else "LOW",
        "signals": signals,
        "scoring": {
            "amt_zscore": round(amt_zscore, 2),
            "distance_from_home": round(distance, 1),
            "rule_applied": rule_applied is not None,
        },
        "user_profile": {
            "txn_count": txn_count,
            "mean_amt": round(mean_amt, 2),
            "std_amt": round(std_amt, 2),
            "age": age,
        },
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


if __name__ == "__main__":
    # Local: skip RunOutput resolution — lifespan falls back to local paths
    serving_env.parameters = []

    # Point to local artifacts (from running the pipeline locally)
    if not os.environ.get(MODEL_PATH_ENV):
        print("Note: Set MODEL_PATH and FEAST_DIR env vars, or run the pipeline first.")
        print("Example: MODEL_PATH=model.joblib FEAST_DIR=feast_artifacts python app.py")

    serve_ctx = flyte.with_servecontext(mode="local")
    local_app = serve_ctx.serve(serving_env)
    local_app.activate(wait=True)
    print(f"App running at {local_app.endpoint}")
    print('Try: curl "http://localhost:8080/score?user_id=42&amt=500.00&category=grocery_pos&merch_lat=40.71&merch_long=-74.01"')

    input("Press Enter to shut down...")
    local_app.deactivate(wait=True)