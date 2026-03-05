"""Fraud scoring app — loads model + Feast from pipeline. Remote: flyte deploy app.py serving_env | Local: python app.py"""

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

from shared import haversine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH_ENV, FEAST_DIR_ENV = "MODEL_PATH", "FEAST_DIR"
FEAST_FEATURES = [
    "user_stats:txn_count", "user_stats:mean_amt", "user_stats:std_amt", "user_stats:max_amt",
    "user_stats:home_lat", "user_stats:home_long", "user_stats:age",
]
PIPELINE_TASK = "fraud-detection-env.fraud_detection_pipeline"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and Feast feature store on startup."""
    from feast import FeatureStore

    model_path = os.environ.get(MODEL_PATH_ENV, "model.joblib")
    feast_dir = os.environ.get(FEAST_DIR_ENV, "feast_artifacts")
    logger.info("Loading model from %s", model_path)
    artifacts = joblib.load(model_path)
    app.state.model = artifacts["model"]
    app.state.feature_cols = artifacts["feature_cols"]
    app.state.category_mapping = artifacts["category_mapping"]
    logger.info("Loading Feast store from %s", feast_dir)
    app.state.store = FeatureStore(repo_path=feast_dir)
    logger.info("Model and feature store loaded.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Fraud Scorer", lifespan=lifespan)

serving_env = FastAPIAppEnvironment(
    name="fraud-scorer",
    app=app,
    description="Real-time fraud scoring using Feast online features",
    parameters=[
        Parameter(name="model", value=RunOutput(task_name=PIPELINE_TASK, type="file", getter=(0,)), download=True, env_var=MODEL_PATH_ENV),
        Parameter(name="feast_store", value=RunOutput(task_name=PIPELINE_TASK, type="directory", getter=(1,)), download=True, env_var=FEAST_DIR_ENV),
    ],
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "joblib", "feast", "scikit-learn", "xgboost", "pandas", "pyarrow"),
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
    """Score a transaction for fraud risk (txn from request, user profile from Feast, derived at runtime)."""
    now = datetime.now(timezone.utc)
    amt_log = float(np.log1p(amt))
    category_encoded = app.state.category_mapping.get(category, 0)
    hour = hour if hour is not None else now.hour
    day_of_week = day_of_week if day_of_week is not None else now.weekday()

    row = app.state.store.get_online_features(
        features=FEAST_FEATURES, entity_rows=[{"user_id": user_id}],
    ).to_df().fillna(0).iloc[0]
    mean_amt, std_amt, max_amt = float(row["mean_amt"]), float(row["std_amt"]), float(row["max_amt"])
    home_lat, home_long = float(row["home_lat"]), float(row["home_long"])
    age, txn_count = int(row["age"]), int(row["txn_count"])

    amt_zscore = (amt - mean_amt) / max(std_amt, 1)
    amt_ratio = amt / max(mean_amt, 1)
    distance = haversine(home_lat, home_long, merch_lat, merch_long)

    feature_row = {
        "amt": amt, "amt_log": amt_log, "category_encoded": category_encoded,
        "merch_lat": merch_lat, "merch_long": merch_long,
        "txn_count": txn_count, "mean_amt": mean_amt, "std_amt": std_amt,
        "max_amt": max_amt, "home_lat": home_lat, "home_long": home_long,
        "age": age, "amt_zscore": amt_zscore, "amt_ratio": amt_ratio,
        "distance_from_home": distance, "hour": hour, "day_of_week": day_of_week,
    }
    X = pd.DataFrame([feature_row])[app.state.feature_cols].values
    model_probability = float(app.state.model.predict_proba(X)[0, 1])

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
        "transaction": {"amt": amt, "category": category, "hour": hour, "day_of_week": day_of_week},
        "fraud_prediction": "Fraud" if prediction else "Legit",
        "fraud_probability": round(probability, 4),
        "model_probability": round(model_probability, 4),
        "risk_level": "HIGH" if probability > 0.5 else "MEDIUM" if probability > 0.1 else "LOW",
        "signals": signals,
        "scoring": {"amt_zscore": round(amt_zscore, 2), "distance_from_home": round(distance, 1), "rule_applied": rule_applied is not None},
        "user_profile": {"txn_count": txn_count, "mean_amt": round(mean_amt, 2), "std_amt": round(std_amt, 2), "age": age},
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


if __name__ == "__main__":
    serving_env.parameters = []
    if not os.environ.get(MODEL_PATH_ENV):
        print("Note: Set MODEL_PATH and FEAST_DIR, or run the pipeline first.")
    serve_ctx = flyte.with_servecontext(mode="local")
    local_app = serve_ctx.serve(serving_env)
    local_app.activate(wait=True)
    print(f"App running at {local_app.endpoint}")
    print('Try: curl "http://localhost:8080/score?user_id=42&amt=500&category=grocery_pos&merch_lat=40.71&merch_long=-74.01"')
    input("Press Enter to shut down...")
    local_app.deactivate(wait=True)
