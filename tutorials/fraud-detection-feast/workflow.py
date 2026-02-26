"""
Fraud detection ML pipeline using Feast feature store + Flyte orchestration.

Uses the Sparkov simulated credit card transactions dataset with fully
interpretable features: amount, merchant category, location, user spending
history, and derived anomaly scores.

Architecture:
    1. prepare_data          — download dataset, engineer features → flyte.io.Dir
    2. train_model           — train XGBoost classifier → flyte.io.File
    3. materialize_features  — feast apply + materialize → flyte.io.Dir
    4. fraud_detection_pipeline — orchestrate all, return model + feast artifacts

Usage:
    flyte run --local workflow.py fraud_detection_pipeline
    flyte run --local --tui workflow.py fraud_detection_pipeline
"""

import asyncio
import json
import logging
import math
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import flyte
import flyte.io
import flyte.report
from config import base_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

env = base_env

# ------------------------------------------------------------------
# Feature definitions
#
# Transaction features: known at scoring time (from the request)
# User features: pre-computed aggregates stored in Feast
# Derived features: computed at both training and scoring time by
#                   comparing the transaction to the user's profile
# ------------------------------------------------------------------

TXN_FEATURE_COLS = ["amt", "amt_log", "category_encoded", "merch_lat", "merch_long"]

USER_FEATURE_COLS = [
    "txn_count", "mean_amt", "std_amt", "max_amt",
    "home_lat", "home_long", "age",
]

DERIVED_FEATURE_COLS = [
    "amt_zscore", "amt_ratio", "distance_from_home", "hour", "day_of_week",
]

ALL_FEATURE_COLS = TXN_FEATURE_COLS + USER_FEATURE_COLS + DERIVED_FEATURE_COLS


def haversine(lat1, lon1, lat2, lon2):
    """Compute distance in miles between two (lat, lon) points."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------
# Task 1: Download dataset and engineer features
# ------------------------------------------------------------------

@env.task(report=True, cache="auto")
async def prepare_data() -> flyte.io.Dir:
    """Download the Sparkov credit card fraud dataset and prepare parquets."""
    import kagglehub

    log.info("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("kartik2112/fraud-detection")
    csv_path = os.path.join(dataset_path, "fraudTrain.csv")
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df):,} transactions ({int(df['is_fraud'].sum()):,} fraudulent)")

    # Sample for workshop speed (stratified to preserve fraud ratio)
    if len(df) > 500_000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=500_000, stratify=df["is_fraud"], random_state=42)
        log.info(f"Sampled to {len(df):,} transactions")

    # ------------------------------------------------------------------
    # Parse timestamps
    # ------------------------------------------------------------------
    df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"])
    df["event_timestamp"] = df["event_timestamp"].dt.tz_localize("UTC")
    df["hour"] = df["event_timestamp"].dt.hour
    df["day_of_week"] = df["event_timestamp"].dt.dayofweek

    # ------------------------------------------------------------------
    # Map cc_num → sequential user_id for clean API
    # ------------------------------------------------------------------
    cc_nums = df["cc_num"].unique()
    cc_to_user = {cc: i for i, cc in enumerate(sorted(cc_nums))}
    df["user_id"] = df["cc_num"].map(cc_to_user)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    df["amt_log"] = np.log1p(df["amt"])

    # Label-encode merchant category
    categories = sorted(df["category"].unique())
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    df["category_encoded"] = df["category"].map(cat_to_int)

    # Compute age from dob
    df["dob"] = pd.to_datetime(df["dob"]).dt.tz_localize("UTC")
    ref_date = df["event_timestamp"].max()
    df["age"] = ((ref_date - df["dob"]).dt.days / 365.25).astype(int)

    # Distance between buyer and merchant
    df["distance"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # ------------------------------------------------------------------
    # Build user aggregates
    # ------------------------------------------------------------------
    user_stats = df.groupby("user_id").agg(
        txn_count=("amt", "count"),
        mean_amt=("amt", "mean"),
        std_amt=("amt", "std"),
        max_amt=("amt", "max"),
        home_lat=("lat", "median"),
        home_long=("long", "median"),
        age=("age", "first"),
    ).reset_index()
    user_stats["std_amt"] = user_stats["std_amt"].fillna(0)
    latest_ts = df.groupby("user_id")["event_timestamp"].max().reset_index()
    user_stats = user_stats.merge(latest_ts, on="user_id")

    # ------------------------------------------------------------------
    # Save to temp directory
    # ------------------------------------------------------------------
    data_dir = tempfile.mkdtemp()

    txn_cols = [
        "user_id", "event_timestamp",
        "amt", "amt_log", "category_encoded", "merch_lat", "merch_long",
        "hour", "day_of_week", "lat", "long", "distance",
        "is_fraud",
    ]
    df[txn_cols].to_parquet(os.path.join(data_dir, "transactions.parquet"), index=False)
    user_stats.to_parquet(os.path.join(data_dir, "user_features.parquet"), index=False)

    # Save category mapping + cc_num mapping for the app
    with open(os.path.join(data_dir, "category_mapping.json"), "w") as f:
        json.dump(cat_to_int, f)
    with open(os.path.join(data_dir, "user_mapping.json"), "w") as f:
        json.dump({str(k): v for k, v in cc_to_user.items()}, f)

    fraud_pct = df["is_fraud"].mean() * 100
    html = (
        f"<h2>Data Prepared</h2>"
        f"<p><b>Transactions:</b> {len(df):,}</p>"
        f"<p><b>Fraudulent:</b> {int(df['is_fraud'].sum()):,} ({fraud_pct:.2f}%)</p>"
        f"<p><b>Users:</b> {user_stats['user_id'].nunique():,}</p>"
        f"<p><b>Categories:</b> {len(categories)}</p>"
    )
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(data_dir)


# ------------------------------------------------------------------
# Task 2: Train XGBoost model
# ------------------------------------------------------------------

@env.task(report=True)
async def train_model(data_dir: flyte.io.Dir) -> flyte.io.File:
    """Train an XGBoost classifier on the prepared dataset."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    from xgboost import XGBClassifier

    data_path = await data_dir.download()
    txn_df = pd.read_parquet(os.path.join(data_path, "transactions.parquet"))
    user_df = pd.read_parquet(os.path.join(data_path, "user_features.parquet"))

    with open(os.path.join(data_path, "category_mapping.json")) as f:
        category_mapping = json.load(f)

    # Join user-level aggregates onto transactions
    training_data = txn_df.merge(
        user_df[["user_id"] + USER_FEATURE_COLS],
        on="user_id",
        how="left",
    )

    # Derived features — compare this transaction to the user's profile
    training_data["amt_zscore"] = (
        (training_data["amt"] - training_data["mean_amt"])
        / training_data["std_amt"].replace(0, 1)
    )
    training_data["amt_ratio"] = (
        training_data["amt"] / training_data["mean_amt"].replace(0, 1)
    )
    training_data["distance_from_home"] = haversine(
        training_data["home_lat"], training_data["home_long"],
        training_data["merch_lat"], training_data["merch_long"],
    )

    training_data = training_data.dropna(subset=ALL_FEATURE_COLS)
    X = training_data[ALL_FEATURE_COLS].values
    y = training_data["is_fraud"].values
    log.info(f"Training on {len(X):,} rows, {int(y.sum()):,} fraud")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    n_legit = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    scale_pos_weight = n_legit / max(n_fraud, 1)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])

    log.info(f"AUC-ROC: {auc:.4f}")
    log.info(f"\n{report}")

    # Report
    html = (
        f"<h2>Model Performance</h2>"
        f"<p><b>AUC-ROC:</b> {auc:.4f}</p>"
        f"<h3>Confusion Matrix</h3>"
        f"<table border='1' cellpadding='8'>"
        f"<tr><th></th><th>Predicted Legit</th><th>Predicted Fraud</th></tr>"
        f"<tr><td><b>Actual Legit</b></td><td>{cm[0][0]:,}</td><td>{cm[0][1]:,}</td></tr>"
        f"<tr><td><b>Actual Fraud</b></td><td>{cm[1][0]:,}</td><td>{cm[1][1]:,}</td></tr>"
        f"</table>"
        f"<h3>Classification Report</h3>"
        f"<pre>{report}</pre>"
    )

    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1]
    html += "<h3>Feature Importance</h3><ol>"
    for i in top_idx:
        html += f"<li>{ALL_FEATURE_COLS[i]}: {importance[i]:.4f}</li>"
    html += "</ol>"

    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    # Save model + metadata
    model_path = os.path.join(tempfile.mkdtemp(), "model.joblib")
    joblib.dump({
        "model": model,
        "auc_roc": auc,
        "feature_cols": ALL_FEATURE_COLS,
        "category_mapping": category_mapping,
    }, model_path)

    return await flyte.io.File.from_local(model_path)


# ------------------------------------------------------------------
# Task 3: Set up Feast and materialize user profiles to online store
# ------------------------------------------------------------------

@env.task(report=True)
async def materialize_features(data_dir: flyte.io.Dir) -> flyte.io.Dir:
    """Apply Feast definitions and materialize user profiles to SQLite online store."""
    from feast import Entity, FeatureStore, FeatureView, Field, FileSource
    from feast.types import Float64, Int64

    data_path = await data_dir.download()

    # Create a self-contained Feast repo in a temp directory
    feast_dir = tempfile.mkdtemp()

    # Write feature_store.yaml
    yaml_content = (
        "project: fraud_detection\n"
        f"registry: {feast_dir}/registry.db\n"
        "provider: local\n"
        "online_store:\n"
        "  type: sqlite\n"
        f"  path: {feast_dir}/online_store.db\n"
        "offline_store:\n"
        "  type: file\n"
        "entity_key_serialization_version: 3\n"
    )
    yaml_path = os.path.join(feast_dir, "feature_store.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    store = FeatureStore(repo_path=feast_dir)

    # Define entity and feature view
    user = Entity(name="user", join_keys=["user_id"], description="Credit card holder")

    user_source = FileSource(
        path=os.path.join(data_path, "user_features.parquet"),
        timestamp_field="event_timestamp",
    )

    user_stats = FeatureView(
        name="user_stats",
        entities=[user],
        ttl=timedelta(days=0),  # No expiry — workshop data has old timestamps
        schema=[
            Field(name="txn_count", dtype=Int64),
            Field(name="mean_amt", dtype=Float64),
            Field(name="std_amt", dtype=Float64),
            Field(name="max_amt", dtype=Float64),
            Field(name="home_lat", dtype=Float64),
            Field(name="home_long", dtype=Float64),
            Field(name="age", dtype=Int64),
        ],
        online=True,
        source=user_source,
    )

    # Apply and materialize
    log.info("Applying Feast definitions...")
    store.apply([user, user_stats])

    log.info("Materializing user profiles to online store...")
    store.materialize(
        start_date=datetime(2018, 1, 1, tzinfo=timezone.utc),
        end_date=datetime.now(timezone.utc),
    )

    # Rewrite feature_store.yaml with relative paths for portability
    portable_yaml = (
        "project: fraud_detection\n"
        "registry: registry.db\n"
        "provider: local\n"
        "online_store:\n"
        "  type: sqlite\n"
        "  path: online_store.db\n"
        "offline_store:\n"
        "  type: file\n"
        "entity_key_serialization_version: 3\n"
    )
    with open(yaml_path, "w") as f:
        f.write(portable_yaml)

    html = (
        f"<h2>Feature Store Materialized</h2>"
        f"<p><b>Feature view:</b> user_stats (spending profile + location + age)</p>"
        f"<p><b>Online store:</b> SQLite</p>"
        f"<p>User profiles are ready for real-time serving.</p>"
    )
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(feast_dir)


# ------------------------------------------------------------------
# Orchestrator: prepare → train + materialize → done
# ------------------------------------------------------------------

@env.task(report=True)
async def fraud_detection_pipeline() -> tuple[flyte.io.File, flyte.io.Dir]:
    """
    Full fraud detection pipeline:
    1. Download and prepare data
    2. Train model + materialize features (in parallel)
    Returns model file and Feast artifacts for serving.
    """
    log.info("Starting fraud detection pipeline")

    await flyte.report.replace.aio("<h2>Fraud Detection Pipeline</h2><p>Preparing data...</p>")
    await flyte.report.flush.aio()

    data_dir = await prepare_data()

    await flyte.report.replace.aio("<h2>Fraud Detection Pipeline</h2><p>Training model + materializing features...</p>")
    await flyte.report.flush.aio()

    # Train model and materialize features in parallel
    model_file, feast_dir = await asyncio.gather(
        train_model(data_dir),
        materialize_features(data_dir),
    )

    # Save copies to working directory for local app testing
    model_local = await model_file.download()
    feast_local = await feast_dir.download()
    shutil.copy2(model_local, "model.joblib")
    if os.path.exists("feast_artifacts"):
        shutil.rmtree("feast_artifacts")
    shutil.copytree(feast_local, "feast_artifacts")
    log.info("Saved local copies: model.joblib, feast_artifacts/")

    await flyte.report.replace.aio(
        f"<h2>Pipeline Complete</h2>"
        f"<p>Model and feature store artifacts are ready for serving.</p>"
        f"<p><b>Local:</b> <code>python app.py</code></p>"
        f"<p><b>Remote:</b> <code>flyte deploy app.py serving_env</code></p>"
    )
    await flyte.report.flush.aio()

    log.info("Pipeline complete")
    return model_file, feast_dir
