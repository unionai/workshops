"""Fraud detection ML pipeline: Feast + Flyte. Usage: flyte run --local workflow.py fraud_detection_pipeline"""

import asyncio
import json
import logging
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
from shared import ALL_FEATURE_COLS, USER_FEATURE_COLS, haversine

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
env = base_env


@env.task(report=True, cache="auto")
async def prepare_data() -> flyte.io.Dir:
    """Download Sparkov fraud dataset and prepare parquets."""
    import kagglehub

    log.info("Downloading dataset...")
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    df = pd.read_csv(os.path.join(path, "fraudTrain.csv"))
    log.info(f"Loaded {len(df):,} transactions ({int(df['is_fraud'].sum()):,} fraudulent)")

    if len(df) > 500_000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=500_000, stratify=df["is_fraud"], random_state=42)
        log.info(f"Sampled to {len(df):,} transactions")

    df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"]).dt.tz_localize("UTC")
    df["hour"] = df["event_timestamp"].dt.hour
    df["day_of_week"] = df["event_timestamp"].dt.dayofweek

    cc_nums = df["cc_num"].unique()
    cc_to_user = {cc: i for i, cc in enumerate(sorted(cc_nums))}
    df["user_id"] = df["cc_num"].map(cc_to_user)

    df["amt_log"] = np.log1p(df["amt"])
    categories = sorted(df["category"].unique())
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    df["category_encoded"] = df["category"].map(cat_to_int)

    df["dob"] = pd.to_datetime(df["dob"]).dt.tz_localize("UTC")
    ref_date = df["event_timestamp"].max()
    df["age"] = ((ref_date - df["dob"]).dt.days / 365.25).astype(int)
    df["distance"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    user_stats = df.groupby("user_id").agg(
        txn_count=("amt", "count"), mean_amt=("amt", "mean"), std_amt=("amt", "std"),
        max_amt=("amt", "max"), home_lat=("lat", "median"), home_long=("long", "median"),
        age=("age", "first"),
    ).reset_index()
    user_stats["std_amt"] = user_stats["std_amt"].fillna(0)
    latest_ts = df.groupby("user_id")["event_timestamp"].max().reset_index()
    user_stats = user_stats.merge(latest_ts, on="user_id")

    data_dir = tempfile.mkdtemp()
    txn_cols = ["user_id", "event_timestamp", "amt", "amt_log", "category_encoded", "merch_lat", "merch_long", "hour", "day_of_week", "lat", "long", "distance", "is_fraud"]
    df[txn_cols].to_parquet(os.path.join(data_dir, "transactions.parquet"), index=False)
    user_stats.to_parquet(os.path.join(data_dir, "user_features.parquet"), index=False)
    for name, data in [("category_mapping.json", cat_to_int), ("user_mapping.json", {str(k): v for k, v in cc_to_user.items()})]:
        with open(os.path.join(data_dir, name), "w") as f:
            json.dump(data, f)

    fraud_pct = df["is_fraud"].mean() * 100
    html = f"<h2>Data Prepared</h2><p><b>Transactions:</b> {len(df):,}</p><p><b>Fraudulent:</b> {int(df['is_fraud'].sum()):,} ({fraud_pct:.2f}%)</p><p><b>Users:</b> {user_stats['user_id'].nunique():,}</p><p><b>Categories:</b> {len(categories)}</p>"
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(data_dir)


@env.task(report=True)
async def train_model(data_dir: flyte.io.Dir) -> flyte.io.File:
    """Train XGBoost on prepared data; output model.joblib."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    from xgboost import XGBClassifier

    data_path = await data_dir.download()
    txn_df = pd.read_parquet(os.path.join(data_path, "transactions.parquet"))
    user_df = pd.read_parquet(os.path.join(data_path, "user_features.parquet"))
    with open(os.path.join(data_path, "category_mapping.json")) as f:
        category_mapping = json.load(f)

    training_data = txn_df.merge(user_df[["user_id"] + USER_FEATURE_COLS], on="user_id", how="left")
    training_data["amt_zscore"] = (training_data["amt"] - training_data["mean_amt"]) / training_data["std_amt"].replace(0, 1)
    training_data["amt_ratio"] = training_data["amt"] / training_data["mean_amt"].replace(0, 1)
    training_data["distance_from_home"] = haversine(training_data["home_lat"], training_data["home_long"], training_data["merch_lat"], training_data["merch_long"])

    training_data = training_data.dropna(subset=ALL_FEATURE_COLS)
    X, y = training_data[ALL_FEATURE_COLS].values, training_data["is_fraud"].values
    log.info(f"Training on {len(X):,} rows, {int(y.sum()):,} fraud")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    n_legit, n_fraud = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = n_legit / max(n_fraud, 1)

    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    log.info(f"AUC-ROC: {auc:.4f}\n{report}")

    importance = model.feature_importances_
    top = np.argsort(importance)[::-1]
    html = (
        f"<h2>Model Performance</h2><p><b>AUC-ROC:</b> {auc:.4f}</p>"
        f"<h3>Confusion Matrix</h3><table border='1' cellpadding='8'><tr><th></th><th>Predicted Legit</th><th>Predicted Fraud</th></tr>"
        f"<tr><td><b>Actual Legit</b></td><td>{cm[0][0]:,}</td><td>{cm[0][1]:,}</td></tr><tr><td><b>Actual Fraud</b></td><td>{cm[1][0]:,}</td><td>{cm[1][1]:,}</td></tr></table>"
        f"<h3>Classification Report</h3><pre>{report}</pre><h3>Feature Importance</h3><ol>"
        + "".join(f"<li>{ALL_FEATURE_COLS[i]}: {importance[i]:.4f}</li>" for i in top) + "</ol>"
    )

    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    model_path = os.path.join(tempfile.mkdtemp(), "model.joblib")
    joblib.dump({"model": model, "auc_roc": auc, "feature_cols": ALL_FEATURE_COLS, "category_mapping": category_mapping}, model_path)
    return await flyte.io.File.from_local(model_path)


@env.task(report=True)
async def materialize_features(data_dir: flyte.io.Dir) -> flyte.io.Dir:
    """Apply Feast definitions and materialize user profiles to SQLite online store."""
    from feast import Entity, FeatureStore, FeatureView, Field, FileSource
    from feast.types import Float64, Int64

    data_path = await data_dir.download()
    feast_dir = tempfile.mkdtemp()
    yaml_content = (
        f"project: fraud_detection\nregistry: {feast_dir}/registry.db\nprovider: local\n"
        f"online_store:\n  type: sqlite\n  path: {feast_dir}/online_store.db\n"
        "offline_store:\n  type: file\nentity_key_serialization_version: 3\n"
    )
    with open(os.path.join(feast_dir, "feature_store.yaml"), "w") as f:
        f.write(yaml_content)

    store = FeatureStore(repo_path=feast_dir)
    user = Entity(name="user", join_keys=["user_id"], description="Credit card holder")
    user_source = FileSource(path=os.path.join(data_path, "user_features.parquet"), timestamp_field="event_timestamp")
    user_stats = FeatureView(
        name="user_stats", entities=[user], ttl=timedelta(days=0),
        schema=[
            Field(name="txn_count", dtype=Int64), Field(name="mean_amt", dtype=Float64),
            Field(name="std_amt", dtype=Float64), Field(name="max_amt", dtype=Float64),
            Field(name="home_lat", dtype=Float64), Field(name="home_long", dtype=Float64),
            Field(name="age", dtype=Int64),
        ],
        online=True, source=user_source,
    )

    log.info("Applying Feast definitions...")
    store.apply([user, user_stats])
    log.info("Materializing user profiles to online store...")
    store.materialize(start_date=datetime(2018, 1, 1, tzinfo=timezone.utc), end_date=datetime.now(timezone.utc))

    portable_yaml = "project: fraud_detection\nregistry: registry.db\nprovider: local\nonline_store:\n  type: sqlite\n  path: online_store.db\noffline_store:\n  type: file\nentity_key_serialization_version: 3\n"
    with open(os.path.join(feast_dir, "feature_store.yaml"), "w") as f:
        f.write(portable_yaml)

    html = "<h2>Feature Store Materialized</h2><p><b>Feature view:</b> user_stats</p><p><b>Online store:</b> SQLite</p>"
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(feast_dir)


@env.task(report=True)
async def fraud_detection_pipeline() -> tuple[flyte.io.File, flyte.io.Dir]:
    """Full pipeline: prepare → (train + materialize) in parallel."""
    log.info("Starting fraud detection pipeline")
    await flyte.report.replace.aio("<h2>Fraud Detection Pipeline</h2><p>Preparing data...</p>")
    await flyte.report.flush.aio()

    data_dir = await prepare_data()

    await flyte.report.replace.aio("<h2>Fraud Detection Pipeline</h2><p>Training model + materializing features...</p>")
    await flyte.report.flush.aio()

    model_file, feast_dir = await asyncio.gather(train_model(data_dir), materialize_features(data_dir))

    model_local = await model_file.download()
    feast_local = await feast_dir.download()
    shutil.copy2(model_local, "model.joblib")
    if os.path.exists("feast_artifacts"):
        shutil.rmtree("feast_artifacts")
    shutil.copytree(feast_local, "feast_artifacts")
    log.info("Saved local copies: model.joblib, feast_artifacts/")

    await flyte.report.replace.aio("<h2>Pipeline Complete</h2><p>Model and feature store ready. Local: python app.py | Remote: flyte deploy app.py serving_env</p>")
    await flyte.report.flush.aio()
    log.info("Pipeline complete")
    return model_file, feast_dir
