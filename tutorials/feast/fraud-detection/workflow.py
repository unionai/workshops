"""
Fraud detection ML pipeline using Feast feature store + Flyte orchestration.

Architecture:
    1. prepare_training_data — join transaction + user features with pandas
    2. train_model           — train XGBoost classifier, return model as flyte.io.File
    3. score_transactions    — materialize to Feast online store + score in real time

Usage:
    flyte run --local workflow.py fraud_detection_pipeline
    flyte run --local --tui workflow.py fraud_detection_pipeline
"""

import json
import logging
import tempfile
from datetime import datetime, timezone

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

FEATURE_COLS = [
    "Amount", "amount_log", "hour_of_day",
] + [f"V{i}" for i in range(1, 29)]

FEAST_FEATURES = [
    "transaction_features:Amount",
    "transaction_features:amount_log",
    "transaction_features:hour_of_day",
] + [f"transaction_features:V{i}" for i in range(1, 29)] + [
    "user_stats:txn_count",
    "user_stats:mean_amount",
    "user_stats:std_amount",
    "user_stats:max_amount",
    "user_stats:total_amount",
]

ALL_FEATURE_COLS = FEATURE_COLS + [
    "txn_count", "mean_amount", "std_amount", "max_amount", "total_amount",
]


# ------------------------------------------------------------------
# Task 1: Build training data by joining transaction + user features
# ------------------------------------------------------------------
# We join features directly with pandas here — it's fast and simple.
# Feast's offline store (get_historical_features) does the same thing
# with point-in-time correctness, but the local file provider is slow
# on 284K rows. In production you'd use BigQuery/Redshift as the
# offline store backend for fast historical joins.
# ------------------------------------------------------------------

@env.task(report=True)
async def prepare_training_data() -> pd.DataFrame:
    """Join transaction and user features into a training dataset."""
    log.info("Loading parquet files...")
    txn_df = pd.read_parquet("data/transactions.parquet")
    user_df = pd.read_parquet("data/user_features.parquet")

    # Join user-level aggregates onto each transaction
    log.info(f"Joining {len(txn_df):,} transactions with {len(user_df):,} user profiles...")
    training_data = txn_df.merge(
        user_df[["user_id", "txn_count", "mean_amount", "std_amount", "max_amount", "total_amount"]],
        on="user_id",
        how="left",
    )

    # Drop rows with missing features
    before = len(training_data)
    training_data = training_data.dropna(subset=ALL_FEATURE_COLS)
    log.info(f"Training data: {len(training_data):,} rows ({before - len(training_data)} dropped)")

    # Report: dataset summary
    fraud_count = int(training_data["Class"].sum())
    legit_count = len(training_data) - fraud_count
    html = (
        f"<h2>Training Data</h2>"
        f"<p><b>Total:</b> {len(training_data):,} transactions</p>"
        f"<p><b>Legitimate:</b> {legit_count:,} ({legit_count/len(training_data)*100:.1f}%)</p>"
        f"<p><b>Fraudulent:</b> {fraud_count:,} ({fraud_count/len(training_data)*100:.3f}%)</p>"
        f"<p><b>Features:</b> {len(ALL_FEATURE_COLS)}</p>"
    )
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    return training_data


# ------------------------------------------------------------------
# Task 2: Train XGBoost model
# ------------------------------------------------------------------

@env.task(report=True)
async def train_model(training_data: pd.DataFrame) -> flyte.io.File:
    """Train an XGBoost classifier on Feast-enriched features."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, roc_auc_score, confusion_matrix,
    )
    from xgboost import XGBClassifier

    X = training_data[ALL_FEATURE_COLS].values
    y = training_data["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Handle class imbalance with scale_pos_weight
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

    # Report: model performance
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

    # Feature importance (top 15)
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[-15:][::-1]
    html += "<h3>Top 15 Features</h3><ol>"
    for i in top_idx:
        html += f"<li>{ALL_FEATURE_COLS[i]}: {importance[i]:.4f}</li>"
    html += "</ol>"

    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    # Save model artifacts to a file — Flyte stores it by reference (no inline blob)
    model_path = tempfile.mktemp(suffix=".joblib")
    joblib.dump({
        "model": model,
        "auc_roc": auc,
        "confusion_matrix": cm.tolist(),
        "feature_cols": ALL_FEATURE_COLS,
    }, model_path)

    return await flyte.io.File.from_local(model_path)


# ------------------------------------------------------------------
# Task 3: Score transactions using Feast online store
# ------------------------------------------------------------------

@env.task(report=True)
async def score_transactions(model_file: flyte.io.File) -> str:
    """Materialize features to online store and score sample transactions."""
    from feast import FeatureStore

    # Load model from Flyte-managed file
    model_path = await model_file.download()
    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    auc = artifacts["auc_roc"]

    # Materialize features to the online store
    log.info("Materializing features to online store...")
    store = FeatureStore(repo_path="feature_repo")
    store.materialize_incremental(end_date=datetime.now(timezone.utc))

    # Pick some sample transactions to score
    txn_df = pd.read_parquet("data/transactions.parquet")
    sample = txn_df.sample(n=20, random_state=42)

    # Fetch features from the online store
    log.info(f"Scoring {len(sample)} sample transactions...")
    entity_rows = [{"user_id": int(row["user_id"])} for _, row in sample.iterrows()]
    online_features = store.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=entity_rows,
    ).to_df()

    # Fill any missing online features with 0
    online_features = online_features.fillna(0)

    # Score
    X_score = online_features[ALL_FEATURE_COLS].values
    probabilities = model.predict_proba(X_score)[:, 1]
    predictions = model.predict(X_score)

    # Build results
    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        results.append({
            "transaction_id": int(row["transaction_id"]),
            "user_id": int(row["user_id"]),
            "amount": float(row["Amount"]),
            "actual": int(row["Class"]),
            "predicted": int(predictions[i]),
            "fraud_probability": float(probabilities[i]),
        })

    # Report: scored transactions
    html = "<h2>Scored Transactions (Sample)</h2>"
    html += "<table border='1' cellpadding='6'>"
    html += "<tr><th>Txn ID</th><th>User</th><th>Amount</th><th>Actual</th><th>Predicted</th><th>Fraud Prob</th></tr>"
    for r in results:
        color = "red" if r["fraud_probability"] > 0.5 else "green"
        html += (
            f"<tr>"
            f"<td>{r['transaction_id']}</td>"
            f"<td>{r['user_id']}</td>"
            f"<td>${r['amount']:.2f}</td>"
            f"<td>{'Fraud' if r['actual'] else 'Legit'}</td>"
            f"<td>{'Fraud' if r['predicted'] else 'Legit'}</td>"
            f"<td style='color:{color}'>{r['fraud_probability']:.4f}</td>"
            f"</tr>"
        )
    html += "</table>"
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    return json.dumps({"scored": results, "auc_roc": auc})


# ------------------------------------------------------------------
# Orchestrator: prepare → train → score
# ------------------------------------------------------------------

@env.task(report=True)
async def fraud_detection_pipeline() -> str:
    """
    Full fraud detection pipeline:
    1. Build training data (pandas join)
    2. Train XGBoost model
    3. Score sample transactions via Feast online store
    """
    log.info("Starting fraud detection pipeline")

    await flyte.report.replace.aio("<h2>Fraud Detection Pipeline</h2><p>Preparing training data...</p>")
    await flyte.report.flush.aio()

    training_data = await prepare_training_data()

    await flyte.report.replace.aio("<h2>Fraud Detection Pipeline</h2><p>Training model...</p>")
    await flyte.report.flush.aio()

    model_file = await train_model(training_data)

    await flyte.report.replace.aio("<h2>Fraud Detection Pipeline</h2><p>Scoring transactions...</p>")
    await flyte.report.flush.aio()

    result = await score_transactions(model_file)

    data = json.loads(result)
    await flyte.report.replace.aio(
        f"<h2>Pipeline Complete</h2>"
        f"<p><b>AUC-ROC:</b> {data['auc_roc']:.4f}</p>"
        f"<p><b>Transactions scored:</b> {len(data['scored'])}</p>"
    )
    await flyte.report.flush.aio()

    log.info("Pipeline complete")
    return result
