"""
Fraud detection activities for Temporal workflow.

Each activity performs a single, well-defined action (equivalent of Flyte tasks).
"""

import json
import os
import shutil
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from temporalio import activity

from shared import ALL_FEATURE_COLS, USER_FEATURE_COLS, haversine


@activity.defn(name="prepare_data")
async def prepare_data(data_dir: str) -> str:
    """Download Sparkov fraud dataset and prepare parquets to data_dir."""
    import kagglehub

    activity.logger.info("Downloading dataset...")
    os.makedirs(data_dir, exist_ok=True)

    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    df = pd.read_csv(os.path.join(path, "fraudTrain.csv"))
    activity.logger.info(f"Loaded {len(df):,} transactions ({int(df['is_fraud'].sum()):,} fraudulent)")

    # Sample for workshop speed (stratified to preserve fraud ratio)
    if len(df) > 500_000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=500_000, stratify=df["is_fraud"], random_state=42)
        activity.logger.info(f"Sampled to {len(df):,} transactions")

    # Parse timestamps
    df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"])
    df["event_timestamp"] = df["event_timestamp"].dt.tz_localize("UTC")
    df["hour"] = df["event_timestamp"].dt.hour
    df["day_of_week"] = df["event_timestamp"].dt.dayofweek

    # Map cc_num → sequential user_id for clean API
    cc_nums = df["cc_num"].unique()
    cc_to_user = {cc: i for i, cc in enumerate(sorted(cc_nums))}
    df["user_id"] = df["cc_num"].map(cc_to_user)

    # Feature engineering
    df["amt_log"] = np.log1p(df["amt"])

    categories = sorted(df["category"].unique())
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    df["category_encoded"] = df["category"].map(cat_to_int)

    df["dob"] = pd.to_datetime(df["dob"]).dt.tz_localize("UTC")
    ref_date = df["event_timestamp"].max()
    df["age"] = ((ref_date - df["dob"]).dt.days / 365.25).astype(int)

    df["distance"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # Build user aggregates
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

    # Save to data_dir
    txn_cols = [
        "user_id", "event_timestamp",
        "amt", "amt_log", "category_encoded", "merch_lat", "merch_long",
        "hour", "day_of_week", "lat", "long", "distance",
        "is_fraud",
    ]
    df[txn_cols].to_parquet(os.path.join(data_dir, "transactions.parquet"), index=False)
    user_stats.to_parquet(os.path.join(data_dir, "user_features.parquet"), index=False)

    for name, data in [("category_mapping.json", cat_to_int), ("user_mapping.json", {str(k): v for k, v in cc_to_user.items()})]:
        with open(os.path.join(data_dir, name), "w") as f:
            json.dump(data, f)

    fraud_pct = df["is_fraud"].mean() * 100
    activity.logger.info(
        f"Data prepared: {len(df):,} txn, {int(df['is_fraud'].sum()):,} fraud ({fraud_pct:.2f}%), "
        f"{user_stats['user_id'].nunique():,} users, {len(categories)} categories"
    )

    return data_dir


@activity.defn(name="train_model")
async def train_model(data_dir: str) -> str:
    """Train XGBoost on prepared data; writes model.joblib to data_dir."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    from xgboost import XGBClassifier

    txn_df = pd.read_parquet(os.path.join(data_dir, "transactions.parquet"))
    user_df = pd.read_parquet(os.path.join(data_dir, "user_features.parquet"))

    with open(os.path.join(data_dir, "category_mapping.json")) as f:
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
    activity.logger.info(f"Training on {len(X):,} rows, {int(y.sum()):,} fraud")

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

    activity.logger.info(f"AUC-ROC: {auc:.4f}\n{report}")

    model_path = os.path.join(data_dir, "model.joblib")
    joblib.dump({
        "model": model,
        "auc_roc": auc,
        "feature_cols": ALL_FEATURE_COLS,
        "category_mapping": category_mapping,
    }, model_path)

    return model_path


@activity.defn(name="materialize_features")
async def materialize_features(data_dir: str) -> str:
    """Apply Feast definitions and materialize user profiles to feast_artifacts/ in data_dir."""
    from feast import Entity, FeatureStore, FeatureView, Field, FileSource
    from feast.types import Float64, Int64

    feast_dir = os.path.join(data_dir, "feast_artifacts")
    os.makedirs(feast_dir, exist_ok=True)

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

    user = Entity(name="user", join_keys=["user_id"], description="Credit card holder")

    user_source = FileSource(
        path=os.path.join(data_dir, "user_features.parquet"),
        timestamp_field="event_timestamp",
    )

    user_stats = FeatureView(
        name="user_stats",
        entities=[user],
        ttl=timedelta(days=0),
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

    activity.logger.info("Applying Feast definitions...")
    store.apply([user, user_stats])

    activity.logger.info("Materializing user profiles to online store...")
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

    activity.logger.info("Feature store materialized to %s", feast_dir)
    return feast_dir


@activity.defn(name="copy_artifacts")
async def copy_artifacts(model_path: str, feast_path: str, output_dir: str) -> dict:
    """Copy model and Feast artifacts to output_dir for serving."""
    os.makedirs(output_dir, exist_ok=True)
    model_dest = os.path.join(output_dir, "model.joblib")
    feast_dest = os.path.join(output_dir, "feast_artifacts")

    shutil.copy2(model_path, model_dest)
    if os.path.exists(feast_dest):
        shutil.rmtree(feast_dest)
    shutil.copytree(feast_path, feast_dest)

    activity.logger.info("Copied artifacts to %s", output_dir)
    return {"model_path": model_dest, "feast_path": feast_dest}
