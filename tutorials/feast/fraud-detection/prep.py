"""
Download the Kaggle Credit Card Fraud dataset and prepare Parquet files for Feast.

Usage:
    python prep.py

Creates:
    data/transactions.parquet  — per-transaction features with timestamps and user IDs
    data/user_features.parquet — aggregated per-user statistics
"""

import os
import numpy as np
import pandas as pd
import kagglehub


def download_dataset() -> pd.DataFrame:
    """Download the Credit Card Fraud dataset via kagglehub."""
    print("Downloading Credit Card Fraud dataset...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(path, "creditcard.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} transactions ({df['Class'].sum():,} fraudulent)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and Feast-required columns."""

    # ------------------------------------------------------------------
    # Timestamps — Feast needs an event_timestamp column.
    # The raw 'Time' column is seconds since the first transaction.
    # We anchor it to a base date so Feast can do point-in-time joins.
    # ------------------------------------------------------------------
    base_time = pd.Timestamp("2024-01-01", tz="UTC")
    df["event_timestamp"] = base_time + pd.to_timedelta(df["Time"], unit="s")

    # ------------------------------------------------------------------
    # Synthetic user IDs — the dataset has no user column.
    # We assign ~1000 users so we can demonstrate per-user aggregates.
    # ------------------------------------------------------------------
    rng = np.random.RandomState(42)
    n_users = 1000
    df["user_id"] = rng.randint(0, n_users, size=len(df))

    # ------------------------------------------------------------------
    # Transaction-level features
    # ------------------------------------------------------------------
    df["hour_of_day"] = (df["Time"] % 86400 / 3600).astype(int)
    df["amount_log"] = np.log1p(df["Amount"])

    # Transaction ID for entity tracking
    df["transaction_id"] = range(len(df))

    return df


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user aggregate features."""
    user_stats = df.groupby("user_id").agg(
        txn_count=("Amount", "count"),
        mean_amount=("Amount", "mean"),
        std_amount=("Amount", "std"),
        max_amount=("Amount", "max"),
        total_amount=("Amount", "sum"),
        fraud_count=("Class", "sum"),
    ).reset_index()

    user_stats["std_amount"] = user_stats["std_amount"].fillna(0)

    # Use the latest timestamp per user as the event timestamp
    latest_ts = df.groupby("user_id")["event_timestamp"].max().reset_index()
    user_stats = user_stats.merge(latest_ts, on="user_id")

    return user_stats


def main():
    os.makedirs("data", exist_ok=True)

    df = download_dataset()
    df = engineer_features(df)

    # ------------------------------------------------------------------
    # Save transaction features
    # ------------------------------------------------------------------
    txn_cols = [
        "transaction_id", "user_id", "event_timestamp",
        "Amount", "amount_log", "hour_of_day",
    ] + [f"V{i}" for i in range(1, 29)] + ["Class"]

    df[txn_cols].to_parquet("data/transactions.parquet", index=False)
    print(f"Saved data/transactions.parquet ({len(df):,} rows)")

    # ------------------------------------------------------------------
    # Save user aggregate features
    # ------------------------------------------------------------------
    user_df = build_user_features(df)
    user_df.to_parquet("data/user_features.parquet", index=False)
    print(f"Saved data/user_features.parquet ({len(user_df):,} rows)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    fraud_pct = df["Class"].mean() * 100
    print(f"\nDataset summary:")
    print(f"  Transactions: {len(df):,}")
    print(f"  Fraudulent:   {df['Class'].sum():,} ({fraud_pct:.3f}%)")
    print(f"  Users:        {user_df['user_id'].nunique():,}")
    print(f"  Time span:    {df['event_timestamp'].min()} → {df['event_timestamp'].max()}")


if __name__ == "__main__":
    main()
