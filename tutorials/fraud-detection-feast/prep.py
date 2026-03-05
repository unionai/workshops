"""Download Sparkov fraud dataset and prepare parquets. Usage: python prep.py"""

import json
import os

import numpy as np
import pandas as pd
import kagglehub

from shared import haversine


def main():
    os.makedirs("data", exist_ok=True)

    print("Downloading Sparkov credit card fraud dataset...")
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    df = pd.read_csv(os.path.join(path, "fraudTrain.csv"))
    print(f"Loaded {len(df):,} transactions ({df['is_fraud'].sum():,} fraudulent)")

    # Sample for workshop speed
    if len(df) > 500_000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=500_000, stratify=df["is_fraud"], random_state=42)
        print(f"Sampled to {len(df):,} transactions")

    # Parse timestamps
    df["event_timestamp"] = pd.to_datetime(df["trans_date_trans_time"])
    df["event_timestamp"] = df["event_timestamp"].dt.tz_localize("UTC")
    df["hour"] = df["event_timestamp"].dt.hour
    df["day_of_week"] = df["event_timestamp"].dt.dayofweek

    # Map cc_num → sequential user_id
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

    # Save transaction features
    txn_cols = [
        "user_id", "event_timestamp",
        "amt", "amt_log", "category_encoded", "merch_lat", "merch_long",
        "hour", "day_of_week", "lat", "long", "distance",
        "is_fraud",
    ]
    df[txn_cols].to_parquet("data/transactions.parquet", index=False)
    print(f"Saved data/transactions.parquet ({len(df):,} rows)")

    # Save user profiles
    user_stats.to_parquet("data/user_features.parquet", index=False)
    print(f"Saved data/user_features.parquet ({len(user_stats):,} rows)")

    # Save mappings
    with open("data/category_mapping.json", "w") as f:
        json.dump(cat_to_int, f, indent=2)
    print(f"Saved data/category_mapping.json ({len(cat_to_int)} categories)")

    # Summary
    fraud_pct = df["is_fraud"].mean() * 100
    print(f"\nDataset summary:")
    print(f"  Transactions: {len(df):,}")
    print(f"  Fraudulent:   {df['is_fraud'].sum():,} ({fraud_pct:.2f}%)")
    print(f"  Users:        {user_stats['user_id'].nunique():,}")
    print(f"  Categories:   {len(categories)}")
    print(f"  Time span:    {df['event_timestamp'].min()} → {df['event_timestamp'].max()}")


if __name__ == "__main__":
    main()
