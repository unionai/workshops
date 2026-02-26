"""
Feast feature definitions for fraud detection.

Entities:
    user — join key for per-user features

Feature Views:
    transaction_features — per-transaction data (amount, PCA features, engineered features)
    user_stats           — aggregated per-user statistics (txn count, avg amount, etc.)
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64

# ------------------------------------------------------------------
# Entities
# ------------------------------------------------------------------

user = Entity(
    name="user",
    join_keys=["user_id"],
    description="A credit card user (synthetic ID)",
)

# ------------------------------------------------------------------
# Data sources
# ------------------------------------------------------------------

transactions_source = FileSource(
    path="../data/transactions.parquet",
    timestamp_field="event_timestamp",
)

user_features_source = FileSource(
    path="../data/user_features.parquet",
    timestamp_field="event_timestamp",
)

# ------------------------------------------------------------------
# Feature Views
# ------------------------------------------------------------------

transaction_features = FeatureView(
    name="transaction_features",
    entities=[user],
    ttl=timedelta(days=3),
    schema=[
        Field(name="Amount", dtype=Float64),
        Field(name="amount_log", dtype=Float64),
        Field(name="hour_of_day", dtype=Int64),
    ] + [
        Field(name=f"V{i}", dtype=Float64) for i in range(1, 29)
    ],
    online=True,
    source=transactions_source,
)

user_stats = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="txn_count", dtype=Int64),
        Field(name="mean_amount", dtype=Float64),
        Field(name="std_amount", dtype=Float64),
        Field(name="max_amount", dtype=Float64),
        Field(name="total_amount", dtype=Float64),
        Field(name="fraud_count", dtype=Int64),
    ],
    online=True,
    source=user_features_source,
)