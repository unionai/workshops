"""Shared feature definitions and utilities."""

import numpy as np

TXN_FEATURE_COLS = ["amt", "amt_log", "category_encoded", "merch_lat", "merch_long"]
USER_FEATURE_COLS = ["txn_count", "mean_amt", "std_amt", "max_amt", "home_lat", "home_long", "age"]
DERIVED_FEATURE_COLS = ["amt_zscore", "amt_ratio", "distance_from_home", "hour", "day_of_week"]
ALL_FEATURE_COLS = TXN_FEATURE_COLS + USER_FEATURE_COLS + DERIVED_FEATURE_COLS


def haversine(lat1, lon1, lat2, lon2):
    """Compute distance in miles between two (lat, lon) points."""
    R = 3959
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
