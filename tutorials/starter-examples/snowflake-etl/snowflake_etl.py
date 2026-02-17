"""
Snowflake ETL on Flyte

Query data from Snowflake, transform in Python, and write results back.

Usage:
    uv run flyte run snowflake_etl.py pipeline
"""

import pandas as pd
from flyteplugins.snowflake import Snowflake, SnowflakeConfig

import flyte

sf_config = SnowflakeConfig(
    account="<your-account>",
    user="<your-user>",
    database="ANALYTICS",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
)

# --- Snowflake queries ---

extract_orders = Snowflake(
    name="extract_orders",
    query_template="""
        SELECT customer_id, order_date, amount
        FROM orders
        WHERE order_date >= '2024-01-01'
    """,
    plugin_config=sf_config,
    output_dataframe_type=pd.DataFrame,
    snowflake_private_key="snowflake",
)

load_summary = Snowflake(
    name="load_summary",
    query_template="""
        INSERT INTO customer_summary (customer_id, total_spend, order_count)
        VALUES (%(customer_id)s, %(total_spend)s, %(order_count)s)
    """,
    plugin_config=sf_config,
    inputs={"customer_id": list[str], "total_spend": list[float], "order_count": list[int]},
    snowflake_private_key="snowflake",
    batch=True,
)

snowflake_env = flyte.TaskEnvironment.from_task("snowflake_env", extract_orders, load_summary)

env = flyte.TaskEnvironment(
    name="etl",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-snowflake"),
    depends_on=[snowflake_env],
)


# --- Transform ---

@env.task
async def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate orders into per-customer summary."""
    summary = df.groupby("customer_id").agg(
        total_spend=("amount", "sum"),
        order_count=("amount", "count"),
    ).reset_index()
    print(f"Summarized {len(df)} orders into {len(summary)} customers")
    return summary


# --- Pipeline ---

@env.task
async def pipeline() -> pd.DataFrame:
    """Extract orders from Snowflake, transform, and load summary back."""
    orders = await extract_orders()
    summary = await transform(orders)
    await load_summary(
        customer_id=summary["customer_id"].tolist(),
        total_spend=summary["total_spend"].tolist(),
        order_count=summary["order_count"].tolist(),
    )
    return summary

# uv run flyte run snowflake_etl.py pipeline