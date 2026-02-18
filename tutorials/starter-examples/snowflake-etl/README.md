# Snowflake ETL Pipeline

Query data from Snowflake, transform in Python, and write results back.

## What it does

- **`extract_orders`** — Queries orders from Snowflake using the Snowflake connector
- **`transform`** — Aggregates orders into per-customer summary with pandas
- **`load_summary`** — Batch inserts the summary back to Snowflake
- **`pipeline`** — Orchestrates extract -> transform -> load

## Setup

```bash
cd tutorials/starter-examples/snowflake-etl

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

## Flyte Cluster (for remote runs)

To run remotely, configure your Flyte cluster endpoint:

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

Don't have a cluster? Request access at [flyte.org](https://flyte.org/).

## Additional setup

1. Update `SnowflakeConfig` with your Snowflake account and user
2. Create the `orders` and `customer_summary` tables in your Snowflake database
3. Configure the `snowflake` private key secret on your Flyte cluster

## Run

**Remote:**
```bash
uv run flyte run snowflake_etl.py pipeline
```

**Local:**
```bash
uv run flyte run --local snowflake_etl.py pipeline
```