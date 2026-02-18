# DuckDB Data Pipeline

Extract CSV data, transform with DuckDB SQL, and display results in a Flyte report.

## What it does

- **`extract`** — Loads the Titanic CSV from a public URL using DuckDB's `read_csv_auto`
- **`transform`** — Aggregates survival statistics by passenger class using SQL
- **`pipeline`** — Orchestrates extract -> transform, renders results as an HTML table in a Flyte report

## Setup

```bash
cd tutorials/starter-examples/duckdb-etl

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

## Run

**Remote:**
```bash
uv run flyte run duckdb_etl.py pipeline
```

**Local:**
```bash
uv run flyte run --local duckdb_etl.py pipeline
```

## Notes

- Fully self-contained — no external services or accounts needed
- DuckDB can query pandas DataFrames directly with SQL