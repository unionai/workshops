import duckdb
import pandas as pd

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="duckdb-etl",
    image=flyte.Image.from_debian_base().with_pip_packages("duckdb", "pandas", "pyarrow"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

SAMPLE_CSV = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

@env.task
async def extract() -> pd.DataFrame:
    """Load raw data from a CSV source."""
    df = duckdb.sql(f"SELECT * FROM read_csv_auto('{SAMPLE_CSV}')").df()
    print(f"Extracted {len(df)} rows")
    return df

@env.task
async def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate survival stats by passenger class using DuckDB SQL."""
    summary = duckdb.sql("""
        SELECT
            Pclass AS passenger_class,
            COUNT(*) AS total,
            SUM(Survived) AS survived,
            ROUND(AVG(Survived) * 100, 1) AS survival_rate,
            ROUND(AVG(Fare), 2) AS avg_fare
        FROM raw
        GROUP BY Pclass
        ORDER BY Pclass
    """).df()
    print(summary.to_string(index=False))
    return summary

@env.task(report=True)
async def pipeline() -> pd.DataFrame:
    """Extract → Transform pipeline."""
    raw = await extract()
    summary = await transform(raw)

    await flyte.report.replace.aio(
        f"<h2>DuckDB ETL Results</h2>"
        f"<p>Processed {len(raw)} rows into {len(summary)} groups</p>"
        f"<h3>Survival by Passenger Class</h3>"
        f"{summary.to_html(index=False)}"
    )
    await flyte.report.flush.aio()

    return summary

# uv run flyte run duckdb_etl.py pipeline