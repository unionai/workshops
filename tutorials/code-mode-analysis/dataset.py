"""The dataset: NYC Yellow Taxi trip records, read straight from S3 as Parquet.

Everything dataset-specific lives here. To point the tutorial at different data,
change this file and nothing else: the tools, the agent, and the app all read the
schema description and the load SQL from here.

The data is the real TLC public dataset — ~3 million trips per month, hosted as
monthly Parquet files. DuckDB reads them over HTTPS with column and row-group
pushdown, so a query touches megabytes, not the whole 47MB file.
"""

from __future__ import annotations

TRIPS_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# The months the agent is allowed to query. One month is one `query` task.
MONTHS = [f"2024-{m:02d}" for m in range(1, 13)]

# Trusted SQL — ours, never the model's. These run once per month, at download
# time, and they are the only statements in the tutorial that touch the network.
# Projecting to just the columns we expose is what keeps the cached file (and the
# pod that loads it) a sensible size: the source file carries 19 columns, we
# publish 11.
LOAD_TRIPS_SQL = """
SELECT
    tpep_pickup_datetime   AS pickup,
    tpep_dropoff_datetime  AS dropoff,
    passenger_count,
    trip_distance,
    PULocationID           AS pu_id,
    DOLocationID           AS do_id,
    payment_type,
    fare_amount,
    tip_amount,
    tolls_amount,
    total_amount
FROM read_parquet('{url}')
"""

LOAD_ZONES_SQL = """
SELECT LocationID AS location_id, Borough AS borough, Zone AS zone
FROM read_csv_auto('{url}')
"""

# This text is handed to the model verbatim. It is the entire contract between
# the LLM and the data — if a column isn't described here, the model won't use it.
DATA_DESCRIPTION = """\
You are analyzing the NYC Yellow Taxi public trip record dataset for 2024.

The `query(sql, month)` tool runs read-only DuckDB SQL against ONE month of data.
`month` is a string like "2024-03". Available months: 2024-01 through 2024-12.

Two tables are in scope for every query:

  trips (one row per taxi trip, ~3 million rows per month)
    pickup           TIMESTAMP  trip start time
    dropoff          TIMESTAMP  trip end time
    passenger_count  BIGINT     passengers (sometimes 0 or NULL — filter these)
    trip_distance    DOUBLE     miles
    pu_id, do_id     BIGINT     pickup / dropoff zone ids, join to zones.location_id
    payment_type     BIGINT     1 = credit card, 2 = cash, 3 = no charge, 4 = dispute
    fare_amount      DOUBLE     metered fare, dollars
    tip_amount       DOUBLE     tip, dollars
    tolls_amount     DOUBLE     tolls, dollars
    total_amount     DOUBLE     total charged, dollars

  zones (265 rows, one per taxi zone)
    location_id      BIGINT
    borough          VARCHAR    Manhattan, Queens, Brooklyn, Bronx, Staten Island, EWR
    zone             VARCHAR    e.g. "JFK Airport", "Upper East Side North"

Data quality notes you MUST account for:
  - Tips are only recorded for CARD payments. Cash tips are always 0. Any tip
    analysis must filter `payment_type = 1`, or it will conclude that cash-heavy
    boroughs don't tip, which is an artifact and not a fact.
  - There are junk rows: negative fares, zero-mile trips, 100+ mile trips, and
    timestamps outside the month. Filter to sane ranges before aggregating.
  - Some boroughs and zones have almost no pickups (EWR has a few dozen trips a
    month against Manhattan's millions). An average over 28 trips is noise, and it
    will top any ranking you build. Require a minimum trip count — `HAVING
    count(*) > 1000` is a reasonable floor — or say plainly that the sample is thin.
  - Aggregate in SQL. Do not ask for raw trip rows — a month is 3 million of them.
"""
