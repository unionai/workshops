"""The tools the model may call from inside the sandbox.

`query` is the expensive one — `analysis.py` wraps it as an @env.task, so every
query the model writes becomes a durable child task. The render tools are
microseconds of pure Python and run in-process, where a round-trip would only add
latency.

The sandbox has no imports, so anything needing json/duckdb/datetime happens here
rather than in the generated code.
"""

from __future__ import annotations

import contextvars
import html
import os
import pathlib
import random
import time

import duckdb

import report
from dataset import (
    DATA_DESCRIPTION,
    LOAD_TRIPS_SQL,
    LOAD_ZONES_SQL,
    MONTHS,
    TRIPS_URL,
    ZONES_URL,
)

__all__ = [
    "DATA_DESCRIPTION",
    "query",
    "create_metric",
    "create_chart",
    "create_table",
    "describe_column",
    "REPORT_TOOLS",
    "ALL_TOOLS",
    "new_report",
    "collect_report",
]

MAX_ROWS = 500


# {{docs-fragment sql_guard}}
def _assert_read_only(sql: str) -> None:
    """Reject anything that is not exactly one SELECT.

    DuckDB's parser classifies the statement, which beats matching keywords by hand.
    Not sufficient on its own: `read_csv('/etc/passwd')` is a valid SELECT, and the
    lockdown in _connect() is what stops that one.

    The error messages are written for the model, not for us: they say what was
    wrong AND what to do about it, so the agent fixes its program in one turn
    instead of guessing.
    """
    sql = (sql or "").strip()

    if not sql:
        raise ValueError(
            "The `sql` argument was empty. It must be a complete SELECT statement, "
            "e.g. \"SELECT count(*) AS n FROM trips\". If you are fanning out with "
            'flyte_map("query", sqls, months), make sure every entry of `sqls` is a '
            "full SELECT string."
        )

    if not sql.lstrip().upper().startswith(("SELECT", "WITH")):
        raise ValueError(
            f"The `sql` argument must be a SELECT statement, but it starts with: "
            f"{sql.splitlines()[0][:60]!r}. You passed something that is not SQL — "
            "probably Python. `query(sql, month)` takes the SQL string as its FIRST "
            "argument and a month like '2024-03' as its second."
        )

    try:
        statements = duckdb.extract_statements(sql)
    except Exception as exc:
        raise ValueError(f"That SQL did not parse: {exc}") from exc

    if len(statements) != 1:
        raise ValueError(
            f"Pass exactly one statement, got {len(statements)}. Run one SELECT per "
            "query() call."
        )

    kind = statements[0].type.name
    if kind != "SELECT":
        raise ValueError(f"Only SELECT is allowed, got {kind}. The data is read-only.")
# {{/docs-fragment sql_guard}}


CACHE_DIR = pathlib.Path(os.environ.get("TAXI_CACHE_DIR", "/tmp/nyc-taxi"))


def _fetch(url_sql: str, path: pathlib.Path) -> None:
    """Pull one file from the TLC, with backoff.

    The host rate-limits bursts, and a fan-out is a burst: twelve tasks starting at
    once means twelve simultaneous downloads. Jittered backoff spreads the retries
    out instead of having all twelve retry in lockstep.
    """
    last_error: Exception | None = None
    for attempt in range(6):
        con = duckdb.connect()
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            # Temp name first: a killed download must not look like a cache hit.
            tmp = path.with_suffix(".partial")
            con.execute(f"COPY ({url_sql}) TO '{tmp}' (FORMAT parquet)")
            tmp.replace(path)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt + random.uniform(0, 3))
        finally:
            con.close()

    raise RuntimeError(f"Could not download after 6 attempts: {last_error}") from last_error


def download_month(month: str) -> tuple[str, str]:
    """Fetch a month from the TLC to local disk; return (trips_path, zones_path).

    The only function here that touches the network. On the cluster it runs inside
    the cached `fetch_*` tasks in analysis.py, so a month is downloaded once and
    then served from blob storage forever after.
    """
    if month not in MONTHS:
        raise ValueError(f"Unknown month {month!r}. Available: {MONTHS[0]}..{MONTHS[-1]}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    trips_path = CACHE_DIR / f"trips-{month}.parquet"
    zones_path = CACHE_DIR / "zones.parquet"

    if not trips_path.exists():
        _fetch(LOAD_TRIPS_SQL.format(url=TRIPS_URL.format(month=month)), trips_path)
    if not zones_path.exists():
        _fetch(LOAD_ZONES_SQL.format(url=ZONES_URL), zones_path)

    return str(trips_path), str(zones_path)


# {{docs-fragment query_tool}}
def _connect(trips_path: str, zones_path: str) -> duckdb.DuckDBPyConnection:
    """Open a connection over the month's files, then lock the door behind us."""
    con = duckdb.connect()
    # Tables, not views: a view is read lazily, i.e. after the lockdown below, and
    # would then be denied along with everything else. Load first, then lock.
    con.execute(f"CREATE TABLE trips AS SELECT * FROM read_parquet('{trips_path}')")
    con.execute(f"CREATE TABLE zones AS SELECT * FROM read_parquet('{zones_path}')")

    # No files, no network, no extensions — and the model's SQL cannot re-enable any
    # of it.
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def query_files(sql: str, trips_path: str, zones_path: str) -> list:
    """Run the model's SQL against a month already on local disk."""
    _assert_read_only(sql)

    con = _connect(trips_path, zones_path)
    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchmany(MAX_ROWS)
    finally:
        con.close()

    # Monty handles only primitives and list/dict/tuple, so dates become strings.
    out = []
    for row in rows:
        record = {}
        for name, value in zip(columns, row):
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif value is not None and not isinstance(value, (int, float, str, bool)):
                value = str(value)
            record[name] = value
        out.append(record)
    return out


def query(sql: str, month: str) -> list:
    """Run a read-only DuckDB SELECT over one month of NYC taxi trips.

    Args:
        sql: A single SELECT against the tables `trips` and `zones`. Aggregate in
            SQL — do not select raw trip rows. At most 500 rows are returned.
        month: Which month to query, as "YYYY-MM" (e.g. "2024-03"). Months
            2024-01 through 2024-12 are available. One call reads one month; to
            cover several months, call this once per month.

    Returns:
        A list of row dicts, with dates and timestamps as ISO strings.
    """
    trips_path, zones_path = download_month(month)
    return query_files(sql, trips_path, zones_path)
# {{/docs-fragment query_tool}}


# The render tools append their HTML here and return a one-line confirmation, so the
# markup never travels back through the sandbox as an observation. The ContextVar
# keeps concurrent runs out of each other's reports.

# {{docs-fragment collector}}
_blocks: contextvars.ContextVar[list] = contextvars.ContextVar("report_blocks")


def new_report() -> None:
    """Start a fresh report for this run."""
    _blocks.set([])


def collect_report() -> list[str]:
    """Take the HTML blocks the render tools produced during this run."""
    return list(_blocks.get([]))


def _emit(block_html: str) -> None:
    blocks = _blocks.get(None)
    if blocks is None:
        blocks = []
        _blocks.set(blocks)
    blocks.append(block_html)
# {{/docs-fragment collector}}


def create_metric(label: str, value: str, note: str = "") -> str:
    """Add a headline number to the report (a big stat card).

    Args:
        label: What the number is, e.g. "Median fare".
        value: The number, already formatted, e.g. "$14.20" or "12.3%".
        note: Optional one-line caveat or comparison shown under the value.
    """
    _emit(
        f'<div class="metric"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div></div>'
    )
    return f"metric added: {label} = {value}"


def create_chart(chart_type: str, title: str, labels: list, values: list) -> str:
    """Add a chart to the report.

    Args:
        chart_type: One of "bar", "line", "pie", "doughnut".
        title: Chart title shown above the plot.
        labels: X-axis labels (or slice labels for pie/doughnut).
        values: Either a flat list of numbers, or — for multiple series — a list
            of dicts like {"label": "Manhattan", "data": [1, 2, 3]}.
    """
    if chart_type not in ("bar", "line", "pie", "doughnut"):
        raise ValueError(f"chart_type must be bar/line/pie/doughnut, got {chart_type!r}")

    palette = ["#6366f1", "#34d399", "#f59e0b", "#f87171", "#38bdf8", "#a78bfa"]

    if values and isinstance(values[0], dict):
        datasets = [
            {
                "label": s.get("label", ""),
                "data": s.get("data", []),
                "backgroundColor": palette[i % len(palette)],
                "borderColor": palette[i % len(palette)],
            }
            for i, s in enumerate(values)
        ]
    elif chart_type in ("pie", "doughnut"):
        datasets = [{"label": title, "data": values, "backgroundColor": palette}]
    else:
        datasets = [{"label": title, "data": values, "backgroundColor": palette[0],
                     "borderColor": palette[0]}]

    spec = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": True, "text": title}},
        },
    }

    blocks = _blocks.get(None) or []
    _emit(report.chart_html(spec, canvas_id=f"chart-{len(blocks)}"))
    return f"chart added: {title} ({len(labels)} points)"


def create_table(title: str, rows: list) -> str:
    """Add a small table to the report. Keep it to ~20 rows — it is for reading,
    not for dumping a result set.

    Args:
        title: Table caption.
        rows: List of row dicts. Keys of the first row become the columns.
    """
    if not rows:
        return "table skipped: no rows"

    columns = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns) + "</tr>"
        for r in rows[:20]
    )
    _emit(
        f'<div class="table"><h3>{html.escape(title)}</h3>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )
    return f"table added: {title} ({len(rows)} rows)"


def describe_column(values: list) -> dict:
    """Summary statistics for a list of numbers.

    Returns a dict with count, mean, median, min, max, and std_dev. Useful for
    turning a column of query results into a single claim without another query.
    """
    nums = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not nums:
        return {"count": 0}

    n = len(nums)
    mean = sum(nums) / n
    mid = n // 2
    median = nums[mid] if n % 2 else (nums[mid - 1] + nums[mid]) / 2
    variance = sum((x - mean) ** 2 for x in nums) / n
    return {
        "count": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "min": nums[0],
        "max": nums[-1],
        "std_dev": round(variance**0.5, 4),
    }


# The render tools — cheap, in-process, no round-trip.
REPORT_TOOLS = [create_metric, create_chart, create_table, describe_column]

# Everything, with the plain (non-durable) query. Steps 1 and 2 use this;
# step 3 swaps `query` for the @env.task version.
ALL_TOOLS = [query, *REPORT_TOOLS]
