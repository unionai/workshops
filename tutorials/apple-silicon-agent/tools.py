"""The agent's tools: three read-only operations on a SQLite database, each a Flyte task.

    flyte run tools.py list_tables
    flyte run tools.py schema
    flyte run tools.py describe_table --table Invoice
    flyte run tools.py run_sql --query "SELECT Country, COUNT(*) n FROM Customer GROUP BY Country ORDER BY n DESC LIMIT 3"

Because they are tasks, every call the agent makes is a durable, retryable action with its own
inputs, outputs and logs in the run graph. The database is Chinook, a small music store (11 tables,
1 MB), downloaded on first use. Nothing here needs a GPU or a model.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import urllib.request
from pathlib import Path

import flyte

from config import PROPAGATED

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CHINOOK_URL = "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
MAX_ROWS = 50

tools_env = flyte.TaskEnvironment(
    name="apple-silicon-agent-tools",
    image=flyte.Image.from_debian_base().with_pip_packages("python-dotenv"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    env_vars=PROPAGATED,
)

_db_path: Path | None = None


def connect() -> sqlite3.Connection:
    """Open the database read-only, downloading it once per process."""
    global _db_path
    if _db_path is None or not _db_path.exists():
        _db_path = Path(tempfile.gettempdir()) / "Chinook_Sqlite.sqlite"
        if not _db_path.exists():
            log.info(f"downloading {CHINOOK_URL}")
            urllib.request.urlretrieve(CHINOOK_URL, _db_path)
    return sqlite3.connect(f"file:{_db_path}?mode=ro", uri=True)


@tools_env.task
async def list_tables() -> list[str]:
    """List the tables in the database."""
    with connect() as con:
        return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


@tools_env.task
async def schema() -> str:
    """Every table with its columns, one line each. The agent reads this before its first query."""
    with connect() as con:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        lines = []
        for t in tables:
            cols = ", ".join(c[1] for c in con.execute(f'PRAGMA table_info("{t}")'))
            lines.append(f"{t}({cols})")
        return "\n".join(lines)


@tools_env.task
async def describe_table(table: str) -> str:
    """The columns of one table, with types, and three sample rows."""
    with connect() as con:
        cols = con.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not cols:
            return f"error: no table named {table!r}. Call list_tables."
        sample = con.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall()
    names = [c[1] for c in cols]
    lines = [f"{table}({', '.join(f'{c[1]} {c[2]}' for c in cols)})", "sample rows:"]
    lines += [json.dumps(dict(zip(names, row)), default=str) for row in sample]
    return "\n".join(lines)


@tools_env.task
async def run_sql(query: str) -> str:
    """Run one read-only SQL query (SQLite dialect). Returns up to 50 rows as JSON, or an error message."""
    q = query.strip().rstrip(";")
    if not q.lower().startswith(("select", "with")):
        return "error: only SELECT queries are allowed"
    try:
        with connect() as con:
            cur = con.execute(q)
            names = [d[0] for d in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        return f"error: {e}"
    truncated = len(rows) > MAX_ROWS
    out = [dict(zip(names, r)) for r in rows[:MAX_ROWS]]
    return json.dumps({"rows": out, "row_count": len(out), "truncated": truncated}, default=str)
