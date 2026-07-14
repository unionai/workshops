"""The durable half: the `query` task, the agent's instructions, and the agent itself.

Steps 3, 4, and 5 all import from here, so the durable task is defined in exactly
one place and the two modes in step 4 differ by exactly one flag.
"""

from __future__ import annotations

from flyte.ai.agents import Agent
from flyte.io import File

import tools
from config import env
from dataset import MONTHS
from llm import MODEL, SANDBOX_RULES, Usage, counting_llm


# {{docs-fragment fetch_task}}
# Cached on the month alone — not on the SQL — so the TLC is hit once per month,
# ever, and every later query (any SQL, any question, any user in this project)
# reads the Parquet back out of blob storage. Without this, a twelve-month fan-out
# is twelve pods downloading from the same host at the same moment, which is a good
# way to get rate-limited.
@env.task(cache="auto", retries=3)
def fetch_trips(month: str) -> File:
    """Fetch one month of trips from the TLC into blob storage."""
    trips_path, _ = tools.download_month(month)
    return File.from_local_sync(trips_path)


@env.task(cache="auto", retries=3)
def fetch_zones() -> File:
    """Fetch the zone lookup (265 rows, shared by every month)."""
    _, zones_path = tools.download_month(MONTHS[0])
    return File.from_local_sync(zones_path)
# {{/docs-fragment fetch_task}}


# {{docs-fragment query_task}}
# Durable, and cached: the same SQL over the same published month is deterministic,
# so identical queries dedupe across every question anyone asks.
@env.task(cache="auto")
def query(sql: str, month: str) -> list:
    """Run a read-only DuckDB SELECT over one month of NYC taxi trips.

    Args:
        sql: A single SELECT against the tables `trips` and `zones`. Aggregate in
            SQL — do not select raw trip rows. At most 500 rows are returned.
        month: Which month to query, as "YYYY-MM" (e.g. "2024-03"). Months
            2024-01 through 2024-12 are available. One call reads one month.

    Returns:
        A list of row dicts, with timestamps as ISO strings.
    """
    trips = fetch_trips(month)  # cache hit after the first time, for everyone
    zones = fetch_zones()
    return tools.query_files(sql, trips.download_sync(), zones.download_sync())
# {{/docs-fragment query_task}}


INSTRUCTIONS = f"""You are a data analyst for NYC's taxi commission.

{tools.DATA_DESCRIPTION}

Available months: {", ".join(MONTHS)}.

How to work:
  - Each `query` call reads ONE month.

  - To cover several months, fan them out with `flyte_map` — they then run in
    PARALLEL, as separate durable tasks on the cluster, so twelve months costs
    about the same wall-clock as one. `flyte_map` takes the name of a registered
    task as a STRING (it cannot take a function you define yourself), followed by
    one list per argument of that task. `query(sql, month)` takes two, so:

        sql = "SELECT count(*) AS trips, sum(tip_amount) AS tips FROM trips WHERE payment_type = 1"

        wanted = ["2024-01", "2024-02", "2024-03"]
        sqls = []
        for m in wanted:
            sqls.append(sql)            # the SAME sql string, once per month
        per_month = flyte_map("query", sqls, wanted, concurrency=4)  # -> list of row-lists

    Always pass concurrency=4. Do NOT call `query` in a plain for-loop over months —
    that runs them one at a time for no reason.

    Define `sql` as a complete SELECT string BEFORE you build the list, exactly as
    above. Every entry of `sqls` must be a full SELECT. The first argument to
    `query` is always SQL and the second is always a month string like "2024-03" —
    never pass Python code, a list, or an empty string as the SQL.

  - Do the whole job in ONE program: fan out the queries, compute, AND call the
    create_* tools — all of it, in the first program you write. Then reply with
    your plain-text answer.

    Do not write an exploratory program that just fetches numbers, planning to
    chart them in a second one. Each program runs in a fresh sandbox, so the
    second program would have to re-type the data as literals — and a chart built
    from numbers you typed by hand is not a chart of the data, it is a chart of
    what you remembered. If you catch yourself pasting numbers into a program, the
    fix belongs in the first program.

  - Do the analysis in code: query for the numbers, then rank, compare, and
    compute growth in plain Python. Don't do arithmetic in your head that you
    could just write down.
  - Put your findings in the report as you go: `create_metric` for headline
    numbers, `create_chart` for anything with a trend or a ranking,
    `create_table` for a small leaderboard.
  - Then reply with one or two plain sentences stating what you found. Lead with
    the answer, not with what you did.

{SANDBOX_RULES}"""


# {{docs-fragment agent}}
def build_agent(code_mode: bool = True, max_turns: int = 6) -> tuple[Agent, Usage]:
    """The agent, plus the tally of what it spends.

    The `tools` list is the interesting part: `query` is an @env.task, so calls to it
    dispatch as durable child tasks, while the render helpers run in-process. The
    model calls them identically — where each one runs is our decision, not its.
    """
    call_llm, usage = counting_llm()
    agent = Agent(
        name=f"taxi-analyst-{'code' if code_mode else 'sequential'}",
        instructions=INSTRUCTIONS,
        model=MODEL,
        tools=[query, *tools.REPORT_TOOLS],
        code_mode=code_mode,
        max_turns=max_turns,
        call_llm=call_llm,
    )
    return agent, usage
# {{/docs-fragment agent}}
