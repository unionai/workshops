"""Step 1 — the sandbox, with no LLM anywhere.

Before we let a model write code, it's worth seeing what actually runs it.

`@env.sandbox.orchestrator` marks a task whose body executes inside Monty, a
Rust-based Python interpreter with no filesystem, no network, no imports, and no
OS access. It can do control flow — loops, conditionals, arithmetic — and it can
call other Flyte tasks. That's all.

When the sandboxed body calls `load_month`, Monty *pauses*, Flyte runs that task
in a real container (which does have network access, and downloads a month of
taxi data), and Monty *resumes* with the result. To the code in the sandbox it
just looks like a function call that returned a value.

Run it:

    uv run flyte run --local step1_sandbox.py monthly_tip_trend
    uv run flyte run step1_sandbox.py monthly_tip_trend
"""

from __future__ import annotations

import tools
from config import sandbox_env as env  # no secret needed — there's no LLM in this step


# --- Worker tasks: these run in full containers, with network and DuckDB -----


@env.task
def load_month(month: str) -> dict:
    """Fetch one month's headline numbers. Runs in a container, hits the network."""
    rows = tools.query(
        """
        SELECT count(*)                                          AS trips,
               round(avg(tip_amount / nullif(fare_amount, 0)) * 100, 2) AS tip_pct,
               round(avg(trip_distance), 2)                      AS avg_miles
        FROM trips
        WHERE payment_type = 1          -- card only: cash tips are never recorded
          AND fare_amount > 0
          AND trip_distance BETWEEN 0.1 AND 100
        """,
        month,
    )
    row = rows[0]
    return {"month": month, **row}


@env.task
def pct_change(first: float, last: float) -> float:
    """A deliberately trivial task, to show what dispatch looks like."""
    if first == 0:
        return 0.0
    return round((last - first) / first * 100, 2)


# --- The sandboxed orchestrator: pure control flow, no I/O ------------------


@env.sandbox.orchestrator
async def monthly_tip_trend(months: list[str] = ["2024-01", "2024-06", "2024-12"]) -> dict:
    """Compare tipping across several months.

    Everything in this function body runs inside Monty. Note what it does *not*
    do: no imports, no file access, no HTTP. The only way it reaches the outside
    world is by calling `load_month`, which Flyte dispatches for it.

    Note also what it *can* do — loop, branch, accumulate, compare. That is the
    part that, in the next step, an LLM will write instead of us.
    """
    summaries = []
    best_month = ""
    best_tip = 0.0

    for month in months:
        summary = load_month(month)
        summaries.append(summary)

        tip = summary["tip_pct"]
        if tip > best_tip:
            best_tip = tip
            best_month = month

    drift = pct_change(summaries[0]["tip_pct"], summaries[-1]["tip_pct"])

    return {
        "months": summaries,
        "most_generous_month": best_month,
        "best_tip_pct": best_tip,
        "tip_pct_change_over_period": drift,
    }


if __name__ == "__main__":
    import flyte

    flyte.init_from_config(image_builder="remote")
    run = flyte.run(monthly_tip_trend, months=["2024-01", "2024-06", "2024-12"])
    print(f"View at: {run.url}")
    run.wait()
    print(run.outputs())
