"""Step 3 — the agent, and the payoff: generated code that fans out.

Same sandbox, same tools, but now `flyte.ai.agents.Agent` runs the loop for us,
and one thing changes that matters a lot.

In steps 1 and 2, `query` was a plain Python function: it ran inside the task that
called it. In `analysis.py` it is an `@env.task`. The code-mode runtime notices,
and every `query(...)` the model writes is dispatched through the Flyte controller
as a durable child task — retried on failure, cached, and visible in the UI.

So when the model writes this (and for a twelve-month question, it will):

    results = flyte_map("query", sqls, months)

...that is not a for-loop. It is twelve containers running at once, each pulling
down its own month of taxi data, each retried independently if it fails. The model
wrote a fan-out without knowing it was writing a fan-out.

That is the thing the sandbox buys you that a plain code interpreter does not.
Meanwhile the cheap render tools stay in-process, where a round-trip would only
add latency.

Run it, then open the run in the UI and look at the child tasks:

    uv run flyte run step3_agent_report.py analyze \\
        --question "Rank the boroughs by tip rate and show how it moved through 2024"
"""

from __future__ import annotations

import tools
from analysis import build_agent
from config import env
from report import render


@env.task(report=True)
async def analyze(question: str) -> str:
    """Answer one question, and render the findings as an HTML report."""
    import flyte.report

    tools.new_report()
    agent, usage = build_agent(code_mode=True)

    result = await agent.run.aio(question)

    if result.error:
        raise RuntimeError(f"The agent could not answer: {result.error}")

    # Every program it wrote, not just the last — see llm.Usage.
    html = render(question, usage.programs or result.code, tools.collect_report(),
                  result.summary)
    html += (
        f'<p style="color:#6b7280;font-size:.8rem">'
        f"{usage.turns} model turns · {usage.total_tokens:,} tokens · "
        f"{usage.seconds:.1f}s in the model</p>"
    )
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    return result.summary


if __name__ == "__main__":
    import flyte

    flyte.init_from_config(image_builder="remote")
    run = flyte.run(
        analyze,
        question="Rank the boroughs by tip rate and show how it moved through 2024",
    )
    print(f"View at: {run.url}")
    run.wait()
    print(run.outputs())
