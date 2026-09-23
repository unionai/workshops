"""Step 2: the ML engineer agent, on Flyte.

A LangGraph graph (see graph.py) drives the work: the model reads the request, lists the
candidates, evaluates what it needs to, fine-tunes if that is what it takes, promotes one
model, and returns a typed decision. Flyte makes it durable: the task retries, every model
turn is recorded so a retry replays it, and every tool call is a child action with its own
container, retries and cache. Evals and fine-tunes run on T4s, in parallel when the
model asks for several at once.

Needs the key for AGENT_MODEL's provider (locally in .env; on the cluster as a secret).
If Grafana is configured, this run is also a conversation in Agent Observability and the
task carries the two Grafana links; step 4 is about looking at them.

    flyte run --local ml_engineer.py ml_engineer_agent --candidates_to_screen 2 --max_fine_tunes 1
    flyte run ml_engineer.py ml_engineer_agent
    flyte run ml_engineer.py ml_engineer_agent --min_accuracy 0.9 --max_latency_ms 120 --budget 6

The task report shows the agent's timeline (model turns and tool calls) on one tab and
the decision, checked against the request, on another. On the cluster, `promote` also
publishes the winner as the `ticket-router` artifact and deploys the serving app.
"""

from __future__ import annotations

import json

import flyte
import flyte.report

from config import GRAFANA_LINKS, observed_env
from graph import Request
from graph import engineer as _engineer
from report import decision_html


@observed_env.task(report=True, retries=2, links=GRAFANA_LINKS)
async def ml_engineer_agent(
    min_accuracy: float = 0.95,
    max_latency_ms: float = 150.0,
    candidates_to_screen: int = 5,
    max_fine_tunes: int = 3,
    budget: int = 12,
    dataset: str = "v1",
    situation: str = "",
    model: str | None = None,
) -> dict:
    """Take the request, run the factory, return the decision with what it cost."""
    request = Request(
        min_accuracy, max_latency_ms, candidates_to_screen, max_fine_tunes, budget, dataset=dataset, situation=situation
    )
    result = await _engineer(request, model_spec=model)
    await flyte.report.replace.aio(decision_html(result))
    await flyte.report.flush.aio()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(ml_engineer_agent)
    print(run.url)
