"""Step 7 (optional): the bake-off. Which model should drive the factory?

The agent's own brain is a model too, and it is graded like any other: did it promote
something that meets the request, and how many runs did it spend getting there? This
task fans the same request out over several agent models, in parallel. Each run is its
own durable child action, and each agent model is its own *agent version* in Grafana,
because the plugin binds the task version and the agent name onto every generation. The
Flyte report scores constraints, runs spent, tokens and wall time; Grafana adds per-turn
cost and latency and lets you read every transcript side by side.

The evals and fine-tunes the agents ask for are cached, so after the first agent has run,
the others mostly hit cache, and the bake-off measures the agents, not the GPUs.
Promotion publishes artifacts but does not deploy (FACTORY_DEPLOY=0 on this environment).

    flyte run --local bakeoff.py bakeoff \
        --models '["anthropic:claude-opus-5", "anthropic:claude-haiku-4-5"]'

    flyte run bakeoff.py bakeoff \
        --models '["anthropic:claude-opus-5", "anthropic:claude-haiku-4-5", "openai:gpt-4.1"]' --trials 2
"""

from __future__ import annotations

import asyncio
import json

import flyte
import flyte.report

from config import GRAFANA_LINKS, factory_env
from graph import Request
from graph import engineer as _engineer
from llm import short_name
from report import bakeoff_html, score

DEFAULT_MODELS = ["anthropic:claude-haiku-4-5", "anthropic:claude-opus-5"]


@factory_env.task(retries=2, links=GRAFANA_LINKS)
async def engineer_with(model: str, trial: int, request: dict) -> dict:
    """One cell of the bake-off: one agent model, one trial. A child action with its own report."""
    result = await _engineer(Request(**request), model_spec=model)
    result["model"] = model
    result["score"] = score(result)
    return result


@factory_env.task(report=True, links=GRAFANA_LINKS)
async def bakeoff(
    models: list[str] = DEFAULT_MODELS,
    trials: int = 1,
    min_accuracy: float = 0.95,
    max_latency_ms: float = 150.0,
    candidates_to_screen: int = 5,
    max_fine_tunes: int = 3,
    budget: int = 12,
) -> dict:
    """Run every agent model on the same request, in parallel, and score them."""
    request = Request(min_accuracy, max_latency_ms, candidates_to_screen, max_fine_tunes, budget).__dict__
    rows: list[dict] = []
    for model in models:
        # One group per agent model in the Flyte UI; the trials inside run concurrently.
        with flyte.group(f"agent-{short_name(model)}"):
            results = await asyncio.gather(
                *(engineer_with(model, t, request) for t in range(trials)), return_exceptions=True
            )
        for t, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"{model} trial {t} failed: {r}")
                continue
            rows.append(r)

    await flyte.report.replace.aio(bakeoff_html(rows))
    await flyte.report.flush.aio()

    summary = {}
    for model in models:
        rs = [r for r in rows if r["model"] == model]
        if not rs:
            summary[model] = {"runs": 0}
            continue
        summary[model] = {
            "runs": len(rs),
            "met_constraints": sum(r["score"]["accuracy_ok"] and r["score"]["latency_ok"] for r in rs),
            "within_budget": sum(r["score"]["budget_ok"] for r in rs),
            "avg_runs_spent": round(sum(r["decision"]["runs_spent"] for r in rs) / len(rs), 1),
            "avg_tokens": round(sum(r["stats"]["input_tokens"] + r["stats"]["output_tokens"] for r in rs) / len(rs)),
            "avg_seconds": round(sum(r["stats"]["seconds"] for r in rs) / len(rs), 1),
        }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(bakeoff)
    print(run.url)
