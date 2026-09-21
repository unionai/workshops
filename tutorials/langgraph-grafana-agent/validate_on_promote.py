"""Every promotion gets validated, automatically, with no agent code.

A Union artifact trigger runs `validate_router` whenever a new version of `ticket-router`
lands, whoever published it: the agent in step 1, a person on the CLI, another team's
pipeline. The task receives the new version as a plain `Dir`, evaluates it on the full
held-out set on a T4, and writes a report. In a room of thirty people promoting, this
fires thirty times, and the artifact's Triggers tab lists every run.

Deploy the trigger once:

    flyte deploy validate_on_promote.py validate_env

Then promote anything (step 1, or by hand) and watch it fire. To run it on demand:

    flyte run validate_on_promote.py validate_router --model artifact:ticket-router
"""

from __future__ import annotations

import json

import flyte
import flyte.report
from flyte.io import Dir

from config import ml_image, tagged
from factory import evaluate
from report import evals_html
from tickets import DATASET_VERSIONS

validate_env = flyte.TaskEnvironment(
    name=tagged("factory-validate"),
    image=ml_image,
    resources=flyte.Resources(cpu=3, memory="12Gi", gpu="T4:1", disk="30Gi"),
)

on_new_router = flyte.Trigger(
    name="validate-on-promote",
    automation=flyte.OnArtifact(name=tagged("ticket-router")),
    inputs={"model": flyte.TriggeredArtifact},
    description="Evaluate every new ticket-router version on the full held-out set",
)


@validate_env.task(report=True, triggers=(on_new_router,))
async def validate_router(model: Dir, n: int = 200) -> str:
    """Evaluate a promoted model on every dataset version it could be serving."""
    local = await model.download()
    meta_path = f"{local}/factory.json"
    try:
        with open(meta_path) as f:
            trained_on = json.load(f).get("dataset", "v1")
    except FileNotFoundError:
        trained_on = "v1"

    results = []
    for version in DATASET_VERSIONS:
        r = evaluate(local, n=n, version=version)
        r.model = f"ticket-router (trained on {trained_on}) / dataset {version}"
        results.append(r.__dict__)

    await flyte.report.replace.aio(evals_html(results))
    await flyte.report.flush.aio()
    lines = [f"{r['model']}: accuracy {r['accuracy']:.1%}, p50 {r['latency_p50_ms']:.0f} ms" for r in results]
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(validate_env)
    print("deployed; the trigger fires on every new ticket-router version")
