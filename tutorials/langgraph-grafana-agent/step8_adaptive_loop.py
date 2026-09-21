"""Step 8: the loop closes itself. New tickets land, the factory turns, nobody clicks.

Step 5 had you run the factory when the data changed. Here the platform does it:

    publish_tickets ─► artifact support-tickets @vN
                              │  trigger: adapt-on-new-tickets
                              ▼
    adapt(tickets)  ─► the engineer agent (steps 1 and 5's graph), told what landed
                    ─► measures the model in production on the new tickets
                    ─► meets the bar? ─► keeps it, promotes nothing, says so
                    ─► below the bar? ─► fine_tune ─► promote ─► artifact ticket-router @vM
                                                                │  trigger: validate-on-promote
                                                                ▼
                                                        validate_router ─► report

Two artifact triggers chained through the agent. The only human act is publishing the
tickets. Deploy the trigger once, then publish a dataset version and watch:

    flyte deploy step8_adaptive_loop.py observed_env          # registers the trigger
    flyte run step8_adaptive_loop.py publish_tickets --version v2

The `support-tickets` artifact is the event and the record: it carries the tickets as
JSONL plus a manifest naming the dataset version, so the trigger run knows what it is
looking at and the artifact's Versions tab is the history of the data.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import flyte
import flyte.artifacts as artifacts
import flyte.report
from flyte.io import Dir

from config import GRAFANA_LINKS, observed_env, tagged, tools_env
from graph import Request
from graph import engineer as _engineer
from report import decision_html
from tickets import DATASET_VERSIONS, load_split

TICKETS_ARTIFACT = tagged("support-tickets")
BAR = 0.95


@tools_env.task(produces_artifacts=True)
async def publish_tickets(version: str = "v2") -> Dir:
    """Publish a dataset version as the `support-tickets` artifact. This is the drift event."""
    if version not in DATASET_VERSIONS:
        raise ValueError(f"unknown dataset {version!r}; known: {', '.join(DATASET_VERSIONS)}")
    out = tempfile.mkdtemp(prefix="tickets-")
    for split in ("train", "test"):
        with open(os.path.join(out, f"{split}.jsonl"), "w") as f:
            for t in load_split(split, version):
                f.write(json.dumps({"id": t.id, "text": t.text, "label": t.label}) + "\n")
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({"dataset": version, "categories": DATASET_VERSIONS[version]}, f)
    d = await Dir.from_local(out)
    meta = artifacts.Metadata(
        name=TICKETS_ARTIFACT,
        # Unique per publish: republishing an existing version is a no-op and fires nothing.
        version=f"{version}-{len(DATASET_VERSIONS[version])}cats-{int(time.time())}",
        description=f"Support tickets, dataset {version}: {len(DATASET_VERSIONS[version])} categories",
        kind="data",
        attrs={"dataset": version, "categories": str(len(DATASET_VERSIONS[version]))},
    )
    return artifacts.new(d, meta)


on_new_tickets = flyte.Trigger(
    name="adapt-on-new-tickets",
    automation=flyte.OnArtifact(name=TICKETS_ARTIFACT),
    inputs={"tickets": flyte.TriggeredArtifact},
    description="When a new version of the support tickets lands, check the router in production and retrain if it fell below the bar",
)


@observed_env.task(report=True, retries=2, triggers=(on_new_tickets,), links=GRAFANA_LINKS)
async def adapt(
    tickets: Dir, min_accuracy: float = BAR, max_latency_ms: float = 150.0, budget: int = 6, model: str | None = None
) -> dict:
    """The engineer, started by the platform: new tickets landed, check production, fix it if it fell below the bar."""
    local = await tickets.download()
    with open(os.path.join(local, "manifest.json")) as f:
        manifest = json.load(f)
    version = manifest["dataset"]

    # Same graph, same tools, same report as steps 1 and 5. The only thing the trigger
    # adds is the situation: what landed, and that production has to be checked first.
    situation = (
        f"New support tickets have landed: dataset {version}, {len(manifest['categories'])} categories "
        f"({', '.join(manifest['categories'])}). A ticket router is in production (see production_status; "
        "measure the exact version it names, not 'latest'). Check whether it still meets the bar on the new "
        "tickets before you train anything. If it does, keep it and promote nothing. If it does not, fix it "
        "with the least retraining that will do: retrain the same base model that is in production on the new "
        "tickets first (it serves on a CPU pod, so keep it small), and only switch candidates if that fails the bar. "
        "Promote the result and test the deployment."
    )
    request = Request(
        min_accuracy,
        max_latency_ms,
        candidates_to_screen=1,
        max_fine_tunes=1,
        budget=budget,
        dataset=version,
        situation=situation,
        fresh=False,
    )
    result = await _engineer(request, model_spec=model)
    result["tickets_version"] = version
    await flyte.report.replace.aio(decision_html(result))
    await flyte.report.flush.aio()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(observed_env)
    run = flyte.run(publish_tickets, version="v2")
    print("published tickets:", run.url)
    print("the adapt-on-new-tickets trigger picks it up from here")
