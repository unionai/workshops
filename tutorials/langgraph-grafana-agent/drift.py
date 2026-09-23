"""Step 5: day two. The support team adds a category, and the factory has to turn again.

Dataset v2 adds `data_request` (GDPR-style tickets that used to be misfiled under
`other`). The model in production has never seen the label: it cannot say it, so its
accuracy on the new tickets drops below the bar. The engineer is told what changed and
what is in production, and has to work out the rest: check the live model against v2,
fine-tune on v2, promote, test the deployment.

Every turn of the factory leaves a new version of the `ticket-router` artifact with its
numbers on the card. Open the artifact in the Union UI: the Versions tab is the history
of the router, and the Lineage tab leads from the app back through each promotion to the
fine-tune that produced it.

    flyte run drift.py day_two
    flyte run drift.py day_two --min_accuracy 0.9
"""

from __future__ import annotations

import json

import flyte
import flyte.report

from config import GRAFANA_LINKS, observed_env
from graph import Request
from graph import engineer as _engineer
from report import decision_html

SITUATION = """\
Day two. The support team has added a ninth ticket category, `data_request`, for privacy
and data-subject requests; those tickets used to be filed under `other`. Our tickets are
now dataset v2. A ticket router is already in production (see production_status), and it
was trained on v1. Find out whether it still meets the bar on v2, and if it does not, fix
it: fine-tune on v2 and promote the result. Retrain the same base model that is in
production first (production_status names it; it serves on a CPU pod, so keep it small);
only switch to another candidate if that fails the bar. Do not retrain more than you have to."""


@observed_env.task(report=True, retries=2, links=GRAFANA_LINKS)
async def day_two(
    min_accuracy: float = 0.95,
    max_latency_ms: float = 150.0,
    candidates_to_screen: int = 1,
    max_fine_tunes: int = 1,
    budget: int = 5,
    model: str | None = None,
) -> dict:
    """The factory's second turn: the world changed, the router in production has to follow."""
    request = Request(
        min_accuracy,
        max_latency_ms,
        candidates_to_screen,
        max_fine_tunes,
        budget,
        dataset="v2",
        situation=SITUATION,
        fresh=False,
    )
    result = await _engineer(request, model_spec=model)
    await flyte.report.replace.aio(decision_html(result))
    await flyte.report.flush.aio()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(day_two)
    print(run.url)
