"""Step 0 — warm the dataset cache. Optional, but worth it before a workshop.

    uv run flyte run step0_download_data.py download

`fetch_trips` is cached on the month (not on the SQL), so months are pulled from the
TLC once, parked in blob storage, and served from there to every later query — any
SQL, any question, any user in the project. That happens on demand, so the tutorial
works without this step.

What this changes is *who pays for the cold start*. Left to itself, the first person
to ask a twelve-month question waits for the download, and forty people asking at
once means forty cold fan-outs reaching for the same rate-limited CDN. Walking the
months serially up front makes everybody's first question instant.
"""

from __future__ import annotations

from analysis import fetch_trips, fetch_zones
from config import env
from dataset import MONTHS


@env.task
def download(months: list[str] = MONTHS) -> str:
    """Pull every month into blob storage, one at a time."""
    fetch_zones()
    for month in months:
        fetch_trips(month)
        print(f"cached {month}")
    return f"cached {len(months)} months + zones"


if __name__ == "__main__":
    import flyte

    flyte.init_from_config(image_builder="remote")
    run = flyte.run(download)
    print(f"View at: {run.url}")
    run.wait()
    print(run.outputs())
