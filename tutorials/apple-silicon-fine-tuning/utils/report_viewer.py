"""Show Flyte HTML reports inside a notebook."""

from __future__ import annotations

import pathlib

LOCAL_METADATA = "/tmp/flyte/metadata"


def show_latest_local():
    """The most recent `flyte run --local` report (tasks marked report=True write one)."""
    from IPython.display import HTML, Markdown

    reports = sorted(pathlib.Path(LOCAL_METADATA).rglob("report.html"), key=lambda p: p.stat().st_mtime)
    if not reports:
        return Markdown(f"No report under `{LOCAL_METADATA}` yet. Run a step with `--local` first.")
    return HTML(reports[-1].read_text())


def run_link(run_name: str, endpoint: str = "http://localhost:30080", project: str = "flytesnacks", domain: str = "development"):
    """A link to a devbox run's page, where the report tab lives."""
    from IPython.display import Markdown

    return Markdown(f"[Open run `{run_name}` in the Flyte UI]({endpoint}/v2/domain/{domain}/project/{project}/runs/{run_name})")
