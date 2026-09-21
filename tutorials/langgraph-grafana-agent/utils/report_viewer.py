"""Show a local run's HTML report inside a notebook.

`flyte run --local ...` in a notebook cell runs in a subshell, so you get no run
object back — just text. These helpers find the report it wrote and render it.

    !flyte run --local step1_retrieve.py search --question "..."

    from utils.report_viewer import show_latest
    show_latest()
"""

from __future__ import annotations

import pathlib

LOCAL_METADATA = "/tmp/flyte/metadata"


def show_latest(n: int = 1):
    """Render the most recently written local report.

    Steps that call step 0 as a subtask write several reports in one run (one
    per task). The last one written is the step you actually invoked, which is
    why this defaults to n=1. Pass n=2 to see the one before it.
    """
    from IPython.display import HTML, Markdown

    reports = sorted(
        pathlib.Path(LOCAL_METADATA).rglob("report.html"),
        key=lambda p: p.stat().st_mtime,
    )
    if not reports:
        return Markdown(
            f"No report found under `{LOCAL_METADATA}`. Run a step with `--local` "
            "first — and note that only tasks marked `@env.task(report=True)` "
            "write one."
        )
    if n > len(reports):
        n = len(reports)
    return HTML(reports[-n].read_text())


def show(run):
    """Render a specific run's report, when you drove Flyte from Python.

        import flyte
        flyte.init()
        run = flyte.run(search, question="...")
        run.wait()
        show(run)

    A local run's `url` is its metadata directory, so the report is on disk. A
    remote run's `url` is a console link, so you get the link instead.
    """
    from IPython.display import HTML, Markdown

    url = str(getattr(run, "url", ""))
    if url.startswith("http"):
        return Markdown(f"Remote run — open the report in the Flyte UI: [{url}]({url})")

    reports = sorted(
        pathlib.Path(url or LOCAL_METADATA).rglob("report.html"),
        key=lambda p: p.stat().st_mtime,
    )
    if not reports:
        return Markdown(f"No report found for run `{getattr(run, 'name', '?')}`.")
    return HTML(reports[-1].read_text())
