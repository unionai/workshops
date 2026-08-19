"""Open a local run's HTML report in your browser.

`flyte run --local` prints a one-line summary to the terminal, but the actual
output of steps 1-4 — the retrieved chunks, the answer with its citations, the
UMAP chart, the memories — is in an HTML report on disk. In the notebook
`show_latest()` renders it inline; from a terminal, this opens it.

    python open_report.py        # the most recent report
    python open_report.py 2      # the one before that
    python open_report.py --path # just print the path, don't open

A step that calls step 0 as a subtask writes more than one report per run (one
per task with `report=True`). The most recent is the step you actually invoked,
which is why this defaults to the last one.
"""

from __future__ import annotations

import pathlib
import sys
import webbrowser

LOCAL_METADATA = "/tmp/flyte/metadata"


def find(n: int = 1) -> pathlib.Path | None:
    reports = sorted(
        pathlib.Path(LOCAL_METADATA).rglob("report.html"),
        key=lambda p: p.stat().st_mtime,
    )
    if not reports:
        return None
    return reports[-min(n, len(reports))]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--path"]
    path_only = "--path" in sys.argv
    which = int(args[0]) if args else 1

    report = find(which)
    if report is None:
        print(
            f"No report found under {LOCAL_METADATA}.\n"
            "Run a step with --local first. Only tasks marked @env.task(report=True) "
            "write one — that's steps 0 through 4."
        )
        raise SystemExit(1)

    print(report)
    if not path_only:
        webbrowser.open(report.as_uri())
