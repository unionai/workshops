"""Show a local run's HTML reports inside a notebook.

On a cluster every task's report is a tab in the Flyte UI. A local run (`run(task)` with
`LOCAL = True`, or `flyte run --local`) writes the same files to its metadata folder
instead, one `report.html` per task that has `report=True`. These helpers find and render
them:

    from utils.workshop import run, show_reports
    run(model_factory, fine_tune_too=True)   # local mode: the step's own report renders under the cell
    show_reports(all=True)                   # every report of that run: each eval, each fine-tune
    show_reports(latest=True)                # after `!flyte run --local ...`: the newest report on disk
"""

from __future__ import annotations

import html
import pathlib
import re

LOCAL_METADATA = pathlib.Path("/tmp/flyte/metadata")  # where `flyte run --local` keeps a run's reports


def reports_in(folder: str | pathlib.Path) -> list[pathlib.Path]:
    """Every report under a local run's metadata folder, oldest first (the step's own report is last)."""
    return sorted(pathlib.Path(folder).rglob("report.html"), key=lambda p: p.stat().st_mtime)


def title_of(path: pathlib.Path) -> str:
    """What the report is about: its first heading, else its tab names."""
    text = path.read_text(errors="ignore")
    m = re.search(r"<h[12][^>]*>(.*?)</h[12]>", text, re.S)
    if m:
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
        if ":" not in title:
            # "Eval results" says nothing about which model; a later heading does
            # ("Confusion matrix: qwen2.5-1.5b").
            sub = re.search(r"<h[3-4][^>]*>[^<:]*:\s*([^<]+)</h[3-4]>", text)
            if sub:
                name = html.unescape(sub.group(1)).strip()
                if "/" in name:  # a fine-tune's weights directory: keep only its name
                    name = name.rstrip("/").rsplit("/", 1)[-1]
                title += " · " + name
        return title
    tabs = re.findall(r'handleLinkClick\(this\)">([^<]*)<', text)
    return ", ".join(t for t in tabs if t != "main") or path.parent.name


def render(path: pathlib.Path, height: int = 640):
    """The report in an iframe, so its own styles and tab script stay inside it."""
    from IPython.display import HTML

    doc = html.escape(path.read_text(errors="ignore"), quote=True)
    return HTML(
        f'<details open style="margin:6px 0"><summary style="cursor:pointer;font-family:sans-serif">'
        f"{html.escape(title_of(path))}</summary>"
        f'<iframe srcdoc="{doc}" style="width:100%;height:{height}px;border:1px solid #ddd;border-radius:4px"></iframe>'
        "</details>"
    )


def show_reports(folder: str | pathlib.Path | None = None, *, all: bool = False, latest: bool = False, height: int = 640):
    """Render a local run's reports in the notebook.

    `folder` is the run's metadata folder (`run()` remembers the last one it ran). With
    `all=True` every report of the run is shown, oldest first; otherwise only the newest,
    which is the step's own. `latest=True` ignores `folder` and shows the newest report on
    disk, for a run started with `!flyte run --local`.
    """
    from IPython.display import Markdown, display

    where = LOCAL_METADATA if latest else folder
    if where is None:
        return Markdown("No local run yet. Run a step with `LOCAL = True` first, or pass `latest=True`.")
    found = reports_in(where)
    if not found:
        return Markdown(
            f"No report found under `{where}`. Only tasks declared with `report=True` write one, "
            "and only after they finish."
        )
    for p in found if all else found[-1:]:
        display(render(p, height=height))
