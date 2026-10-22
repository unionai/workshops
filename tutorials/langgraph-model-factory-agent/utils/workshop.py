"""Notebook helpers for running the steps from Python instead of the shell.

    from utils.workshop import run, show, show_reports
    from ml_engineer import ml_engineer_agent
    result = run(ml_engineer_agent)                 # on the cluster in your flyte config
    result = run(ml_engineer_agent, local=True)     # in this process, no cluster; its report renders below the cell
    show_reports(all=True)                          # and every tool's report from that local run

`run` prints the run URL as soon as the run exists, waits for it (reconnecting if the
watch stream drops), prints the phase, and returns the task's outputs. Open the URL for the
run graph and the reports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# The tutorial folder: where this file's parent folder lives, by default.
ROOT = Path(__file__).resolve().parent.parent


def _root_for(task) -> Path:
    """The folder holding the task's source file. The code bundle is rooted there.

    The pod imports `config`, `llm`, `tools` and friends by their bare names, so the bundle
    root has to be the folder they sit in. Left to the SDK, the root is guessed from the
    environment, and in Colab that guess is wrong: the modules land in the bundle under a
    nested path and the first import fails with "No module named 'config'". Rooting on the
    task's own file also survives a notebook that keeps `utils/` somewhere else.
    """
    import inspect

    func = getattr(task, "func", None)
    try:
        return Path(inspect.getfile(func)).resolve().parent if func else ROOT
    except (TypeError, OSError):
        return ROOT


def run(task, *, local: bool = False, **kwargs):
    import flyte

    root = _root_for(task)
    if local:
        # No cluster: the task and every tool it calls run in this process. Nothing to open;
        # the output of each step is the result below.
        global LAST_LOCAL_RUN
        _reports_as_text()
        flyte.init(root_dir=root)
        print(f"local run: {getattr(task, 'name', task)} in this kernel (no cluster; it takes as long as the work does)")
        r = flyte.with_runcontext(mode="local").run(task, **kwargs)
        outputs = r.outputs()
        # A local run's `url` is its metadata folder, where every task's report.html landed.
        # Show the step's own report here, the way the UI would; show_reports(all=True) has the rest.
        LAST_LOCAL_RUN = Path(str(r.url)) if str(r.url).startswith("/") else None
        if LAST_LOCAL_RUN is not None and _in_ipython():
            show_reports(LAST_LOCAL_RUN)
        return _first(outputs)

    from flyte.remote import ActionDetails, Run

    flyte.init_from_config(_config_path(root), root_dir=root)
    # In a notebook the SDK would pickle the task instead of bundling the source files. The
    # pickle carries the task's module by value but only *references* the modules it imports,
    # so the pod dies on the first one: "No module named 'llm'". Ship the source instead.
    r = flyte.with_runcontext(interactive_mode=False).run(task, **kwargs)
    print(f"run {r.name}: {r.url}")
    for _ in range(60):
        try:
            r.wait()
            break
        except Exception as e:  # noqa: BLE001  (a dropped watch stream, not a failed run)
            print("reconnecting:", str(e)[:80])
            time.sleep(10)
            r = Run.get(r.name)
    r = Run.get(r.name)
    phase = str(r.action.phase).rsplit(".", 1)[-1]
    print(f"phase: {phase}")
    if phase != "SUCCEEDED":
        print(str(ActionDetails.get(run_name=r.name, name="a0").pb2.error_info)[:1500])
        return None
    return _first(r.outputs())


LAST_LOCAL_RUN: Path | None = None  # the metadata folder of the last local run (its reports)


def show_reports(folder: Path | str | None = None, *, all: bool = False, latest: bool = False, height: int = 640):
    """Render the reports of the last local run (or `folder`) in the notebook; see utils/report_viewer.py."""
    from utils.report_viewer import show_reports as _show

    return _show(LAST_LOCAL_RUN if folder is None else folder, all=all, latest=latest, height=height)


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _reports_as_text() -> None:
    """Keep task reports working inside a Jupyter kernel.

    flyte 2.12 renders a report as an IPython `HTML` object whenever an IPython shell is
    active, and `flyte.report.flush()` then fails its own `isinstance(str)` check, so every
    task that writes a report dies in local mode in a notebook (not from the CLI). Hand
    `flush` the HTML text instead. Harmless once the SDK fixes it.
    """
    from flyte.report import _report

    if getattr(_report.Report, "_workshop_patched", False):
        return
    original = _report.Report.get_final_report

    def as_text(self):
        out = original(self)
        return getattr(out, "data", out)  # IPython's HTML keeps the markup in .data

    _report.Report.get_final_report = as_text
    _report.Report._workshop_patched = True


def _config_path(root: Path = ROOT):
    """The cluster config: next to the code, else in the current directory.

    `flyte create config` writes `config.yaml` where it is run; the notebook runs it in the
    tutorial folder, and a notebook run from elsewhere writes it there instead. With no file
    in either place the SDK's own search (FLYTECTL_CONFIG, ~/.flyte, ...) applies.
    """
    cwd = Path.cwd()
    for candidate in (root / "config.yaml", root / ".flyte" / "config.yaml", cwd / "config.yaml"):
        if candidate.exists():
            return candidate
    return None


def _first(outputs):
    """A single-output task comes back as a one-element ActionOutputs; unwrap it."""
    try:
        return outputs[0] if len(outputs) == 1 else outputs
    except TypeError:
        return outputs


def show(result, keys: list[str] | None = None) -> None:
    """Print the interesting part of a result dict."""
    if result is None:
        print("no result")
        return
    if keys:
        result = {k: result.get(k) for k in keys}
    print(json.dumps(result, indent=2, default=str))
