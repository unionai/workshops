"""Notebook helpers for running the steps from Python instead of the shell.

    from utils.workshop import run, show
    from step2_engineer import engineer
    result = run(engineer)                 # on the cluster in your flyte config
    result = run(engineer, local=True)     # in this process, no cluster

`run` prints the run URL as soon as the run exists, waits for it (reconnecting if the
watch stream drops), prints the phase, and returns the task's outputs. Open the URL for the
run graph, the reports, and the Grafana links.
"""

from __future__ import annotations

import json
import time


def run(task, *, local: bool = False, **kwargs):
    import flyte

    if local:
        flyte.init()
        return _first(flyte.with_runcontext(mode="local").run(task, **kwargs).outputs())

    from flyte.remote import ActionDetails, Run

    flyte.init_from_config()
    r = flyte.run(task, **kwargs)
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
