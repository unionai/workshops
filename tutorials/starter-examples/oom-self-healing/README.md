# OOM Self-Healing

Catch an out-of-memory (OOM) error and automatically retry the task with more memory — no manual intervention, no lost run.

## What it does

- **`oomer`** — Allocates a large list that exceeds the task's memory limit, triggering an OOM kill
- **`always_succeeds`** — A lightweight task showing the pipeline keeps making progress
- **`failure_recovery`** — Catches `flyte.errors.OOMError` and self-heals by re-running the task with a bigger memory allocation via `.override(...)`

Flyte surfaces OOM kills as a typed `flyte.errors.OOMError`, so you can catch it in code and react — bump resources, fall back, or give up gracefully — instead of failing the whole run.

## Setup

```bash
cd tutorials/starter-examples/oom-self-healing

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

## Flyte Cluster (for remote runs)

To run remotely, configure your Flyte cluster endpoint:

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

Don't have a cluster? Request access at [flyte.org](https://flyte.org/).

## Run

**Remote:**
```bash
uv run flyte run oom_self_healing.py failure_recovery
```

**Local:**
```bash
uv run flyte run --local oom_self_healing.py failure_recovery
```

## Notes

- The first `oomer` attempt runs under the environment default (`250Mi`) and gets OOM-killed; the retry runs with `1Gi` and succeeds.
- `.override(resources=...)` re-runs the same task with different resources — the same pattern works for CPU, GPU, and other overrides.
- The `finally` block runs regardless of the OOM outcome, so downstream work still moves forward.
