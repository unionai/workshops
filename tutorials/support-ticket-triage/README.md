# Support Ticket Triage — Intro to Flyte

A pure-Python workflow that triages customer support tickets in parallel using Flyte's `map` for fan-out.

**What it shows:**
- Workflows as plain Python functions (`@env.task`, tasks calling tasks)
- Parallel fan-out with `flyte.map` — every ticket scored simultaneously
- Same code runs locally or on a cluster, no config changes

## Setup

```bash
cd tutorials/support-ticket-triage
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run locally

```bash
flyte run --local --tui workflow.py triage_pipeline
```

## Run on Union

```bash
flyte run workflow.py triage_pipeline
```

## How it works

1. **`triage_pipeline`** — entry point, holds a batch of sample tickets
2. **`score_ticket`** — map task, scores each ticket's urgency & sentiment using keyword matching (no ML, no API keys)
3. **`prioritize`** — sorts by combined priority score, prints a ranked report

The fan-out happens in one line: `flyte.map(score_ticket, tickets)` — Flyte runs all 10 scoring tasks in parallel on the cluster.
