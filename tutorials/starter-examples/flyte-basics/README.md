# Flyte Basics

Get started with Flyte 2 — tasks, ML pipelines, error handling, and AI agents.

## What it covers

- `flyte.TaskEnvironment` — defining task environments and images
- `flyte.ReusePolicy` — reusable containers for faster iteration
- `flyte.map()` — parallel task execution
- Building an ML pipeline end-to-end
- Error handling patterns

## Setup

```bash
cd tutorials/starter-examples/flyte-basics

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

Open `00_flyte2-starter.ipynb` in Jupyter or VS Code and follow along.