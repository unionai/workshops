# Autoresearch-style self-healing agent (PyTorch LM)

This folder implements an agent in the spirit of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): **short
experiments**, a **single scalar metric** (**val_bpb** — validation bits per
byte, **lower is better**), and **iteration** after failure — on Flyte with
explicit recovery paths.

# Setup

Go to the autoresearch project directory:

```bash
cd tutorials/autoresearch
```

Install dependencies and activate the virtual environment:

```bash
pip install uv
uv sync
source .venv/bin/activate
```

```bash
export FLYTE_PROJECT=<project>
```

```bash
# Configure Union CLI
flyte create config \
    --endpoint tryv2.hosted.unionai.cloud \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project $FLYTE_PROJECT
```

## Set up the Anthropic API key secret (optional):

```bash
flyte create secret --project $FLYTE_PROJECT --domain development my-anthropic-api-key
```

Then edit the `autoresearch_agent.py` file to use the new secret name:
```python
secrets=[flyte.Secret(key="my-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
```

## Dataset (same source as upstream `prepare.py`)

Text data comes from the **climbmix** parquet shards referenced in upstream
[`prepare.py`](https://github.com/karpathy/autoresearch/blob/master/prepare.py)
(HuggingFace `karpathy/climbmix-400b-shuffle`). `build_autoresearch_bundle`
downloads a configurable number of training shards plus the **pinned validation
shard** (`shard_06542`), trains a **BPE tokenizer** (rustbpe + tiktoken), and
uploads **data** and **tokenizer** as separate directories (Flyte `Dir` values).

For each training run, `run_training_subjob` turns those trees into **two
`tar.gz` inputs** (`data_tgz`, `tokenizer_tgz`) and passes the workspace
`prepare.py` into the sandbox as **`prepare.py`** so training code can import
dataloaders and `evaluate_bpb` from that module (see baseline `train.py`).

## Files

| File | Role |
|------|------|
| `program.md` | Human-editable mission/constraints (upstream-style); **not** read by the Flyte workflow — documentation for operators. |
| `prepare.py` | Download shards, train BPE, dataloaders, **evaluate_bpb** (aligned with upstream; `AUTORESEARCH_CACHE` / `AUTORESEARCH_EVAL_TOKENS`). |
| `train.py` | Single editable **PyTorch** training script the LLM rewrites each round; baseline extracts tarballs, sets cache, writes `/var/outputs/metrics_json_str`. |
| `autoresearch_types.py` | Dataclasses: `DatasetProfile`, metrics, history, workflow output. |
| `report.py` | HTML for Flyte run reports (history tab, val_bpb chart, summary). |
| `autoresearch_agent.py` | Flyte workflow: bundle → profile → optional arXiv → write code → **provision** (CPU/RAM/**T4 GPU**) → sandbox sub-job → heal loop; optional **HITL** outer loop. |


## Run the agent

Requires `flyte.init_from_config()` and an Anthropic API key. The agent
environment maps Flyte secret **`internal-anthropic-api-key`** to
`ANTHROPIC_API_KEY`.

The first task run downloads parquet shards from HuggingFace and trains the
tokenizer — **large network use** and several minutes of CPU/RAM.

```bash
cd /path/to/tutorials/autoresearch
uv run python autoresearch_agent.py
```

The default `__main__` launches **`infinite_research_loop`**, which runs
`autoresearch_agent` end-to-end and then uses **flyteplugins-hitl** to prompt
whether to continue (indefinite rounds of full agent runs until you decline).

Tune `num_prepare_shards`, `prepare_download_workers`, `max_experiment_rounds`,
and `research_topic` in `autoresearch_agent.py` (or via Flyte overrides).
With **`research_topic=None`**, the **literature / arXiv** step is skipped (no
extra network). With a non-empty topic, `search_arxiv_with_retry` queries the
arXiv Atom API; transient HTTP failures use the task’s **Flyte retries**
(`retries=3`).

### Environment tuning

| Variable | Effect |
|----------|--------|
| `AUTORESEARCH_CACHE` | Root for `data/` and `tokenizer/` (set in sandboxes after extract). |
| `AUTORESEARCH_EVAL_TOKENS` | Validation token budget in `evaluate_bpb` (default in `prepare.py` matches upstream scale; baseline `train.py` sets a smaller default for demos). |
| `AUTORESEARCH_TIME_BUDGET` | Wall-clock training seconds in baseline `train.py` (default 120). |

### Dependencies note

`prepare.py` / sandboxes require **`rustbpe`** (Rust extension). If image builds
fail, install a Rust toolchain or use a platform with prebuilt wheels; see
upstream [autoresearch](https://github.com/karpathy/autoresearch) discussion.

## Self-healing hooks

1. **Provisioning** — `provision_resources` proposes **CPU, memory, and GPU**
   (e.g. `T4:1`) from bundle profile + current `train.py`; **OOM**
   (`flyte.errors.OOMError`) triggers a new proposal with the error and prior
   JSON in context.
2. **Training code** — `run_training_subjob` runs the rewritten script in
   `flyte.sandbox` with **`auto_io=False`** (verbatim code; tar inputs +
   `prepare.py`); failures feed **LLM code repair** on the next round with the
   traceback (non-OOM exceptions still advance rounds when below
   `max_experiment_rounds`).
3. **Literature** — Optional arXiv snippets when `research_topic` is set; empty
   or whitespace query skips the request entirely.

`autoresearch_agent.py` combines **optional retrieval**, **provisioning**
(including GPU), and a **training sandbox sub-job** aimed at iterative
“research” runs on **text tokens** and **val_bpb**.
