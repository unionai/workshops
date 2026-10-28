# Fine-tune on your Mac's GPU: LoRA on `mps` with Flyte, then serve it

Flyte's devbox can now run tasks on an Apple Silicon GPU. Start it with `--apple-silicon`, ask for `gpu="AppleSilicon"` in a task's resources, and that task runs natively on your Mac, with PyTorch on the `mps` device, while Flyte still gives you the run graph, inputs and outputs, caching, retries, logs and reports.

This tutorial fine-tunes [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) for text-to-SQL with LoRA, scores it before and after, and serves the adapter as an API. All of it on the laptop. Then the same code runs on a cluster GPU by changing one environment variable.

```bash
flyte start devbox --apple-silicon
flyte create config --devbox
flyte run workflow.py gpu_check      # is this task really on the Mac's GPU?
flyte run workflow.py pipeline       # data -> baseline -> LoRA -> evaluate
python serve.py                      # serve the adapter, on the same GPU
```

## Results

Measured on an M5 Max (128 GB), flyte 2.12.0, torch 2.14.1, defaults everywhere: 500 training examples, 1 epoch, rank 16, 100 held-out questions, exact match on the normalized SQL.

| | |
|---|---|
| Where the task ran | `Darwin arm64`, device `mps`, bf16 |
| 2048x2048 fp32 matmul | 12.0 TFLOPS on `mps` vs 1.8 on the CPU |
| Base model exact match | 5% |
| Fine-tuned exact match | **62%** |
| Training | 62 optimizer steps in **14 s** |
| Loss | 2.9 → 0.85 |
| GPU memory | 15 GB reported by the Metal driver (that figure includes PyTorch's cache; see below) |

The base model's 5% is not because it cannot write SQL. It explains, hedges and adds code fences; the fine-tuned model answers and stops. Exact match is a floor: read the side-by-side table in the **evaluate** report for the queries that are right but phrased differently.

Smaller Macs are slower, not different. A 500-example run is a few minutes on an M1. If you have 16 GB, pass `--batch_size 2 --grad_accum 4`.

## What is actually happening

A devbox container is a Linux VM. Linux has no Metal, so code inside the container can never see the GPU. A task that requests `gpu="AppleSilicon"` therefore runs **outside** the container:

1. The task's pod starts as usual, but its entrypoint hands the invocation to a bridge process on the Mac.
2. The bridge builds a macOS Python environment from the image's pip packages with `uv` (cached under `~/.flyte/apple-silicon/envs/`), rewrites in-cluster endpoints to devbox ports, and runs the Flyte entrypoint natively.
3. The container streams the process's logs, mirrors its exit code and forwards aborts. In the UI, nothing looks different.

Because the process is a real macOS process with your `HOME`, it shares your Hugging Face cache and can use `torch.device("mps")`. `platform.system()` says `Darwin`. `gpu_check` prints exactly that, plus a matmul benchmark, and its report shows the GPU beating the CPU by an order of magnitude.

There is no silent fallback. Request `AppleSilicon` on a devbox started without the flag, on a Linux box or on a cluster, and the task fails with `APPLE_SILICON_NOT_AVAILABLE` instead of quietly training on the CPU.

## Files

| File | What it does |
|---|---|
| `config.py` | The `ACCELERATOR` knob: pod shapes for `apple`, `T4`, `L40s`, `cpu`; the pip list; the two task environments |
| `sql.py` | Chat rendering, SQL normalization and exact-match scoring, plus `pick_device()` and `pick_dtype()` |
| `workflow.py` | `gpu_check`, `prepare_data` → `evaluate` → `train` → `evaluate`, and the `baseline` and `pipeline` entrypoints |
| `serve.py` | FastAPI app: base model plus the adapter a pipeline run produced, served on the same accelerator |
| `report_helpers.py` | Report CSS and SVG charts (shared with the sibling tutorials) |
| `apple-silicon-fine-tuning-tutorial.ipynb` | The same steps as a notebook |

## Setup

You need an Apple Silicon Mac (M1 or later), Docker Desktop, [`uv`](https://docs.astral.sh/uv/) and `flyte` 2.12 or later.

```bash
cd tutorials/apple-silicon-fine-tuning
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

flyte start devbox --apple-silicon    # recreate an older devbox first: flyte delete devbox
flyte create config --devbox          # writes .flyte/config.yaml pointing at localhost:30080
```

The ready panel ends with a line about the Apple Silicon bridge on `127.0.0.1:30090`. `flyte get devbox` shows the bridge and every workload that has run on the host.

The model and dataset are ungated; no Hugging Face token needed.

## Run

```bash
# 0. Prove it: platform, chip, torch device, GPU vs CPU matmul. No downloads.
flyte run workflow.py gpu_check

# 1. Score the untouched model on 100 held-out questions.
flyte run workflow.py baseline

# 2. The whole pipeline: data -> baseline -> LoRA -> evaluate.
flyte run workflow.py pipeline

# A smaller first run
flyte run workflow.py pipeline --train_samples 100 --num_eval_examples 20

# 3. Serve the adapter from the latest pipeline run, then ask it something.
python serve.py
curl -s -X POST "$URL/generate" -H "Content-Type: application/json" -d '{
  "schema": "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
  "question": "What is the average salary by department?"
}'
```

Every `flyte run` prints a URL into the devbox UI. Open the **train** task while it runs for the live loss chart, and the **evaluate** tasks for base vs fine-tuned side by side. The pipeline's own report has the before/after bar chart.

The first run is slow twice: the devbox builds the Linux image (torch has to be installed in it, because the pod still exists), and the bridge builds the Mac environment. Both are cached.

### Parameters

Flyte derives flags from parameter names, so they use underscores.

| Flag | Default | Notes |
|---|---|---|
| `--train_samples` | `500` | 500 examples at batch 4 x 2 accumulation is 62 optimizer steps |
| `--eval_samples` / `--num_eval_examples` | `100` / `100` | Held-out pool, and how many of them are scored |
| `--epochs` | `1` | |
| `--lr` | `2e-4` | One-cycle cosine schedule |
| `--batch_size` / `--grad_accum` | `4` / `2` | Effective batch 8. Use `2` / `4` on a 16 GB Mac |
| `--max_length` | `512` | SQL examples average about 110 tokens |
| `--lora_r` / `--lora_alpha` | `16` / `32` | Attention and MLP projections; 8.8M trainable parameters (1.7%) |

## The same code on a cluster

The only thing that differs between a Mac GPU and a cluster GPU is the resource request and the device string. Both live behind one knob:

```bash
ACCELERATOR=T4   flyte run workflow.py pipeline     # 16 GB CUDA card, fp32 training
ACCELERATOR=L40s flyte run workflow.py pipeline     # 48 GB CUDA card, bf16
ACCELERATOR=T4   python serve.py
```

`config.py` maps the knob to a pod shape (`gpu="AppleSilicon"` versus `gpu="T4:1"`), and `sql.pick_device()` maps it to `mps` or `cuda` at run time. Point `.flyte/config.yaml` at your cluster and nothing else changes.

Verified on the Union demo cluster (2026-10-20) with `ACCELERATOR=T4 flyte run workflow.py pipeline --train_samples 200 --num_eval_examples 30`: device `cuda`, fp32, 25 steps in 25 s, 4.1 GB peak VRAM, 0% → 50% exact match on the smaller run. Same files, same run graph, different GPU. Note that the pipeline task launches its children from inside its own container, so the knob is also passed to every task as an environment variable; otherwise a child would decide its own GPU request from a container that has no `.env`.

`flyte run --local workflow.py pipeline` also works, with no devbox at all. On a Mac, `pick_device("apple")` still gives you `mps`.

## Things to know

- **Workload code runs on your Mac, unsandboxed.** That is what the opt-in flag means. With it, every devbox port binds to `127.0.0.1`.
- **Pip packages only.** The Mac environment is built from `with_pip_packages(...)`. Apt packages, files baked into the image and Kubernetes-mounted secrets do not exist on the host. Environment variables do, so `env_vars=` is how configuration travels.
- **Memory is the constraint, not the pod request.** The pod only proxies the process, so `cpu=1, memory="2Gi"` is fine. Size the run by the Mac's unified memory. `torch.mps.driver_allocated_memory()` reports what the Metal driver holds, including PyTorch's cache, so the 15 GB above overstates what the run needs.
- **No QLoRA.** bitsandbytes has no Metal backend. On a 0.5B model that is no loss; the base weights are 1 GB in bf16.
- **The MPS limit you may remember is gone.** Earlier PyTorch releases crashed on `mps` with models above a few hundred million parameters. torch 2.14 trains this 0.5B model and serves a 1.7B one without trouble.
- **Shared GPU.** Several Apple Silicon tasks can run at once (there is no queue), but they time-share one GPU. The pipeline runs its steps sequentially for clean timings.
