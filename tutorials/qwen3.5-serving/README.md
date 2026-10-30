# Serve Qwen3.5-4B as an endpoint, and measure it

Take a model off the Hub, put it behind an OpenAI-compatible endpoint on a GPU you control, and
check it the way you would check any service: with a scored request set and a load test, from
code, before anyone points an app at it.

```bash
python serve.py                    # prefetch the weights once, deploy vLLM, scale to zero when idle
flyte run benchmark.py sanity      # 12 questions with known answers, scored, with tokens and seconds
flyte run benchmark.py benchmark   # concurrent requests: tokens/s, time to first token, latency
flyte run benchmark.py compare     # thinking off vs on, one report
python app_gradio.py               # a chat UI that follows the endpoint wherever it is deployed
```

This is leg 1 of the [model-drop template](../MODEL_DROP.md): the same four tutorials get built
for every new model, and this one is the endpoint the other three use.

<a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/qwen3.5-serving/qwen3.5-serving-tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Results

RESULTS_PENDING

## How it works

**The weights live in object storage, not in the image.** `serve.py` runs `flyte.prefetch.hf_model`
once: a task downloads the repo from Hugging Face into the cluster's blob store and the run's
output is the directory. The app is deployed with `model_path=RunOutput(run_name=...)`, and
vLLM's Flyte loader streams the safetensors from there straight to the GPU on every cold start.
The image never contains the model, so a new model is a new prefetch, not a new build.

**Scale to zero.** `Scaling(replicas=(0, 1), scaledown_after=900)`: nothing runs between
requests, and the first request after idle pays a cold start (image pull, weight stream, CUDA
graph capture; a few minutes on an L40S). The benchmark task's first request shows it.

**The endpoint is measured before it is used.** `sanity` sends twelve questions with gold answers
and scores them with a regex, then puts the score on the same row as the cost (output tokens,
seconds). That habit is the point: "deployed" and "answers correctly" are different claims, and
the second is cheap to make true. `benchmark` streams concurrent requests and reports aggregate
and per-request tokens/s, time to first token and latency. `compare` runs both with Qwen's
thinking off and on, because thinking is a per-request switch
(`chat_template_kwargs.enable_thinking`) and its cost is a number you should know before you
default to it.

**Reports.** Every task writes a `flyte.report`: a stat strip, SVG bar charts (per-request
tokens/s, TTFT vs total latency), the scored table, and the first replies in full.

**What Qwen3.5 needs from vLLM.** `--reasoning-parser qwen3` moves the model's `<think>` block
into `reasoning_content` so `content` is just the answer. `--enable-auto-tool-choice
--tool-call-parser qwen3_coder` turns on OpenAI-style tool calling, which the agent tutorial
depends on. `--language-model-only` skips the vision encoder Qwen3.5 ships with. These are the
flags the model card recommends; a different model family changes this list and nothing else.

## Files

| File | What it does |
|---|---|
| `config.py` | `MODEL`, `MODEL_ID`, `APP_NAME`, `GPU`, optional `API_KEY`; the pod shapes; `endpoint_for()` for reaching the app from a task |
| `serve.py` | The `VLLMAppEnvironment` with the Qwen3.5 flags; `python serve.py` prefetches and deploys |
| `benchmark.py` | `sanity`, `benchmark`, `compare`: CPU tasks that measure the endpoint and write reports |
| `app_gradio.py` | A Gradio chat app with a thinking toggle; gets the endpoint as an `AppEndpoint` parameter |
| `report_helpers.py` | Report CSS and SVG charts (shared with the sibling tutorials) |
| `qwen3.5-serving-tutorial.ipynb` | The same steps as a notebook |

## Setup

**Attach an accelerator first.** vLLM needs an NVIDIA GPU: the Union cluster (`GPU=L40s`, the
default) or `flyte start devbox --gpu`. Colab cannot host the app and an Apple Silicon devbox cannot
run vLLM; both run the client tasks against a cluster endpoint (`LLM_BASE_URL`). To serve a model on
the Mac's own GPU, the [`apple-silicon-serving`](../apple-silicon-serving/) tutorial is the pattern.

Three ways to run it; pick one. The `flyte run` commands below are the same on a Union cluster
and on a devbox; `flyte run --local` runs a task in your own Python with no cluster at all.

| | Union cluster | Devbox (your machine) | No cluster (`--local`) |
|---|---|---|---|
| What runs the work | the cluster's GPUs | a Docker container on your machine, your GPU | your Python process, your GPU if torch finds one |
| Needs | Python 3.10+, `uv`, a browser sign-in | Python 3.10+, `uv`, Docker running | Python 3.10+, `uv` |
| GPU legs (serve, fine-tune, RL) | yes | yes, with an NVIDIA GPU (`--gpu`) | fine-tune and RL only, if you have a CUDA GPU |
| Reports, run graph, retries, cache | yes | yes | no: plain Python, printed output |

In every case, start here:

```bash
git clone https://github.com/unionai/workshops
cd workshops/tutorials/qwen3.5-serving
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
flyte --version          # 2.12 or newer
```

<details>
<summary><b>Union cluster</b></summary>

```bash
flyte create config --endpoint dns:///<your-union-host> --project flytesnacks --domain development --builder remote
flyte get project         # opens a browser sign-in the first time
```

Optional: protect the endpoint with a bearer token. `flyte create secret VLLM_API_KEY --value <token>`,
then `API_KEY=VLLM_API_KEY` in `.env`. The tasks and the chat app read the same secret. Without it the endpoint is open,
which is fine on a devbox and not fine on a shared cluster.
</details>

<details>
<summary><b>Devbox on your machine</b></summary>

A devbox is a Flyte control plane in one container. Workloads that ask for a GPU use the
machine's own:

```bash
flyte start devbox --gpu            # an NVIDIA machine; `flyte start devbox` for CPU only
flyte create config --devbox        # writes .flyte/config.yaml pointing at localhost:30080
```

vLLM needs an NVIDIA GPU. The `GPU` name in `config.py` is matched against the devbox's node labels; if
yours is not an `L40s`, set `SHAPES` to `gpu=1` (any GPU) in `config.py`. An Apple Silicon devbox cannot run vLLM;
it can run `benchmark.py` and the chat app against a cluster endpoint (next section).
</details>

<details>
<summary><b>No cluster: <code>--local</code></b></summary>

The vLLM app itself has no `--local` form (it is a deployment). The client tasks do, against any endpoint:

```bash
LLM_BASE_URL=https://<app-host> flyte run --local benchmark.py sanity
SERVER_URL=https://<app-host> python app_gradio.py --local
```
</details>

## Run

### 1. Serve

```bash
python serve.py
# Prefetching Qwen/Qwen3.5-4B into object storage…  (a few minutes, once)
# Deploying qwen3.5-4b on L40s…
# Endpoint: https://<app-host>/v1
PREFETCH_RUN=<run name> python serve.py     # redeploy without re-downloading
```

The deploy returns when the app is registered; the first request wakes it.

### 2. Measure it

```bash
flyte run benchmark.py sanity                                   # correctness, cheap
flyte run benchmark.py benchmark --prompts 16 --concurrency 8   # throughput under load
flyte run benchmark.py compare                                  # thinking off vs on
```

Open the run's report for the charts.

### 3. A chat UI

```bash
python app_gradio.py
# Chat UI: https://<chat-host>
```

### 4. From anywhere

```python
from openai import OpenAI
client = OpenAI(base_url="https://<app-host>/v1", api_key="<token or anything>")
r = client.chat.completions.create(model="qwen3.5-4b", messages=[{"role": "user", "content": "hi"}],
                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

## Swap the model

`config.py` is the whole surface: `MODEL=Qwen/Qwen3.8-27B-FP8 MODEL_ID=qwen3.8-27b APP_NAME=qwen38-27b-vllm python serve.py`
serves the 27B FP8 checkpoint on the same L40S shape (28 GB of weights). Check the new model's
card for its vLLM parser flags and edit `extra_args` in `serve.py` if they differ.

## Union or open-source Flyte

Everything here runs on open-source Flyte. On Union the prefetch also registers the weights as a
model artifact named after the repo (`Qwen3.5-4B`, versioned by the Hugging Face commit), and
`serve.py` binds the app to that artifact by name, so the deployment records exactly which version
it serves and a re-prefetch never swaps weights under a running app. Without a registry the app
points at the prefetch run's output: the same files. `ARTIFACTS=0` forces the fallback.

## Things to know

- **The app name is the address.** The agent tutorial finds this endpoint by `APP_NAME`; keep it stable.
- **`requires_auth=False` is not "open".** It turns off platform auth so a task can call the app with a plain bearer token; the token (`API_KEY`) is what protects it. Leave both off only on a devbox.
- **Cold starts are real.** After `scaledown_after` seconds idle the pod is gone. The first request in a benchmark run includes the restart; the report shows it as a long TTFT on request #1.
- **Thinking changes the cost, not the API.** Same endpoint, per-request switch. `compare` measures it so you decide with numbers.
