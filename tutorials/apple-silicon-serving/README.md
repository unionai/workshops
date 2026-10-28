# Serve an LLM on your Mac's GPU as a Flyte app

A Flyte app with `gpu="AppleSilicon"` runs natively on your Mac, so it can load a model onto the Metal GPU through PyTorch's `mps` device. The devbox still gives it a URL, logs, port forwarding and a place in the console; only the process is on the laptop.

This tutorial serves [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) behind a small OpenAI-compatible API (chat completions, streaming, tool calls), measures it from a Flyte task, compares it with MLX running natively, and puts a chat UI in front of it. The same app runs on a cluster GPU by changing one environment variable.

```bash
flyte start devbox --apple-silicon
flyte create config --devbox
python serve.py                           # the model, on the Mac's GPU
flyte run benchmark.py compare            # torch on mps vs MLX, from Flyte tasks
python app_gradio.py                      # a chat UI that finds the model app by name
```

## Results

Measured on an M5 Max (128 GB), flyte 2.12.0, torch 2.14.1, mlx-lm 0.32.2. Eight prompts, 128 new tokens each.

| Backend | Model | Generation speed | Latency |
|---|---|---|---|
| `transformers` on `mps`, bf16, behind the app | Qwen3-1.7B | **79 tokens/s** per request | 3.4 s mean at 4 concurrent (requests queue on one GPU lock) |
| `mlx-lm`, 4-bit, in-process | Qwen3-1.7B-4bit | **277 tokens/s** | 0.34 s mean, sequential |

Two different things are compared on purpose. The torch app is portable: `ACCELERATOR=L40s` and it serves from a CUDA card with no code change. MLX is Mac-only and about 3.5x faster here, at 4-bit. Reach for torch when the Mac is a development stand-in for a cluster, and for MLX when the Mac is the deployment target. Flyte runs both the same way.

## Files

| File | What it does |
|---|---|
| `config.py` | The `ACCELERATOR` knob and pod shapes, the model names, and `reach()` for calling an app from a task |
| `openai_server.py` | The server: `/v1/chat/completions` with streaming and Hermes-style tool-call parsing, `/v1/models`, `/health`. No Flyte in it |
| `serve.py` | Wraps the server in a `FastAPIAppEnvironment` with `gpu="AppleSilicon"` and deploys it |
| `benchmark.py` | `benchmark`: concurrent requests against the app from a CPU task. `compare`: that plus the MLX task, one report |
| `mlx_native.py` | `mlx_generate`: the same prompts through mlx-lm, natively on the Mac's GPU |
| `app_gradio.py` | A chat UI app that gets the model app's address as an `AppEndpoint` parameter |
| `apple-silicon-serving-tutorial.ipynb` | The same steps as a notebook |

## Setup

You need an Apple Silicon Mac (M1 or later), Docker Desktop, [`uv`](https://docs.astral.sh/uv/) and `flyte` 2.12 or later.

```bash
cd tutorials/apple-silicon-serving
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

flyte start devbox --apple-silicon
flyte create config --devbox
```

Qwen3-1.7B is ungated (3.4 GB in bf16). It downloads into your own `~/.cache/huggingface` on the app's first start, because the app's process is yours.

## Run

### 1. Serve

```bash
python serve.py
# Endpoint: http://apple-silicon-llm-flytesnacks-development.localhost:30081
```

The app loads the model onto `mps`, runs one short generation to compile the Metal kernels, and only then takes traffic. From then on it answers the OpenAI chat API:

```bash
URL=http://apple-silicon-llm-flytesnacks-development.localhost:30081
curl -s $URL/health
curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-1.7B",
  "messages": [{"role": "user", "content": "Write a haiku about a laptop."}]
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url=f"{URL}/v1", api_key="not-used")
for chunk in client.chat.completions.create(model="Qwen/Qwen3-1.7B", stream=True,
        messages=[{"role": "user", "content": "Count to ten."}]):
    print(chunk.choices[0].delta.content or "", end="")
```

Qwen3 thinks by default. The server turns that off unless a request asks for it with `extra_body={"chat_template_kwargs": {"enable_thinking": True}}`. Pass `tools=[...]` and the model's `<tool_call>` blocks come back as OpenAI `tool_calls`, which is what the agent tutorial next door relies on.

### 2. Measure it from a task

```bash
flyte run benchmark.py benchmark --prompts 8 --concurrency 4
flyte run mlx_native.py mlx_generate
flyte run benchmark.py compare
```

`benchmark` is an ordinary CPU task. It stays in its container and calls the app over the devbox network, the way any other service would. `mlx_generate` asks for `gpu="AppleSilicon"` and runs on the Mac, where MLX exists. `compare` runs both and draws the bar chart.

### 3. A chat UI

```bash
python app_gradio.py
# Chat UI: http://apple-silicon-llm-chat-flytesnacks-development.localhost:30081
```

The UI is a plain CPU app. It is told where the model app lives through an `AppEndpoint` parameter, so it follows the model app wherever that is deployed.

## The same app on a cluster

```bash
ACCELERATOR=L40s python serve.py
ACCELERATOR=L40s flyte run benchmark.py benchmark
```

`config.py` turns the knob into a pod shape, and `openai_server.pick_device()` turns it into `cuda`. Everything else, the server, the UI, the benchmark task, is unchanged. `mlx_generate` is the one thing that cannot follow: MLX only exists on macOS, which is the point of including it.

## Things to know

- **How an app lands on the Mac.** The app's pod starts, but its `fserve` entrypoint hands the work to the bridge on the host, which builds a macOS environment from the image's pip packages and runs the server there. The host picks a free port; the container forwards the app's port to it. `app_env.get_port()` returns the right one inside the server function, which is why the server binds to whatever it is given.
- **Calling an app from a task on a devbox.** Apps are told the in-cluster address pattern of their neighbours, but tasks on a devbox are not, and a pod cannot resolve the devbox's public `*.localhost:30081` names. `config.reach(app_env)` swaps in the in-cluster service name (`<app>-<project>-<domain>.flyte.svc.cluster.local`) when it detects that situation. On a cluster it returns `app_env.endpoint` untouched.
- **One lock, one GPU.** The server generates one request at a time. Per-request speed stays flat under load; latency grows with concurrency. Batching is what a real inference server adds, which is what vLLM does on the cluster and oMLX does on a Mac (there is a `flyteplugins-omlx` app environment for it).
- **Memory.** The pod request is a formality (`cpu=1, memory="2Gi"`); the model lives in the Mac's unified memory. bf16 Qwen3-1.7B takes about 4 GB. `MODEL_NAME=Qwen/Qwen3-4B python serve.py` wants about 9 GB.
- **No silent fallback.** Request `AppleSilicon` on a devbox started without the flag and the deploy is refused with `APPLE_SILICON_NOT_AVAILABLE`.
- **Gradio 6.** `ChatInterface` no longer takes `type=`; messages format is the default.
