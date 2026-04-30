# Gemma 4 Chat on Flyte 2

Three-app port of the original Ollama+Gradio Gemma 4 demos to a Flyte 2 devbox running on a GPU host (tested on DGX Spark, NVIDIA GB10, aarch64):

- **vLLM model server** (`vllm_server.py`) — serves Gemma 4 IT via vLLM's OpenAI-compatible API. Streams safetensors directly from Flyte's object store to GPU. Autoscales to zero.
- **Gradio chat UI** (`chat_app.py`) — text chat. Pickled into a separate Flyte app. Talks to vLLM over the cluster-internal Knative DNS. Has a thinking-mode toggle and a thinking-budget slider.
- **Gradio live-camera UI** (`live_camera_app.py`) — webcam → vision caption every few seconds. Same vLLM backend (Gemma 4 is multimodal). Optionally exposes a public HTTPS URL via Gradio's tunnel.

Both Gradio apps preserve the 🧠 Thinking panel from the originals — Gemma 4 IT's thinking is wrapped in `<|channel>...<channel|>` special-token markers, which we keep visible in the response by setting `skip_special_tokens=False` and parse client-side.

## Files

| File | What it does |
|------|--------------|
| `config.py` | Model + GPU choice. Default is `gemma-4-26B-A4B-it`; flip via `GEMMA_VARIANT=31b`. |
| `prefetch_model.py` | One-shot `flyte.prefetch.hf_model` — downloads HF weights into Flyte object store. |
| `vllm_server.py` | `VLLMAppEnvironment` for the chosen Gemma. `__main__` runs prefetch + deploys. |
| `chat_app.py` | Gradio chat `AppEnvironment` + `@env.server`. |
| `live_camera_app.py` | Gradio webcam vision-caption `AppEnvironment` + `@env.server`. |
| `requirements.txt` | Local-side deps (no vllm — that runs in the Flyte container). |
| `SPARK_SETUP.md` | Quick-start setup guide specific to DGX Spark. |

## Setup

```bash
cd tutorials/gemma4-chat

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Start the devbox

```bash
flyte start devbox --gpu
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

The `--gpu` flag is critical — without it, the devbox container is started without `--gpus all` and workload pods can't see the GPU. It also swaps the default image to `cr.flyte.org/flyteorg/flyte-devbox:gpu-latest`, which has the NVIDIA runtime hooks baked in. Verify with:

```bash
docker exec flyte-devbox nvidia-smi -L
```

## Add your HF token

Gemma is gated. Create a Flyte secret with a HuggingFace token that has accepted the Gemma license:

```bash
flyte create secret HF_TOKEN
# paste your hf_xxx token
```

## Deploy

Two steps. Order matters — the chat app's `depends_on` references the vLLM app by name.

```bash
# 1. Serve Gemma 4 via vLLM. Prefetches weights on first run.
python vllm_server.py
# → vLLM app deployed: https://...

# 2. Deploy the Gradio chat UI.
python chat_app.py
# → Chat UI deployed: https://...
```

If a prior prefetch run already wrote the weights to the object store and you don't want to re-download, point at it directly:

```bash
GEMMA_PREFETCH_RUN=<run-name> python vllm_server.py
```

Get the run name from the Flyte UI (`http://localhost:30080/v2`) under successful `prefetch-hf-model` runs, or from the `Prefetch run:` line printed by an earlier successful invocation.

Open the chat UI URL. First message will spin up the vLLM replica (cold start ~30–60s), subsequent messages are warm.

## Switching models

```bash
GEMMA_VARIANT=31b python vllm_server.py
GEMMA_VARIANT=31b python chat_app.py
```

| Variant | Params | GPU spec in `config.py` | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B` (default) | 26B total / 4B active (MoE) | `H100:1` | Fast — only 4B active params per forward pass. Comfortable on 80GB+ GPUs and on GB10 unified memory. |
| `gemma-4-31B` | 31B dense | `H100:2` | Dense bf16 ≈ 62GB; needs TP=2 on a multi-GPU box. On a single-GPU GB10 it's tight even with unified memory — try `--max-model-len 4096` and watch for OOMs. |

The `gpu` field in `config.py` uses Flyte's `<accelerator>:<count>` format. The accelerator label is matched against node labels in real clusters; on the local devbox it's effectively just a count. Edit if you're on different hardware (A100/L40s/B200/GB10/etc.).

## Architecture

```
┌────────────────────┐        AppEndpoint        ┌──────────────────────┐
│  gemma4-chat-ui    │  ────────────────────────▶│ gemma4-26b-a4b-vllm  │
│  (Gradio, CPU)     │   wired via depends_on    │ (vLLM, 1 GPU)        │
│  port 7860         │   + Parameter env_var     │ port 8080            │
└────────────────────┘                            └──────────────────────┘
        ▲                                                    ▲
        │ user                                               │ stream safetensors
        │                                                    │
   browser                                              Flyte object store
                                                        (prefetched HF weights)
```

## Why vLLM (not Ollama)?

Flyte 2's first-class GPU serving is `flyteplugins.vllm.VLLMAppEnvironment` (and `flyteplugins.sglang.SGLangAppEnvironment`). Both expose an OpenAI-compatible API, handle GPU resources, autoscale, and stream model weights directly from blob store to GPU. Ollama would mean managing a sidecar process, manual model pulls inside the container, and no scale-to-zero.

vLLM over SGLang for chat: simpler image (no Rust/CUDA toolkit install at build time), broader model support today. SGLang wins for structured/JSON output — swap by importing `SGLangAppEnvironment` and adjusting `extra_args` (`--context-length` instead of `--max-model-len`, `--tp` instead of `--tensor-parallel-size`).

## Troubleshooting

**`Repository google/gemma-4-26B-A4B does not exist in HuggingFace`** — your HF token hasn't accepted the Gemma license, or the repo path drifted. Visit the model page and click "Acknowledge license", then retry.

**vLLM pod OOMs at startup** — drop `--max-model-len` in `config.py`, or move to the larger GPU spec.

**`<think>` tags showing inline in the answer** — Gemma chose not to produce a thinking block for that prompt; or the tag name differs. Check what the model actually emits via vLLM's `/docs` UI, then update `OPEN`/`CLOSE` in `chat_app.py:_split_thinking`.

**Chat UI shows the URL but `/v1/chat/completions` fails** — vLLM replica is still cold-starting. Wait ~6 min on first request after idle (image pull + safetensors stream + Inductor compile + CUDA-graph capture). Watch the vLLM app logs in the Flyte UI at http://localhost:30080/v2.

**vLLM image build fails on aarch64 / Blackwell (GB10) host** — vanilla `vllm==0.11.0` + `flashinfer` wheels are x86_64-only and even when they build, they don't recognize Gemma 4. Use NVIDIA's prebuilt `vllm/vllm-openai:gemma4-cu130` image instead (already wired up in `vllm_server.py`). See [build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm).

## Why this is the way it is — gotchas we hit

A bunch of small issues we tripped over getting this running on a DGX Spark devbox; documenting so future-you doesn't repeat them.

1. **Devbox needs `--gpu`** — `flyte start devbox` (no flag) starts the container with no GPU passthrough. The k3s cluster will schedule pods but they won't see the GPU. `flyte start devbox --gpu` adds `--gpus all` and uses `flyte-devbox:gpu-latest`.

2. **GPU spec must be `gpu=1`, not `gpu="H100:1"`** — Flyte filters typed-accelerator requests against node labels, and the GB10 node isn't labeled `H100`. A typed mismatch silently drops the GPU request and the pod schedules with zero GPU.

3. **Image registry defaults to `ghcr.io/flyteorg`** unless you set it. The detection runs at `from_debian_base()` time, which happens at module import — *before* `flyte.init_from_config()` runs and tells the SDK we're on a localhost endpoint. Pass `registry="localhost:30000"` explicitly.

4. **Multi-arch builds on aarch64 segfault under QEMU** — buildx's default `linux/amd64,linux/arm64` runs the amd64 layers via QEMU emulation, and `uv venv` segfaults inside it. Pin `platform=("linux/arm64",)`.

5. **Gemma 4 needs the custom vLLM image, not vanilla** — `vllm/vllm-openai:gemma4-cu130` has the architecture patches for `Gemma4ForConditionalGeneration`, the right CUDA 13 + Blackwell sm_120 kernels, and is multi-arch (aarch64-ready). Vanilla `vllm==0.11.0` produces `libcudart.so.12: cannot open shared object file` because torch 2.11 brings in cu13 wheels but vLLM was built against cu12.

6. **`from_base()` returns a non-extendable image** — to layer `flyteplugins-vllm` on top, call `.clone(extendable=True)`. Setting platform also requires `object.__setattr__` since `Image` is a frozen dataclass and `from_base()` doesn't expose a platform parameter.

7. **`flyteplugins-vllm` must be installed in the base image's system Python**, not in `/opt/venv` — Flyte's image builder creates a fresh venv at `/opt/venv`, but the base image's torch + vllm are at `/usr/lib/python3.12/dist-packages`. `with_pip_packages` writes to `/opt/venv` and `vllm-fserve` then crashes with `ModuleNotFoundError: No module named 'torch'`. Use `with_commands(["/usr/bin/python3 -m pip install --pre flyteplugins-vllm"])` to target the system Python.

8. **GB10 unified memory is a tight fit** — 119.7 GiB total but only ~106.87 GiB free at startup (drivers, GUI, etc.). vLLM's default `--gpu-memory-utilization=0.9` requests 107.7 GiB → `ValueError`. Set `0.85`.

9. **Gemma 4 base model has no chat template** — neither in `tokenizer_config.json` nor as a separate `.jinja` file in the HF repo. Calling `/v1/chat/completions` returns `default chat template is no longer allowed`. The chat app sidesteps this by formatting the prompt manually and using `/v1/completions` (see `_format_gemma_prompt` in `chat_app.py`).

10. **`AppEndpoint` doesn't work with pkl-bundle interactive deploy + `depends_on`** — the framework tries to deploy the dep in pkl mode, but `VLLMAppEnvironment` has no `@server` function. Without `depends_on`, `AppEndpoint(public=False)` can't read `INTERNAL_APP_ENDPOINT_PATTERN`, and `public=True` returns the `.localhost` URL which only resolves on the host, not inside other pods. Easiest fix: skip `AppEndpoint` and pass the cluster-internal URL directly as a string parameter:

    ```python
    f"http://{MODEL.app_name}-flytesnacks-development.flyte.svc.cluster.local"
    ```

11. **Gradio version matters** — `gr.Chatbot(type="messages")` needs Gradio 5.x; 6.x dropped that kwarg. We pin `gradio==5.42.0`.

12. **Knative scales to zero by default** — first request after idle pays the full ~6 min cold start. We bump `scaledown_after` to 1800s (30 min) so an active session doesn't trip over it.

13. **Webcam needs HTTPS or localhost** — `getUserMedia` is blocked over plain HTTP from a non-localhost origin. Two ways to handle it for `live_camera_app.py`:

    - On the Spark itself, browse to `http://localhost:30081/...` — `localhost` is exempt and webcam works.
    - From a remote/Tailscaled machine, deploy with `GRADIO_SHARE=1`. Gradio inside the pod opens an outbound TLS tunnel to `gradio.live` and gets a public `https://<random>.gradio.live` URL that proxies back to port 7867 in the pod. Gives webcam access from anywhere.

14. **Gradio share traffic bypasses Knative's queue-proxy** — when using `GRADIO_SHARE=1`, requests come in via the gradio.live tunnel directly to gradio's port inside the pod, **not** through Knative's ingress + queue-proxy. So Knative sees zero activity and scales the pod (and tunnel) down at `scaledown_after`. This actually doubles as auto-cleanup — the public link auto-expires when the pod dies. To extend a live session, periodically hit the Knative URL too (`curl http://localhost:30081 -H "Host: gemma4-live-camera-flytesnacks-development.localhost"` from the Spark) to reset the idle timer. To pin always-on, set `replicas=(1, 1)` on the AppEnvironment.
