"""vLLM model-serving app for Gemma 4.

This defines the `vllm_app` AppEnvironment (importable by `chat_app.py`) and,
when run as `__main__`, prefetches the model from HF and deploys the server
to the Flyte 2 devbox. The vLLM server speaks the OpenAI-compatible API on
port 8080; `/v1/chat/completions`, `/docs`, etc.

Deploy:
    python vllm_server.py
    # or, for the dense 31B variant:
    GEMMA_VARIANT=31b python vllm_server.py
"""

from __future__ import annotations

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app

from config import MODEL


# vLLM image: pinned vllm + flashinfer kernels + the flyte vLLM plugin
# (which provides `vllm-fserve` and the streaming model loader).
# Gemma 4 needs a custom vLLM container (per NVIDIA DGX Spark playbook —
# https://build.nvidia.com/spark/vllm) because the architecture is too new
# for vanilla vLLM. The `gemma4-cu130` tag bundles the right vllm patches,
# CUDA 13 runtime, and Blackwell (sm_120) kernels.
#
# Notes:
# - `from_base()` returns a non-extendable Image; clone with extendable=True
#   so we can layer flyteplugins-vllm on top (it provides `vllm-fserve`).
# - `from_base()` defaults to platform=linux/amd64 and clone() doesn't expose
#   a platform parameter, so we use dataclasses.replace() to set arm64.
#   Without this the build runs amd64 layers under QEMU on this aarch64 host
#   and segfaults inside `uv venv` (qemu signal 11).
_base = flyte.Image.from_base("vllm/vllm-openai:gemma4-cu130")
# Image is a frozen dataclass and blocks direct __init__, so dataclasses.replace
# doesn't work. Use object.__setattr__ to bypass the freeze.
object.__setattr__(_base, "platform", ("linux/arm64",))
image = (
    _base.clone(
        registry="localhost:30000",
        name="gemma4-vllm-image",
        extendable=True,
    )
    # Install flyteplugins-vllm into the BASE IMAGE's system Python
    # (/usr/bin/python3, where vllm + torch already live) instead of
    # into Flyte's /opt/venv. Otherwise vllm-fserve crashes at startup
    # with `ModuleNotFoundError: No module named 'torch'` because the
    # /opt/venv only sees flyteplugins-vllm and its direct deps.
    .with_commands([
        "/usr/bin/python3 -m pip install --no-cache-dir --pre flyteplugins-vllm"
    ])
)

# `model_hf_path` here is a placeholder — at deploy time we override it with
# `model_path=RunOutput(...)` pointing at the prefetched directory in object
# storage so vLLM streams safetensors directly to GPU.
vllm_app = VLLMAppEnvironment(
    name=MODEL.app_name,
    image=image,
    model_hf_path=MODEL.hf_repo,
    model_id=MODEL.model_id,
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu=MODEL.gpu, disk="20Gi"),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),        # scale to zero when idle
        scaledown_after=1800,   # 30 min — cold starts take ~6 min (image pull + model stream + kernel compile), so we want to amortize warm-up over a generous idle window
    ),
    requires_auth=False,       # devbox: skip auth so the Gradio app can call freely
    extra_args=[
        "--max-model-len", str(MODEL.max_model_len),
        "--trust-remote-code",
        # GB10 unified memory: ~119.7 GiB total but only ~106.87 GiB free at
        # vLLM startup (rest reserved by drivers / display / other procs).
        # Default 0.9 utilization tries to claim more than free → ValueError.
        "--gpu-memory-utilization", "0.85",
    ],
)


if __name__ == "__main__":
    import os

    flyte.init_from_config()

    # Skip the prefetch and reuse a known-good prefetch run. Useful when HF
    # downloads are flaky or to avoid re-uploading the same weights.
    existing_run = os.environ.get("GEMMA_PREFETCH_RUN")
    if existing_run:
        run_name = existing_run
        print(f"Reusing prefetched model from run: {run_name}")
    else:
        import flyte.prefetch
        from flyte.remote import Run

        print(f"Prefetching {MODEL.hf_repo}…")
        run: Run = flyte.prefetch.hf_model(repo=MODEL.hf_repo)
        run.wait()
        print(f"Prefetch run: {run.url}")
        run_name = run.name

    print(f"Deploying vLLM server for {MODEL.model_id} on {MODEL.gpu}…")
    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run_name),
            model_hf_path=None,
        )
    )
    print(f"vLLM app deployed: {app.url}")
    print(f"  OpenAI base URL: {app.url}/v1")
    print(f"  OpenAPI docs:    {app.url}/docs")
