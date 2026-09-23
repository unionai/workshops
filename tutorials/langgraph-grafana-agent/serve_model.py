"""Serve an open model on a Union app, OpenAI-compatible, so the ML engineer agent itself
can run on a model you own.

Step 7 points the same agent at this endpoint with `vllm:qwen3-8b`. vLLM speaks the
OpenAI chat-completions API, so the only thing that changes in the agent is the model
string; LangChain's `ChatOpenAI` does the rest. One app serves the whole room; it is not
namespaced per attendee.

    flyte create secret VLLM_API_KEY --value <any string you choose>   # once
    python serve_model.py                                          # prefetch + deploy
    VLLM_PREFETCH_RUN=<run name> python serve_model.py             # reuse a prefetch

Then, in .env:

    VLLM_BASE_URL=https://<app url>/v1

The weights are prefetched from Hugging Face into the cluster's object store once, and
streamed from there straight to the GPU on every cold start. The app scales to zero when
idle, so it costs nothing between bake-offs. Qwen3-8B fits comfortably on one L40s.
"""

from __future__ import annotations

import os

import flyte
import flyte.app
from flyteplugins.vllm import VLLMAppEnvironment

from config import VLLM_SECRET

HF_REPO = os.environ.get("VLLM_HF_REPO", "Qwen/Qwen3-8B")
MODEL_ID = os.environ.get("VLLM_MODEL_ID", "qwen3-8b")
APP_NAME = os.environ.get("VLLM_APP_NAME", "factory-qwen3-8b")

vllm_app = VLLMAppEnvironment(
    name=APP_NAME,
    model_hf_path=HF_REPO,  # replaced with the prefetched path at deploy time (see below)
    model_id=MODEL_ID,
    resources=flyte.Resources(cpu="6", memory="32Gi", gpu="L40s:1", disk="60Gi"),
    stream_model=True,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    # Platform auth off so a task (or your laptop) can call it with a plain bearer token;
    # vLLM's own --api-key, fed from the secret, is what protects it.
    requires_auth=False,
    secrets=[VLLM_SECRET],
    extra_args=[
        "--api-key",
        "$VLLM_API_KEY",
        "--max-model-len",
        "16384",
        # Tool calling: Qwen3 emits Hermes-style <tool_call> blocks.
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        # Keep any thinking the model does out of the answer text.
        "--reasoning-parser",
        "qwen3",
    ],
)


if __name__ == "__main__":
    flyte.init_from_config()

    run_name = os.environ.get("VLLM_PREFETCH_RUN")
    if run_name:
        print(f"Reusing prefetched weights from run {run_name}")
    else:
        import flyte.prefetch

        print(f"Prefetching {HF_REPO} into object storage…")
        run = flyte.prefetch.hf_model(repo=HF_REPO)
        print(run.url)
        run.wait()
        run_name = run.name

    print(f"Deploying {MODEL_ID} on L40s…")
    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run_name),
            model_hf_path=None,
        )
    )
    # app.url is the console page; app.endpoint is what vLLM answers on.
    print(f"Console: {app.url}")
    print(f"Put this in .env:  VLLM_BASE_URL={app.endpoint}/v1")
    print(
        f'Then:              flyte run bakeoff.py bakeoff --models \'["anthropic:claude-opus-5", "vllm:{MODEL_ID}"]\''
    )
