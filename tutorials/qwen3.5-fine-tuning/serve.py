"""Serve the fine-tuned model the same way the base is served: vLLM, OpenAI-compatible, scale to zero.

    python serve.py                         # the latest `pipeline` run's output, base from Hugging Face
    python serve.py --run <run name>        # a specific run
    python serve.py --artifact              # Union: base and adapter from the artifact registry, by name
    PREFETCH_RUN=<run> python serve.py      # base from the serving tutorial's prefetch run instead of the Hub

A LoRA or QLoRA run produced an adapter: vLLM loads the base and attaches the adapter with
`--enable-lora`, and the API exposes the adapter as its own model name (`qwen3.5-4b-support`) next
to the base. A `full` run produced the whole model: it is served as the base. Either way the agent
tutorial can point at this endpoint by changing one model string.

This is a plain `AppEnvironment` on the vLLM plugin's image rather than `VLLMAppEnvironment`: the
plugin owns its model mount and takes no extra parameters, and an adapter is a second mounted
directory. Flyte downloads every mounted parameter before the server starts, so a cold start reads
the weights from object storage, not from the Hub, in the two modes that mount the base. The adapter
mounted is the `vllm/` copy the train task writes (tensor names under vLLM's multimodal module layout);
see `vllm_adapter_copy` in workflow.py for why the PEFT original would be silently ignored.

On Union, `--artifact` binds the app to the `qwen3.5-4b-support` model artifact the pipeline
published (and to the prefetched base's artifact), by name: the resolved versions are recorded on
the deployment, and a later pipeline run does not swap weights under a running app until you
redeploy. On open-source Flyte the run outputs are the same files without the registry.

    curl $ENDPOINT/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "model": "qwen3.5-4b-support",
      "messages": [{"role": "system", "content": "<the SYSTEM_PROMPT in support.py>"},
                   {"role": "user", "content": "i want to cancel order {{Order Number}}"}],
      "chat_template_kwargs": {"enable_thinking": false}}'
"""

from __future__ import annotations

import argparse
import os
import re

import flyte
import flyte.app
from flyteplugins.vllm import DEFAULT_VLLM_IMAGE

from config import MODEL, MODEL_ID, SHAPES

ADAPTER_NAME = f"{MODEL_ID}-support"
APP_NAME = os.getenv("FT_APP_NAME", "qwen35-4b-support-vllm")
GPU = os.getenv("GPU", "L40s")
BASE_ARTIFACT = re.sub(r"[^A-Za-z0-9_-]", "-", MODEL.rsplit("/", 1)[-1])  # what flyte.prefetch names the base: Qwen3-5-4B
MODEL_MOUNT, ADAPTER_MOUNT = "/tmp/model", "/tmp/adapter"
VLLM_FLAGS = ["--max-model-len", "8192", "--reasoning-parser", "qwen3", "--language-model-only"]


def app_for(run_name: str | None, method: str, artifact: bool = False) -> flyte.app.AppEnvironment:
    if artifact:
        trained = flyte.app.ArtifactValue(name=ADAPTER_NAME, type="directory")
    elif run_name:
        trained = flyte.app.RunOutput(type="directory", run_name=run_name)
    else:
        trained = flyte.app.RunOutput(type="directory", task_name="qwen35-ft-cpu.pipeline")

    common = dict(
        name=APP_NAME,
        image=DEFAULT_VLLM_IMAGE,
        resources=SHAPES[GPU],
        port=8080,
        requires_auth=False,
        scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
        env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "0"},
    )
    if method == "full":  # the whole model: serve it as the base
        return flyte.app.AppEnvironment(
            args=["vllm", "serve", MODEL_MOUNT, "--served-model-name", MODEL_ID, "--port", "8080", *VLLM_FLAGS],
            parameters=[flyte.app.Parameter(name="model", value=trained, mount=MODEL_MOUNT)],
            **common,
        )

    # The base: from the registry, from the prefetch run, or (slowest cold start) from the Hub.
    params = [flyte.app.Parameter(name="adapter", value=trained, mount=ADAPTER_MOUNT)]
    prefetch = os.getenv("PREFETCH_RUN")
    if artifact:
        base, params = MODEL_MOUNT, params + [flyte.app.Parameter(name="model", value=flyte.app.ArtifactValue(name=BASE_ARTIFACT, type="directory"), mount=MODEL_MOUNT)]
    elif prefetch:
        base, params = MODEL_MOUNT, params + [flyte.app.Parameter(name="model", value=flyte.app.RunOutput(type="directory", run_name=prefetch), mount=MODEL_MOUNT)]
    else:
        base = MODEL
    return flyte.app.AppEnvironment(
        args=["vllm", "serve", base, "--served-model-name", MODEL_ID, "--port", "8080", *VLLM_FLAGS,
              "--enable-lora", "--max-lora-rank", "64", "--lora-modules", f"{ADAPTER_NAME}={ADAPTER_MOUNT}/vllm"],
        parameters=params,
        **common,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="pipeline run name (default: the latest)")
    parser.add_argument("--method", default="lora", choices=["full", "lora", "qlora"], help="what that run trained")
    parser.add_argument("--artifact", action="store_true", help="Union: bind base and adapter to their artifacts by name")
    args = parser.parse_args()

    flyte.init_from_config()
    app = flyte.serve(app_for(args.run, args.method, args.artifact))
    print(f"Console:  {app.url}")
    print(f"Endpoint: {app.endpoint}/v1")
    print(f"Models:   {MODEL_ID}" + ("" if args.method == "full" else f", {ADAPTER_NAME}"))
