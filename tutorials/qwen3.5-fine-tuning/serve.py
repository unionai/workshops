"""Serve the fine-tuned model the same way the base is served: vLLM, OpenAI-compatible, scale to zero.

    python serve.py                         # the latest `pipeline` run's output
    python serve.py --run <run name>        # a specific run
    python serve.py --artifact              # Union: the latest version of the `qwen3.5-4b-support` artifact
    PREFETCH_RUN=<run> python serve.py      # reuse the base weights the serving tutorial prefetched

On Union, `--artifact` binds the app to the model artifact the pipeline published, by name: the app
definition is complete on its own, the resolved version is recorded on the deployment, and a later
pipeline run does not swap weights under a running app until you redeploy. On open-source Flyte the
run output (the default) is the same files without the registry.

A LoRA or QLoRA run produced an adapter: vLLM loads the base and attaches the adapter with
`--enable-lora`, and the API exposes the adapter as its own model name (`qwen3.5-4b-support`) next
to the base. A `full` run produced the whole model: it is served as the base, from its run output.
Either way the agent tutorial can point at this endpoint by changing one model string.

    curl $ENDPOINT/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "model": "qwen3.5-4b-support",
      "messages": [{"role": "system", "content": "<the SYSTEM_PROMPT in support.py>"},
                   {"role": "user", "content": "i want to cancel order {{Order Number}}"}],
      "chat_template_kwargs": {"enable_thinking": false}}'
"""

from __future__ import annotations

import argparse
import json
import os

import flyte
import flyte.app
from flyteplugins.vllm import VLLMAppEnvironment

from config import MODEL, MODEL_ID, SHAPES

ADAPTER_NAME = f"{MODEL_ID}-support"
APP_NAME = os.getenv("FT_APP_NAME", "qwen35-4b-support-vllm")
ADAPTER_MOUNT = "/tmp/adapter"
GPU = os.getenv("GPU", "L40s")


def app_for(run_name: str | None, method: str, artifact: bool = False) -> VLLMAppEnvironment:
    if artifact:
        trained = flyte.app.ArtifactValue(name=ADAPTER_NAME, type="directory")
    elif run_name:
        trained = flyte.app.RunOutput(type="directory", run_name=run_name)
    else:
        trained = flyte.app.RunOutput(type="directory", task_name="qwen35-ft-cpu.pipeline")
    common = dict(
        name=APP_NAME,
        resources=SHAPES[GPU],
        scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
        requires_auth=False,
        stream_model=True,
    )
    base_args = ["--max-model-len", "8192", "--reasoning-parser", "qwen3", "--language-model-only"]
    prefetch = os.getenv("PREFETCH_RUN")
    if method == "full":
        return VLLMAppEnvironment(model_path=trained, model_id=MODEL_ID, extra_args=base_args, **common)
    base = dict(model_path=flyte.app.RunOutput(type="directory", run_name=prefetch)) if prefetch else dict(model_hf_path=MODEL, stream_model=False)
    common.update(base)
    return VLLMAppEnvironment(
        model_id=MODEL_ID,
        extra_args=[*base_args, "--enable-lora", "--max-lora-rank", "64", "--lora-modules", f"{ADAPTER_NAME}={ADAPTER_MOUNT}"],
        parameters=[flyte.app.Parameter(name="adapter", value=trained, mount=ADAPTER_MOUNT)],
        **common,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="pipeline run name (default: the latest)")
    parser.add_argument("--method", default="lora", choices=["full", "lora", "qlora"], help="what that run trained")
    parser.add_argument("--artifact", action="store_true", help="Union: bind to the published model artifact by name")
    args = parser.parse_args()

    flyte.init_from_config()
    app = flyte.serve(app_for(args.run, args.method, args.artifact))
    print(f"Console:  {app.url}")
    print(f"Endpoint: {app.endpoint}/v1")
    print(f"Models:   {MODEL_ID}" + ("" if args.method == "full" else f", {ADAPTER_NAME}"))
