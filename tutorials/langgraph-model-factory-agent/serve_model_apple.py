"""Serve an open model on your Mac's GPU as a Flyte app, OpenAI-compatible, so the ML
engineer agent itself can run on a model you own. The Apple Silicon devbox counterpart of
`serve_model.py` (which does the same on a Union GPU with vLLM).

    python serve_model_apple.py                       # prefetch Qwen3-8B, deploy it on the Mac, print the endpoint
    VLLM_PREFETCH_RUN=<run name> python serve_model_apple.py                           # reuse a prefetch
    VLLM_HF_REPO=Qwen/Qwen3-4B VLLM_MODEL_ID=qwen3-4b python serve_model_apple.py   # a 16 GB Mac

The weights are fetched once by a Flyte task (`flyte.prefetch.hf_model`, the same one
`serve_model.py` uses) into the devbox's object store, so the download is a run in the UI
with the weights as its output, and the app mounts that output. Redeploys and other apps
reuse it instead of pulling from Hugging Face again.

Then, in .env:

    VLLM_BASE_URL=http://<the printed endpoint>/v1
    FACTORY_PROVIDERS=anthropic,vllm

and step 6 can bake off `vllm:qwen3-8b` against the API models. `openai_server.py` is the
same small server as the `apple-silicon-serving` tutorial: transformers on `mps`, tool
calls rendered by the model's own chat template and parsed back out of the Hermes-style
blocks Qwen emits, thinking off. Generation is serialized, one GPU, one stream, so a
bake-off with several open-model agents at once takes turns on it.

The engineer asks for its typed decision through a tool call (see `llm.py`), so no JSON
schema support is needed on the server. The app needs a `VLLM_API_KEY` secret to exist
on the devbox only because the agent's environment asks for one; any value works.
"""

from __future__ import annotations

import logging
import os

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

from config import ACCELERATOR, PROPAGATED, SHAPES
from openai_server import build_app

logging.basicConfig(level=logging.INFO, format="%(message)s")

HF_REPO = os.environ.get("VLLM_HF_REPO", "Qwen/Qwen3-8B")
MODEL_ID = os.environ.get("VLLM_MODEL_ID", "qwen3-8b")
APP_NAME = os.environ.get("VLLM_APP_NAME", f"factory-{MODEL_ID}")
SERVER_PIP = ("torch>=2.4", "transformers>=4.51", "fastapi", "uvicorn", "python-dotenv")

MODEL_DIR = "/tmp/llm"  # where the app mounts the prefetched weights

app = build_app(MODEL_DIR, ACCELERATOR, model_id=MODEL_ID)
app.title = f"{MODEL_ID} ({HF_REPO}), OpenAI-compatible, on {ACCELERATOR}"


def make_app_env(prefetch_run: str) -> FastAPIAppEnvironment:
    return FastAPIAppEnvironment(
        name=APP_NAME,
        app=app,
        description=f"{HF_REPO} for the ML engineer agent, OpenAI-compatible, on {ACCELERATOR}",
        image=flyte.Image.from_debian_base().with_pip_packages(*SERVER_PIP),
        resources=SHAPES[ACCELERATOR],
        requires_auth=False,
        # VLLM_PREFETCH_RUN travels with the app so the server process rebuilds the same `app_env`.
        env_vars={**PROPAGATED, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "VLLM_PREFETCH_RUN": prefetch_run},
        parameters=[
            flyte.app.Parameter(
                name="model",
                value=flyte.app.RunOutput(type="directory", run_name=prefetch_run),
                mount=MODEL_DIR,
            )
        ],
        links=[flyte.app.Link(path="/v1/models", title="Models", is_relative=True)],
    )


# Module level on purpose: `flyte.serve` records the app as `<module>:<variable>` and the
# server process imports it by that name. An environment that only exists inside a
# function deploys fine and then dies with "No command provided to execute".
app_env = make_app_env(os.environ.get("VLLM_PREFETCH_RUN", ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prefetch an open model and serve it on this Mac's GPU")
    parser.add_argument(
        "--config", help="Flyte config file (default: the SDK's search, which finds .flyte/config.yaml)"
    )
    args = parser.parse_args()
    flyte.init_from_config(args.config)

    run_name = os.environ.get("VLLM_PREFETCH_RUN")
    if run_name:
        print(f"Reusing prefetched weights from run {run_name}")
    else:
        import flyte.prefetch

        print(f"Prefetching {HF_REPO} into the object store…")
        run = flyte.prefetch.hf_model(repo=HF_REPO, hf_token_key=None)  # ungated: no token needed
        print(run.url)
        run.wait()
        run_name = run.name
        os.environ["VLLM_PREFETCH_RUN"] = run_name
        app_env = make_app_env(run_name)

    print(f"Deploying {MODEL_ID} on {ACCELERATOR}…")
    deployed = flyte.serve(app_env)
    print(f"Console:  {deployed.url}")
    print(f"Endpoint: {deployed.endpoint}")
    print(f"\nIn .env:  VLLM_BASE_URL={deployed.endpoint}/v1   (prefetch run: {run_name})")
