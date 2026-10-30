"""Serve Qwen3.5-4B as an OpenAI-compatible endpoint on a GPU, with vLLM.

    python serve.py                         # prefetch the weights, then deploy
    PREFETCH_RUN=<run name> python serve.py # reuse an earlier prefetch

The weights are pulled from Hugging Face into the cluster's object store once (`flyte.prefetch`),
and streamed from there straight to the GPU on every cold start, so a restart never touches the
Hub. The app scales to zero when idle and comes back on the first request.

On Union the prefetch also registers the weights as a model artifact named after the repo
(`Qwen3.5-4B`, versioned by the Hugging Face commit), and the app binds to that artifact by name:
the deployment records exactly which version it serves. On open-source Flyte there is no registry,
and the app points at the prefetch run's output instead: the same files, no card.

What the flags do:
    --reasoning-parser qwen3        Qwen3.5 thinks in a <think> block by default; the parser moves it
                                    into `reasoning_content` so `content` is just the answer.
    --enable-auto-tool-choice       OpenAI-style tool calling (the agent tutorial depends on this),
    --tool-call-parser qwen3_coder  with the parser the model card recommends for this family.
    --language-model-only           Qwen3.5 ships a vision encoder; this tutorial serves text only
                                    and skips loading it.
"""

from __future__ import annotations

import os

import flyte
import flyte.app
import flyte.prefetch
from flyteplugins.vllm import VLLMAppEnvironment

from config import API_KEY, APP_NAME, ARTIFACTS, GPU, MODEL, MODEL_ID, SECRETS, SHAPES

MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))

vllm_app = VLLMAppEnvironment(
    name=APP_NAME,
    model_hf_path=MODEL,  # swapped for the prefetched object-store path at deploy time (below)
    model_id=MODEL_ID,
    resources=SHAPES[GPU],
    stream_model=True,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    # Platform auth off so tasks and laptops call it with a plain bearer token. If API_KEY names a
    # secret, vLLM's own --api-key is what protects the endpoint; otherwise it is open.
    requires_auth=False,
    secrets=SECRETS,
    extra_args=[
        "--max-model-len", str(MAX_MODEL_LEN),
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--language-model-only",
        *(["--api-key", "$VLLM_API_KEY"] if API_KEY else []),
    ],
)


def _has_registry(artifact_name: str) -> bool:
    """True when the platform has an artifact registry with this model in it (Union); False on open-source Flyte."""
    from flyte.remote import Artifact

    try:
        for _ in Artifact.listall(name=artifact_name):
            return True
        return False
    except Exception as e:  # noqa: BLE001
        if "not found" in str(e).lower() or "unimplemented" in str(e).lower():
            return False
        raise


def deploy():
    flyte.init_from_config()

    run_name = os.getenv("PREFETCH_RUN")
    if run_name:
        print(f"Reusing prefetched weights from run {run_name}")
    else:
        print(f"Prefetching {MODEL} into object storage…")
        run = flyte.prefetch.hf_model(repo=MODEL)
        print(run.url)
        run.wait()
        run_name = run.name
        print(f"Prefetched. Reuse with: PREFETCH_RUN={run_name} python serve.py")

    model_path = flyte.app.RunOutput(type="directory", run_name=run_name)
    if ARTIFACTS and _has_registry(MODEL.rsplit("/", 1)[-1]):
        model_path = flyte.app.ArtifactValue(name=MODEL.rsplit("/", 1)[-1], type="directory")
        print(f"Binding to the model artifact {MODEL.rsplit('/', 1)[-1]} (Union registry)")
    print(f"Deploying {MODEL_ID} on {GPU}…")
    app = flyte.serve(vllm_app.clone_with(name=vllm_app.name, model_path=model_path, model_hf_path=None))
    print(f"Console:  {app.url}")
    print(f"Endpoint: {app.endpoint}/v1")
    print(f"Try it:   flyte run benchmark.py sanity")
    return app


if __name__ == "__main__":
    deploy()
