"""The knobs. Change MODEL for a new model; everything else follows.

    MODEL        the Hugging Face repo to serve
    MODEL_ID     the name the OpenAI-compatible API answers to
    APP_NAME     the Flyte app name (lowercase, DNS-safe)
    GPU          which pod shape to serve on: L40s (default) or A100
    API_KEY      optional: the name of a Flyte secret holding a bearer token for the endpoint
    ARTIFACTS    on Union, bind the app to the prefetched model artifact by name (default on; falls back to the run output elsewhere)

Runs on the Union demo cluster or a CUDA devbox (`flyte start devbox --gpu`). vLLM needs an
NVIDIA GPU; see the README for what an Apple Silicon devbox can run.
"""

from __future__ import annotations

import os

import flyte
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "Qwen/Qwen3.5-4B")
MODEL_ID = os.getenv("MODEL_ID", "qwen3.5-4b")
APP_NAME = os.getenv("APP_NAME", "qwen35-4b-vllm")
GPU = os.getenv("GPU", "L40s")
API_KEY = os.getenv("API_KEY", "")  # e.g. VLLM_API_KEY, a secret you created with `flyte create secret`

# Union's artifact registry (model cards, versions, lineage) is used when it is there and skipped
# when it is not, so everything runs on open-source Flyte too. ARTIFACTS=0 forces it off.
ARTIFACTS = os.getenv("ARTIFACTS", "1") not in ("0", "false", "no")

SHAPES = {
    "L40s": flyte.Resources(cpu=6, memory="32Gi", gpu="L40s:1", disk="60Gi"),
    "A100": flyte.Resources(cpu=8, memory="48Gi", gpu="A100:1", disk="60Gi"),
}
if GPU not in SHAPES:
    raise ValueError(f"GPU={GPU!r}; choose one of {list(SHAPES)}")

# Child tasks are serialized inside the parent's pod, which has no .env: hand these along.
PROPAGATED = {"MODEL": MODEL, "MODEL_ID": MODEL_ID, "APP_NAME": APP_NAME, "GPU": GPU, "API_KEY": API_KEY}
SECRETS = [flyte.Secret(key=API_KEY, as_env_var="VLLM_API_KEY")] if API_KEY else []


def endpoint_for(app_name: str = APP_NAME) -> str:
    """The OpenAI-compatible base URL of the app, from wherever this code runs.

    `LLM_BASE_URL` overrides everything (for `flyte run --local` against a public URL). Inside a pod,
    Flyte hands apps the in-cluster address pattern for their neighbours; tasks ask the control plane
    for the app's endpoint. On a devbox a pod cannot resolve the public `*.localhost` name, so it is
    swapped for the in-cluster service name.
    """
    from urllib.parse import urlparse

    override = os.getenv("LLM_BASE_URL", "")
    if override:
        return override.rstrip("/").removesuffix("/v1") + "/v1"
    from flyte.app._app_environment import INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR

    pattern = os.getenv(INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR)
    if pattern:
        return pattern.format(app_fqdn=app_name).rstrip("/") + "/v1"
    import flyte.remote

    endpoint = flyte.remote.App.get(app_name).endpoint
    host = urlparse(endpoint).hostname or ""
    if host.endswith(".localhost") and os.getenv("KUBERNETES_SERVICE_HOST"):
        endpoint = f"http://{host.removesuffix('.localhost')}.flyte.svc.cluster.local"
    return endpoint.rstrip("/") + "/v1"


def api_key() -> str:
    return os.getenv("VLLM_API_KEY") or "not-used"
