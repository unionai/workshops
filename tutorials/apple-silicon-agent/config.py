"""Flyte environments for an agent whose model runs on your Mac's GPU.

Three environments, one knob each:

    llm app     serve_model.py. ACCELERATOR picks where the model runs: `apple` (this Mac, mps),
                `T4`/`L40s` (a cluster GPU) or `cpu`.
    tools_env   tools.py. Plain CPU tasks over a SQLite database. Each tool call is a Flyte action.
    agent_env   agent.py. The LangGraph loop. AGENT_MODEL picks the brain:

                    local                       the app above, via its OpenAI-compatible API (default)
                    anthropic:claude-haiku-4-5  Claude, if you have a key (ANTHROPIC_API_KEY)
                    openai:gpt-4.1-mini         OpenAI, if you have a key (OPENAI_API_KEY)

Everything read from the environment here is also handed to the tasks as `env_vars`, because the
agent task launches the tool tasks from inside its own container, which has no `.env`.
"""

from __future__ import annotations

import os
import platform

import flyte
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")  # Qwen/Qwen3-1.7B fits a 16 GB Mac; see the README
APP_NAME = os.getenv("APP_NAME", "apple-silicon-agent-llm")
AGENT_MODEL = os.getenv("AGENT_MODEL", "local")
# Where the agent finds the model when it is not running inside the devbox (flyte run --local):
# the LLM app's public URL, or a bare `uvicorn openai_server:app --port 8000`.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")


def _default_accelerator() -> str:
    return "apple" if platform.system() == "Darwin" and platform.machine() == "arm64" else "cpu"


ACCELERATOR = os.getenv("ACCELERATOR", _default_accelerator())

SHAPES = {
    "apple": flyte.Resources(cpu=1, memory="2Gi", gpu="AppleSilicon"),
    "T4": flyte.Resources(cpu=4, memory="24Gi", gpu="T4:1", disk="32Gi"),
    "L40s": flyte.Resources(cpu=6, memory="32Gi", gpu="L40s:1", disk="32Gi"),
    "cpu": flyte.Resources(cpu=2, memory="8Gi"),
}
if ACCELERATOR not in SHAPES:
    raise ValueError(f"ACCELERATOR={ACCELERATOR!r}; choose one of {list(SHAPES)}")

SERVER_PIP = ("torch>=2.4", "transformers>=4.51", "fastapi", "uvicorn", "python-dotenv")

PROPAGATED = {
    "ACCELERATOR": ACCELERATOR,
    "MODEL_NAME": MODEL_NAME,
    "APP_NAME": APP_NAME,
    "AGENT_MODEL": AGENT_MODEL,
    "LLM_BASE_URL": LLM_BASE_URL,
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}

# A task only asks for the secret its provider needs; `local` needs none.
PROVIDER_SECRETS = {
    "anthropic": flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
    "openai": flyte.Secret(key="OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
}
AGENT_PROVIDER = AGENT_MODEL.split(":", 1)[0]
AGENT_SECRETS = [PROVIDER_SECRETS[AGENT_PROVIDER]] if AGENT_PROVIDER in PROVIDER_SECRETS else []


def llm_base_url() -> str:
    """The OpenAI-compatible base URL of the LLM app, from wherever this code is running.

    Apps are handed the in-cluster address pattern for their neighbours; tasks on a devbox are not, and
    a pod cannot resolve the devbox's public `*.localhost` names, so inside a pod the public URL is
    swapped for the in-cluster service name. `LLM_BASE_URL` overrides everything (flyte run --local).
    """
    from urllib.parse import urlparse

    if LLM_BASE_URL:
        return LLM_BASE_URL.rstrip("/").removesuffix("/v1") + "/v1"
    from flyte.app._app_environment import INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR

    pattern = os.getenv(INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR)
    if pattern:
        return pattern.format(app_fqdn=APP_NAME).rstrip("/") + "/v1"
    import flyte.remote

    endpoint = flyte.remote.App.get(APP_NAME).endpoint
    host = urlparse(endpoint).hostname or ""
    if host.endswith(".localhost") and os.getenv("KUBERNETES_SERVICE_HOST"):
        endpoint = f"http://{host.removesuffix('.localhost')}.flyte.svc.cluster.local"
    return endpoint.rstrip("/") + "/v1"
