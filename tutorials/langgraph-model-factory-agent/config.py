"""Task environments, images and secrets.

Five environments, so each step only asks for the credentials it actually uses:

    tools_env     list_candidates, fine_tune (the wrapper), promote, test_deployment. No secrets.
    gpu_env       run_eval and train_model. One T4 each. Step 1 runs on these alone.
    cpu_env       run_eval_cpu. A small CPU node; what the serving app looks like.
    agent_env     the ML engineer and the support agent. Needs the key for AGENT_MODEL's
                  provider. Steps 0, 2, 3, 4, 5 and 7.
    factory_env   the bake-off. Keys for every provider in FACTORY_PROVIDERS. Step 6.

Everything the environments read from `.env` is also passed to the tasks as `env_vars`
(see PROPAGATED below), because a child task's spec is serialized inside the parent's pod,
which has no `.env`.

Where the GPU work runs is one knob, ACCELERATOR (see below): a T4 or an L40s on a cluster,
the GPU in your Mac on an Apple Silicon devbox, or the CPU anywhere else.
"""

from __future__ import annotations

import os
import platform
import re

import flyte
from dotenv import load_dotenv

load_dotenv()  # API keys for `flyte run --local`; on the cluster, secrets arrive as env vars

# ── Your corner of a shared cluster ───────────────────────────────────────────

# In a workshop, thirty people run this against one project. FACTORY_TAG (say, your
# handle) namespaces the things that must be yours: the serving app, the production
# artifacts, and the environments that carry triggers. The GPU eval and training tasks
# stay shared on purpose, so the eval matrix is cached once for the whole room.
TAG = re.sub(r"[^a-z0-9-]+", "-", os.environ.get("FACTORY_TAG", "").lower()).strip("-")


def tagged(name: str) -> str:
    return f"{name}-{TAG}" if TAG else name


# ── Where the GPU work runs ──────────────────────────────────────────────────

# ACCELERATOR picks the pod shape for run_eval and train_model, and factory.device() picks
# the matching torch device at run time:
#
#     T4      a 16 GB CUDA card on a cluster. The numbers in the README. The default.
#     L40s    a 48 GB CUDA card on a cluster.
#     apple   the GPU in this Mac. The task runs natively on the Mac, PyTorch uses `mps`,
#             and the devbox still records the run. Needs `flyte start devbox --apple-silicon`.
#     cpu     no GPU. Slow for the chat models; fine for the encoder.
#
# On a Mac whose Flyte config points at a devbox the default is `apple`; a cluster config
# keeps the cluster default. The config is the one the CLI reads: `.flyte/config.yaml`
# next to the code, or the file named by FLYTECTL_CONFIG (or UCTL_CONFIG) when you keep
# a devbox config and a cluster config side by side. Set ACCELERATOR to override.


def _default_accelerator() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = os.environ.get("UCTL_CONFIG") or os.environ.get("FLYTECTL_CONFIG")
        cfg = os.path.join(here, cfg) if cfg and not os.path.isabs(cfg) else cfg
        try:
            if "localhost:30080" in open(cfg or os.path.join(here, ".flyte", "config.yaml")).read():
                return "apple"
        except OSError:
            pass
    return "T4"


ACCELERATOR = os.environ.get("ACCELERATOR") or _default_accelerator()
SHAPES = {
    "T4": flyte.Resources(cpu=3, memory="12Gi", gpu="T4:1", disk="30Gi"),
    "L40s": flyte.Resources(cpu=6, memory="32Gi", gpu="L40s:1", disk="32Gi"),
    # The pod only proxies a process on the Mac; the request reserves devbox capacity, not the
    # work. Size the run by the Mac's memory instead (the 1.5B LoRA peaks around 12 GB).
    "apple": flyte.Resources(cpu=1, memory="2Gi", gpu="AppleSilicon"),
    "cpu": flyte.Resources(cpu=2, memory="8Gi", disk="20Gi"),
}
if ACCELERATOR not in SHAPES:
    raise ValueError(f"ACCELERATOR={ACCELERATOR!r}; choose one of {list(SHAPES)}")
ON_MAC = ACCELERATOR == "apple"
# How the GPU is named in prompts, reports and the request's latency bar.
GPU_LABEL = {"T4": "T4", "L40s": "L40s", "apple": "Apple Silicon GPU", "cpu": "CPU"}[ACCELERATOR]
GPU_ARTICLE = {"T4": "a T4", "L40s": "an L40s", "apple": "the Mac's GPU", "cpu": "the CPU"}[ACCELERATOR]

# A run with no platform at all (`flyte run --local`, Colab) keeps what it promotes here: the
# weights of each version and a pointer to the current one. The support agent's `--router oss`
# loads the current version in-process from this directory.
LOCAL_DIR = os.environ.get("FACTORY_LOCAL_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".factory")

# Whether the platform has an artifact registry (Union does; a devbox does not). Decided at
# run time by tools.registry(): one failed lookup and the factory passes models around as
# task outputs instead. FACTORY_ARTIFACTS=0 forces that off anywhere.
FACTORY_ARTIFACTS = os.environ.get("FACTORY_ARTIFACTS", "1") not in ("0", "false", "no")

# ── Secrets ──────────────────────────────────────────────────────────────────

# On a shared cluster, prefix your secret names so they don't collide with someone else's
# (FLYTE_SECRET_PREFIX=SAGE_ turns ANTHROPIC_API_KEY into SAGE_ANTHROPIC_API_KEY). To use a
# secret with a different name entirely, set <NAME>_SECRET_NAME, e.g. OPENAI_API_KEY_SECRET_NAME=my-key.
SECRET_PREFIX = os.environ.get("FLYTE_SECRET_PREFIX", "")


def secret(name: str, env_var: str) -> flyte.Secret:
    key = os.environ.get(f"{name}_SECRET_NAME") or f"{SECRET_PREFIX}{name}"
    return flyte.Secret(key=key, as_env_var=env_var)


# One secret per model provider (see llm.py). A task only asks for the ones it can use.
PROVIDER_SECRETS = {
    "anthropic": secret("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    "openai": secret("OPENAI_API_KEY", "OPENAI_API_KEY"),
    "vllm": secret("VLLM_API_KEY", "VLLM_API_KEY"),
}
VLLM_SECRET = PROVIDER_SECRETS["vllm"]

AGENT_MODEL = os.environ.get("AGENT_MODEL", "anthropic:claude-haiku-4-5")
AGENT_PROVIDER = AGENT_MODEL.split(":", 1)[0]

# Every provider a task may be asked to use: the agent's own, plus FACTORY_PROVIDERS (the
# bake-off in step 6, or a `model=` override on any step). Add vllm once serve_model.py is
# deployed. A task only gets the secrets it lists, so a provider missing here fails on the
# cluster with an authentication error even though the key is in .env.
FACTORY_PROVIDERS = [p.strip() for p in os.environ.get("FACTORY_PROVIDERS", "").split(",") if p.strip()]
if AGENT_PROVIDER not in FACTORY_PROVIDERS:
    FACTORY_PROVIDERS.insert(0, AGENT_PROVIDER)
FACTORY_SECRETS = [PROVIDER_SECRETS[p] for p in FACTORY_PROVIDERS]

# Everything above that was read from the environment has to travel with the tasks. A
# child task's spec is serialized where it is launched, and for the agent's tools that is
# inside the agent's pod, which has no .env. Without this, a pod would fall back to the
# defaults (no secret prefix, the Anthropic provider) and request the wrong secrets.
_KNOBS = (
    "ACCELERATOR",
    "FACTORY_ARTIFACTS",
    "FLYTE_SECRET_PREFIX",
    "FACTORY_TAG",
    "AGENT_MODEL",
    "FACTORY_PROVIDERS",
    "FACTORY_DEPLOY",
    "FACTORY_APPROVAL",
    "FACTORY_MAX_STEPS",
    "VLLM_BASE_URL",
    "FACTORY_MAX_PARALLEL_GPU",
    "ROUTER_CPU",
    "ROUTER_MEMORY",
)
PROPAGATED = {k: v for k, v in os.environ.items() if (k in _KNOBS or k.endswith("_SECRET_NAME")) and v}
# The accelerator travels resolved, not as read: its default comes from this Mac's config
# file, which a pod cannot see, and a pod left to guess would ask for a T4 that a devbox
# does not have.
PROPAGATED["ACCELERATOR"] = ACCELERATOR

# How many GPU tools one run may have in flight at once. 0 means no limit: "evaluate these
# five" is five T4s. In a room of thirty people on one GPU pool, set it to 1 or 2 so a
# single run cannot take the whole pool; the calls still go out in one turn, they just queue.
# A Mac has one GPU, so there the default is 1: the calls still go out together, and run one
# after another. Raise it on a Mac with memory to spare.
MAX_PARALLEL_GPU = int(os.environ.get("FACTORY_MAX_PARALLEL_GPU", "1" if ON_MAC else "0") or 0)
GPU_TOOLS = {"run_eval", "fine_tune"}  # the tools that end up on a T4


def gpu_slots():
    """A semaphore for GPU tools, or None when there is no limit. One per run.

    A run with no platform (`flyte run --local`, Colab) has one process and one GPU, so
    its GPU tools take turns unless FACTORY_MAX_PARALLEL_GPU says otherwise.
    """
    import asyncio

    limit = MAX_PARALLEL_GPU
    if limit == 0 and (flyte.ctx() is None or flyte.ctx().mode == "local"):
        limit = 1
    return asyncio.Semaphore(limit) if limit > 0 else None


# ── Image ────────────────────────────────────────────────────────────────────

# Listed inline rather than via .with_requirements("requirements.txt"): that stores a
# relative path and re-resolves it inside the pod, where the file is not in the bundle.
# Keep this list and requirements.txt in sync.
image = flyte.Image.from_debian_base(name="langgraph-model-factory-agent").with_pip_packages(
    "flyteplugins-agents-langgraph>=2.8.1",
    "langchain-anthropic>=1.0",
    "langchain-openai>=1.0",
    "langgraph>=1.0",
    "python-dotenv",
    "fastapi",  # promote deploys router_app.py, which imports FastAPI at module scope
    "uvicorn",
)

# The factory floor: the same image plus the training stack. Used by the T4 tasks and the
# serving app. torch's CUDA wheels make this the slow build; it happens once per cluster.
ml_image = image.with_pip_packages(
    "torch>=2.4",
    "transformers>=4.45",
    "peft>=0.13",
    "trl>=0.12",
    "datasets>=3.0",
    "accelerate>=0.34",
)

# Whether promote() deploys the router app after publishing the artifact. Off for the
# bake-off, where four agents promoting four models should not deploy four apps.
FACTORY_DEPLOY = os.environ.get("FACTORY_DEPLOY", "1") not in ("0", "false", "no")

# Whether promote() pauses for a human to approve the deployment in the Flyte UI
# (a native `flyte.new_condition`). Off by default; turn it on for the live demo.
FACTORY_APPROVAL = os.environ.get("FACTORY_APPROVAL", "0") in ("1", "true", "yes")

# ── Environments ─────────────────────────────────────────────────────────────

# The expensive tools: run_eval and train_model. One GPU per call (ACCELERATOR says which);
# the agent can ask for several at once and on a cluster they run side by side.
gpu_env = flyte.TaskEnvironment(
    name="factory-gpu",
    image=ml_image,
    resources=SHAPES[ACCELERATOR],
    env_vars={**PROPAGATED},
)

# CPU inference: the same stack on a small CPU node, for evaluating models the way the
# serving app will run them (the app is a CPU pod). Training always goes to the GPU.
# Sized under the demo cluster's t3a.xlarge nodes. On a Mac it is the Mac's CPU: the task
# runs on the host like the GPU ones (a devbox container cannot run this stack natively).
cpu_env = flyte.TaskEnvironment(
    name="factory-cpu",
    image=ml_image,
    resources=SHAPES["apple"] if ON_MAC else SHAPES["cpu"],
    env_vars={**PROPAGATED, "FACTORY_DEVICE": "cpu"},
)

# The cheap tools: list_candidates, the fine_tune wrapper (which launches train_model on
# the GPU or CPU env), promote. No secrets. On a Mac these run on the host as well: promote
# deploys the app by uploading its spec to a signed URL, and a devbox signs URLs for
# `localhost:30002`, which the Mac can reach and a pod cannot.
tools_env = flyte.TaskEnvironment(
    name=tagged("factory-tools"),
    image=image,
    resources=SHAPES["apple"] if ON_MAC else flyte.Resources(cpu=1, memory="2Gi"),
    env_vars={**PROPAGATED, "FACTORY_DEPLOY": "1" if FACTORY_DEPLOY else "0"},
    depends_on=[gpu_env, cpu_env],
)

# The agents: the ML engineer (steps 2, 4, 5 and 7) and the support agent (steps 0 and 3).
# One key, for AGENT_MODEL's provider.
agent_env = flyte.TaskEnvironment(
    name=tagged("factory-agent"),
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[*FACTORY_SECRETS],
    env_vars={**PROPAGATED, "AGENT_MODEL": AGENT_MODEL},
    depends_on=[tools_env, gpu_env, cpu_env],
)

# The bake-off. Step 6: the same agent driven by several models, compared.
# Deploys are off here, whatever .env says: four agents promoting four models must not
# redeploy one app four times (and race each other for which version answers).
factory_env = flyte.TaskEnvironment(
    name=tagged("factory-bakeoff"),
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[*FACTORY_SECRETS],
    env_vars={
        **PROPAGATED,
        "AGENT_MODEL": AGENT_MODEL,
        "FACTORY_DEPLOY": "0",
    },
    depends_on=[tools_env, gpu_env, cpu_env],
)
