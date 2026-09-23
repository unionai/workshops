"""Task environments, secrets, and the one line that turns Grafana on.

Five environments, so each step only asks for the credentials it actually uses:

    tools_env     list_candidates, fine_tune (the wrapper), promote, test_deployment. No secrets.
    gpu_env       run_eval and train_model. One T4 each. Step 1 runs on these alone.
    cpu_env       run_eval_cpu. A small CPU node; what the serving app looks like.
    agent_env     the ML engineer. Needs the key for AGENT_MODEL's provider. Step 2.
    observed_env  the engineer (and the support agent), observed by Grafana. Adds the
                  Grafana token when configured. Steps 0, 3, 4, 5, 7, 8.
    factory_env   the bake-off. Keys for every provider in FACTORY_PROVIDERS. Step 7.

Grafana configuration is four plain values and one secret. The four go in `.env` (or the
environment) and are baked into the task as `env_vars`; the token is a Flyte secret:

    GRAFANA_HOST=https://<stack>.grafana.net
    AGENTO11Y_ENDPOINT=<API URL from Agent Observability > Configuration>
    AGENTO11Y_AUTH_TENANT_ID=<Instance ID from the same page>
    OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-<region>.grafana.net/otlp
    TEMPO_DATASOURCE_UID=grafanacloud-traces          # optional; this is the Cloud default

    flyte create secret GRAFANA_TOKEN --value <glc_... token with sigil:write + traces:write>

Locally, put GRAFANA_TOKEN in `.env` too. `config.py` derives the OTLP basic-auth header
from the tenant id and token, so you never paste a base64 string anywhere.
"""

from __future__ import annotations

import base64
import os
import re

import flyte
from dotenv import load_dotenv

load_dotenv()  # API keys for `flyte run --local`; on the cluster, secrets arrive as env vars

# ── Grafana: values, derived header, module-scope init ───────────────────────

GRAFANA_HOST = os.environ.get("GRAFANA_HOST", "").rstrip("/")
AGENTO11Y_ENDPOINT = os.environ.get("AGENTO11Y_ENDPOINT", "")
AGENTO11Y_TENANT = os.environ.get("AGENTO11Y_AUTH_TENANT_ID", "")
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
TEMPO_DATASOURCE_UID = os.environ.get("TEMPO_DATASOURCE_UID", "grafanacloud-traces")
GRAFANA_TOKEN = os.environ.get("GRAFANA_TOKEN", "")

GRAFANA_CONFIGURED = bool(AGENTO11Y_ENDPOINT and AGENTO11Y_TENANT)

# These four are not secret; they ride along as plain env vars on the observed tasks.
GRAFANA_ENV_VARS = {
    "GRAFANA_HOST": GRAFANA_HOST,
    "AGENTO11Y_ENDPOINT": AGENTO11Y_ENDPOINT,
    "AGENTO11Y_AUTH_TENANT_ID": AGENTO11Y_TENANT,
    "OTEL_EXPORTER_OTLP_ENDPOINT": OTLP_ENDPOINT,
    "TEMPO_DATASOURCE_UID": TEMPO_DATASOURCE_UID,
    # agento11y defaults the auth mode to "none", in which case the token is never sent and
    # Grafana Cloud answers 401. Cloud wants Basic auth with the instance id as the username.
    "AGENTO11Y_AUTH_MODE": "basic",
    "AGENTO11Y_PROTOCOL": "http",
}

if GRAFANA_TOKEN:
    # Two channels, one token. Generations go to Agent Observability (agento11y reads
    # AGENTO11Y_AUTH_TOKEN); spans go to Tempo over OTLP, which wants a pre-built Basic header.
    # Grafana Cloud wants HTTP and Basic auth; agento11y defaults to gRPC and no auth, which
    # comes back as a 401 with the token never sent. Set here (not only as task env_vars) so
    # a local run behaves like a pod.
    os.environ.setdefault("AGENTO11Y_AUTH_MODE", "basic")
    os.environ.setdefault("AGENTO11Y_PROTOCOL", "http")
    os.environ.setdefault("AGENTO11Y_AUTH_TOKEN", GRAFANA_TOKEN)
    _basic = base64.b64encode(f"{AGENTO11Y_TENANT}:{GRAFANA_TOKEN}".encode()).decode()
    os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", f"Authorization=Basic {_basic}")

if GRAFANA_CONFIGURED and GRAFANA_TOKEN:
    # Module scope on purpose. The Flyte task span opens before the task body runs, and the
    # binding that names this run as the Grafana "conversation" rides on that span. Calling
    # init() inside a task would miss it. This is the entire Grafana integration.
    from flyteplugins.agento11y import init as _grafana_init

    _grafana_init(service_name="model-factory-agent")

# Links rendered on the task in the Flyte UI: one to this run's conversation in Agent
# Observability, one to its spans in Tempo. Skipped when there is no stack to point at.
if GRAFANA_HOST:
    from flyteplugins.agento11y import GrafanaAgentObservability
    from flyteplugins.otel.grafana import GrafanaTrace

    GRAFANA_LINKS = (
        GrafanaAgentObservability(host=GRAFANA_HOST),
        GrafanaTrace(host=GRAFANA_HOST, datasource_uid=TEMPO_DATASOURCE_UID),
    )
else:
    GRAFANA_LINKS = ()

# ── Your corner of a shared cluster ───────────────────────────────────────────

# In a workshop, thirty people run this against one project. FACTORY_TAG (say, your
# handle) namespaces the things that must be yours: the serving app, the production
# artifacts, and the environments that carry triggers. The GPU eval and training tasks
# stay shared on purpose, so the eval matrix is cached once for the whole room.
TAG = re.sub(r"[^a-z0-9-]+", "-", os.environ.get("FACTORY_TAG", "").lower()).strip("-")


def tagged(name: str) -> str:
    return f"{name}-{TAG}" if TAG else name


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
# Only requested when Grafana is configured, so steps 2-4 also run without a stack (they
# just print that nothing is being exported).
GRAFANA_SECRETS = [secret("GRAFANA_TOKEN", "GRAFANA_TOKEN")] if GRAFANA_CONFIGURED else []
VLLM_SECRET = PROVIDER_SECRETS["vllm"]

AGENT_MODEL = os.environ.get("AGENT_MODEL", "anthropic:claude-haiku-4-5")
AGENT_PROVIDER = AGENT_MODEL.split(":", 1)[0]

# Every provider a task may be asked to use: the agent's own, plus FACTORY_PROVIDERS (the
# bake-off in step 7, or a `model=` override on any step). Add vllm once serve_model.py is
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
    "FLYTE_SECRET_PREFIX",
    "FACTORY_TAG",
    "AGENT_MODEL",
    "FACTORY_PROVIDERS",
    "FACTORY_DEPLOY",
    "FACTORY_APPROVAL",
    "FACTORY_MAX_STEPS",
    "VLLM_BASE_URL",
    "ROUTER_CPU",
    "ROUTER_MEMORY",
)
PROPAGATED = {k: v for k, v in os.environ.items() if (k in _KNOBS or k.endswith("_SECRET_NAME")) and v}

# ── Image ────────────────────────────────────────────────────────────────────

# Listed inline rather than via .with_requirements("requirements.txt"): that stores a
# relative path and re-resolves it inside the pod, where the file is not in the bundle.
# Keep this list and requirements.txt in sync.
image = (
    flyte.Image.from_debian_base(name="langgraph-grafana-agent")
    # flyteplugins-agento11y is installed from git, which needs git in the image.
    .with_apt_packages("git")
    .with_pip_packages(
        "flyteplugins-agents-langgraph>=2.8.1",
        "flyteplugins-otel>=2.8.1",
        "flyteplugins-agento11y[langgraph] @ git+https://github.com/flyteorg/flyte-sdk.git#subdirectory=plugins/agento11y",
        "langchain-anthropic>=1.0",
        "langchain-openai>=1.0",
        "langgraph>=1.0",
        "python-dotenv",
        "fastapi",  # promote deploys router_app.py, which imports FastAPI at module scope
        "uvicorn",
    )
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

# The expensive tools: run_eval and train_model. One T4 per call; the agent can ask for
# several at once and they run side by side.
gpu_env = flyte.TaskEnvironment(
    name="factory-gpu",
    image=ml_image,
    resources=flyte.Resources(cpu=3, memory="12Gi", gpu="T4:1", disk="30Gi"),
    env_vars={**PROPAGATED},
)

# CPU inference: the same stack on a small CPU node, for evaluating models the way the
# serving app will run them (the app is a CPU pod). Training always goes to the T4.
# Sized under the demo cluster's t3a.xlarge nodes.
cpu_env = flyte.TaskEnvironment(
    name="factory-cpu",
    image=ml_image,
    resources=flyte.Resources(cpu=2, memory="8Gi", disk="20Gi"),
    env_vars={**PROPAGATED, "FACTORY_DEVICE": "cpu"},
)

# The cheap tools: list_candidates, the fine_tune wrapper (which launches train_model on
# the GPU or CPU env), promote. No secrets.
tools_env = flyte.TaskEnvironment(
    name=tagged("factory-tools"),
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    env_vars={**PROPAGATED, "FACTORY_DEPLOY": "1" if FACTORY_DEPLOY else "0"},
    depends_on=[gpu_env, cpu_env],
)

# The agent, plain. Steps 0 and 1.
agent_env = flyte.TaskEnvironment(
    name=tagged("factory-agent"),
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[*FACTORY_SECRETS],
    env_vars={**PROPAGATED, "AGENT_MODEL": AGENT_MODEL},
    depends_on=[tools_env, gpu_env, cpu_env],
)

# The agent, observed. Steps 2 and 3.
observed_env = flyte.TaskEnvironment(
    name=tagged("factory-agent-observed"),
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[*FACTORY_SECRETS, *GRAFANA_SECRETS],
    env_vars={**PROPAGATED, **GRAFANA_ENV_VARS, "AGENT_MODEL": AGENT_MODEL},
    depends_on=[tools_env, gpu_env, cpu_env],
)

# The bake-off. Step 7: the same agent driven by several models, compared.
factory_env = flyte.TaskEnvironment(
    name=tagged("factory-bakeoff"),
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[*FACTORY_SECRETS, *GRAFANA_SECRETS],
    env_vars={
        **PROPAGATED,
        **GRAFANA_ENV_VARS,
        "AGENT_MODEL": AGENT_MODEL,
        "FACTORY_DEPLOY": "1" if FACTORY_DEPLOY else "0",
    },
    depends_on=[tools_env, gpu_env, cpu_env],
)
