"""Settings and task environments.

Every setting is an environment variable, read here at import time. For a local run that is your
shell (or `.env`). For a devbox or cluster run, `flyte run` bakes the values into the task
environments below, so tasks in the cluster see the values your shell had.
"""

from __future__ import annotations

import os

import flyte
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- where the target VM is

VM_HOST = os.environ.get("RPA_VM_HOST", "localhost")
SSH_PORT = int(os.environ.get("RPA_SSH_PORT", "2222"))
SSH_USER = os.environ.get("RPA_SSH_USER", "rpa")
# The private key: its text (a Flyte secret in the cluster) or a path to it (local runs).
SSH_KEY = os.environ.get("RPA_SSH_KEY", "")
SSH_KEY_PATH = os.environ.get("RPA_SSH_KEY_PATH", "")

CUA_URL = os.environ.get("RPA_CUA_URL", f"http://{VM_HOST}:8000")
ERP_URL = os.environ.get("RPA_ERP_URL", f"http://{VM_HOST}:8001")
ERP_API_KEY = os.environ.get("ACME_API_KEY", "demo-key")
# The portal as the task sees it (Playwright) and as the VM's own desktop browser sees it.
PORTAL_URL = os.environ.get("RPA_PORTAL_URL", f"http://{VM_HOST}:8002")
PORTAL_URL_ON_VM = "http://localhost:8002"
PORTAL_USER = os.environ.get("PORTAL_USER", "acme-buyer")
PORTAL_PASSWORD = os.environ.get("PORTAL_PASSWORD", "demo-password")

# The VM's desktop and folders (see target_vm/).
DISPLAY = ":1"
SCREEN_W, SCREEN_H = 1280, 800
SHARE_DIR = "/srv/share"
DOWNLOADS_DIR = "/home/rpa/Downloads"

# ---------------------------------------------------------------- models

# "anthropic": the Claude API with ANTHROPIC_API_KEY. "vertex": Claude on Vertex AI with Google ADC.
LLM_PROVIDER = os.environ.get("RPA_LLM_PROVIDER", "anthropic")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")
VERTEX_REGION = os.environ.get("VERTEX_REGION", "global")
ORCHESTRATOR_MODEL = os.environ.get("RPA_ORCHESTRATOR_MODEL", "claude-opus-5-5")
COMPUTER_USE_MODEL = os.environ.get("RPA_COMPUTER_USE_MODEL", "claude-opus-5-5")

# Ask a person before the bot commits more than this many price changes.
APPROVAL_THRESHOLD = int(os.environ.get("RPA_APPROVAL_THRESHOLD", "10"))

# ---------------------------------------------------------------- task environments

ENV_VARS = {
    "RPA_VM_HOST": VM_HOST,
    "RPA_SSH_PORT": str(SSH_PORT),
    "RPA_LLM_PROVIDER": LLM_PROVIDER,
    "VERTEX_PROJECT": VERTEX_PROJECT,
    "VERTEX_REGION": VERTEX_REGION,
    "RPA_ORCHESTRATOR_MODEL": ORCHESTRATOR_MODEL,
    "RPA_COMPUTER_USE_MODEL": COMPUTER_USE_MODEL,
    "RPA_APPROVAL_THRESHOLD": str(APPROVAL_THRESHOLD),
}

# In the cluster these come from Flyte secrets; in a local run, from your environment.
SECRETS = [flyte.Secret(key="rpa-vm-ssh-key", as_env_var="RPA_SSH_KEY")]
if LLM_PROVIDER == "anthropic":
    SECRETS.append(flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"))

PACKAGES = ("anthropic[vertex]==1.13.0", "asyncssh==2.24.1", "httpx", "openpyxl", "pillow", "python-dotenv")

base_image = flyte.Image.from_debian_base(python_version=(3, 12), name="agentic-rpa").with_pip_packages(*PACKAGES)

browser_image = (
    flyte.Image.from_debian_base(python_version=(3, 12), name="agentic-rpa-browser")
    .with_pip_packages(*PACKAGES, "playwright==1.63.0")
    # A shared path: the image builds as root, but tasks run as the `flyte` user.
    .with_env_vars({"PLAYWRIGHT_BROWSERS_PATH": "/opt/ms-playwright"})
    .with_commands(["playwright install --with-deps chromium && chmod -R a+rX /opt/ms-playwright"])
)

# Rung 5: headless Chromium in the task's own container, pointed at the portal on the VM.
browser_env = flyte.TaskEnvironment(
    name="agentic-rpa-browser",
    image=browser_image,
    env_vars=ENV_VARS,
    secrets=SECRETS,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)

# Rung 6: the computer-use loop. Light on CPU, long-running, and the only environment that records.
cu_env = flyte.TaskEnvironment(
    name="agentic-rpa-computer-use",
    image=base_image,
    env_vars=ENV_VARS,
    secrets=SECRETS,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
)

# The orchestrator and the cheap rungs (API, system state, shell, files). It calls the other two.
env = flyte.TaskEnvironment(
    name="agentic-rpa",
    image=base_image,
    env_vars=ENV_VARS,
    secrets=SECRETS,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[browser_env, cu_env],
    include=("instructions.md",),  # non-Python files aren't bundled by default
)
