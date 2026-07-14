"""Task environments: image, compute, and secrets."""

from dotenv import load_dotenv

import flyte

load_dotenv()  # ANTHROPIC_API_KEY for `flyte run --local`; ignored on the cluster

# Packages are listed inline rather than via .with_requirements("requirements.txt"):
# that stores a *relative path* and re-resolves it at runtime, which blows up inside
# a pod (cwd is not this directory, and the file is not in the code bundle) — e.g.
# when step 5's app pod launches the `answer` task. Keep this list and
# requirements.txt in sync; requirements.txt is for your local venv.
image = flyte.Image.from_debian_base(name="code-mode-analysis").with_pip_packages(
    "duckdb>=1.1.0",
    "pydantic-monty",  # the Monty sandbox runtime
    "anthropic",  # the LLM client (see llm.py)
    "python-dotenv",
    "fastapi",  # step 5's chat app
    "uvicorn",
)

# Step 1 has no LLM in it, so no secret — you can run it before registering a key.
sandbox_env = flyte.TaskEnvironment(
    name="code-mode-sandbox",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),  # a month of trips is ~600MB
)

# Steps 2-5. The secret must live in the project/domain you run in:
#   flyte create secret ANTHROPIC_API_KEY -p flytesnacks -d development
env = flyte.TaskEnvironment(
    name="code-mode-analysis",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[flyte.Secret(key="SAGE_ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY")],
)
