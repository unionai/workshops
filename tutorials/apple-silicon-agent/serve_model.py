"""Serve the agent's model on your Mac's GPU as a Flyte app, OpenAI-compatible.

    python serve_model.py                   # deploy; prints the endpoint
    ACCELERATOR=L40s python serve_model.py  # the same app on a cluster GPU

The agent (agent.py) finds this app by name. `openai_server.py` is the same small server as the
`apple-silicon-serving` tutorial: transformers on `mps`, Hermes-style tool calls parsed back out,
Qwen3 thinking off by default.
"""

from __future__ import annotations

import logging

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

from config import ACCELERATOR, APP_NAME, MODEL_NAME, PROPAGATED, SERVER_PIP, SHAPES
from openai_server import build_app

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = build_app(MODEL_NAME, ACCELERATOR)

app_env = FastAPIAppEnvironment(
    name=APP_NAME,
    app=app,
    description=f"{MODEL_NAME} for the SQL agent, OpenAI-compatible, on {ACCELERATOR}",
    image=flyte.Image.from_debian_base().with_pip_packages(*SERVER_PIP),
    resources=SHAPES[ACCELERATOR],
    requires_auth=False,
    env_vars=PROPAGATED,
    links=[flyte.app.Link(path="/v1/models", title="Models", is_relative=True)],
)

if __name__ == "__main__":
    flyte.init_from_config()
    deployed = flyte.serve(app_env)
    print(f"Console:  {deployed.url}")
    print(f"Endpoint: {deployed.endpoint}")
    print(f"\nFor flyte run --local:  export LLM_BASE_URL={deployed.endpoint}")
