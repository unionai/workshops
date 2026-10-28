"""Serve an open LLM on your Mac's GPU as a Flyte app, OpenAI-compatible.

    python serve.py                         # deploy; prints the endpoint
    ACCELERATOR=L40s python serve.py        # the same app on a cluster GPU

    curl -s $URL/v1/models
    curl -s $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{
      "model": "Qwen/Qwen3-1.7B",
      "messages": [{"role": "user", "content": "Write a haiku about a laptop."}]
    }'

Or with any OpenAI client:

    from openai import OpenAI
    client = OpenAI(base_url=f"{URL}/v1", api_key="not-used")
    client.chat.completions.create(model="Qwen/Qwen3-1.7B", messages=[...], stream=True)

The devbox gives the app its URL, logs and port forwarding exactly as it does for a container; only
the process is on the Mac. The model downloads into the Mac's own Hugging Face cache on first start.
"""

from __future__ import annotations

import logging

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

# Imported at module top on purpose: the app bundle only carries modules imported at serialization.
from config import ACCELERATOR, APP_NAME, MODEL_NAME, PIP, PROPAGATED, SHAPES
from openai_server import build_app

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = build_app(MODEL_NAME, ACCELERATOR)

app_env = FastAPIAppEnvironment(
    name=APP_NAME,
    app=app,
    description=f"{MODEL_NAME}, OpenAI-compatible, on {ACCELERATOR}",
    image=flyte.Image.from_debian_base().with_pip_packages(*PIP),
    resources=SHAPES[ACCELERATOR],
    requires_auth=False,
    env_vars=PROPAGATED,
    links=[flyte.app.Link(path="/v1/models", title="Models", is_relative=True), flyte.app.Link(path="/docs", title="OpenAPI docs", is_relative=True)],
)

if __name__ == "__main__":
    flyte.init_from_config()
    deployed = flyte.serve(app_env)
    print(f"Console:  {deployed.url}")
    print(f"Endpoint: {deployed.endpoint}")
    print(f"\ncurl -s {deployed.endpoint}/v1/models")
