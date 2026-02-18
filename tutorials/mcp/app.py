# /// script
# requires-python = "==3.12"
# dependencies = ["flyte==2.0.0b56"]
# ///

"""Flyte App deployment for the Recipe MCP Server."""

import os
import logging
import pathlib
from contextlib import asynccontextmanager

import flyte
from flyte.app import AppEnvironment, Domain, Scaling, Link
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.responses import Response
from starlette.requests import Request

from server import mcp


APP_NAME = os.getenv("APP_NAME")
APP_SUBDOMAIN = os.getenv("APP_SUBDOMAIN")
APP_PORT = int(os.getenv("APP_PORT", 8000))
REQUIRES_AUTH = os.getenv("REQUIRES_AUTH", "false").lower() == "true"

if not APP_NAME:
    raise ValueError("set an app name with export APP_NAME= is not set")

# -----------------
# App Environment
# -----------------

image = (
    flyte.Image.from_debian_base(name="recipe-mcp-server")
    .with_requirements("requirements.txt")
)

app_env = AppEnvironment(
    name=APP_NAME,
    port=APP_PORT,
    domain=Domain(subdomain=APP_SUBDOMAIN) if APP_SUBDOMAIN else None,
    include=["./server.py", "./tools/"],
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[flyte.Secret("SPOONACULAR_API_KEY")],
    requires_auth=REQUIRES_AUTH,
    scaling=Scaling(replicas=(1, 2)),
    env_vars={"APP_NAME": APP_NAME},
    links=[
        Link(
            path="/spoonacular/mcp",
            title="Streamable HTTP transport endpoint",
            is_relative=True,
        ),
        Link(
            path="/health",
            title="Health check endpoint",
            is_relative=True,
        )
    ],
)

@asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp.session_manager.run():
        yield


starlette_app = Starlette(
    lifespan=lifespan,
    routes=[Mount("/spoonacular", app=mcp.streamable_http_app())],
)

@starlette_app.route("/health")
async def health(request: Request) -> Response:
    return Response(content="OK", media_type="text/plain", status_code=200)


@app_env.server
async def server():
    import uvicorn

    uvicorn_server = uvicorn.Server(uvicorn.Config(starlette_app, port=APP_PORT))
    await uvicorn_server.serve()


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.INFO,
    )

    served_app = flyte.serve(app_env)
    print(f"Served app: {served_app.url}")
