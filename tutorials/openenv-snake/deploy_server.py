"""Deploy the Snake OpenEnv server as a Union app."""

import sys
import os

import fastapi

import flyte
from flyte.app import Timeouts
from flyte.app.extras import FastAPIAppEnvironment

app = fastapi.FastAPI()

env = FastAPIAppEnvironment(
    name="snake-env",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages(
        "openenv-core",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
    timeouts=Timeouts(request=3600),  # 1 hour — long RL episodes
    include=[
        "snake_env/__init__.py",
        "snake_env/models.py",
        "snake_env/client.py",
        "snake_env/server/__init__.py",
        "snake_env/server/environment.py",
    ],
)


@env.on_startup
async def startup():
    """Mount the OpenEnv app at startup (avoids pickle issues with create_app)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from openenv.core.env_server import create_app

    from snake_env.models import SnakeAction, SnakeObservation
    from snake_env.server.environment import SnakeEnvironment

    openenv_app = create_app(
        SnakeEnvironment,
        SnakeAction,
        SnakeObservation,
        env_name="snake",
        max_concurrent_envs=8,
    )
    app.mount("/", openenv_app)