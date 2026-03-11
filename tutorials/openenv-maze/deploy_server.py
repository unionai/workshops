"""Deploy the Maze OpenEnv server as a Union app."""

import sys
import os

import fastapi

import flyte
from flyte.app import Timeouts
from flyte.app.extras import FastAPIAppEnvironment

app = fastapi.FastAPI()

env = FastAPIAppEnvironment(
    name="maze-env",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages(
        "openenv-core",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
    timeouts=Timeouts(request=3600),
    include=[
        "maze_env/__init__.py",
        "maze_env/models.py",
        "maze_env/server/__init__.py",
        "maze_env/server/environment.py",
    ],
)


@env.on_startup
async def startup():
    """Mount the OpenEnv app at startup (avoids pickle issues with create_app)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from openenv.core.env_server import create_app

    from maze_env.models import MazeAction, MazeObservation
    from maze_env.server.environment import MazeEnvironment

    openenv_app = create_app(
        MazeEnvironment,
        MazeAction,
        MazeObservation,
        env_name="maze",
        max_concurrent_envs=8,
    )
    app.mount("/", openenv_app)