"""FastAPI app serving the Maze environment via OpenEnv protocol."""

import sys
import os

# Add parent dirs so maze_env is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import fastapi

import flyte
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
    include=["../../maze_env/**/*.py"],
)


@env.on_startup
async def startup():
    """Mount the OpenEnv app at startup (avoids pickle issues with create_app)."""
    for candidate in [
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "/root",
    ]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

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


if __name__ == "__main__":
    import uvicorn

    from openenv.core.env_server import create_app

    from maze_env.models import MazeAction, MazeObservation
    from maze_env.server.environment import MazeEnvironment

    local_app = create_app(
        MazeEnvironment,
        MazeAction,
        MazeObservation,
        env_name="maze",
        max_concurrent_envs=8,
    )
    uvicorn.run(local_app, host="0.0.0.0", port=8000)
