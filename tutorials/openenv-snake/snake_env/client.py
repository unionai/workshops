"""OpenEnv client for the Snake environment."""

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from snake_env.models import SnakeAction, SnakeObservation, SnakeState


class SnakeEnv(EnvClient[SnakeAction, SnakeObservation, SnakeState]):
    """Client that connects to a Snake OpenEnv server."""

    def _step_payload(self, action: SnakeAction) -> dict:
        return {"direction": action.direction}

    def _parse_result(self, payload: dict) -> StepResult[SnakeObservation]:
        obs_data = payload.get("observation", payload)
        observation = SnakeObservation(
            grid=obs_data.get("grid", []),
            snake=[tuple(p) for p in obs_data.get("snake", [])],
            apple=tuple(obs_data.get("apple", (0, 0))),
            score=obs_data.get("score", 0),
            death_reason=obs_data.get("death_reason", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SnakeState:
        return SnakeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            score=payload.get("score", 0),
            grid_size=payload.get("grid_size", 10),
        )
