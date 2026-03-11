"""Pydantic models for the Maze OpenEnv environment."""

from typing import List, Tuple

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class MazeAction(Action):
    """Action: choose a direction to move in the maze."""
    direction: str = "RIGHT"


class MazeObservation(Observation):
    """What the agent sees after each step."""
    grid: List[List[str]] = Field(default_factory=list)
    agent_pos: Tuple[int, int] = (1, 1)
    exit_pos: Tuple[int, int] = (5, 5)
    steps_taken: int = 0


class MazeState(State):
    """Metadata about the current maze episode."""
    maze_seed: int = 0
    optimal_path_length: int = 0
