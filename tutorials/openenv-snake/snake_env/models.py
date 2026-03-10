"""Data models for the Snake OpenEnv environment."""

from typing import List, Optional, Tuple

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


class SnakeAction(Action):
    """Action: choose a direction to move the snake."""

    direction: str = "RIGHT"  # UP, DOWN, LEFT, RIGHT


class SnakeObservation(Observation):
    """What the agent sees after each step."""

    grid: List[List[str]]  # "." empty, "S" body, "H" head, "A" apple
    snake: List[Tuple[int, int]] = Field(default_factory=list)
    apple: Tuple[int, int] = (0, 0)
    score: int = 0
    death_reason: str = ""  # "wall", "self", or ""


class SnakeState(State):
    """Metadata about the current episode."""

    score: int = 0
    grid_size: int = 10
