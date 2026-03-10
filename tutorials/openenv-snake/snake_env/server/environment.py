"""Snake game environment implementing the OpenEnv interface."""

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server import Environment

from snake_env.models import SnakeAction, SnakeObservation, SnakeState


class SnakeEnvironment(Environment[SnakeAction, SnakeObservation, SnakeState]):
    """10x10 Snake game with wall/self collision and apple collection."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._grid_size = 10
        self._snake = []
        self._apple = (0, 0)
        self._direction = "RIGHT"
        self._done = False
        self._score = 0
        self._step_count = 0
        self._episode_id = ""
        self._death_reason = ""
        self._prev_distance = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> SnakeObservation:
        if seed is not None:
            random.seed(seed)

        self._episode_id = episode_id or str(uuid4())
        mid = self._grid_size // 2
        self._snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self._direction = "RIGHT"
        self._done = False
        self._score = 0
        self._step_count = 0
        self._death_reason = ""
        self._place_apple()
        self._prev_distance = self._manhattan_distance()

        return self._make_observation(reward=0.0)

    def step(
        self,
        action: SnakeAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SnakeObservation:
        if self._done:
            return self._make_observation(reward=0.0)

        # Prevent 180-degree reversal
        opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        direction = action.direction.upper()
        if direction in opposites and opposites[direction] != self._direction:
            self._direction = direction

        # Move head
        head_r, head_c = self._snake[0]
        dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[
            self._direction
        ]
        new_head = (head_r + dr, head_c + dc)

        # Wall collision
        r, c = new_head
        if r < 0 or r >= self._grid_size or c < 0 or c >= self._grid_size:
            self._done = True
            self._death_reason = "wall"
            self._step_count += 1
            return self._make_observation(reward=-1.0)

        # Self collision
        if new_head in self._snake:
            self._done = True
            self._death_reason = "self"
            self._step_count += 1
            return self._make_observation(reward=-1.0)

        # Move snake
        self._snake.insert(0, new_head)

        # Apple collection
        if new_head == self._apple:
            self._score += 1
            self._place_apple()
            reward = 1.0
        else:
            self._snake.pop()
            # Distance-based shaping
            curr_dist = self._manhattan_distance()
            reward = 0.01 if curr_dist < self._prev_distance else -0.01
            self._prev_distance = curr_dist

        self._step_count += 1

        # Max steps safety
        if self._step_count >= 200:
            self._done = True

        return self._make_observation(reward=reward)

    @property
    def state(self) -> SnakeState:
        return SnakeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            score=self._score,
            grid_size=self._grid_size,
        )

    def _place_apple(self):
        """Place apple on a random empty cell."""
        occupied = set(self._snake)
        free = [
            (r, c)
            for r in range(self._grid_size)
            for c in range(self._grid_size)
            if (r, c) not in occupied
        ]
        self._apple = random.choice(free) if free else (0, 0)
        self._prev_distance = self._manhattan_distance()

    def _manhattan_distance(self) -> int:
        hr, hc = self._snake[0]
        ar, ac = self._apple
        return abs(hr - ar) + abs(hc - ac)

    def _build_grid(self) -> list[list[str]]:
        grid = [["." for _ in range(self._grid_size)] for _ in range(self._grid_size)]
        for r, c in self._snake[1:]:
            if 0 <= r < self._grid_size and 0 <= c < self._grid_size:
                grid[r][c] = "S"
        hr, hc = self._snake[0]
        if 0 <= hr < self._grid_size and 0 <= hc < self._grid_size:
            grid[hr][hc] = "H"
        ar, ac = self._apple
        grid[ar][ac] = "A"
        return grid

    def _make_observation(self, reward: float) -> SnakeObservation:
        return SnakeObservation(
            grid=self._build_grid(),
            snake=list(self._snake),
            apple=self._apple,
            score=self._score,
            death_reason=self._death_reason,
            done=self._done,
            reward=reward,
        )
