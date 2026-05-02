"""
Discrete obstacle maze2D environment for the existing QRL training pipeline.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces

from .base import BaseNavigationEnv


Cell = Tuple[int, int]


class Maze2DNavigation(BaseNavigationEnv):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    ACTIONS: Tuple[Cell, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(
        self,
        grid_size: Tuple[int, int] = (15, 15),
        walls: Optional[Iterable[Cell]] = None,
        start_pos: Optional[Cell] = None,
        goal_pos: Optional[Cell] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 200,
    ):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.height, self.width = self.grid_size
        self.walls = set(walls if walls is not None else self._default_walls(self.height, self.width))
        self.start_pos = start_pos if start_pos is not None else (1, 1)
        self.goal_pos = goal_pos if goal_pos is not None else (self.height - 2, self.width - 2)
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.valid_cells: List[Cell] = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.walls
        ]
        self.valid_set = set(self.valid_cells)
        if self.start_pos not in self.valid_set:
            self.start_pos = self.valid_cells[0]
        if self.goal_pos not in self.valid_set:
            self.goal_pos = self.valid_cells[-1]

        self.agent_pos = self.start_pos
        self._t = 0
        self._distance_cache = {}

    @staticmethod
    def _default_walls(height: int, width: int) -> List[Cell]:
        walls = set()
        for r in range(height):
            walls.add((r, 0))
            walls.add((r, width - 1))
        for c in range(width):
            walls.add((0, c))
            walls.add((height - 1, c))

        barrier_col = width // 2
        gap_row = height // 2
        for r in range(1, height - 1):
            if r != gap_row:
                walls.add((r, barrier_col))
        return sorted(walls)

    def _cell_to_obs(self, cell: Cell) -> np.ndarray:
        return np.array(
            [cell[0] / (self.height - 1), cell[1] / (self.width - 1)],
            dtype=np.float32,
        )

    def _obs_to_cell(self, state: np.ndarray) -> Cell:
        state = np.asarray(state, dtype=np.float32).reshape(2)
        r = int(round(float(state[0]) * (self.height - 1)))
        c = int(round(float(state[1]) * (self.width - 1)))
        return (
            int(np.clip(r, 0, self.height - 1)),
            int(np.clip(c, 0, self.width - 1)),
        )

    def _nearest_valid_cell(self, cell: Cell) -> Cell:
        if cell in self.valid_set:
            return cell
        best = min(self.valid_cells, key=lambda x: abs(x[0] - cell[0]) + abs(x[1] - cell[1]))
        return best

    def _next_cell(self, cell: Cell, action: int) -> Cell:
        dr, dc = self.ACTIONS[int(action)]
        nxt = (cell[0] + dr, cell[1] + dc)
        return nxt if nxt in self.valid_set else cell

    def _valid_neighbors(self, cell: Cell) -> List[Cell]:
        neighbors = []
        for action in range(4):
            nxt = self._next_cell(cell, action)
            if nxt != cell:
                neighbors.append(nxt)
        return neighbors

    def is_valid_state(self, state: np.ndarray) -> bool:
        return self._obs_to_cell(state) in self.valid_set

    def sample_valid_state(self, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cell = self.valid_cells[int(rng.integers(0, len(self.valid_cells)))]
        return self._cell_to_obs(cell)

    def sample_goal(self, seed: Optional[int] = None) -> np.ndarray:
        return self.sample_valid_state(seed=seed)

    def compute_shortest_path_distance(
        self,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
    ) -> float:
        start_cell = self.agent_pos if start is None else self._nearest_valid_cell(self._obs_to_cell(start))
        goal_cell = self.goal_pos if goal is None else self._nearest_valid_cell(self._obs_to_cell(goal))
        if goal_cell not in self._distance_cache:
            dist = np.full((self.height, self.width), np.inf, dtype=np.float32)
            dist[goal_cell] = 0.0
            queue = deque([goal_cell])
            while queue:
                cell = queue.popleft()
                for nxt in self._valid_neighbors(cell):
                    if dist[nxt] > dist[cell] + 1.0:
                        dist[nxt] = dist[cell] + 1.0
                        queue.append(nxt)
            self._distance_cache[goal_cell] = dist
        return float(self._distance_cache[goal_cell][start_cell])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "start" in options:
            self.start_pos = self._nearest_valid_cell(self._obs_to_cell(options["start"]))
        self.agent_pos = self.start_pos
        self._t = 0
        return self._cell_to_obs(self.agent_pos), {}

    def step(self, action: int):
        self.agent_pos = self._next_cell(self.agent_pos, int(action))
        self._t += 1
        terminated = self.agent_pos == self.goal_pos
        truncated = self._t >= self.max_episode_steps
        reward = 1.0 if terminated else -0.01
        return self._cell_to_obs(self.agent_pos), float(reward), bool(terminated), bool(truncated), {"is_success": bool(terminated)}

    def get_state(self) -> dict:
        return {"agent_pos": tuple(self.agent_pos), "start_pos": tuple(self.start_pos), "t": int(self._t)}

    def set_state(self, state: dict) -> None:
        self.agent_pos = tuple(state["agent_pos"])
        self.start_pos = tuple(state["start_pos"])
        self._t = int(state["t"])

    def render(self):
        if self.render_mode != "human":
            return
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for r, c in self.walls:
            grid[r][c] = "#"
        grid[self.start_pos[0]][self.start_pos[1]] = "S"
        grid[self.goal_pos[0]][self.goal_pos[1]] = "G"
        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"
        print("\n".join("".join(row) for row in grid))
