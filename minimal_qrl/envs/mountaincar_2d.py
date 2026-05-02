"""
MountainCar navigation environment for the existing QRL training pipeline.

The observation follows the QRL paper's discretized MountainCar setting:

    [position, velocity, indicator]

Regular physical states have indicator=0. The original MountainCar goal set is
represented by a special abstract goal [0.5, 0.0, 1.0].
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import gym
import numpy as np
from gym import spaces

from .base import BaseNavigationEnv


class MountainCar2D(BaseNavigationEnv):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    min_position: float = -1.2
    max_position: float = 0.6
    min_velocity: float = -0.07
    max_velocity: float = 0.07
    force: float = 0.001
    gravity: float = 0.0025

    def __init__(
        self,
        goal_position: float = 0.5,
        goal_velocity: float = 0.0,
        goal_tolerance_pos: float = 0.015,
        goal_tolerance_vel: float = 0.01,
        max_episode_steps: int = 200,
        gt_pos_bins: int = 160,
        gt_vel_bins: int = 160,
        gt_goal_mode: str = "threshold",
        dataset_mode: str = "random_policy_paper",
        abstract_goal_transition_repeats: int = 15,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.goal_position = float(goal_position)
        self.goal_velocity = float(goal_velocity)
        self.goal_tolerance_pos = float(goal_tolerance_pos)
        self.goal_tolerance_vel = float(goal_tolerance_vel)
        self.max_episode_steps = int(max_episode_steps)
        self.gt_pos_bins = int(gt_pos_bins)
        self.gt_vel_bins = int(gt_vel_bins)
        self.gt_goal_mode = str(gt_goal_mode)
        self.dataset_mode = str(dataset_mode)
        self.abstract_goal_transition_repeats = int(abstract_goal_transition_repeats)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([self.min_position, self.min_velocity, 0.0], dtype=np.float32),
            high=np.array([self.max_position, self.max_velocity, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.start = self._as_observation(np.array([-0.5, 0.0], dtype=np.float32))
        self.goal = self.abstract_goal()
        self.state = self.start.copy()
        self._t = 0
        self._reverse_graph = None
        self._distance_cache = {}

    def is_valid_state(self, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] == 3 and state[2] >= 0.5:
            return True
        return (
            self.min_position <= float(state[0]) <= self.max_position
            and self.min_velocity <= float(state[1]) <= self.max_velocity
        )

    def sample_valid_state(self, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        physical = np.array(
            [
                rng.uniform(self.min_position, self.max_position),
                rng.uniform(self.min_velocity, self.max_velocity),
            ],
            dtype=np.float32,
        )
        return self._as_observation(self._discretize_physical_state(physical))

    def sample_goal(self, seed: Optional[int] = None) -> np.ndarray:
        return self.abstract_goal()

    def abstract_goal(self) -> np.ndarray:
        return np.array([self.goal_position, 0.0, 1.0], dtype=np.float32)

    def _as_physical_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return state[:2].astype(np.float32)

    def _as_observation(self, physical_state: np.ndarray, indicator: float = 0.0) -> np.ndarray:
        physical_state = np.asarray(physical_state, dtype=np.float32).reshape(2)
        return np.array([physical_state[0], physical_state[1], indicator], dtype=np.float32)

    def _discretize_physical_state(self, state: np.ndarray) -> np.ndarray:
        idx = self._state_to_index(state)
        return self._index_to_state(idx)

    def _dynamics(self, state: np.ndarray, action: int) -> np.ndarray:
        state = self._as_physical_state(state)
        position = float(state[0])
        velocity = float(state[1])
        velocity += (int(action) - 1) * self.force - math.cos(3.0 * position) * self.gravity
        velocity = float(np.clip(velocity, self.min_velocity, self.max_velocity))
        position += velocity
        position = float(np.clip(position, self.min_position, self.max_position))
        if position <= self.min_position and velocity < 0.0:
            velocity = 0.0
        return self._discretize_physical_state(np.array([position, velocity], dtype=np.float32))

    def iter_discrete_transitions(self):
        """Yield every one-step edge in the discretized MountainCar graph."""
        for idx in range(self.gt_pos_bins * self.gt_vel_bins):
            state = self._index_to_state(idx)
            for action in range(3):
                next_state = self._dynamics(state, action)
                done = self._is_terminal_goal(next_state)
                reward = 0.0 if done else -1.0
                yield self._as_observation(state), int(action), self._as_observation(next_state), float(reward), bool(done)
        yield from self.iter_added_goal_transitions()

    def iter_added_goal_transitions(self):
        """Added edges from goal-set states to the paper's abstract MountainCar goal."""
        goal = self.abstract_goal()
        goal_states = []
        for idx in range(self.gt_pos_bins * self.gt_vel_bins):
            state = self._index_to_state(idx)
            if self._in_original_goal_set(state):
                goal_states.append(state)
        for _ in range(max(1, self.abstract_goal_transition_repeats)):
            for state in goal_states:
                yield self._as_observation(state), 1, goal.copy(), 0.0, True

    def _is_terminal_goal(self, state: np.ndarray, goal: Optional[np.ndarray] = None, mode: Optional[str] = None) -> bool:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        goal = self.goal if goal is None else np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal.shape[0] == 3 and goal[2] >= 0.5:
            return self._in_original_goal_set(state, goal_position=float(goal[0]))
        mode = self.gt_goal_mode if mode is None else mode
        if mode == "threshold":
            return bool(state[0] >= goal[0] and state[1] >= 0.0)
        return bool(
            abs(float(state[0] - goal[0])) <= self.goal_tolerance_pos
            and abs(float(state[1] - goal[1])) <= self.goal_tolerance_vel
        )

    def _in_original_goal_set(self, state: np.ndarray, goal_position: Optional[float] = None) -> bool:
        state = self._as_physical_state(state)
        goal_position = self.goal_position if goal_position is None else float(goal_position)
        return bool(goal_position <= state[0] <= self.max_position and 0.0 <= state[1] <= self.max_velocity)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "start" in options:
            self.start = self._as_observation(self._discretize_physical_state(options["start"]))
        else:
            rng = np.random.default_rng(seed)
            self.start = self._as_observation(self._discretize_physical_state(np.array(
                [rng.uniform(-0.6, -0.4), rng.uniform(-0.005, 0.005)],
                dtype=np.float32,
            )))
        self.state = self.start.astype(np.float32)
        self._t = 0
        return self.state.copy(), {}

    def step(self, action: int):
        next_physical = self._dynamics(self.state, int(action))
        self.state = self._as_observation(next_physical)
        self._t += 1
        terminated = self._is_terminal_goal(self.state)
        truncated = self._t >= self.max_episode_steps
        reward = 0.0 if terminated else -1.0
        return self.state.copy(), float(reward), bool(terminated), bool(truncated), {"is_success": bool(terminated)}

    def _state_to_index(self, state: np.ndarray) -> int:
        ip = int(round((float(state[0]) - self.min_position) / (self.max_position - self.min_position) * (self.gt_pos_bins - 1)))
        iv = int(round((float(state[1]) - self.min_velocity) / (self.max_velocity - self.min_velocity) * (self.gt_vel_bins - 1)))
        ip = int(np.clip(ip, 0, self.gt_pos_bins - 1))
        iv = int(np.clip(iv, 0, self.gt_vel_bins - 1))
        return ip * self.gt_vel_bins + iv

    def _index_to_state(self, index: int) -> np.ndarray:
        ip = int(index) // self.gt_vel_bins
        iv = int(index) % self.gt_vel_bins
        pos = self.min_position + (self.max_position - self.min_position) * ip / (self.gt_pos_bins - 1)
        vel = self.min_velocity + (self.max_velocity - self.min_velocity) * iv / (self.gt_vel_bins - 1)
        return np.array([pos, vel], dtype=np.float32)

    def _get_reverse_graph(self):
        if self._reverse_graph is not None:
            return self._reverse_graph
        n_states = self.gt_pos_bins * self.gt_vel_bins
        reverse = [[] for _ in range(n_states)]
        for src in range(n_states):
            state = self._index_to_state(src)
            for action in range(3):
                dst = self._state_to_index(self._dynamics(state, action))
                reverse[dst].append(src)
        self._reverse_graph = reverse
        return reverse

    def _goal_indices(self, goal: np.ndarray, mode: Optional[str] = None):
        mode = self.gt_goal_mode if mode is None else mode
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        indices = []
        for idx in range(self.gt_pos_bins * self.gt_vel_bins):
            state = self._index_to_state(idx)
            if self._is_terminal_goal(state, goal=goal, mode=mode):
                indices.append(idx)
        return indices

    def _distance_grid(self, goal: np.ndarray, mode: Optional[str] = None) -> np.ndarray:
        mode = self.gt_goal_mode if mode is None else mode
        goal_key = tuple(np.round(np.asarray(goal, dtype=np.float32).reshape(-1), 4).tolist()) + (mode,)
        if goal_key in self._distance_cache:
            return self._distance_cache[goal_key]
        n_states = self.gt_pos_bins * self.gt_vel_bins
        dist = np.full(n_states, np.inf, dtype=np.float32)
        queue = deque()
        for idx in self._goal_indices(goal, mode=mode):
            dist[idx] = 0.0
            queue.append(idx)
        reverse = self._get_reverse_graph()
        while queue:
            node = queue.popleft()
            nd = dist[node] + 1.0
            for pred in reverse[node]:
                if nd < dist[pred]:
                    dist[pred] = nd
                    queue.append(pred)
        self._distance_cache[goal_key] = dist
        return dist

    def compute_shortest_path_distance(
        self,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
    ) -> float:
        start = self.state if start is None else np.asarray(start, dtype=np.float32).reshape(-1)
        goal = self.goal if goal is None else np.asarray(goal, dtype=np.float32).reshape(-1)
        dist = self._distance_grid(goal)
        value = float(dist[self._state_to_index(self._as_physical_state(start))])
        return value

    def get_state(self) -> dict:
        return {"state": self.state.copy(), "start": self.start.copy(), "t": int(self._t)}

    def set_state(self, state: dict) -> None:
        self.state = np.asarray(state["state"], dtype=np.float32).copy()
        self.start = np.asarray(state["start"], dtype=np.float32).copy()
        self._t = int(state["t"])
