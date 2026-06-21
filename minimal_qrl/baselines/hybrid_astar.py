from __future__ import annotations

import heapq
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np

from minimal_qrl.baselines.base import BaselineController
from minimal_qrl.envs import CommInspectionDubinsUAV2D


@dataclass
class HybridAStarConfig:
    position_resolution: float = 0.25
    heading_bins: int = 24
    primitive_steps: int = 5
    primitive_scales: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    heuristic_weight: float = 1.0
    max_expansions: int = 50_000
    timeout_sec: float = 10.0


@dataclass
class _Node:
    state: np.ndarray
    env_state: dict
    cost: float
    parent: Optional[int]
    actions: tuple[float, ...]


class HybridAStarController(BaselineController):
    name = "hybrid_astar"

    def __init__(self, cfg: HybridAStarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._actions: list[np.ndarray] = []
        self._cursor = 0

    def _key(self, env: CommInspectionDubinsUAV2D, state: np.ndarray) -> tuple[int, int, int]:
        x_bin = int(np.floor((float(state[0]) - env.x_min) / self.cfg.position_resolution + 0.5))
        y_bin = int(np.floor((float(state[1]) - env.y_min) / self.cfg.position_resolution + 0.5))
        angle = (float(state[2]) + np.pi) % (2.0 * np.pi)
        theta_bin = int(np.floor(angle / (2.0 * np.pi) * self.cfg.heading_bins)) % self.cfg.heading_bins
        return x_bin, y_bin, theta_bin

    def _heuristic(self, env: CommInspectionDubinsUAV2D, state: np.ndarray) -> float:
        target = np.asarray(env.inspection_target, dtype=np.float32)
        distance = float(np.linalg.norm(np.asarray(state[:2]) - target))
        remaining_distance = max(0.0, distance - float(env.observation_radius))
        return remaining_distance / max(float(env.v), 1e-8)

    def _simulate_primitive(
        self,
        env: CommInspectionDubinsUAV2D,
        node: _Node,
        omega: float,
    ) -> Optional[tuple[dict, np.ndarray, float, tuple[float, ...], bool]]:
        env.set_state(node.env_state)
        total_cost = 0.0
        actions = []
        success = False
        for _ in range(max(1, int(self.cfg.primitive_steps))):
            _obs, _reward, terminated, truncated, info = env.step(
                np.array([omega], dtype=np.float32)
            )
            total_cost += float(info.get("cost_total", -float(_reward)))
            actions.append(float(omega))
            if bool(info.get("collision", False)) or bool(info.get("out_of_bounds", False)):
                return None
            success = bool(info.get("success", False))
            if truncated and not success:
                return None
            if terminated or truncated:
                break
        return env.get_state(), env.state.copy(), total_cost, tuple(actions), success

    def _reconstruct(self, nodes: list[_Node], goal_index: int) -> list[np.ndarray]:
        chunks = []
        idx: Optional[int] = goal_index
        while idx is not None:
            node = nodes[idx]
            if node.actions:
                chunks.append(node.actions)
            idx = node.parent
        result = []
        for chunk in reversed(chunks):
            result.extend(np.array([omega], dtype=np.float32) for omega in chunk)
        return result

    def _plan(self, env: CommInspectionDubinsUAV2D) -> Dict[str, Any]:
        base_state = env.get_state()
        start_time = perf_counter()
        root = _Node(env.state.copy(), base_state, 0.0, None, ())
        nodes = [root]
        best_cost = {self._key(env, root.state): 0.0}
        queue: list[tuple[float, int, int]] = []
        counter = 0
        heapq.heappush(queue, (self.cfg.heuristic_weight * self._heuristic(env, root.state), counter, 0))
        expansions = 0
        generated = 1
        goal_index: Optional[int] = None
        failure_reason = "open_set_exhausted"

        try:
            while queue:
                if perf_counter() - start_time >= float(self.cfg.timeout_sec):
                    failure_reason = "timeout"
                    break
                if expansions >= int(self.cfg.max_expansions):
                    failure_reason = "max_expansions"
                    break
                _priority, _order, node_index = heapq.heappop(queue)
                node = nodes[node_index]
                key = self._key(env, node.state)
                if node.cost > best_cost.get(key, float("inf")) + 1e-9:
                    continue
                if env.is_task_feasible(node.state):
                    goal_index = node_index
                    failure_reason = ""
                    break

                expansions += 1
                for scale in self.cfg.primitive_scales:
                    omega = float(scale) * float(env.omega_max)
                    simulated = self._simulate_primitive(env, node, omega)
                    if simulated is None:
                        continue
                    child_env_state, child_state, edge_cost, actions, success = simulated
                    child_cost = node.cost + edge_cost
                    child_key = self._key(env, child_state)
                    if child_cost >= best_cost.get(child_key, float("inf")) - 1e-9:
                        continue
                    best_cost[child_key] = child_cost
                    child = _Node(child_state, child_env_state, child_cost, node_index, actions)
                    nodes.append(child)
                    child_index = len(nodes) - 1
                    generated += 1
                    counter += 1
                    priority = child_cost + self.cfg.heuristic_weight * self._heuristic(env, child_state)
                    heapq.heappush(queue, (priority, counter, child_index))
                    if success:
                        goal_index = child_index
                        failure_reason = ""
                        queue.clear()
                        break
        finally:
            env.set_state(base_state)

        elapsed = perf_counter() - start_time
        self._actions = self._reconstruct(nodes, goal_index) if goal_index is not None else []
        self._cursor = 0
        path_cost = float(nodes[goal_index].cost) if goal_index is not None else float("inf")
        return {
            "planner_success": bool(goal_index is not None),
            "planner_failure_reason": failure_reason,
            "initial_planning_time_sec": float(elapsed),
            "expanded_nodes": int(expansions),
            "generated_nodes": int(generated),
            "planned_action_count": int(len(self._actions)),
            "planned_cost": path_cost,
        }

    def begin_episode(
        self,
        env: CommInspectionDubinsUAV2D,
        goal_obs: np.ndarray,
        seed: int,
    ) -> Dict[str, Any]:
        super().begin_episode(env, goal_obs, seed)
        diagnostics = self._plan(env)
        self._episode_diagnostics.update(diagnostics)
        return diagnostics

    def act(
        self,
        obs: np.ndarray,
        env: CommInspectionDubinsUAV2D,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        del obs
        if self._cursor < len(self._actions):
            action = self._actions[self._cursor]
            self._cursor += 1
            return action.copy(), {"planned_action_index": int(self._cursor - 1)}
        return np.array([0.0], dtype=np.float32), {
            "plan_exhausted": True,
            "planner_failure_reason": self._episode_diagnostics.get("planner_failure_reason", ""),
        }
