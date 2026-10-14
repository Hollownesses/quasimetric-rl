from __future__ import annotations

import heapq
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np

from minimal_qrl.baselines.base import BaselineController
from minimal_qrl.envs import CommInspectionDubinsUAV2D


# Included in oracle cache provenance: old first-generated-goal costs must not
# be silently reused after changing search termination, keys or heuristics.
HYBRID_ASTAR_ALGORITHM_VERSION = 2


@dataclass
class HybridAStarConfig:
    position_resolution: float = 0.25
    heading_bins: int = 24
    primitive_steps: int = 5
    primitive_scales: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    heuristic_weight: float = 1.0
    max_expansions: int = 50_000
    timeout_sec: float = 30.0
    terminal_samples: int = 128

    def __post_init__(self) -> None:
        if (not np.isfinite(self.heuristic_weight) or self.heuristic_weight < 0
                or not np.isfinite(self.position_resolution) or self.position_resolution <= 0
                or not np.isfinite(self.timeout_sec) or self.timeout_sec <= 0
                or min(self.heading_bins, self.primitive_steps, self.max_expansions) <= 0
                or self.terminal_samples < 0
                or not self.primitive_scales
                or not all(np.isfinite(s) for s in self.primitive_scales)):
            raise ValueError("invalid Hybrid A* search configuration")


@dataclass
class _Node:
    state: np.ndarray
    env_state: dict
    cost: float
    parent: Optional[int]
    actions: tuple[float, ...]
    terminal: bool = False


class HybridAStarController(BaselineController):
    """Feasible reference paths, not certified continuous-state optima.

    Pose-bin pruning merges distinct physical states with different successors.
    Correct A* termination cannot remove that approximation, nor the restriction
    to a finite set of motion primitives. Budget-limited incumbents are retained
    and explicitly distinguished from goals selected from the open queue.
    """
    name = "hybrid_astar"

    def __init__(self, cfg: HybridAStarConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._actions: list[np.ndarray] = []
        self._cursor = 0
        self._terminal_states = np.zeros((0, 3), dtype=np.float32)

    def _key(self, env: CommInspectionDubinsUAV2D, state: np.ndarray) -> tuple[int, int, int]:
        x_bin = int(np.floor((float(state[0]) - env.x_min) / self.cfg.position_resolution + 0.5))
        y_bin = int(np.floor((float(state[1]) - env.y_min) / self.cfg.position_resolution + 0.5))
        angle = (float(state[2]) + np.pi) % (2.0 * np.pi)
        theta_bin = int(np.floor(angle / (2.0 * np.pi) * self.cfg.heading_bins)) % self.cfg.heading_bins
        return x_bin, y_bin, theta_bin

    def _heuristic(self, env: CommInspectionDubinsUAV2D, state: np.ndarray) -> float:
        # Every feasible terminal lies in this annulus. Distance to this
        # superset / speed lower-bounds travel time, and cost_total >= time.
        # Finite terminal samples need not include the closest terminal; adding
        # turning time also double-counts simultaneous translation and turning.
        # Neither is a valid lower bound. Samples remain diagnostic only.
        target = np.asarray(env.inspection_target, dtype=np.float32)
        distance = float(np.linalg.norm(np.asarray(state[:2]) - target))
        remaining_distance = max(0.0, distance - float(env.observation_max_distance),
                                 float(env.observation_min_distance) - distance)
        return remaining_distance / max(float(env.v), 1e-8)

    def _search_key(self, env: CommInspectionDubinsUAV2D, node: _Node) -> tuple:
        # A cheaper but later arrival can have too little episode budget left.
        # A nearby nonterminal must not suppress a genuine terminal in its bin.
        return (self._key(env, node.state), int(node.env_state["t"]), node.terminal)

    def _sample_terminal_states(self, env: CommInspectionDubinsUAV2D, seed: int) -> np.ndarray:
        n = max(0, int(self.cfg.terminal_samples))
        if n <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        samples = []
        for i in range(n):
            try:
                samples.append(
                    env.sample_task_terminal_state(seed=int(seed) * 100_003 + i + 1)
                )
            except RuntimeError:
                break
        return np.asarray(samples, dtype=np.float32).reshape((-1, 3)) if samples else np.zeros((0, 3), dtype=np.float32)

    def _simulate_primitive(
        self,
        env: CommInspectionDubinsUAV2D,
        node: _Node,
        omega: float,
    ) -> Optional[tuple[dict, np.ndarray, float, tuple[float, ...], bool]]:
        if int(node.env_state["t"]) >= int(env.max_episode_steps):
            return None
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
        if any(not np.isfinite(w) or w < 0 for w in (
            env.observation_violation_cost_weight, env.communication_violation_cost_weight
        )):
            raise ValueError("Hybrid A* requires nonnegative finite cost weights")
        base_state = env.get_state()
        start_time = perf_counter()
        root = _Node(env.state.copy(), base_state, 0.0, None, (),
                     terminal=env.is_task_feasible(env.state))
        nodes = [root]
        best_cost = {self._search_key(env, root): 0.0}
        # Within each approximate pose/terminal bin keep a time-cost Pareto
        # frontier. Keeping every time layer would needlessly expand later,
        # more expensive loops; keeping only cheapest would lose early arrivals.
        root_key = self._search_key(env, root)
        arrival_costs = {(root_key[0], root_key[2]): {root_key[1]: 0.0}}
        queue: list[tuple[float, int, int]] = []
        counter = 0
        heapq.heappush(queue, (self.cfg.heuristic_weight * self._heuristic(env, root.state), counter, 0))
        expansions = 0
        generated = 1
        goal_index: Optional[int] = 0 if root.terminal else None
        search_stop_reason = "open_set_exhausted"

        try:
            while queue:
                if perf_counter() - start_time >= float(self.cfg.timeout_sec):
                    search_stop_reason = "timeout"
                    break
                if expansions >= int(self.cfg.max_expansions):
                    search_stop_reason = "max_expansions"
                    break
                _priority, _order, node_index = heapq.heappop(queue)
                node = nodes[node_index]
                key = self._search_key(env, node)
                if key not in best_cost or node.cost > best_cost[key] + 1e-9:
                    continue
                if node.terminal:
                    if goal_index is None or node.cost < nodes[goal_index].cost:
                        goal_index = node_index
                    search_stop_reason = "goal_popped"
                    break

                expansions += 1
                for scale in self.cfg.primitive_scales:
                    omega = float(scale) * float(env.omega_max)
                    simulated = self._simulate_primitive(env, node, omega)
                    if simulated is None:
                        continue
                    child_env_state, child_state, edge_cost, actions, success = simulated
                    if not np.isfinite(edge_cost) or edge_cost < 0:
                        raise ValueError("Hybrid A* requires nonnegative finite edge costs")
                    child_cost = node.cost + edge_cost
                    child = _Node(child_state, child_env_state, child_cost, node_index, actions,
                                  terminal=bool(success))
                    child_key = self._search_key(env, child)
                    pose_key, arrival_time, terminal = child_key
                    frontier = arrival_costs.setdefault((pose_key, terminal), {})
                    if any(t <= arrival_time and c <= child_cost + 1e-9
                           for t, c in frontier.items()):
                        continue
                    for t in [t for t, c in frontier.items()
                              if t >= arrival_time and c >= child_cost - 1e-9]:
                        del frontier[t]
                        best_cost.pop((pose_key, t, terminal), None)
                    frontier[arrival_time] = child_cost
                    best_cost[child_key] = child_cost
                    nodes.append(child)
                    child_index = len(nodes) - 1
                    generated += 1
                    counter += 1
                    heuristic = 0.0 if success else self._heuristic(env, child_state)
                    priority = child_cost + self.cfg.heuristic_weight * heuristic
                    heapq.heappush(queue, (priority, counter, child_index))
                    if success:
                        # Keep the best feasible incumbent if a budget expires,
                        # but still compare other actions and queued paths.
                        if goal_index is None or child_cost < nodes[goal_index].cost:
                            goal_index = child_index
        finally:
            env.set_state(base_state)

        elapsed = perf_counter() - start_time
        self._actions = self._reconstruct(nodes, goal_index) if goal_index is not None else []
        self._cursor = 0
        path_cost = float(nodes[goal_index].cost) if goal_index is not None else float("inf")
        return {
            "planner_success": bool(goal_index is not None),
            "planner_failure_reason": "" if goal_index is not None else search_stop_reason,
            "planner_search_stop_reason": search_stop_reason,
            "planner_algorithm_version": HYBRID_ASTAR_ALGORITHM_VERSION,
            "planner_cost_kind": "feasible_path_upper_bound",
            "planner_optimality_guaranteed": False,
            "initial_planning_time_sec": float(elapsed),
            "expanded_nodes": int(expansions),
            "generated_nodes": int(generated),
            "planned_action_count": int(len(self._actions)),
            "planned_cost": path_cost,
            "terminal_sample_count": int(len(self._terminal_states)),
        }

    def begin_episode(
        self,
        env: CommInspectionDubinsUAV2D,
        goal_obs: np.ndarray,
        seed: int,
    ) -> Dict[str, Any]:
        super().begin_episode(env, goal_obs, seed)
        self._terminal_states = self._sample_terminal_states(env, int(seed))
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
