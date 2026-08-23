from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Protocol

import numpy as np

from minimal_qrl.baselines.base import BaselineController
from minimal_qrl.envs import CircleObstacle, CommInspectionDubinsUAV2D
from minimal_qrl.gc_agents import GoalConditionedAgentBase


@dataclass
class MPPIConfig:
    horizon: int = 20
    num_samples: int = 256
    noise_sigma: float = 0.8
    temperature: float = 1.0
    invalid_penalty: float = 1_000_000.0
    terminal_weight: float = 1.0
    terminal_samples: int = 128


class TerminalValueProvider(Protocol):
    """Batch terminal-value interface used by non-learned MPPI oracles."""

    def begin_episode(
        self,
        env: CommInspectionDubinsUAV2D,
        *,
        seed: int,
    ) -> Dict[str, Any]: ...

    def batch_value(
        self,
        env: CommInspectionDubinsUAV2D,
        states: np.ndarray,
    ) -> np.ndarray: ...

    def end_episode(self) -> Dict[str, Any]: ...


def _normalize_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _circle_segment_hits(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    obstacle: CircleObstacle,
) -> np.ndarray:
    vx = x2 - x1
    vy = y2 - y1
    wx = float(obstacle.x) - x1
    wy = float(obstacle.y) - y1
    denom = vx * vx + vy * vy
    t = np.divide(vx * wx + vy * wy, denom, out=np.zeros_like(denom), where=denom > 1e-12)
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * vx
    py = y1 + t * vy
    return (px - float(obstacle.x)) ** 2 + (py - float(obstacle.y)) ** 2 <= float(obstacle.radius) ** 2


def _rectangle_segment_hits(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    obstacle,
) -> np.ndarray:
    """Vectorized equivalent of ``Obstacle.intersects_segment``."""

    x_min = float(obstacle.x_min)
    x_max = float(obstacle.x_max)
    y_min = float(obstacle.y_min)
    y_max = float(obstacle.y_max)
    hits = (
        ((x1 >= x_min) & (x1 <= x_max) & (y1 >= y_min) & (y1 <= y_max))
        | ((x2 >= x_min) & (x2 <= x_max) & (y2 >= y_min) & (y2 <= y_max))
    )
    dx = x2 - x1
    dy = y2 - y1
    for boundary in (x_min, x_max):
        nonzero = np.abs(dx) > 1e-12
        t = np.divide(
            boundary - x1,
            dx,
            out=np.zeros_like(dx),
            where=nonzero,
        )
        y_intersection = y1 + t * dy
        hits |= nonzero & (t >= 0.0) & (t <= 1.0) & (y_intersection >= y_min) & (y_intersection <= y_max)
    for boundary in (y_min, y_max):
        nonzero = np.abs(dy) > 1e-12
        t = np.divide(
            boundary - y1,
            dy,
            out=np.zeros_like(dy),
            where=nonzero,
        )
        x_intersection = x1 + t * dx
        hits |= nonzero & (t >= 0.0) & (t <= 1.0) & (x_intersection >= x_min) & (x_intersection <= x_max)
    return hits


def _segment_hits_obstacle(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    obstacle,
) -> np.ndarray:
    if isinstance(obstacle, CircleObstacle):
        return _circle_segment_hits(x1, y1, x2, y2, obstacle)
    if all(hasattr(obstacle, field) for field in ("x_min", "x_max", "y_min", "y_max")):
        return _rectangle_segment_hits(x1, y1, x2, y2, obstacle)
    return np.asarray(
        [
            obstacle.intersects_segment(float(a), float(b), float(c), float(d))
            for a, b, c, d in zip(x1, y1, x2, y2)
        ],
        dtype=bool,
    )


def _los_to_point(
    env: CommInspectionDubinsUAV2D,
    states: np.ndarray,
    point,
    *,
    allow_endpoint_contact: bool = False,
) -> np.ndarray:
    point = np.asarray(point, dtype=np.float32)
    result = np.ones((states.shape[0],), dtype=bool)
    x2 = np.full((states.shape[0],), float(point[0]), dtype=np.float32)
    y2 = np.full((states.shape[0],), float(point[1]), dtype=np.float32)
    if allow_endpoint_contact:
        dx = x2 - states[:, 0]
        dy = y2 - states[:, 1]
        length = np.hypot(dx, dy)
        eps = np.minimum(
            1e-4 * max(float(env.x_max - env.x_min), float(env.y_max - env.y_min), 1.0),
            0.1 * length,
        )
        valid = length > 1e-9
        x2[valid] -= eps[valid] * dx[valid] / length[valid]
        y2[valid] -= eps[valid] * dy[valid] / length[valid]
    for obstacle in env.obstacles:
        result &= ~_segment_hits_obstacle(states[:, 0], states[:, 1], x2, y2, obstacle)
    return result


def _state_terms(
    env: CommInspectionDubinsUAV2D,
    states: np.ndarray,
    collision: np.ndarray,
    out_of_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(env.inspection_target, dtype=np.float32)
    anchor = np.asarray(env.observation_anchor, dtype=np.float32)
    station = np.asarray(env.ground_station, dtype=np.float32)

    device_to_uav = states[:, :2] - target[None, :]
    target_distance = np.linalg.norm(device_to_uav, axis=1)
    position_bearing = np.arctan2(device_to_uav[:, 1], device_to_uav[:, 0])
    sector_error = np.abs(_normalize_angle(position_bearing - float(env.preferred_bearing)))
    anchor_delta = anchor[None, :] - states[:, :2]
    anchor_bearing = np.arctan2(anchor_delta[:, 1], anchor_delta[:, 0])
    heading_error = np.abs(_normalize_angle(anchor_bearing - states[:, 2]))
    distance_margin = np.minimum(
        target_distance - float(env.observation_min_distance),
        float(env.observation_max_distance) - target_distance,
    )
    sector_margin = float(env.bearing_tolerance) - sector_error
    heading_margin = 0.5 * float(env.fov_angle) - heading_error
    obs_margin = np.minimum(np.minimum(distance_margin, sector_margin), heading_margin)
    target_los = _los_to_point(env, states, anchor, allow_endpoint_contact=True)
    if env.require_target_los:
        obs_margin = np.minimum(obs_margin, np.where(target_los, np.inf, -1.0))
    observation_feasible = (
        (distance_margin >= 0.0)
        & (sector_margin >= 0.0)
        & (heading_margin >= 0.0)
        & (target_los | (not env.require_target_los))
    )

    station_distance = np.linalg.norm(station[None, :] - states[:, :2], axis=1)
    station_los = _los_to_point(
        env,
        states,
        env.ground_station_los_anchor,
        allow_endpoint_contact=True,
    )
    comm_quality = float(env.comm_bias) - float(env.comm_alpha) * np.log(station_distance + 1e-6)
    comm_quality = comm_quality - (~station_los).astype(np.float32) * float(env.comm_occlusion_penalty)
    comm_margin = comm_quality - float(env.comm_threshold)
    communication_feasible = comm_quality >= float(env.comm_threshold)
    if env.require_ground_station_los:
        communication_feasible &= station_los

    obs_shortfall = np.maximum(0.0, -obs_margin) / max(float(env.observation_max_distance), 1e-6)
    comm_shortfall = np.maximum(0.0, -comm_margin) / max(abs(float(env.comm_threshold)) + 1.0, 1.0)
    costs = np.full((states.shape[0],), float(env.dt), dtype=np.float32)
    costs += float(env.observation_violation_cost_weight) * obs_shortfall
    costs += float(env.communication_violation_cost_weight) * comm_shortfall
    costs += (~observation_feasible).astype(np.float32) * float(env.observation_failure_cost)
    costs += (~communication_feasible).astype(np.float32) * float(env.communication_break_cost)
    costs += collision.astype(np.float32) * float(env.collision_cost)
    costs += out_of_bounds.astype(np.float32) * float(env.out_of_bounds_cost)

    valid = ~(collision | out_of_bounds)
    success = valid & observation_feasible & communication_feasible
    return costs, success


def simulate_action_sequences(
    env: CommInspectionDubinsUAV2D,
    initial_state: np.ndarray,
    action_sequences: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Vectorized model rollout matching the environment's deterministic step semantics."""

    actions = np.asarray(action_sequences, dtype=np.float32)
    n, requested_horizon = actions.shape
    remaining_steps = max(0, int(env.max_episode_steps) - int(env._t))
    horizon = min(requested_horizon, remaining_steps)
    states = np.repeat(np.asarray(initial_state, dtype=np.float32)[None, :], n, axis=0)
    total_cost = np.zeros((n,), dtype=np.float32)
    active = np.ones((n,), dtype=bool)
    invalid = np.zeros((n,), dtype=bool)
    success = np.zeros((n,), dtype=bool)
    steps = np.zeros((n,), dtype=np.int32)

    for t in range(horizon):
        if not np.any(active):
            break
        current = states.copy()
        omega = np.clip(actions[:, t], -float(env.omega_max), float(env.omega_max))
        theta_new = _normalize_angle(current[:, 2] + omega * float(env.dt))
        x_new = current[:, 0] + float(env.v) * np.cos(theta_new) * float(env.dt)
        y_new = current[:, 1] + float(env.v) * np.sin(theta_new) * float(env.dt)

        collision = np.zeros((n,), dtype=bool)
        for obstacle in env.obstacles:
            collision |= _segment_hits_obstacle(current[:, 0], current[:, 1], x_new, y_new, obstacle)
        out_of_bounds = (
            (x_new < float(env.x_min))
            | (x_new > float(env.x_max))
            | (y_new < float(env.y_min))
            | (y_new > float(env.y_max))
        )
        collision &= active
        out_of_bounds &= active & ~collision

        proposed = np.stack([x_new, y_new, theta_new], axis=1).astype(np.float32)
        moved = active & ~collision & ~out_of_bounds
        states[moved] = proposed[moved]
        step_cost, step_success = _state_terms(env, states, collision, out_of_bounds)
        total_cost[active] += step_cost[active]
        steps[active] += 1
        success |= active & step_success
        invalid |= collision | out_of_bounds
        active &= ~(success | invalid)

    return {
        "final_states": states,
        "costs": total_cost,
        "success": success,
        "invalid": invalid,
        "steps": steps,
    }


class MPPIController(BaselineController):
    def __init__(
        self,
        cfg: MPPIConfig,
        *,
        terminal_mode: str,
        qrl_agent: Optional[GoalConditionedAgentBase] = None,
        terminal_value_provider: Optional[TerminalValueProvider] = None,
        static_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if terminal_mode not in {"none", "model", "qrl", "oracle"}:
            raise ValueError(f"Unknown MPPI terminal mode: {terminal_mode}")
        if terminal_mode == "qrl" and qrl_agent is None:
            raise ValueError("qrl terminal mode requires qrl_agent")
        if terminal_mode == "oracle" and terminal_value_provider is None:
            raise ValueError("oracle terminal mode requires terminal_value_provider")
        self.cfg = cfg
        self.terminal_mode = terminal_mode
        self.qrl_agent = qrl_agent
        self.terminal_value_provider = terminal_value_provider
        self.static_diagnostics = dict(static_diagnostics or {})
        if terminal_mode == "none":
            self.name = "mppi_no_terminal"
        elif terminal_mode == "model":
            self.name = "model_mppi"
        elif terminal_mode == "qrl":
            self.name = "qrl_mppi"
        else:
            self.name = "oracle_mppi"
        self._nominal = np.zeros((max(1, int(cfg.horizon)),), dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._terminal_states = np.zeros((0, 3), dtype=np.float32)
        self._model_rollouts = 0

    def begin_episode(
        self,
        env: CommInspectionDubinsUAV2D,
        goal_obs: np.ndarray,
        seed: int,
    ) -> Dict[str, Any]:
        super().begin_episode(env, goal_obs, seed)
        self._nominal = np.zeros((max(1, int(self.cfg.horizon)),), dtype=np.float32)
        self._rng = np.random.default_rng(int(seed) + 7001)
        self._model_rollouts = 0
        if self.terminal_mode == "model":
            sampled = [
                env.sample_task_terminal_state(seed=int(seed) * 100_003 + i + 1)
                for i in range(max(1, int(self.cfg.terminal_samples)))
            ]
            self._terminal_states = np.asarray(sampled, dtype=np.float32)
        else:
            self._terminal_states = np.zeros((0, 3), dtype=np.float32)
        provider_diagnostics: Dict[str, Any] = {}
        if self.terminal_mode == "oracle":
            if self.terminal_value_provider is None:
                raise RuntimeError("MPPI oracle terminal provider is unavailable")
            provider_diagnostics = self.terminal_value_provider.begin_episode(
                env,
                seed=int(seed),
            )
        self._episode_diagnostics.update(self.static_diagnostics)
        self._episode_diagnostics.update(provider_diagnostics)
        return {
            "terminal_sample_count": int(len(self._terminal_states)),
            **self.static_diagnostics,
            **provider_diagnostics,
        }

    def _terminal_cost(self, env: CommInspectionDubinsUAV2D, states: np.ndarray) -> np.ndarray:
        if self.terminal_mode == "none":
            return np.zeros((states.shape[0],), dtype=np.float32)
        if self.terminal_mode == "qrl":
            if self.goal_obs is None or self.qrl_agent is None:
                raise RuntimeError("MPPI controller was not initialized for the episode")
            obs_batch = np.stack([env.state_to_observation(state) for state in states], axis=0)
            goal_batch = np.repeat(self.goal_obs[None, :], states.shape[0], axis=0)
            return self.qrl_agent.batch_value(obs_batch, goal_batch).astype(np.float32)
        if self.terminal_mode == "oracle":
            if self.terminal_value_provider is None:
                raise RuntimeError("MPPI oracle terminal provider is unavailable")
            return self.terminal_value_provider.batch_value(env, states).astype(np.float32)

        delta = states[:, None, :2] - self._terminal_states[None, :, :2]
        position_time = np.linalg.norm(delta, axis=2) / max(float(env.v), 1e-8)
        angle_delta = _normalize_angle(
            states[:, None, 2] - self._terminal_states[None, :, 2]
        )
        angle_time = np.abs(angle_delta) / max(float(env.omega_max), 1e-8)
        return np.min(position_time + angle_time, axis=1).astype(np.float32)

    def act(
        self,
        obs: np.ndarray,
        env: CommInspectionDubinsUAV2D,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        del obs
        start = perf_counter()
        n = max(1, int(self.cfg.num_samples))
        noise = self._rng.normal(
            0.0,
            float(self.cfg.noise_sigma),
            size=(n, len(self._nominal)),
        ).astype(np.float32)
        candidates = np.clip(
            self._nominal[None, :] + noise,
            -float(env.omega_max),
            float(env.omega_max),
        )
        candidates[0] = self._nominal
        rollout = simulate_action_sequences(env, env.state, candidates)
        costs = rollout["costs"].astype(np.float64)
        unfinished = ~(rollout["success"] | rollout["invalid"])
        if np.any(unfinished) and self.terminal_mode != "none" and float(self.cfg.terminal_weight) != 0.0:
            costs[unfinished] += float(self.cfg.terminal_weight) * self._terminal_cost(
                env, rollout["final_states"][unfinished]
            )
        costs[rollout["invalid"]] += float(self.cfg.invalid_penalty)
        shifted = costs - float(np.min(costs))
        weights = np.exp(-shifted / max(float(self.cfg.temperature), 1e-6))
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            best = int(np.argmin(costs))
            self._nominal = candidates[best].copy()
        else:
            weights = weights / weight_sum
            self._nominal = np.sum(weights[:, None] * candidates, axis=0).astype(np.float32)

        action = np.array([self._nominal[0]], dtype=np.float32)
        self._nominal[:-1] = self._nominal[1:]
        self._nominal[-1] = 0.0
        self._model_rollouts += n
        elapsed = perf_counter() - start
        return action, {
            "planning_time_sec": float(elapsed),
            "model_rollouts": int(n),
            "best_rollout_cost": float(np.min(costs)),
            "invalid_rollout_ratio": float(np.mean(rollout["invalid"])),
        }

    def end_episode(self) -> Dict[str, Any]:
        diagnostics = {"model_rollouts": int(self._model_rollouts)}
        if self.terminal_mode == "oracle" and self.terminal_value_provider is not None:
            diagnostics.update(self.terminal_value_provider.end_episode())
        return diagnostics
