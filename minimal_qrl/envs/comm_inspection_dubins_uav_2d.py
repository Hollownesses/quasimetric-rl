"""
Task-conditioned point-goal communication-aware inspection Dubins UAV 2D environment.

该环境保留 point-goal 训练形式：每个 episode 仍使用单个目标状态 g=(x, y, theta)，
但该 goal 的语义不再是普通几何终点，而是在给定 inspection target / ground
station 任务上下文下采样得到的 task-conditioned terminal state。
"""
import math
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
from gymnasium import spaces

from .dubins_uav_2d import CircleObstacle, DubinsUAV2D, Obstacle


class CommInspectionDubinsUAV2D(DubinsUAV2D):
    """
    通信感知巡检 Dubins 环境。

    主训练语义：
    - 状态仍为 (x, y, theta)
    - goal 仍为单个终态 (x_g, y_g, theta_g)
    - 但 goal 来自当前任务上下文中的 joint task feasible region
    - 环境返回 reward = - stage_cost，使问题保持为路径累积 cost 的 goal reaching

    Legacy 模式：
    - observation_mode="state" 或 "cos_sin" 时，行为与普通 Dubins 更接近
    - observation_mode="task_context" 时，agent 会接收任务实体和任务可行性上下文
    """

    TASK_CONTEXT_DIM = 20

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
        omega_max: float = 1.0,
        v: float = 1.0,
        dt: float = 0.1,
        max_steps: int = 200,
        observation_mode: str = "task_context",
        obstacles: Optional[List[Union[Obstacle, CircleObstacle]]] = None,
        start: Optional[Tuple[float, float, float]] = None,
        goal: Optional[Tuple[float, float, float]] = None,
        inspection_target: Optional[Tuple[float, float]] = None,
        ground_station: Optional[Tuple[float, float]] = None,
        randomize_inspection_target: bool = False,
        randomize_ground_station: bool = False,
        min_entity_separation: float = 0.5,
        observation_radius: float = 1.5,
        fov_angle: float = np.pi / 2.0,
        require_target_los: bool = True,
        comm_alpha: float = 2.0,
        comm_bias: float = 5.0,
        comm_occlusion_penalty: float = 6.0,
        comm_threshold: float = 0.0,
        require_ground_station_los: bool = False,
        goal_sampling_mode: str = "task_feasible",
        goal_position_tolerance: float = 0.15,
        goal_heading_tolerance: float = 0.2,
        collision_cost: float = 10.0,
        out_of_bounds_cost: float = 10.0,
        communication_break_cost: float = 1.0,
        observation_violation_cost_weight: float = 1.0,
        communication_violation_cost_weight: float = 0.5,
        observation_failure_cost: float = 0.25,
        taskscore_beta_obs: float = 1.0,
        taskscore_beta_comm: float = 1.0,
        taskscore_beta_feas: float = 0.5,
        taskscore_margin_clip: float = 2.0,
        render_mode: Optional[str] = None,
        sample_max_attempts: int = 2000,
    ):
        obs_mode = observation_mode.lower()
        if obs_mode not in {"task_context", "state", "cos_sin", "xycs"}:
            raise ValueError(f"未知的 observation_mode: {observation_mode}")

        use_cos_sin_obs = obs_mode in {"cos_sin", "xycs", "task_context"}
        super().__init__(
            bounds=bounds,
            omega_max=omega_max,
            v=v,
            dt=dt,
            max_episode_steps=max_steps,
            epsilon_pos=goal_position_tolerance,
            epsilon_theta=goal_heading_tolerance,
            obstacles=obstacles,
            collision_penalty=-abs(float(collision_cost)),
            start=start,
            goal=goal,
            render_mode=render_mode,
            use_cos_sin_obs=use_cos_sin_obs,
        )

        self.observation_mode = obs_mode
        self.inspection_target = tuple(inspection_target) if inspection_target is not None else None
        self.ground_station = tuple(ground_station) if ground_station is not None else None
        self.randomize_inspection_target = bool(randomize_inspection_target)
        self.randomize_ground_station = bool(randomize_ground_station)
        self.min_entity_separation = float(min_entity_separation)

        self.observation_radius = float(observation_radius)
        self.fov_angle = float(fov_angle)
        self.require_target_los = bool(require_target_los)

        self.comm_alpha = float(comm_alpha)
        self.comm_bias = float(comm_bias)
        self.comm_occlusion_penalty = float(comm_occlusion_penalty)
        self.comm_threshold = float(comm_threshold)
        self.require_ground_station_los = bool(require_ground_station_los)

        self.goal_sampling_mode = str(goal_sampling_mode)
        self.goal_position_tolerance = float(goal_position_tolerance)
        self.goal_heading_tolerance = float(goal_heading_tolerance)
        self.collision_cost = abs(float(collision_cost))
        self.out_of_bounds_cost = abs(float(out_of_bounds_cost))
        self.communication_break_cost = abs(float(communication_break_cost))

        self.observation_violation_cost_weight = float(observation_violation_cost_weight)
        self.communication_violation_cost_weight = float(communication_violation_cost_weight)
        self.observation_failure_cost = abs(float(observation_failure_cost))
        self.taskscore_beta_obs = float(taskscore_beta_obs)
        self.taskscore_beta_comm = float(taskscore_beta_comm)
        self.taskscore_beta_feas = float(taskscore_beta_feas)
        self.taskscore_margin_clip = max(float(taskscore_margin_clip), 1e-6)

        self.sample_max_attempts = int(sample_max_attempts)
        self._ever_task_feasible = False
        self._first_task_feasible_step: Optional[int] = None

        self._configure_observation_space()

    def _configure_observation_space(self) -> None:
        if self.observation_mode == "task_context":
            max_dx = self.x_max - self.x_min
            max_dy = self.y_max - self.y_min
            max_dist = float(math.hypot(max_dx, max_dy))
            low = np.array(
                [
                    self.x_min,
                    self.y_min,
                    -1.0,
                    -1.0,
                    -max_dx,
                    -max_dy,
                    0.0,
                    -1.0,
                    -1.0,
                    -max_dx,
                    -max_dy,
                    0.0,
                    -1.0,
                    -1.0,
                    -50.0,
                    -50.0,
                    0.0,
                    0.0,
                    -50.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            high = np.array(
                [
                    self.x_max,
                    self.y_max,
                    1.0,
                    1.0,
                    max_dx,
                    max_dy,
                    max_dist,
                    1.0,
                    1.0,
                    max_dx,
                    max_dy,
                    max_dist,
                    1.0,
                    1.0,
                    50.0,
                    50.0,
                    1.0,
                    1.0,
                    50.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        elif self.observation_mode == "state":
            self.observation_space = spaces.Box(
                low=np.array([self.x_min, self.y_min, -np.pi], dtype=np.float32),
                high=np.array([self.x_max, self.y_max, np.pi], dtype=np.float32),
                shape=(3,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([self.x_min, self.y_min, -1.0, -1.0], dtype=np.float32),
                high=np.array([self.x_max, self.y_max, 1.0, 1.0], dtype=np.float32),
                shape=(4,),
                dtype=np.float32,
            )

    def _get_rng(self, seed: Optional[int] = None):
        if seed is not None:
            return np.random.default_rng(seed)
        if hasattr(self, "np_random") and self.np_random is not None:
            return self.np_random
        return np.random.default_rng()

    def _sample_valid_point(self, seed: Optional[int] = None) -> np.ndarray:
        rng = self._get_rng(seed)
        for _ in range(self.sample_max_attempts):
            x = rng.uniform(self.x_min, self.x_max)
            y = rng.uniform(self.y_min, self.y_max)
            if self._is_valid_position(float(x), float(y)):
                return np.array([x, y], dtype=np.float32)
        center = np.array(
            [(self.x_min + self.x_max) * 0.5, (self.y_min + self.y_max) * 0.5],
            dtype=np.float32,
        )
        if self._is_valid_position(float(center[0]), float(center[1])):
            return center
        for radius in np.linspace(0.15, 1.5, 10):
            for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
                cand = np.array(
                    [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)],
                    dtype=np.float32,
                )
                if self._is_valid_position(float(cand[0]), float(cand[1])):
                    return cand
        return center

    def _sample_distinct_valid_point(
        self,
        reference: Optional[Tuple[float, float]],
        min_distance: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = self._get_rng(seed)
        if reference is None:
            return self._sample_valid_point(seed=seed)
        ref = np.asarray(reference, dtype=np.float32)
        for _ in range(self.sample_max_attempts):
            cand = self._sample_valid_point(seed=int(rng.integers(0, 1_000_000_000)))
            if float(np.linalg.norm(cand - ref)) >= float(min_distance):
                return cand
        fallback = self._sample_valid_point(seed=seed)
        if float(np.linalg.norm(fallback - ref)) >= float(min_distance):
            return fallback
        direction = fallback - ref
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
            norm = 1.0
        direction = direction / norm
        shifted = ref + direction * float(min_distance)
        shifted[0] = np.clip(shifted[0], self.x_min, self.x_max)
        shifted[1] = np.clip(shifted[1], self.y_min, self.y_max)
        if self._is_valid_position(float(shifted[0]), float(shifted[1])):
            return shifted.astype(np.float32)
        return fallback.astype(np.float32)

    def sample_valid_state(self, seed: Optional[int] = None) -> np.ndarray:
        rng = self._get_rng(seed)
        for _ in range(self.sample_max_attempts):
            x = rng.uniform(self.x_min, self.x_max)
            y = rng.uniform(self.y_min, self.y_max)
            theta = rng.uniform(-np.pi, np.pi)
            state = np.array([x, y, theta], dtype=np.float32)
            if self.is_valid_state(state):
                return state
        center_x = (self.x_min + self.x_max) * 0.5
        center_y = (self.y_min + self.y_max) * 0.5
        return np.array([center_x, center_y, 0.0], dtype=np.float32)

    def _segment_has_los(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        for obs in self.obstacles:
            if obs.intersects_segment(x1, y1, x2, y2):
                return False
        return True

    def _heading_error_to_point(self, state: np.ndarray, point: Tuple[float, float]) -> float:
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        target_bearing = np.arctan2(point[1] - y, point[0] - x)
        return self._normalize_angle(target_bearing - theta)

    def _distance_to_point(self, state: np.ndarray, point: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(state[:2], dtype=np.float32) - np.asarray(point, dtype=np.float32)))

    def _ensure_valid_task_entities(self, seed: Optional[int] = None) -> None:
        if self.inspection_target is None:
            sampled = self._sample_valid_point(seed=seed)
            self.inspection_target = (float(sampled[0]), float(sampled[1]))
        else:
            tgt = np.asarray(self.inspection_target, dtype=np.float32)
            if not self._is_valid_position(float(tgt[0]), float(tgt[1])):
                sampled = self._sample_valid_point(seed=seed)
                self.inspection_target = (float(sampled[0]), float(sampled[1]))

        if self.ground_station is None:
            sampled = self._sample_distinct_valid_point(
                reference=self.inspection_target,
                min_distance=self.min_entity_separation,
                seed=None if seed is None else seed + 701,
            )
            self.ground_station = (float(sampled[0]), float(sampled[1]))
        else:
            gs = np.asarray(self.ground_station, dtype=np.float32)
            tgt = np.asarray(self.inspection_target, dtype=np.float32)
            if (not self._is_valid_position(float(gs[0]), float(gs[1]))) or float(np.linalg.norm(gs - tgt)) < self.min_entity_separation:
                sampled = self._sample_distinct_valid_point(
                    reference=self.inspection_target,
                    min_distance=self.min_entity_separation,
                    seed=None if seed is None else seed + 701,
                )
                self.ground_station = (float(sampled[0]), float(sampled[1]))

    def _sample_entities(self, seed: Optional[int] = None) -> None:
        if self.randomize_inspection_target:
            sampled = self._sample_valid_point(seed=seed)
            self.inspection_target = (float(sampled[0]), float(sampled[1]))
        if self.randomize_ground_station:
            sampled = self._sample_distinct_valid_point(
                reference=self.inspection_target,
                min_distance=self.min_entity_separation,
                seed=None if seed is None else seed + 701,
            )
            self.ground_station = (float(sampled[0]), float(sampled[1]))
        self._ensure_valid_task_entities(seed=seed)

    def compute_observation_score(self, state: np.ndarray) -> float:
        if self.inspection_target is None:
            raise ValueError("inspection_target 尚未设置")
        state = np.asarray(state, dtype=np.float32).reshape(3)
        distance = self._distance_to_point(state, self.inspection_target)
        heading_error = abs(self._heading_error_to_point(state, self.inspection_target))
        distance_margin = self.observation_radius - distance
        angle_margin = 0.5 * self.fov_angle - heading_error
        score = min(distance_margin, angle_margin)
        if self.require_target_los and not self._segment_has_los(tuple(state[:2]), self.inspection_target):
            score -= 1.0
        denom = max(self.observation_radius, 1e-6)
        return float(np.clip(score / denom, -1.0, 1.0))

    def compute_observation_margin(self, state: np.ndarray) -> float:
        if self.inspection_target is None:
            raise ValueError("inspection_target 尚未设置")
        state = np.asarray(state, dtype=np.float32).reshape(3)
        distance = self._distance_to_point(state, self.inspection_target)
        heading_error = abs(self._heading_error_to_point(state, self.inspection_target))
        distance_margin = self.observation_radius - distance
        angle_margin = 0.5 * self.fov_angle - heading_error
        margin = min(distance_margin, angle_margin)
        if self.require_target_los and not self._segment_has_los(tuple(state[:2]), self.inspection_target):
            margin -= 1.0
        return float(margin)

    def is_observation_feasible(self, state: np.ndarray) -> bool:
        if self.inspection_target is None:
            raise ValueError("inspection_target 尚未设置")
        state = np.asarray(state, dtype=np.float32).reshape(3)
        distance = self._distance_to_point(state, self.inspection_target)
        if distance > self.observation_radius:
            return False
        heading_error = abs(self._heading_error_to_point(state, self.inspection_target))
        if heading_error > 0.5 * self.fov_angle:
            return False
        if self.require_target_los and not self._segment_has_los(tuple(state[:2]), self.inspection_target):
            return False
        return True

    def compute_comm_quality(self, state: np.ndarray) -> Dict[str, float]:
        if self.ground_station is None:
            raise ValueError("ground_station 尚未设置")
        state = np.asarray(state, dtype=np.float32).reshape(3)
        distance = self._distance_to_point(state, self.ground_station)
        has_los = self._segment_has_los(tuple(state[:2]), self.ground_station)
        quality = self.comm_bias - self.comm_alpha * np.log(distance + 1e-6)
        if not has_los:
            quality -= self.comm_occlusion_penalty
        return {
            "quality": float(quality),
            "has_los": bool(has_los),
            "distance": float(distance),
            "margin": float(quality - self.comm_threshold),
        }

    def compute_communication_score(self, state: np.ndarray) -> float:
        comm = self.compute_comm_quality(state)
        denom = max(abs(self.comm_threshold) + 1.0, 1.0)
        return float(np.clip(comm["margin"] / denom, -1.0, 1.0))

    def is_communication_feasible(self, state: np.ndarray) -> bool:
        comm = self.compute_comm_quality(state)
        if self.require_ground_station_los and not comm["has_los"]:
            return False
        return bool(comm["quality"] >= self.comm_threshold)

    def is_task_feasible(self, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        return self.is_valid_state(state) and self.is_observation_feasible(state) and self.is_communication_feasible(state)

    def clip_task_margin(self, margin: float) -> float:
        clip = float(self.taskscore_margin_clip)
        return float(np.clip(float(margin), -clip, clip))

    def normalize_task_margin(self, margin: float) -> float:
        return float(self.clip_task_margin(margin) / float(self.taskscore_margin_clip))

    def compute_task_score(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        obs_norm = self.normalize_task_margin(self.compute_observation_margin(state))
        comm_norm = self.normalize_task_margin(self.compute_comm_quality(state)["margin"])
        feasible = 1.0 if self.is_task_feasible(state) else 0.0
        return float(
            self.taskscore_beta_obs * obs_norm
            + self.taskscore_beta_comm * comm_norm
            + self.taskscore_beta_feas * feasible
        )

    def is_subgoal_reached(
        self,
        state: np.ndarray,
        subgoal: np.ndarray,
        *,
        pos_tolerance: float = 0.35,
        theta_tolerance: float = 0.35,
    ) -> bool:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        subgoal = np.asarray(subgoal, dtype=np.float32).reshape(3)
        pos_dist = float(np.linalg.norm(state[:2] - subgoal[:2]))
        theta_diff = abs(self._normalize_angle(float(state[2]) - float(subgoal[2])))
        return bool(pos_dist <= float(pos_tolerance) and theta_diff <= float(theta_tolerance))

    def compute_repair_metrics(
        self,
        raw_state: np.ndarray,
        repaired_state: np.ndarray,
    ) -> Dict[str, float]:
        raw = np.asarray(raw_state, dtype=np.float32).reshape(3)
        repaired = np.asarray(repaired_state, dtype=np.float32).reshape(3)
        pos_dist = float(np.linalg.norm(raw[:2] - repaired[:2]))
        theta_diff = abs(self._normalize_angle(float(raw[2]) - float(repaired[2])))
        return {
            "repair_distance": pos_dist,
            "repair_dtheta": theta_diff,
        }

    def repair_state_with_info(self, state: np.ndarray) -> Dict[str, Union[np.ndarray, bool, float]]:
        raw = np.asarray(state, dtype=np.float32).reshape(3).copy()
        repaired = raw.copy()
        repaired[0] = np.clip(repaired[0], self.x_min, self.x_max)
        repaired[1] = np.clip(repaired[1], self.y_min, self.y_max)
        repaired[2] = self._normalize_angle(float(repaired[2]))
        info: Dict[str, Union[np.ndarray, bool, float]] = {
            "raw_state": raw.astype(np.float32),
            "used_nearby_repair": False,
            "used_global_fallback": False,
        }
        if self.is_valid_state(repaired):
            info["repaired_state"] = repaired.astype(np.float32)
            return info

        base_xy = repaired[:2].copy()
        search_radius = max(self.x_max - self.x_min, self.y_max - self.y_min) * 0.25
        best_state: Optional[np.ndarray] = None
        best_dist = float("inf")

        if self._is_valid_position(float(base_xy[0]), float(base_xy[1])):
            best_state = np.array([base_xy[0], base_xy[1], repaired[2]], dtype=np.float32)
            best_dist = float(np.linalg.norm(best_state[:2] - raw[:2]))

        for radius in np.linspace(0.05, max(search_radius, 0.05), 20):
            for angle in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
                cand_x = float(np.clip(base_xy[0] + radius * np.cos(angle), self.x_min, self.x_max))
                cand_y = float(np.clip(base_xy[1] + radius * np.sin(angle), self.y_min, self.y_max))
                if not self._is_valid_position(cand_x, cand_y):
                    continue
                cand = np.array([cand_x, cand_y, repaired[2]], dtype=np.float32)
                dist = float(np.linalg.norm(cand[:2] - raw[:2]))
                if dist < best_dist:
                    best_state = cand
                    best_dist = dist
            if best_state is not None:
                break

        if best_state is not None:
            info["repaired_state"] = best_state.astype(np.float32)
            info["used_nearby_repair"] = True
            return info
        fallback = self.sample_valid_state()
        fallback[2] = repaired[2]
        info["repaired_state"] = fallback.astype(np.float32)
        info["used_global_fallback"] = True
        return info

    def repair_state(self, state: np.ndarray) -> np.ndarray:
        repair_info = self.repair_state_with_info(state)
        return np.asarray(repair_info["repaired_state"], dtype=np.float32)

    def sample_task_feasible_goal(self, seed: Optional[int] = None) -> np.ndarray:
        rng = self._get_rng(seed)
        for _ in range(self.sample_max_attempts):
            state = np.array(
                [
                    rng.uniform(self.x_min, self.x_max),
                    rng.uniform(self.y_min, self.y_max),
                    rng.uniform(-np.pi, np.pi),
                ],
                dtype=np.float32,
            )
            if self.is_task_feasible(state):
                return state
        raise RuntimeError("未能在给定尝试次数内采样到任务可行目标状态")

    def sample_goal(self, seed: Optional[int] = None) -> np.ndarray:
        if self.goal_sampling_mode == "task_feasible":
            return self.sample_task_feasible_goal(seed=seed)
        if self.goal_sampling_mode == "valid":
            return self.sample_valid_state(seed=seed)
        raise ValueError(f"未知的 goal_sampling_mode: {self.goal_sampling_mode}")

    def _build_task_context_observation(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        x, y, theta = float(state[0]), float(state[1]), float(state[2])
        dx_t = float(self.inspection_target[0] - x)
        dy_t = float(self.inspection_target[1] - y)
        dist_t = float(np.hypot(dx_t, dy_t))
        bearing_err_t = self._heading_error_to_point(state, self.inspection_target)

        dx_gs = float(self.ground_station[0] - x)
        dy_gs = float(self.ground_station[1] - y)
        dist_gs = float(np.hypot(dx_gs, dy_gs))
        bearing_err_gs = self._heading_error_to_point(state, self.ground_station)

        comm = self.compute_comm_quality(state)
        obs_margin = self.compute_observation_margin(state)
        target_los = float(self._segment_has_los(tuple(state[:2]), self.inspection_target))
        station_los = float(comm["has_los"])
        task_feasible = float(self.is_task_feasible(state))

        return np.array(
            [
                x,
                y,
                np.cos(theta),
                np.sin(theta),
                dx_t,
                dy_t,
                dist_t,
                np.sin(bearing_err_t),
                np.cos(bearing_err_t),
                dx_gs,
                dy_gs,
                dist_gs,
                np.sin(bearing_err_gs),
                np.cos(bearing_err_gs),
                comm["quality"],
                comm["margin"],
                target_los,
                station_los,
                obs_margin,
                task_feasible,
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        return self.state_to_observation(self.state)

    def state_to_observation(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        x, y, theta = state[0], state[1], state[2]
        if self.observation_mode == "task_context":
            return self._build_task_context_observation(state)
        if self.observation_mode in {"cos_sin", "xycs"}:
            return np.array([x, y, np.cos(theta), np.sin(theta)], dtype=np.float32)
        return state.copy()

    def observation_to_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).flatten()
        if self.observation_mode == "task_context":
            x, y, c, s = obs[0], obs[1], obs[2], obs[3]
            theta = np.arctan2(s, c)
            return np.array([x, y, theta], dtype=np.float32)
        if self.observation_mode in {"cos_sin", "xycs"} and len(obs) >= 4:
            x, y, c, s = obs[0], obs[1], obs[2], obs[3]
            theta = np.arctan2(s, c)
            return np.array([x, y, theta], dtype=np.float32)
        if len(obs) >= 3:
            return np.array([obs[0], obs[1], obs[2]], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _build_info(
        self,
        state: np.ndarray,
        success: bool,
        collision: bool,
        out_of_bounds: bool,
        step_terms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Union[bool, float, int, None]]:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        comm = self.compute_comm_quality(state)
        target_has_los = True
        if self.inspection_target is not None:
            target_has_los = self._segment_has_los(tuple(state[:2]), self.inspection_target)
        distance_to_goal = self._distance_to_point(state, self.goal[:2]) if self.goal is not None else 0.0
        heading_error = abs(self._normalize_angle(float(state[2]) - float(self.goal[2]))) if self.goal is not None else 0.0
        observation_feasible = self.is_observation_feasible(state)
        communication_feasible = self.is_communication_feasible(state)
        task_feasible = observation_feasible and communication_feasible
        info: Dict[str, Union[bool, float, int, None]] = {
            "success": bool(success),
            "is_success": bool(success),
            "collision": bool(collision),
            "out_of_bounds": bool(out_of_bounds),
            "observation_feasible": bool(observation_feasible),
            "communication_feasible": bool(communication_feasible),
            "task_feasible": bool(task_feasible),
            "ever_task_feasible": bool(self._ever_task_feasible),
            "first_task_feasible_step": self._first_task_feasible_step,
            "comm_quality": float(comm["quality"]),
            "comm_margin": float(comm["margin"]),
            "comm_has_los": bool(comm["has_los"]),
            "target_has_los": bool(target_has_los),
            "distance_to_goal": float(distance_to_goal),
            "heading_error": float(heading_error),
            "distance_to_target": float(self._distance_to_point(state, self.inspection_target)),
            "distance_to_ground_station": float(comm["distance"]),
            "obs_margin": float(self.compute_observation_margin(state)),
            "task_score": float(self.compute_task_score(state)),
            "pos_dist": float(distance_to_goal),
            "theta_diff": float(heading_error),
        }
        if step_terms is not None:
            info.update({k: float(v) for k, v in step_terms.items()})
        return info

    def compute_step_terms(
        self,
        new_state: np.ndarray,
        collision: bool,
        out_of_bounds: bool,
    ) -> Dict[str, float]:
        obs_margin = self.compute_observation_margin(new_state)
        comm = self.compute_comm_quality(new_state)
        comm_margin = float(comm["margin"])
        observation_feasible = self.is_observation_feasible(new_state)
        communication_feasible = self.is_communication_feasible(new_state)

        obs_scale = max(self.observation_radius, 1e-6)
        comm_scale = max(abs(self.comm_threshold) + 1.0, 1.0)
        obs_shortfall = max(0.0, -float(obs_margin)) / obs_scale
        comm_shortfall = max(0.0, -float(comm_margin)) / comm_scale

        step_terms = {
            "cost_time": float(self.dt),
            "cost_obs_violation": float(self.observation_violation_cost_weight) * float(obs_shortfall),
            "cost_comm_violation": float(self.communication_violation_cost_weight) * float(comm_shortfall),
            "cost_obs_fail": self.observation_failure_cost if not observation_feasible else 0.0,
            "cost_comm_break": self.communication_break_cost if not communication_feasible else 0.0,
            "cost_collision": self.collision_cost if collision else 0.0,
            "cost_oob": self.out_of_bounds_cost if out_of_bounds else 0.0,
        }
        step_terms["cost_total"] = float(sum(step_terms.values()))
        step_terms["reward_total"] = -float(step_terms["cost_total"])
        return step_terms

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        gym.Env.reset(self, seed=seed)

        options = options or {}
        if "inspection_target" in options:
            self.inspection_target = tuple(options["inspection_target"])
        if "ground_station" in options:
            self.ground_station = tuple(options["ground_station"])

        self._sample_entities(seed=seed)

        start_override = options.get("start", self._fixed_start)
        goal_override = options.get("goal", self._fixed_goal)

        if start_override is not None:
            start_state = np.asarray(start_override, dtype=np.float32).reshape(3)
            start_state[2] = self._normalize_angle(float(start_state[2]))
        else:
            start_state = self.sample_valid_state(seed=seed)

        if goal_override is not None:
            goal_state = np.asarray(goal_override, dtype=np.float32).reshape(3)
            goal_state[2] = self._normalize_angle(float(goal_state[2]))
        else:
            goal_seed = None if seed is None else seed + 1000
            goal_state = self.sample_goal(seed=goal_seed)

        self.start = tuple(float(v) for v in start_state)
        self.goal = tuple(float(v) for v in goal_state)
        self.state = start_state.copy()
        self._t = 0
        self._ever_task_feasible = False
        self._first_task_feasible_step = None

        info = self._build_info(self.state, success=False, collision=False, out_of_bounds=False)
        info["inspection_target"] = tuple(self.inspection_target)
        info["ground_station"] = tuple(self.ground_station)
        info["goal"] = tuple(self.goal)
        info["observation_mode"] = self.observation_mode
        info["goal_sampling_mode"] = self.goal_sampling_mode
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        if isinstance(action, (int, float, np.number)):
            omega = float(action)
        else:
            action = np.asarray(action, dtype=np.float32).flatten()
            if len(action) != 1:
                raise ValueError(f"动作维度应为 1，得到 {len(action)}")
            omega = float(action[0])

        omega = np.clip(omega, -self.omega_max, self.omega_max)

        x, y, theta = float(self.state[0]), float(self.state[1]), float(self.state[2])
        theta_new = self._normalize_angle(theta + omega * self.dt)
        x_new = x + self.v * np.cos(theta_new) * self.dt
        y_new = y + self.v * np.sin(theta_new) * self.dt

        collision = False
        out_of_bounds = False

        if self.obstacles and self._check_collision(x, y, x_new, y_new):
            collision = True
            x_new, y_new = x, y
        elif not (self.x_min <= x_new <= self.x_max and self.y_min <= y_new <= self.y_max):
            out_of_bounds = True
            x_new, y_new = x, y
        elif any(obs.contains(x_new, y_new) for obs in self.obstacles):
            collision = True
            x_new, y_new = x, y

        self.state = np.array([x_new, y_new, theta_new], dtype=np.float32)
        self._t += 1

        distance_to_goal = self._distance_to_point(self.state, self.goal[:2])
        heading_error = abs(self._normalize_angle(float(self.state[2]) - float(self.goal[2])))
        reached_goal = (
            distance_to_goal <= self.goal_position_tolerance
            and heading_error <= self.goal_heading_tolerance
        )
        success = reached_goal and self.is_task_feasible(self.state)
        truncated = self._t >= self.max_episode_steps

        if self.is_task_feasible(self.state):
            self._ever_task_feasible = True
            if self._first_task_feasible_step is None:
                self._first_task_feasible_step = self._t

        step_terms = self.compute_step_terms(
            new_state=self.state,
            collision=collision,
            out_of_bounds=out_of_bounds,
        )
        reward = step_terms["reward_total"]

        info = self._build_info(
            self.state,
            success=success,
            collision=collision,
            out_of_bounds=out_of_bounds,
            step_terms=step_terms,
        )
        return self._get_obs(), float(reward), bool(success), bool(truncated), info

    def compute_goal_reaching_cost_estimate(
        self,
        state: np.ndarray,
        goal: np.ndarray,
    ) -> float:
        return self.compute_min_time_to_go(start=state, goal=goal)

    def get_state(self) -> dict:
        state = super().get_state()
        state.update(
            {
                "inspection_target": list(self.inspection_target) if self.inspection_target else None,
                "ground_station": list(self.ground_station) if self.ground_station else None,
                "observation_mode": self.observation_mode,
                "goal_sampling_mode": self.goal_sampling_mode,
                "taskscore_beta_obs": self.taskscore_beta_obs,
                "taskscore_beta_comm": self.taskscore_beta_comm,
                "taskscore_beta_feas": self.taskscore_beta_feas,
                "taskscore_margin_clip": self.taskscore_margin_clip,
                "ever_task_feasible": self._ever_task_feasible,
                "first_task_feasible_step": self._first_task_feasible_step,
            }
        )
        return state

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        if state.get("inspection_target") is not None:
            self.inspection_target = tuple(state["inspection_target"])
        if state.get("ground_station") is not None:
            self.ground_station = tuple(state["ground_station"])
        if state.get("observation_mode") is not None:
            self.observation_mode = str(state["observation_mode"])
            self._configure_observation_space()
        self.taskscore_beta_obs = float(state.get("taskscore_beta_obs", self.taskscore_beta_obs))
        self.taskscore_beta_comm = float(state.get("taskscore_beta_comm", self.taskscore_beta_comm))
        self.taskscore_beta_feas = float(state.get("taskscore_beta_feas", self.taskscore_beta_feas))
        self.taskscore_margin_clip = max(float(state.get("taskscore_margin_clip", self.taskscore_margin_clip)), 1e-6)
        self._ever_task_feasible = bool(state.get("ever_task_feasible", False))
        self._first_task_feasible_step = state.get("first_task_feasible_step")

    def render(self):
        if self.render_mode != "human":
            return
        info = self._build_info(self.state, success=False, collision=False, out_of_bounds=False)
        print(f"Step {self._t}:")
        print(f"  State: x={self.state[0]:.2f}, y={self.state[1]:.2f}, theta={self.state[2]:.3f}")
        print(f"  Goal (task terminal state): x={self.goal[0]:.2f}, y={self.goal[1]:.2f}, theta={self.goal[2]:.3f}")
        print(f"  Inspection target: x={self.inspection_target[0]:.2f}, y={self.inspection_target[1]:.2f}")
        print(f"  Ground station: x={self.ground_station[0]:.2f}, y={self.ground_station[1]:.2f}")
        print(
            "  Feasibility:"
            f" obs={info['observation_feasible']},"
            f" comm={info['communication_feasible']},"
            f" task={info['task_feasible']}"
        )
        print(
            f"  Distances: goal={info['distance_to_goal']:.3f},"
            f" target={info['distance_to_target']:.3f},"
            f" station={info['distance_to_ground_station']:.3f}"
        )
        print(f"  Comm quality: {info['comm_quality']:.3f} (margin={info['comm_margin']:.3f})")
        print(f"  Observation margin: {info['obs_margin']:.3f}")
        print()
