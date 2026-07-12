"""Device-catalog-driven communication-aware Dubins UAV inspection environment."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple, Union

import gym
import numpy as np
from gymnasium import spaces

from .dubins_uav_2d import CircleObstacle, DubinsUAV2D, Obstacle
from .industrial_inspection_catalog import (
    CatalogInput,
    DeviceTaskSpec,
    IndustrialInspectionCatalog,
    TaskContextInfeasibleError,
    load_device_catalog,
)


class CommInspectionDubinsUAV2D(DubinsUAV2D):
    """Fixed-site, fixed-base-station industrial inspection task environment."""

    TASK_CONTEXT_FIELDS = (
        "state_x",
        "state_y",
        "state_heading_cos",
        "state_heading_sin",
        "device_x",
        "device_y",
        "anchor_x",
        "anchor_y",
        "station_x",
        "station_y",
        "target_dx",
        "target_dy",
        "target_distance",
        "position_bearing_error_sin",
        "position_bearing_error_cos",
        "anchor_dx",
        "anchor_dy",
        "anchor_distance",
        "anchor_heading_error_sin",
        "anchor_heading_error_cos",
        "observation_min_distance",
        "observation_max_distance",
        "preferred_bearing_sin",
        "preferred_bearing_cos",
        "bearing_tolerance",
        "fov_angle",
        "distance_margin",
        "sector_margin",
        "heading_margin",
        "observation_margin",
        "target_los",
        "station_dx",
        "station_dy",
        "station_distance",
        "comm_quality",
        "comm_margin",
        "station_los",
        "valid_state",
        "observation_feasible",
        "communication_feasible",
        "task_feasible",
        "abstract_goal",
    )
    TASK_CONTEXT_DIM = len(TASK_CONTEXT_FIELDS)

    def __init__(
        self,
        *,
        device_catalog: CatalogInput,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
        omega_max: float = 1.0,
        v: float = 1.0,
        dt: float = 0.1,
        max_steps: int = 200,
        obstacles: Optional[List[Union[Obstacle, CircleObstacle]]] = None,
        start: Optional[Tuple[float, float, float]] = None,
        comm_alpha: float = 2.0,
        comm_bias: float = 5.0,
        comm_occlusion_penalty: float = 6.0,
        comm_threshold: float = 0.0,
        require_ground_station_los: bool = False,
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
        min_start_target_distance: float = 0.5,
        render_mode: Optional[str] = None,
        sample_max_attempts: int = 2000,
        validate_catalog: bool = True,
    ):
        super().__init__(
            bounds=bounds,
            omega_max=omega_max,
            v=v,
            dt=dt,
            max_episode_steps=max_steps,
            epsilon_pos=0.25,
            epsilon_theta=0.3,
            obstacles=obstacles,
            collision_penalty=-abs(float(collision_cost)),
            start=start,
            goal=None,
            render_mode=render_mode,
            use_cos_sin_obs=True,
        )
        self.device_catalog: IndustrialInspectionCatalog = load_device_catalog(device_catalog)
        self.observation_mode = "task_context"
        self.sample_max_attempts = int(sample_max_attempts)
        self.min_start_target_distance = float(min_start_target_distance)

        station = self.device_catalog.ground_station
        self.ground_station = tuple(station.position)
        self.ground_station_los_anchor = tuple(station.los_anchor)
        self.comm_alpha = float(comm_alpha)
        self.comm_bias = float(comm_bias)
        self.comm_occlusion_penalty = float(comm_occlusion_penalty)
        self.comm_threshold = float(comm_threshold)
        self.require_ground_station_los = bool(require_ground_station_los)

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

        self._active_task: DeviceTaskSpec = self.device_catalog.devices[0]
        self._active_device_index = 0
        self._activate_task(self._active_task)
        self._ever_task_feasible = False
        self._first_task_feasible_step: Optional[int] = None
        self._configure_observation_space()
        self._validate_catalog_points()
        if validate_catalog:
            self.validate_catalog_feasibility()

    @property
    def device_ids(self) -> Tuple[str, ...]:
        return tuple(device.device_id for device in self.device_catalog.devices)

    @property
    def active_device_id(self) -> str:
        return self._active_task.device_id

    @property
    def active_device_index(self) -> int:
        return int(self._active_device_index)

    @property
    def active_task(self) -> DeviceTaskSpec:
        return self._active_task

    def _configure_observation_space(self) -> None:
        low = np.full((self.TASK_CONTEXT_DIM,), -np.inf, dtype=np.float32)
        high = np.full((self.TASK_CONTEXT_DIM,), np.inf, dtype=np.float32)
        indices = {name: i for i, name in enumerate(self.TASK_CONTEXT_FIELDS)}
        self.task_context_indices = indices
        for name in (
            "state_heading_cos",
            "state_heading_sin",
            "position_bearing_error_sin",
            "position_bearing_error_cos",
            "anchor_heading_error_sin",
            "anchor_heading_error_cos",
            "preferred_bearing_sin",
            "preferred_bearing_cos",
        ):
            low[indices[name]], high[indices[name]] = -1.0, 1.0
        for name in (
            "target_distance",
            "anchor_distance",
            "observation_min_distance",
            "observation_max_distance",
            "bearing_tolerance",
            "fov_angle",
            "station_distance",
        ):
            low[indices[name]] = 0.0
        for name in (
            "target_los",
            "station_los",
            "valid_state",
            "observation_feasible",
            "communication_feasible",
            "task_feasible",
            "abstract_goal",
        ):
            low[indices[name]], high[indices[name]] = 0.0, 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_rng(self, seed: Optional[int] = None):
        if seed is not None:
            return np.random.default_rng(seed)
        if getattr(self, "np_random", None) is not None:
            return self.np_random
        return np.random.default_rng()

    def _validate_catalog_points(self) -> None:
        def in_bounds(point: Tuple[float, float]) -> bool:
            return self.x_min <= point[0] <= self.x_max and self.y_min <= point[1] <= self.y_max

        errors = []
        station = self.device_catalog.ground_station
        if not in_bounds(station.position) or not in_bounds(station.los_anchor):
            errors.append("ground_station")
        for device in self.device_catalog.devices:
            if not in_bounds(device.position) or not in_bounds(device.observation_anchor):
                errors.append(device.device_id)
        if errors:
            raise ValueError(f"catalog point(s) outside environment bounds: {', '.join(errors)}")

    def _activate_task(self, task: DeviceTaskSpec) -> None:
        self._active_task = task
        self.inspection_target = tuple(task.position)
        self.observation_anchor = tuple(task.observation_anchor)
        self.observation_min_distance = float(task.observation.min_distance)
        self.observation_max_distance = float(task.observation.max_distance)
        self.observation_radius = self.observation_max_distance
        self.preferred_bearing = float(task.observation.preferred_bearing_rad)
        self.bearing_tolerance = float(task.observation.bearing_tolerance_rad)
        self.fov_angle = float(task.observation.fov_angle_rad)
        self.require_target_los = bool(task.observation.require_los)
        try:
            self._active_device_index = self.device_ids.index(task.device_id)
        except ValueError:
            self._active_device_index = -1

    def set_task_by_device_id(self, device_id: str) -> DeviceTaskSpec:
        task = self.device_catalog.get_device(device_id)
        self._activate_task(task)
        return task

    def sample_task_context(self, seed: Optional[int] = None) -> Dict[str, object]:
        rng = self._get_rng(seed)
        index = int(rng.integers(0, len(self.device_catalog.devices)))
        self._active_device_index = index
        self._activate_task(self.device_catalog.devices[index])
        return self.task_context_info()

    def task_context_info(self) -> Dict[str, object]:
        return {
            "device_id": self.active_device_id,
            "device_index": self.active_device_index,
            "inspection_target": tuple(self.inspection_target),
            "observation_anchor": tuple(self.observation_anchor),
            "ground_station": tuple(self.ground_station),
            "ground_station_los_anchor": tuple(self.ground_station_los_anchor),
            "observation_min_distance": self.observation_min_distance,
            "observation_max_distance": self.observation_max_distance,
            "preferred_bearing_rad": self.preferred_bearing,
            "bearing_tolerance_rad": self.bearing_tolerance,
            "fov_angle_rad": self.fov_angle,
            "require_los": self.require_target_los,
        }

    def validate_catalog_feasibility(self) -> None:
        original = self._active_task
        invalid = []
        for device in self.device_catalog.devices:
            self._activate_task(device)
            try:
                self.sample_task_terminal_state(seed=1729 + self.active_device_index)
            except RuntimeError:
                invalid.append(device.device_id)
        self._activate_task(original)
        if invalid:
            raise TaskContextInfeasibleError(
                "device(s) have empty task-terminal sets: " + ", ".join(invalid)
            )

    def sample_valid_state(self, seed: Optional[int] = None) -> np.ndarray:
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
            if self.is_valid_state(state):
                return state
        raise RuntimeError("failed to sample a valid UAV state")

    def _segment_has_los(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        *,
        allow_endpoint_contact: bool = False,
    ) -> bool:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        if allow_endpoint_contact:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length > 1e-9:
                eps = min(1e-4 * max(self.x_max - self.x_min, self.y_max - self.y_min, 1.0), 0.1 * length)
                x2 -= eps * dx / length
                y2 -= eps * dy / length
        return not any(obs.intersects_segment(x1, y1, x2, y2) for obs in self.obstacles)

    def _heading_error_to_point(self, state: np.ndarray, point: Tuple[float, float]) -> float:
        bearing = math.atan2(float(point[1]) - float(state[1]), float(point[0]) - float(state[0]))
        return self._normalize_angle(bearing - float(state[2]))

    def _distance_to_point(self, state: np.ndarray, point: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(state[:2], dtype=np.float32) - np.asarray(point, dtype=np.float32)))

    def _observation_components(self, state: np.ndarray) -> Dict[str, Union[float, bool]]:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        distance = self._distance_to_point(state, self.inspection_target)
        uav_bearing = math.atan2(
            float(state[1]) - self.inspection_target[1],
            float(state[0]) - self.inspection_target[0],
        )
        position_bearing_error = self._normalize_angle(uav_bearing - self.preferred_bearing)
        heading_error = self._heading_error_to_point(state, self.observation_anchor)
        target_los = self._segment_has_los(
            tuple(state[:2]),
            self.observation_anchor,
            allow_endpoint_contact=True,
        )
        distance_margin = min(
            distance - self.observation_min_distance,
            self.observation_max_distance - distance,
        )
        sector_margin = self.bearing_tolerance - abs(position_bearing_error)
        heading_margin = 0.5 * self.fov_angle - abs(heading_error)
        margin = min(distance_margin, sector_margin, heading_margin)
        if self.require_target_los and not target_los:
            margin = min(margin, -1.0)
        feasible = (
            distance_margin >= 0.0
            and sector_margin >= 0.0
            and heading_margin >= 0.0
            and (target_los or not self.require_target_los)
        )
        return {
            "distance": distance,
            "position_bearing_error": float(position_bearing_error),
            "heading_error": float(heading_error),
            "distance_margin": float(distance_margin),
            "sector_margin": float(sector_margin),
            "heading_margin": float(heading_margin),
            "margin": float(margin),
            "target_los": bool(target_los),
            "feasible": bool(feasible),
        }

    def compute_observation_score(self, state: np.ndarray) -> float:
        margin = float(self._observation_components(state)["margin"])
        return float(np.clip(margin / max(self.observation_max_distance, 1e-6), -1.0, 1.0))

    def compute_observation_margin(self, state: np.ndarray) -> float:
        return float(self._observation_components(state)["margin"])

    def is_observation_feasible(self, state: np.ndarray) -> bool:
        return bool(self._observation_components(state)["feasible"])

    def compute_comm_quality(self, state: np.ndarray) -> Dict[str, Union[float, bool]]:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        distance = self._distance_to_point(state, self.ground_station)
        has_los = self._segment_has_los(
            tuple(state[:2]),
            self.ground_station_los_anchor,
            allow_endpoint_contact=True,
        )
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
        return float(np.clip(float(comm["margin"]) / max(abs(self.comm_threshold) + 1.0, 1.0), -1.0, 1.0))

    def is_communication_feasible(self, state: np.ndarray) -> bool:
        comm = self.compute_comm_quality(state)
        if self.require_ground_station_los and not bool(comm["has_los"]):
            return False
        return bool(float(comm["quality"]) >= self.comm_threshold)

    def is_task_feasible(self, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float32).reshape(3)
        return bool(
            self.is_valid_state(state)
            and self.is_observation_feasible(state)
            and self.is_communication_feasible(state)
        )

    def is_terminal_goal_state(self, state: np.ndarray) -> bool:
        return self.is_task_feasible(state)

    def normalize_task_margin(self, margin: float) -> float:
        return float(np.clip(float(margin), -self.taskscore_margin_clip, self.taskscore_margin_clip) / self.taskscore_margin_clip)

    def compute_task_score(self, state: np.ndarray) -> float:
        obs_norm = self.normalize_task_margin(self.compute_observation_margin(state))
        comm_norm = self.normalize_task_margin(float(self.compute_comm_quality(state)["margin"]))
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
        return bool(
            np.linalg.norm(state[:2] - subgoal[:2]) <= float(pos_tolerance)
            and abs(self._normalize_angle(float(state[2] - subgoal[2]))) <= float(theta_tolerance)
        )

    def compute_repair_metrics(self, raw_state: np.ndarray, repaired_state: np.ndarray) -> Dict[str, float]:
        raw = np.asarray(raw_state, dtype=np.float32).reshape(3)
        repaired = np.asarray(repaired_state, dtype=np.float32).reshape(3)
        return {
            "repair_distance": float(np.linalg.norm(raw[:2] - repaired[:2])),
            "repair_dtheta": abs(self._normalize_angle(float(raw[2] - repaired[2]))),
        }

    def repair_state_with_info(self, state: np.ndarray) -> Dict[str, Union[np.ndarray, bool, float]]:
        raw = np.asarray(state, dtype=np.float32).reshape(3).copy()
        repaired = raw.copy()
        repaired[0] = np.clip(repaired[0], self.x_min, self.x_max)
        repaired[1] = np.clip(repaired[1], self.y_min, self.y_max)
        repaired[2] = self._normalize_angle(float(repaired[2]))
        info: Dict[str, Union[np.ndarray, bool, float]] = {
            "raw_state": raw,
            "used_nearby_repair": False,
            "used_global_fallback": False,
        }
        if self.is_valid_state(repaired):
            info["repaired_state"] = repaired
            return info
        for radius in np.linspace(0.05, 0.25 * max(self.x_max - self.x_min, self.y_max - self.y_min), 20):
            for angle in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
                candidate = np.array(
                    [
                        np.clip(repaired[0] + radius * np.cos(angle), self.x_min, self.x_max),
                        np.clip(repaired[1] + radius * np.sin(angle), self.y_min, self.y_max),
                        repaired[2],
                    ],
                    dtype=np.float32,
                )
                if self.is_valid_state(candidate):
                    info["repaired_state"] = candidate
                    info["used_nearby_repair"] = True
                    return info
        info["repaired_state"] = self.sample_valid_state()
        info["used_global_fallback"] = True
        return info

    def repair_state(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(self.repair_state_with_info(state)["repaired_state"], dtype=np.float32)

    def sample_task_terminal_state(self, seed: Optional[int] = None) -> np.ndarray:
        rng = self._get_rng(seed)
        target = np.asarray(self.inspection_target, dtype=np.float32)
        r_min = self.observation_min_distance
        r_max = self.observation_max_distance
        for _ in range(self.sample_max_attempts):
            radius = math.sqrt(float(rng.uniform(r_min * r_min, r_max * r_max)))
            bearing = self._normalize_angle(
                self.preferred_bearing + float(rng.uniform(-self.bearing_tolerance, self.bearing_tolerance))
            )
            pos = target + radius * np.array([math.cos(bearing), math.sin(bearing)], dtype=np.float32)
            heading = math.atan2(
                self.observation_anchor[1] - float(pos[1]),
                self.observation_anchor[0] - float(pos[0]),
            )
            theta = self._normalize_angle(heading + float(rng.uniform(-0.5 * self.fov_angle, 0.5 * self.fov_angle)))
            state = np.array([pos[0], pos[1], theta], dtype=np.float32)
            if self.is_terminal_goal_state(state):
                return state

        radii = np.linspace(r_min, r_max, 24, dtype=np.float32)
        bearings = np.linspace(
            self.preferred_bearing - self.bearing_tolerance,
            self.preferred_bearing + self.bearing_tolerance,
            49,
            dtype=np.float32,
        )
        heading_offsets = np.linspace(-0.45 * self.fov_angle, 0.45 * self.fov_angle, 5)
        for radius in radii:
            for raw_bearing in bearings:
                bearing = self._normalize_angle(float(raw_bearing))
                pos = target + float(radius) * np.array([math.cos(bearing), math.sin(bearing)], dtype=np.float32)
                heading = math.atan2(
                    self.observation_anchor[1] - float(pos[1]),
                    self.observation_anchor[0] - float(pos[0]),
                )
                for offset in heading_offsets:
                    state = np.array([pos[0], pos[1], self._normalize_angle(heading + float(offset))], dtype=np.float32)
                    if self.is_terminal_goal_state(state):
                        return state
        raise RuntimeError(f"device {self.active_device_id!r} has no sampleable task-terminal state")

    def sample_task_feasible_goal(self, seed: Optional[int] = None) -> np.ndarray:
        return self.sample_task_terminal_state(seed=seed)

    def sample_goal(self, seed: Optional[int] = None) -> np.ndarray:
        return self.sample_task_terminal_state(seed=seed)

    def sample_nonterminal_valid_state(self, seed: Optional[int] = None) -> np.ndarray:
        rng = self._get_rng(seed)
        best = None
        best_distance = -np.inf
        for _ in range(self.sample_max_attempts):
            state = self.sample_valid_state(seed=int(rng.integers(0, 1_000_000_000)))
            if self.is_terminal_goal_state(state):
                continue
            distance = self._distance_to_point(state, self.inspection_target)
            if distance > best_distance:
                best, best_distance = state, distance
            if distance >= self.min_start_target_distance:
                return state
        if best is not None:
            return np.asarray(best, dtype=np.float32)
        raise RuntimeError(f"failed to sample a nonterminal start for device {self.active_device_id!r}")

    def _empty_context_values(self) -> Dict[str, float]:
        return {name: 0.0 for name in self.TASK_CONTEXT_FIELDS}

    def _context_array(self, values: Mapping[str, float]) -> np.ndarray:
        return np.asarray([float(values[name]) for name in self.TASK_CONTEXT_FIELDS], dtype=np.float32)

    def _build_task_context_observation(self, state: np.ndarray, *, abstract_goal: bool = False) -> np.ndarray:
        values = self._empty_context_values()
        values.update(
            {
                "device_x": self.inspection_target[0],
                "device_y": self.inspection_target[1],
                "anchor_x": self.observation_anchor[0],
                "anchor_y": self.observation_anchor[1],
                "station_x": self.ground_station[0],
                "station_y": self.ground_station[1],
                "observation_min_distance": self.observation_min_distance,
                "observation_max_distance": self.observation_max_distance,
                "preferred_bearing_sin": math.sin(self.preferred_bearing),
                "preferred_bearing_cos": math.cos(self.preferred_bearing),
                "bearing_tolerance": self.bearing_tolerance,
                "fov_angle": self.fov_angle,
                "abstract_goal": 1.0 if abstract_goal else 0.0,
            }
        )
        if abstract_goal:
            values["state_heading_cos"] = 1.0
            return self._context_array(values)

        state = np.asarray(state, dtype=np.float32).reshape(3)
        obs = self._observation_components(state)
        comm = self.compute_comm_quality(state)
        target_dx = self.inspection_target[0] - float(state[0])
        target_dy = self.inspection_target[1] - float(state[1])
        anchor_dx = self.observation_anchor[0] - float(state[0])
        anchor_dy = self.observation_anchor[1] - float(state[1])
        station_dx = self.ground_station[0] - float(state[0])
        station_dy = self.ground_station[1] - float(state[1])
        observation_feasible = self.is_observation_feasible(state)
        communication_feasible = self.is_communication_feasible(state)
        task_feasible = self.is_valid_state(state) and observation_feasible and communication_feasible
        values.update(
            {
                "state_x": state[0],
                "state_y": state[1],
                "state_heading_cos": math.cos(float(state[2])),
                "state_heading_sin": math.sin(float(state[2])),
                "target_dx": target_dx,
                "target_dy": target_dy,
                "target_distance": obs["distance"],
                "position_bearing_error_sin": math.sin(float(obs["position_bearing_error"])),
                "position_bearing_error_cos": math.cos(float(obs["position_bearing_error"])),
                "anchor_dx": anchor_dx,
                "anchor_dy": anchor_dy,
                "anchor_distance": math.hypot(anchor_dx, anchor_dy),
                "anchor_heading_error_sin": math.sin(float(obs["heading_error"])),
                "anchor_heading_error_cos": math.cos(float(obs["heading_error"])),
                "distance_margin": obs["distance_margin"],
                "sector_margin": obs["sector_margin"],
                "heading_margin": obs["heading_margin"],
                "observation_margin": obs["margin"],
                "target_los": float(bool(obs["target_los"])),
                "station_dx": station_dx,
                "station_dy": station_dy,
                "station_distance": comm["distance"],
                "comm_quality": comm["quality"],
                "comm_margin": comm["margin"],
                "station_los": float(bool(comm["has_los"])),
                "valid_state": float(self.is_valid_state(state)),
                "observation_feasible": float(observation_feasible),
                "communication_feasible": float(communication_feasible),
                "task_feasible": float(task_feasible),
            }
        )
        return self._context_array(values)

    def _get_obs(self) -> np.ndarray:
        return self.state_to_observation(self.state)

    def abstract_goal_observation(self) -> np.ndarray:
        return self._build_task_context_observation(np.zeros(3, dtype=np.float32), abstract_goal=True)

    def state_to_observation(self, state: np.ndarray, *, abstract_goal: bool = False) -> np.ndarray:
        return self._build_task_context_observation(state, abstract_goal=abstract_goal)

    def observation_to_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).flatten()
        return np.array([obs[0], obs[1], math.atan2(float(obs[3]), float(obs[2]))], dtype=np.float32)

    def _build_info(
        self,
        state: np.ndarray,
        success: bool,
        collision: bool,
        out_of_bounds: bool,
        step_terms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        obs = self._observation_components(state)
        comm = self.compute_comm_quality(state)
        observation_feasible = bool(obs["feasible"])
        communication_feasible = self.is_communication_feasible(state)
        task_feasible = self.is_terminal_goal_state(state)
        info: Dict[str, object] = {
            "device_id": self.active_device_id,
            "device_index": self.active_device_index,
            "success": bool(success),
            "is_success": bool(success),
            "collision": bool(collision),
            "out_of_bounds": bool(out_of_bounds),
            "observation_feasible": observation_feasible,
            "communication_feasible": communication_feasible,
            "task_feasible": task_feasible,
            "ever_task_feasible": bool(self._ever_task_feasible),
            "first_task_feasible_step": self._first_task_feasible_step,
            "comm_quality": float(comm["quality"]),
            "comm_margin": float(comm["margin"]),
            "comm_has_los": bool(comm["has_los"]),
            "target_has_los": bool(obs["target_los"]),
            "distance_to_target": float(obs["distance"]),
            "distance_to_ground_station": float(comm["distance"]),
            "obs_margin": float(obs["margin"]),
            "distance_margin": float(obs["distance_margin"]),
            "sector_margin": float(obs["sector_margin"]),
            "heading_margin": float(obs["heading_margin"]),
            "task_score": float(self.compute_task_score(state)),
        }
        if step_terms:
            info.update({key: float(value) for key, value in step_terms.items()})
        return info

    def compute_step_terms(self, new_state: np.ndarray, collision: bool, out_of_bounds: bool) -> Dict[str, float]:
        obs_margin = self.compute_observation_margin(new_state)
        comm_margin = float(self.compute_comm_quality(new_state)["margin"])
        observation_feasible = self.is_observation_feasible(new_state)
        communication_feasible = self.is_communication_feasible(new_state)
        obs_shortfall = max(0.0, -obs_margin) / max(self.observation_max_distance, 1e-6)
        comm_shortfall = max(0.0, -comm_margin) / max(abs(self.comm_threshold) + 1.0, 1.0)
        terms = {
            "cost_time": float(self.dt),
            "cost_obs_violation": self.observation_violation_cost_weight * obs_shortfall,
            "cost_comm_violation": self.communication_violation_cost_weight * comm_shortfall,
            "cost_obs_fail": self.observation_failure_cost if not observation_feasible else 0.0,
            "cost_comm_break": self.communication_break_cost if not communication_feasible else 0.0,
            "cost_collision": self.collision_cost if collision else 0.0,
            "cost_oob": self.out_of_bounds_cost if out_of_bounds else 0.0,
        }
        terms["cost_total"] = float(sum(terms.values()))
        terms["reward_total"] = -terms["cost_total"]
        return terms

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        gym.Env.reset(self, seed=seed)
        options = options or {}
        if "task_context" in options:
            raw_task = options["task_context"]
            task = raw_task if isinstance(raw_task, DeviceTaskSpec) else DeviceTaskSpec.from_dict(raw_task)
            self._activate_task(task)
            try:
                self.sample_task_terminal_state(seed=None if seed is None else seed + 311)
            except RuntimeError as exc:
                raise TaskContextInfeasibleError(
                    f"requested device {task.device_id!r} has an empty task-terminal set"
                ) from exc
        elif "device_id" in options:
            self.set_task_by_device_id(str(options["device_id"]))
        else:
            self.sample_task_context(seed=seed)

        start_override = options.get("start", self._fixed_start)
        if start_override is None:
            start_state = self.sample_nonterminal_valid_state(seed=None if seed is None else seed + 9176)
        else:
            start_state = np.asarray(start_override, dtype=np.float32).reshape(3)
            start_state[2] = self._normalize_angle(float(start_state[2]))
            if not self.is_valid_state(start_state):
                raise ValueError("explicit UAV start state is invalid")

        self.start = tuple(float(v) for v in start_state)
        self.goal = None
        self.state = start_state.copy()
        self._t = 0
        initial_feasible = self.is_terminal_goal_state(self.state)
        self._ever_task_feasible = bool(initial_feasible)
        self._first_task_feasible_step = 0 if initial_feasible else None
        info = self._build_info(self.state, success=False, collision=False, out_of_bounds=False)
        info.update(self.task_context_info())
        info["abstract_goal_observation"] = tuple(float(v) for v in self.abstract_goal_observation())
        info["observation_mode"] = self.observation_mode
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        if isinstance(action, (int, float, np.number)):
            omega = float(action)
        else:
            action = np.asarray(action, dtype=np.float32).flatten()
            if len(action) != 1:
                raise ValueError(f"action must have dimension one, got {len(action)}")
            omega = float(action[0])
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))
        x, y, theta = (float(v) for v in self.state)
        theta_new = self._normalize_angle(theta + omega * self.dt)
        x_new = x + self.v * math.cos(theta_new) * self.dt
        y_new = y + self.v * math.sin(theta_new) * self.dt
        collision = False
        out_of_bounds = False
        if self.obstacles and self._check_collision(x, y, x_new, y_new):
            collision = True
            x_new, y_new, theta_new = x, y, theta
        elif not (self.x_min <= x_new <= self.x_max and self.y_min <= y_new <= self.y_max):
            out_of_bounds = True
            x_new, y_new, theta_new = x, y, theta
        elif any(obs.contains(x_new, y_new) for obs in self.obstacles):
            collision = True
            x_new, y_new, theta_new = x, y, theta
        self.state = np.array([x_new, y_new, theta_new], dtype=np.float32)
        self._t += 1
        success = self.is_terminal_goal_state(self.state)
        terminated = bool(success or collision or out_of_bounds)
        truncated = bool(self._t >= self.max_episode_steps and not terminated)
        if success:
            self._ever_task_feasible = True
            if self._first_task_feasible_step is None:
                self._first_task_feasible_step = self._t
        terms = self.compute_step_terms(self.state, collision, out_of_bounds)
        info = self._build_info(self.state, success, collision, out_of_bounds, terms)
        return self._get_obs(), float(terms["reward_total"]), terminated, truncated, info

    def compute_goal_reaching_cost_estimate(self, state: np.ndarray, goal: np.ndarray) -> float:
        return self.compute_min_time_to_go(start=state, goal=goal)

    def get_state(self) -> dict:
        snapshot = super().get_state()
        snapshot.update(
            {
                "device_id": self.active_device_id,
                "active_task": {
                    "id": self._active_task.device_id,
                    "position": list(self._active_task.position),
                    "observation_anchor": list(self._active_task.observation_anchor),
                    "observation": {
                        "min_distance": self.observation_min_distance,
                        "max_distance": self.observation_max_distance,
                        "preferred_bearing_rad": self.preferred_bearing,
                        "bearing_tolerance_rad": self.bearing_tolerance,
                        "fov_angle_rad": self.fov_angle,
                        "require_los": self.require_target_los,
                    },
                },
                "ever_task_feasible": self._ever_task_feasible,
                "first_task_feasible_step": self._first_task_feasible_step,
            }
        )
        return snapshot

    def set_state(self, state: dict) -> None:
        super().set_state(state)
        self.goal = None
        if state.get("active_task") is not None:
            self._activate_task(DeviceTaskSpec.from_dict(state["active_task"]))
        elif state.get("device_id") is not None:
            self.set_task_by_device_id(state["device_id"])
        self._ever_task_feasible = bool(state.get("ever_task_feasible", False))
        self._first_task_feasible_step = state.get("first_task_feasible_step")

    def render(self):
        if self.render_mode != "human":
            return
        info = self._build_info(self.state, False, False, False)
        print(f"Step {self._t}: state={self.state.tolist()}")
        print(f"  Device: {self.active_device_id} at {self.inspection_target}")
        print(f"  Observation anchor: {self.observation_anchor}")
        print(f"  Ground station: {self.ground_station}")
        print(
            "  Feasibility: "
            f"obs={info['observation_feasible']}, "
            f"comm={info['communication_feasible']}, "
            f"task={info['task_feasible']}"
        )
