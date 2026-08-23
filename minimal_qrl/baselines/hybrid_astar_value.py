"""Exhaustive cost-to-go oracle on the Hybrid A* state/action lattice.

The regular Hybrid A* controller performs a forward search from one state.  MPPI
needs values for hundreds of terminal states at every decision, so repeated
forward searches are prohibitively expensive.  This module freezes the same
position/heading discretization and motion primitives into a directed graph,
then runs one reverse Dijkstra search from the task-feasible set.  The resulting
table is the exact cost-to-go for that finite Hybrid A* lattice and uses the
environment's unchanged running cost.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np

from minimal_qrl.baselines.hybrid_astar import HybridAStarConfig
from minimal_qrl.baselines.mppi import (
    _normalize_angle,
    _segment_hits_obstacle,
    _state_terms,
)
from minimal_qrl.envs import CircleObstacle, CommInspectionDubinsUAV2D


ORACLE_VALUE_SCHEMA_VERSION = 1


def _obstacle_payload(obstacle: Any) -> Dict[str, Any]:
    fields = {
        key: float(value)
        for key, value in vars(obstacle).items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    return {"type": type(obstacle).__name__, **fields}


def _signature_payload(
    env: CommInspectionDubinsUAV2D,
    config: HybridAStarConfig,
) -> Dict[str, Any]:
    task = env.get_state().get("active_task")
    return {
        "schema_version": ORACLE_VALUE_SCHEMA_VERSION,
        "task": task,
        "ground_station": [float(value) for value in env.ground_station],
        "ground_station_los_anchor": [
            float(value) for value in env.ground_station_los_anchor
        ],
        "bounds": [float(value) for value in env.bounds],
        "omega_max": float(env.omega_max),
        "v": float(env.v),
        "dt": float(env.dt),
        "obstacles": [_obstacle_payload(item) for item in env.obstacles],
        "comm_alpha": float(env.comm_alpha),
        "comm_bias": float(env.comm_bias),
        "comm_occlusion_penalty": float(env.comm_occlusion_penalty),
        "comm_threshold": float(env.comm_threshold),
        "require_ground_station_los": bool(env.require_ground_station_los),
        "collision_cost": float(env.collision_cost),
        "out_of_bounds_cost": float(env.out_of_bounds_cost),
        "communication_break_cost": float(env.communication_break_cost),
        "observation_violation_cost_weight": float(
            env.observation_violation_cost_weight
        ),
        "communication_violation_cost_weight": float(
            env.communication_violation_cost_weight
        ),
        "observation_failure_cost": float(env.observation_failure_cost),
        "hybrid_astar_config": asdict(config),
    }


def _digest(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HybridAStarValueOracle:
    """Cached exact cost-to-go table for the finite Hybrid A* lattice."""

    def __init__(
        self,
        config: HybridAStarConfig,
        *,
        cache_dir: Optional[str | Path] = None,
        unreachable_cost: float = 1_000_000.0,
    ) -> None:
        if float(config.position_resolution) <= 0.0:
            raise ValueError("position_resolution must be positive")
        if int(config.heading_bins) <= 0:
            raise ValueError("heading_bins must be positive")
        if int(config.primitive_steps) <= 0:
            raise ValueError("primitive_steps must be positive")
        if not config.primitive_scales:
            raise ValueError("primitive_scales must not be empty")
        self.config = config
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.unreachable_cost = float(unreachable_cost)
        self._tables: Dict[str, np.ndarray] = {}
        self._table_diagnostics: Dict[str, Dict[str, Any]] = {}
        self._active_digest: Optional[str] = None
        self._active_values = np.zeros((0,), dtype=np.float32)
        self._grid_shape = (0, 0, 0)
        self._episode_queries = 0
        self._episode_unreachable = 0
        self._episode_diagnostics: Dict[str, Any] = {}

    def _grid(self, env: CommInspectionDubinsUAV2D) -> tuple[np.ndarray, tuple[int, int, int]]:
        resolution = float(self.config.position_resolution)
        nx = int(math.floor((float(env.x_max) - float(env.x_min)) / resolution + 0.5)) + 1
        ny = int(math.floor((float(env.y_max) - float(env.y_min)) / resolution + 0.5)) + 1
        nt = int(self.config.heading_bins)
        xs = np.minimum(
            float(env.x_min) + np.arange(nx, dtype=np.float32) * resolution,
            float(env.x_max),
        )
        ys = np.minimum(
            float(env.y_min) + np.arange(ny, dtype=np.float32) * resolution,
            float(env.y_max),
        )
        width = 2.0 * np.pi / nt
        headings = -np.pi + (np.arange(nt, dtype=np.float32) + 0.5) * width
        x_grid, y_grid, heading_grid = np.meshgrid(
            xs,
            ys,
            headings,
            indexing="ij",
        )
        states = np.stack(
            [x_grid.ravel(), y_grid.ravel(), heading_grid.ravel()],
            axis=1,
        ).astype(np.float32)
        return states, (nx, ny, nt)

    @staticmethod
    def _valid_grid_states(
        env: CommInspectionDubinsUAV2D,
        states: np.ndarray,
    ) -> np.ndarray:
        valid = (
            (states[:, 0] >= float(env.x_min))
            & (states[:, 0] <= float(env.x_max))
            & (states[:, 1] >= float(env.y_min))
            & (states[:, 1] <= float(env.y_max))
        )
        for obstacle in env.obstacles:
            if isinstance(obstacle, CircleObstacle):
                occupied = (
                    (states[:, 0] - float(obstacle.x)) ** 2
                    + (states[:, 1] - float(obstacle.y)) ** 2
                    <= float(obstacle.radius) ** 2
                )
            else:
                occupied = (
                    (states[:, 0] >= float(obstacle.x_min))
                    & (states[:, 0] <= float(obstacle.x_max))
                    & (states[:, 1] >= float(obstacle.y_min))
                    & (states[:, 1] <= float(obstacle.y_max))
                )
            valid &= ~occupied
        return valid

    def _state_indices(
        self,
        env: CommInspectionDubinsUAV2D,
        states: np.ndarray,
    ) -> np.ndarray:
        nx, ny, nt = self._grid_shape
        resolution = float(self.config.position_resolution)
        ix = np.floor(
            (states[:, 0] - float(env.x_min)) / resolution + 0.5
        ).astype(np.int64)
        iy = np.floor(
            (states[:, 1] - float(env.y_min)) / resolution + 0.5
        ).astype(np.int64)
        angle = (states[:, 2] + np.pi) % (2.0 * np.pi)
        it = np.floor(angle / (2.0 * np.pi) * nt).astype(np.int64) % nt
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        flat = ((np.clip(ix, 0, nx - 1) * ny + np.clip(iy, 0, ny - 1)) * nt + it)
        flat[~inside] = -1
        return flat

    def _primitive_edges(
        self,
        env: CommInspectionDubinsUAV2D,
        grid_states: np.ndarray,
        valid_sources: np.ndarray,
        terminal_sources: np.ndarray,
        omega: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = grid_states.copy()
        source_active = valid_sources & ~terminal_sources
        active = source_active.copy()
        invalid = np.zeros((len(states),), dtype=bool)
        success = np.zeros((len(states),), dtype=bool)
        costs = np.zeros((len(states),), dtype=np.float32)

        for _ in range(max(1, int(self.config.primitive_steps))):
            if not np.any(active):
                break
            current = states.copy()
            theta_new = _normalize_angle(current[:, 2] + float(omega) * float(env.dt))
            x_new = current[:, 0] + float(env.v) * np.cos(theta_new) * float(env.dt)
            y_new = current[:, 1] + float(env.v) * np.sin(theta_new) * float(env.dt)
            collision = np.zeros((len(states),), dtype=bool)
            for obstacle in env.obstacles:
                collision |= _segment_hits_obstacle(
                    current[:, 0],
                    current[:, 1],
                    x_new,
                    y_new,
                    obstacle,
                )
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
            step_cost, step_success = _state_terms(
                env,
                states,
                collision,
                out_of_bounds,
            )
            costs[active] += step_cost[active]
            newly_successful = active & step_success
            success |= newly_successful
            invalid |= collision | out_of_bounds
            active &= ~(newly_successful | collision | out_of_bounds)

        usable = source_active & ~invalid
        sources = np.flatnonzero(usable)
        destinations = self._state_indices(env, states[usable])
        edge_costs = costs[usable]
        edge_success = success[usable]
        return sources, destinations, edge_costs, edge_success

    def _build_table(
        self,
        env: CommInspectionDubinsUAV2D,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        started = perf_counter()
        grid_states, self._grid_shape = self._grid(env)
        valid = self._valid_grid_states(env, grid_states)
        zeros = np.zeros((len(grid_states),), dtype=bool)
        _unused_cost, feasible = _state_terms(env, grid_states, zeros, zeros)
        feasible &= valid

        edge_sources = []
        edge_destinations = []
        edge_costs = []
        direct_goal_sources = []
        direct_goal_costs = []
        for scale in self.config.primitive_scales:
            sources, destinations, costs, success = self._primitive_edges(
                env,
                grid_states,
                valid,
                feasible,
                float(scale) * float(env.omega_max),
            )
            successful = success
            if np.any(successful):
                direct_goal_sources.append(sources[successful])
                direct_goal_costs.append(costs[successful])
            unfinished = ~successful & (destinations >= 0)
            if np.any(unfinished):
                destination_valid = valid[destinations[unfinished]]
                selected = np.flatnonzero(unfinished)[destination_valid]
                edge_sources.append(sources[selected])
                edge_destinations.append(destinations[selected])
                edge_costs.append(costs[selected])

        count = len(grid_states)
        sources = np.concatenate(edge_sources) if edge_sources else np.zeros((0,), dtype=np.int64)
        destinations = (
            np.concatenate(edge_destinations)
            if edge_destinations
            else np.zeros((0,), dtype=np.int64)
        )
        costs = np.concatenate(edge_costs) if edge_costs else np.zeros((0,), dtype=np.float32)
        order = np.argsort(destinations, kind="stable")
        destinations = destinations[order]
        sources = sources[order]
        costs = costs[order]
        offsets = np.zeros((count + 1,), dtype=np.int64)
        if len(destinations):
            offsets[1:] = np.cumsum(np.bincount(destinations, minlength=count))

        values = np.full((count,), np.inf, dtype=np.float64)
        values[feasible] = 0.0
        if direct_goal_sources:
            goal_sources = np.concatenate(direct_goal_sources)
            goal_costs = np.concatenate(direct_goal_costs).astype(np.float64)
            np.minimum.at(values, goal_sources, goal_costs)
        queue = [(float(values[index]), int(index)) for index in np.flatnonzero(np.isfinite(values))]
        heapq.heapify(queue)
        while queue:
            current_cost, destination = heapq.heappop(queue)
            if current_cost > float(values[destination]) + 1e-12:
                continue
            begin = int(offsets[destination])
            end = int(offsets[destination + 1])
            for edge_index in range(begin, end):
                source = int(sources[edge_index])
                candidate = current_cost + float(costs[edge_index])
                if candidate + 1e-12 < float(values[source]):
                    values[source] = candidate
                    heapq.heappush(queue, (candidate, source))

        values = values.astype(np.float32)
        elapsed = perf_counter() - started
        reachable = np.isfinite(values) & valid
        diagnostics = {
            "oracle_value_source": "hybrid_astar_lattice_reverse_dijkstra",
            "oracle_value_grid_states": int(count),
            "oracle_value_valid_grid_states": int(np.sum(valid)),
            "oracle_value_goal_grid_states": int(np.sum(feasible)),
            "oracle_value_graph_edges": int(len(sources)),
            "oracle_value_reachable_grid_states": int(np.sum(reachable)),
            "oracle_value_reachable_fraction": float(
                np.sum(reachable) / max(int(np.sum(valid)), 1)
            ),
            "oracle_value_build_time_sec": float(elapsed),
        }
        return values, diagnostics

    def _cache_path(self, digest: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"hybrid_astar_value_{digest[:20]}.npz"

    def _load_cache(
        self,
        path: Path,
        *,
        digest: str,
        expected_count: int,
    ) -> Optional[tuple[np.ndarray, Dict[str, Any]]]:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["digest"].item()) != digest:
                    return None
                values = np.asarray(data["values"], dtype=np.float32)
                diagnostics = json.loads(str(data["diagnostics"].item()))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None
        if values.shape != (expected_count,):
            return None
        diagnostics["oracle_value_cache_hit"] = True
        diagnostics["oracle_value_cache_path"] = str(path.resolve())
        diagnostics["oracle_value_build_time_sec"] = 0.0
        return values, diagnostics

    @staticmethod
    def _write_cache(
        path: Path,
        *,
        digest: str,
        values: np.ndarray,
        diagnostics: Dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                digest=np.asarray(digest),
                values=np.asarray(values, dtype=np.float32),
                diagnostics=np.asarray(json.dumps(diagnostics, sort_keys=True)),
            )
        os.replace(temporary, path)

    def begin_episode(
        self,
        env: CommInspectionDubinsUAV2D,
        *,
        seed: int,
    ) -> Dict[str, Any]:
        del seed
        payload = _signature_payload(env, self.config)
        digest = _digest(payload)
        _grid, self._grid_shape = self._grid(env)
        count = int(np.prod(self._grid_shape))
        cache_path = self._cache_path(digest)
        cache_hit = False
        diagnostics: Dict[str, Any]
        if digest in self._tables:
            values = self._tables[digest]
            diagnostics = {
                **self._table_diagnostics[digest],
                "oracle_value_cache_hit": True,
                "oracle_value_build_time_sec": 0.0,
            }
            cache_hit = True
        else:
            loaded = (
                self._load_cache(cache_path, digest=digest, expected_count=count)
                if cache_path is not None
                else None
            )
            if loaded is not None:
                values, diagnostics = loaded
                cache_hit = True
            else:
                values, diagnostics = self._build_table(env)
                diagnostics["oracle_value_cache_hit"] = False
                if cache_path is not None:
                    diagnostics["oracle_value_cache_path"] = str(cache_path.resolve())
                    self._write_cache(
                        cache_path,
                        digest=digest,
                        values=values,
                        diagnostics=diagnostics,
                    )
            self._tables[digest] = values
            self._table_diagnostics[digest] = dict(diagnostics)
        diagnostics["oracle_value_cache_hit"] = bool(cache_hit)
        diagnostics["oracle_value_table_digest"] = digest
        self._active_digest = digest
        self._active_values = values
        self._episode_queries = 0
        self._episode_unreachable = 0
        self._episode_diagnostics = dict(diagnostics)
        return dict(diagnostics)

    def batch_value(
        self,
        env: CommInspectionDubinsUAV2D,
        states: np.ndarray,
    ) -> np.ndarray:
        if self._active_digest is None or self._active_values.size == 0:
            raise RuntimeError("begin_episode must be called before batch_value")
        states = np.asarray(states, dtype=np.float32).reshape((-1, 3))
        indices = self._state_indices(env, states)
        values = np.full((len(states),), self.unreachable_cost, dtype=np.float32)
        inside = indices >= 0
        if np.any(inside):
            looked_up = self._active_values[indices[inside]]
            finite = np.isfinite(looked_up)
            values_inside = np.full(
                (int(np.sum(inside)),),
                self.unreachable_cost,
                dtype=np.float32,
            )
            values_inside[finite] = looked_up[finite]
            values[inside] = values_inside
        feasible = np.asarray([env.is_task_feasible(state) for state in states], dtype=bool)
        values[feasible] = 0.0
        unreachable = (~feasible) & (values >= self.unreachable_cost)
        self._episode_queries += int(len(states))
        self._episode_unreachable += int(np.sum(unreachable))
        return values

    def end_episode(self) -> Dict[str, Any]:
        return {
            "oracle_value_queries": int(self._episode_queries),
            "oracle_value_unreachable_queries": int(self._episode_unreachable),
            "oracle_value_unreachable_ratio": float(
                self._episode_unreachable / max(self._episode_queries, 1)
            ),
        }
