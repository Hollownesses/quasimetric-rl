"""Hand-designed long-horizon diagnostic scenario and fixed challenge tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs


SCENARIO_ID = "diagnostic_u_shadow_corridors_v1"
CHALLENGE_STRATA = ("u_trap", "comm_shadow_corridor", "easy_open")
TASK_BANK_SEED = 20260802


def _observation(preferred_bearing: float) -> dict[str, Any]:
    return {
        "min_distance": 0.35,
        "max_distance": 0.85,
        "preferred_bearing_rad": float(preferred_bearing),
        "bearing_tolerance_rad": 0.5,
        "fov_angle_rad": 0.5 * math.pi,
        "require_los": True,
    }


def _device(device_id: str, position: Sequence[float], preferred_bearing: float) -> dict[str, Any]:
    point = [float(position[0]), float(position[1])]
    return {
        "id": str(device_id),
        "position": point,
        "observation_anchor": list(point),
        "observation": _observation(preferred_bearing),
    }


def build_diagnostic_scenario() -> dict[str, Any]:
    """Build one map containing three deliberately different task regimes.

    The U opens to the west while its target is east of the closed wall.  The
    long centre block creates geometrically comparable upper/lower passages;
    the upper passage is radio-shadowed from the ground station, while the
    lower passage keeps line of sight.  Two devices remain in open space so
    aggregate scores cannot improve merely by solving hard cases differently.
    """

    scenario = {
        "schema_version": 2,
        "scenario_family": "diagnostic",
        "scenario_id": SCENARIO_ID,
        "experiment_axes": ["topological_challenge"],
        "meters_per_env_unit": 10.0,
        "physical_width_m": 180.0,
        "physical_height_m": 120.0,
        "physical_area_m2": 21_600.0,
        "bounds": [0.0, 0.0, 18.0, 12.0],
        "device_count": 4,
        "topology": "u_trap_plus_radio_shadow_corridors",
        "max_episode_steps": 300,
        "device_catalog": {
            "ground_station": {
                "position": [7.0, 3.0],
                "los_anchor": [7.0, 3.0],
            },
            "devices": [
                _device("u_trap_target", [6.7, 3.65], 0.0),
                _device("corridor_target", [16.0, 6.5], -0.5 * math.pi),
                _device("easy_north", [7.8, 10.4], 0.0),
                _device("easy_south", [11.0, 1.8], math.pi),
            ],
        },
        "obstacles": [
            {
                "id": "u_top_wall",
                "type": "rectangle",
                "x_min": 2.8,
                "x_max": 5.7,
                "y_min": 5.4,
                "y_max": 5.9,
            },
            {
                "id": "u_bottom_wall",
                "type": "rectangle",
                "x_min": 2.8,
                "x_max": 5.7,
                "y_min": 1.4,
                "y_max": 1.9,
            },
            {
                "id": "u_closed_wall",
                "type": "rectangle",
                "x_min": 5.2,
                "x_max": 5.7,
                "y_min": 1.4,
                "y_max": 5.9,
            },
            {
                "id": "long_radio_shadow_block",
                "type": "rectangle",
                "x_min": 9.0,
                "x_max": 15.0,
                "y_min": 6.0,
                "y_max": 7.5,
            },
        ],
        "omega_max": 3.0,
        "v": 1.0,
        "dt": 0.1,
        "comm_alpha": 2.0,
        "comm_bias": 8.0,
        "comm_occlusion_penalty": 4.0,
        "comm_threshold": 0.5,
        "require_ground_station_los": False,
        "collision_cost": 10.0,
        "out_of_bounds_cost": 10.0,
        "communication_break_cost": 1.0,
        "observation_violation_cost_weight": 1.0,
        "communication_violation_cost_weight": 0.5,
        "observation_failure_cost": 0.25,
        "taskscore_beta_obs": 1.0,
        "taskscore_beta_comm": 1.0,
        "taskscore_beta_feas": 0.5,
        "taskscore_margin_clip": 2.0,
        "min_start_target_distance": 0.5,
        "metadata": {
            "coordinate_interpretation": "1 environment unit = 10 metres",
            "diagnostic_only": True,
            "u_opening_direction": "west",
            "u_target_side": "east_of_closed_wall",
            "corridor_safe_route": "lower",
            "corridor_shadow_route": "upper",
            "task_strata": list(CHALLENGE_STRATA),
            "exploration_diagnostic_regions": {
                "u_trap_interior": [2.8, 1.9, 5.2, 5.4],
                "u_trap_west_exit": [0.0, 1.9, 2.8, 5.4],
                "corridor_west_entry": [7.5, 4.5, 9.0, 9.0],
                "corridor_upper": [9.0, 7.5, 15.0, 12.0],
                "corridor_lower": [9.0, 0.0, 15.0, 6.0],
                "corridor_east_exit": [15.0, 4.5, 17.0, 9.0],
            },
            "exploration_diagnostic_routes": {
                "u_inside_to_exit": ["u_trap_interior", "u_trap_west_exit"],
                "corridor_upper_complete": [
                    "corridor_west_entry",
                    "corridor_upper",
                    "corridor_east_exit",
                ],
                "corridor_lower_complete": [
                    "corridor_west_entry",
                    "corridor_lower",
                    "corridor_east_exit",
                ],
            },
            "exploration_start_strata": [
                {
                    "name": "u_trap_interior",
                    "weight": 0.30,
                    "bounds": [2.9, 2.0, 5.1, 5.3],
                },
                {
                    "name": "u_trap_exit",
                    "weight": 0.15,
                    "bounds": [1.5, 2.0, 2.8, 5.3],
                },
                {
                    "name": "corridor_fork",
                    "weight": 0.20,
                    "bounds": [7.5, 4.0, 8.9, 9.0],
                },
                {
                    "name": "comm_shadow_boundary",
                    "weight": 0.15,
                    "bounds": [8.5, 7.6, 10.5, 10.5],
                },
                {
                    "name": "uniform_free_space",
                    "weight": 0.20,
                    "bounds": [0.0, 0.0, 18.0, 12.0],
                },
            ],
            "u_trap_local_navigability_probes": {
                "device_id": "u_trap_target",
                "centerline_y": 3.65,
                "positions": [
                    {"label": "deep", "x": 4.8},
                    {"label": "middle", "x": 4.0},
                    {"label": "mouth", "x": 3.2},
                    {"label": "outside", "x": 2.4},
                ],
                "headings": [
                    {"label": "west", "theta": math.pi},
                    {"label": "north", "theta": 0.5 * math.pi},
                    {"label": "east", "theta": 0.0},
                    {"label": "south", "theta": -0.5 * math.pi},
                ],
            },
        },
    }
    # This validates both the schema and catalog terminal-set feasibility.
    CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    return scenario


def _fixed_starts() -> dict[str, list[tuple[str, tuple[float, float, float]]]]:
    """Source-controlled starts; ordering determines validation/test split."""

    return {
        "u_trap": [
            ("u_trap_target", (3.70, 2.60, 3.05)),
            ("u_trap_target", (3.70, 4.60, -3.05)),
            ("u_trap_target", (4.10, 3.10, 3.12)),
            ("u_trap_target", (4.10, 4.20, -3.12)),
            ("u_trap_target", (4.45, 2.45, 2.95)),
            ("u_trap_target", (4.45, 4.80, -2.95)),
            ("u_trap_target", (4.75, 3.15, 3.00)),
            ("u_trap_target", (4.75, 4.15, -3.00)),
            ("u_trap_target", (3.35, 2.30, 3.08)),
            ("u_trap_target", (3.35, 5.00, -3.08)),
            ("u_trap_target", (3.95, 2.25, 2.90)),
            ("u_trap_target", (3.95, 5.05, -2.90)),
            ("u_trap_target", (4.60, 2.20, 2.85)),
            ("u_trap_target", (4.60, 5.10, -2.85)),
            ("u_trap_target", (4.90, 3.45, 3.10)),
            ("u_trap_target", (4.90, 3.90, -3.10)),
        ],
        "comm_shadow_corridor": [
            ("corridor_target", (7.75, 6.45, 0.00)),
            ("corridor_target", (7.75, 7.05, 0.05)),
            ("corridor_target", (8.05, 6.55, -0.05)),
            ("corridor_target", (8.05, 6.95, 0.10)),
            ("corridor_target", (7.60, 6.65, -0.10)),
            ("corridor_target", (7.60, 6.85, 0.15)),
            ("corridor_target", (7.90, 6.35, -0.15)),
            ("corridor_target", (7.90, 7.15, 0.20)),
            ("corridor_target", (8.20, 6.40, -0.20)),
            ("corridor_target", (8.20, 7.10, 0.25)),
            ("corridor_target", (7.45, 6.50, -0.25)),
            ("corridor_target", (7.45, 7.00, 0.12)),
            ("corridor_target", (7.70, 6.25, -0.12)),
            ("corridor_target", (7.70, 7.25, 0.18)),
            ("corridor_target", (8.30, 6.60, -0.18)),
            ("corridor_target", (8.30, 6.90, 0.08)),
        ],
        "easy_open": [
            ("easy_north", (6.10, 10.10, 0.05)),
            ("easy_north", (6.10, 10.70, -0.05)),
            ("easy_south", (8.00, 1.55, 0.08)),
            ("easy_south", (8.00, 2.05, -0.08)),
            ("easy_north", (5.80, 10.25, 0.10)),
            ("easy_north", (5.80, 10.55, -0.10)),
            ("easy_south", (7.70, 1.45, 0.12)),
            ("easy_south", (7.70, 2.15, -0.12)),
            ("easy_north", (6.35, 9.95, 0.15)),
            ("easy_north", (6.35, 10.85, -0.15)),
            ("easy_south", (8.30, 1.35, 0.18)),
            ("easy_south", (8.30, 2.25, -0.18)),
            ("easy_north", (5.55, 10.00, 0.20)),
            ("easy_north", (5.55, 10.80, -0.20)),
            ("easy_south", (7.45, 1.65, 0.22)),
            ("easy_south", (7.45, 1.95, -0.22)),
        ],
    }


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_diagnostic_task_bank(
    scenario: Mapping[str, Any] | None = None,
    *,
    validation_per_stratum: int = 4,
) -> dict[str, Any]:
    """Build the fixed, balanced task library used for paired comparisons."""

    scenario = dict(scenario or build_diagnostic_scenario())
    if str(scenario["scenario_id"]) != SCENARIO_ID:
        raise ValueError(f"task bank requires scenario_id={SCENARIO_ID}")
    starts = _fixed_starts()
    validation_count = int(validation_per_stratum)
    if validation_count <= 0:
        raise ValueError("validation_per_stratum must be positive")
    if any(validation_count >= len(rows) for rows in starts.values()):
        raise ValueError("each stratum must retain at least one test task")

    bounds = [float(v) for v in scenario["bounds"]]
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    device_ids = [str(item["id"]) for item in scenario["device_catalog"]["devices"]]
    device_indices = {device_id: index for index, device_id in enumerate(device_ids)}
    records: list[dict[str, Any]] = []
    for stratum_index, stratum in enumerate(CHALLENGE_STRATA):
        for index, (device_id, start) in enumerate(starts[stratum]):
            split = "validation" if index < validation_count else "test"
            split_index = index if split == "validation" else index - validation_count
            seed = TASK_BANK_SEED + stratum_index * 100_003 + index * 101
            records.append(
                {
                    "task_id": f"{SCENARIO_ID}:{split}:{stratum}:{split_index:03d}",
                    "scenario_id": SCENARIO_ID,
                    "split": split,
                    "stratum": stratum,
                    "difficulty": "easy" if stratum == "easy_open" else "hard",
                    "device_id": device_id,
                    "device_index": int(device_indices[device_id]),
                    "sample_index": int(split_index),
                    "stratum_index": int(split_index),
                    "seed": int(seed),
                    "start_normalized": [
                        (float(start[0]) - bounds[0]) / width,
                        (float(start[1]) - bounds[1]) / height,
                        float(start[2]),
                    ],
                }
            )

    payload: dict[str, Any] = {
        "schema_version": 3,
        "generation_mode": "fixed_source_controlled_stratified_starts",
        "generation_seed": TASK_BANK_SEED,
        "scenario_id": SCENARIO_ID,
        "strata": list(CHALLENGE_STRATA),
        "validation_per_stratum": validation_count,
        "test_per_stratum": len(starts[CHALLENGE_STRATA[0]]) - validation_count,
        "records": records,
    }
    _validate_task_bank(payload, scenario)
    payload["content_digest"] = _digest(payload)
    return payload


def _validate_task_bank(bank: Mapping[str, Any], scenario: Mapping[str, Any]) -> None:
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    bounds = [float(v) for v in scenario["bounds"]]
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    counts: dict[tuple[str, str], int] = {}
    for row in bank["records"]:
        normalized = row["start_normalized"]
        start = np.asarray(
            [
                bounds[0] + float(normalized[0]) * width,
                bounds[1] + float(normalized[1]) * height,
                float(normalized[2]),
            ],
            dtype=np.float32,
        )
        env.reset(
            seed=int(row["seed"]),
            options={"device_id": str(row["device_id"]), "start": start},
        )
        if not env.is_valid_state(start) or env.is_terminal_goal_state(start):
            raise ValueError(f"invalid diagnostic task start: {row['task_id']}")
        key = (str(row["split"]), str(row["stratum"]))
        counts[key] = counts.get(key, 0) + 1
        if row["stratum"] == "u_trap" and env._segment_has_los(  # noqa: SLF001
            tuple(start[:2]), tuple(env.inspection_target)
        ):
            raise ValueError(f"U-trap direct path is not blocked: {row['task_id']}")
        if row["stratum"] == "u_trap":
            target_delta = np.asarray(env.inspection_target, dtype=np.float32) - start[:2]
            forward = np.asarray([math.cos(float(start[2])), math.sin(float(start[2]))])
            if float(np.dot(target_delta, forward)) >= 0.0:
                raise ValueError(f"U-trap start does not initially move away: {row['task_id']}")
            if not env.is_communication_feasible(start):
                raise ValueError(f"U-trap start is not initially connected: {row['task_id']}")
        if row["stratum"] == "comm_shadow_corridor":
            if not env.is_communication_feasible(start):
                raise ValueError(f"corridor start is not initially connected: {row['task_id']}")
            if env._segment_has_los(tuple(start[:2]), tuple(env.inspection_target)):  # noqa: SLF001
                raise ValueError(f"corridor direct path is not blocked: {row['task_id']}")
        if row["stratum"] == "easy_open":
            if not env.is_communication_feasible(start):
                raise ValueError(f"easy start is not connected: {row['task_id']}")
            if not env._segment_has_los(tuple(start[:2]), tuple(env.inspection_target)):  # noqa: SLF001
                raise ValueError(f"easy task direct path is blocked: {row['task_id']}")

    for split in ("validation", "test"):
        split_counts = [counts.get((split, stratum), 0) for stratum in CHALLENGE_STRATA]
        if not split_counts or len(set(split_counts)) != 1 or split_counts[0] <= 0:
            raise ValueError(f"{split} tasks are not balanced across strata: {split_counts}")

    env.set_task_by_device_id("corridor_target")
    upper = [
        env.is_communication_feasible(np.asarray([x, 8.0, 0.0], dtype=np.float32))
        for x in np.linspace(10.0, 14.8, 9)
    ]
    lower = [
        env.is_communication_feasible(np.asarray([x, 5.5, 0.0], dtype=np.float32))
        for x in np.linspace(10.0, 14.8, 9)
    ]
    if sum(not value for value in upper) < 7 or not all(lower):
        raise ValueError("corridor probes do not form a long upper shadow and connected lower route")


def write_diagnostic_bundle(directory: str | Path) -> tuple[Path, Path]:
    """Write canonical scenario/task-bank JSON files and return their paths."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = build_diagnostic_scenario()
    bank = build_diagnostic_task_bank(scenario)
    scenario_path = output_dir / "diagnostic_scenario.json"
    bank_path = output_dir / "diagnostic_task_bank.json"
    with scenario_path.open("w", encoding="utf-8") as handle:
        json.dump(scenario, handle, ensure_ascii=False, indent=2)
    with bank_path.open("w", encoding="utf-8") as handle:
        json.dump(bank, handle, ensure_ascii=False, indent=2)
    return scenario_path, bank_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed QRL diagnostic bundle")
    parser.add_argument(
        "--output-dir",
        default="results/diagnostic_u_shadow_corridors/config",
    )
    args = parser.parse_args()
    scenario_path, bank_path = write_diagnostic_bundle(args.output_dir)
    print(f"scenario: {scenario_path}")
    print(f"task bank: {bank_path}")


if __name__ == "__main__":
    main()
