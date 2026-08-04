"""Metric industrial-inspection scenarios used by the scalability study.

The simulator keeps its established numerical convention and attaches an
explicit physical interpretation: one environment unit represents ten metres.
This module is the source of truth for the controlled scenarios.  Generated
result directories are deliberately not used as inputs.  The formal-study
defaults remain unchanged, while callers can select supported subsets.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from minimal_qrl.envs import CircleObstacle, Obstacle


METERS_PER_ENV_UNIT = 10.0
BASE_PHYSICAL_SIDE_M = 100.0
BASE_ENV_SIDE = BASE_PHYSICAL_SIDE_M / METERS_PER_ENV_UNIT
AREA_SIDE_METRES = (100, 200, 300, 500, 1000)
DEFAULT_AREA_SIDE_METRES = (100, 200, 500, 1000)
DEVICE_COUNTS = (4, 12, 24)

BASE_OBSTACLES = (
    {"x": 3.5, "y": 5.0, "radius": 1.0},
    {"x": 6.5, "y": 5.0, "radius": 1.0},
    {"x": 5.0, "y": 3.0, "radius": 0.8},
)

# A deterministic, spatially distributed, nested ordering.  It is encoded in
# source rather than read from an old result artifact.
DEVICE_ORDER = (
    "relief_valve_psv101",
    "reactor_r101",
    "reboiler_e101",
    "emergency_vent_ev01",
    "scrubber_t301",
    "process_compressor_k101",
    "cooling_water_pump_p401",
    "distillation_column_c101",
    "feed_pump_p101b",
    "pipe_rack_node_pr02",
    "filter_f201",
    "feed_pump_p101a",
    "gas_detector_gd01",
    "pipe_rack_node_pr01",
    "reactor_r102",
    "reflux_drum_v101",
    "control_valve_fcv101",
    "shutdown_valve_esdv101",
    "condenser_e102",
    "absorber_t302",
    "boiler_b401",
    "flare_header_node_fh01",
    "heat_exchanger_e201",
    "separator_v201",
)
BASE_DEVICE_COUNT = len(DEVICE_ORDER)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def base_catalog_path() -> Path:
    return _repo_root() / "minimal_qrl" / "configs" / "chemical_process_plant_devices.json"


def _load_base_catalog() -> dict[str, Any]:
    with base_catalog_path().open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    by_id = {str(item["id"]): item for item in catalog["devices"]}
    missing = [device_id for device_id in DEVICE_ORDER if device_id not in by_id]
    if missing:
        raise ValueError(f"base device catalog is missing: {', '.join(missing)}")
    catalog["devices"] = [copy.deepcopy(by_id[device_id]) for device_id in DEVICE_ORDER]
    return catalog


def _scale_point(point: list[float], scale: float) -> list[float]:
    return [float(point[0]) * scale, float(point[1]) * scale]


def _translate_point(point: Sequence[float], dx: float, dy: float) -> list[float]:
    return [float(point[0]) + float(dx), float(point[1]) + float(dy)]


def tiled_device_id(tile_grid_size: int, row: int, col: int, base_device_id: str) -> str:
    """Return a globally unique device id for one tiled park scenario."""

    return f"g{int(tile_grid_size)}_r{int(row)}_c{int(col)}__{str(base_device_id)}"


def build_metric_scenario(physical_side_m: int, device_count: int) -> dict[str, Any]:
    """Build one controlled metric scenario.

    Spatial locations and obstacle geometry are homothetically scaled so that
    topology and obstacle fraction stay fixed.  Inspection stand-off distances
    are intentionally *not* scaled: they keep their 3.5--8.5 metre meaning.
    """

    if int(physical_side_m) not in AREA_SIDE_METRES:
        raise ValueError(f"unsupported physical side: {physical_side_m}")
    if int(device_count) not in DEVICE_COUNTS:
        raise ValueError(f"unsupported device count: {device_count}")
    if physical_side_m != 100 and device_count != 24:
        raise ValueError("only the 100 m device-count axis may use fewer than 24 devices")

    scale = float(physical_side_m) / BASE_PHYSICAL_SIDE_M
    env_side = BASE_ENV_SIDE * scale
    catalog = _load_base_catalog()
    catalog["devices"] = catalog["devices"][: int(device_count)]
    catalog["ground_station"]["position"] = _scale_point(
        catalog["ground_station"]["position"], scale
    )
    catalog["ground_station"]["los_anchor"] = _scale_point(
        catalog["ground_station"]["los_anchor"], scale
    )
    for device in catalog["devices"]:
        device["position"] = _scale_point(device["position"], scale)
        device["observation_anchor"] = _scale_point(device["observation_anchor"], scale)

    obstacles = [
        {
            "x": float(item["x"]) * scale,
            "y": float(item["y"]) * scale,
            "radius": float(item["radius"]) * scale,
        }
        for item in BASE_OBSTACLES
    ]
    comm_alpha = 2.0
    # The controlled scalability study assumes an engineered private-network
    # link budget.  A base bias of 12 keeps even the farthest occluded state in
    # the 100 m reference layout above the 0.5 threshold; alpha*log(scale)
    # then preserves that relative margin at every larger scale.
    comm_bias_base = 12.0
    scenario_id = f"metric_l{int(physical_side_m)}_k{int(device_count)}"
    axes: list[str] = []
    if device_count == 24:
        axes.append("area")
    if physical_side_m == 100:
        axes.append("device_count")

    scenario = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "experiment_axes": axes,
        "meters_per_env_unit": METERS_PER_ENV_UNIT,
        "physical_side_m": float(physical_side_m),
        "physical_area_m2": float(physical_side_m) ** 2,
        "scale_factor": scale,
        "bounds": [0.0, 0.0, env_side, env_side],
        "device_count": int(device_count),
        "topology": "medium",
        "max_episode_steps": int(round(180 * scale)),
        "device_catalog": catalog,
        "obstacles": obstacles,
        "omega_max": 3.0,
        "v": 1.0,
        "dt": 0.1,
        "comm_alpha": comm_alpha,
        "comm_bias": comm_bias_base + comm_alpha * math.log(scale),
        "comm_occlusion_penalty": 6.0,
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
            "catalog_source": str(base_catalog_path().relative_to(_repo_root())),
            "inspection_distances_scaled": False,
            "communication_relative_coverage_fixed": True,
        },
    }
    validate_metric_scenario(scenario)
    return scenario


def build_tiled_metric_scenario(tile_grid_size: int = 2) -> dict[str, Any]:
    """Build a density-preserving tiled industrial park.

    Each tile is a translated copy of the 100 m reference park.  Device and
    obstacle *sizes* are not scaled.  A single engineered base station is
    placed at the centre of the complete park and its link budget is adjusted
    for the larger side length.
    """

    grid = int(tile_grid_size)
    if grid <= 0:
        raise ValueError("tile_grid_size must be positive")

    physical_side_m = int(BASE_PHYSICAL_SIDE_M) * grid
    env_side = BASE_ENV_SIDE * grid
    catalog = _load_base_catalog()
    base_devices = list(catalog["devices"])
    tiled_devices: list[dict[str, Any]] = []
    obstacles: list[dict[str, float]] = []

    for row in range(grid):
        for col in range(grid):
            dx = float(col) * BASE_ENV_SIDE
            dy = float(row) * BASE_ENV_SIDE
            for base_device in base_devices:
                device = copy.deepcopy(base_device)
                device["id"] = tiled_device_id(
                    grid,
                    row,
                    col,
                    str(base_device["id"]),
                )
                device["position"] = _translate_point(device["position"], dx, dy)
                device["observation_anchor"] = _translate_point(
                    device["observation_anchor"],
                    dx,
                    dy,
                )
                tiled_devices.append(device)
            for obstacle in BASE_OBSTACLES:
                obstacles.append(
                    {
                        "x": float(obstacle["x"]) + dx,
                        "y": float(obstacle["y"]) + dy,
                        "radius": float(obstacle["radius"]),
                    }
                )

    centre = [0.5 * env_side, 0.5 * env_side]
    catalog["devices"] = tiled_devices
    catalog["ground_station"]["position"] = list(centre)
    catalog["ground_station"]["los_anchor"] = list(centre)

    comm_alpha = 2.0
    comm_bias_base = 12.0
    device_count = BASE_DEVICE_COUNT * grid * grid
    scenario = {
        "schema_version": 1,
        "scenario_id": f"tiled_g{grid}_l{physical_side_m}_k{device_count}",
        "experiment_axes": ["area", "device_count", "joint_scale"],
        "meters_per_env_unit": METERS_PER_ENV_UNIT,
        "physical_side_m": float(physical_side_m),
        "physical_area_m2": float(physical_side_m) ** 2,
        "scale_factor": float(grid),
        "bounds": [0.0, 0.0, env_side, env_side],
        "device_count": int(device_count),
        "topology": "tiled_medium",
        "max_episode_steps": int(round(180 * grid)),
        "device_catalog": catalog,
        "obstacles": obstacles,
        "omega_max": 3.0,
        "v": 1.0,
        "dt": 0.1,
        "comm_alpha": comm_alpha,
        "comm_bias": comm_bias_base + comm_alpha * math.log(float(grid)),
        "comm_occlusion_penalty": 6.0,
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
            "layout_kind": "density_preserving_tiled",
            "tile_grid_size": int(grid),
            "base_tile_side_m": BASE_PHYSICAL_SIDE_M,
            "base_devices_per_tile": BASE_DEVICE_COUNT,
            "base_obstacles_per_tile": len(BASE_OBSTACLES),
            "coordinate_interpretation": "1 environment unit = 10 metres",
            "catalog_source": str(base_catalog_path().relative_to(_repo_root())),
            "inspection_distances_scaled": False,
            "obstacle_sizes_scaled": False,
            "single_central_ground_station": True,
            "communication_relative_coverage_fixed": True,
        },
    }
    validate_metric_scenario(scenario)
    return scenario


def build_scalability_scenarios(
    *,
    area_sides: Sequence[int] = DEFAULT_AREA_SIDE_METRES,
    device_counts: Sequence[int] = DEVICE_COUNTS,
) -> list[dict[str, Any]]:
    """Return the selected scenario union; the l100/k24 baseline is shared."""

    selected_sides = tuple(dict.fromkeys(int(side) for side in area_sides))
    selected_counts = tuple(dict.fromkeys(int(count) for count in device_counts))
    if not selected_sides:
        raise ValueError("area_sides must not be empty")
    if not selected_counts:
        raise ValueError("device_counts must not be empty")

    scenarios_by_id: dict[str, dict[str, Any]] = {}
    for side in selected_sides:
        scenario = build_metric_scenario(side, 24)
        scenarios_by_id[scenario["scenario_id"]] = scenario
    for count in selected_counts:
        scenario = build_metric_scenario(100, count)
        scenarios_by_id[scenario["scenario_id"]] = scenario
    return list(scenarios_by_id.values())


def _validate_common_scenario(scenario: Mapping[str, Any]) -> None:
    """Validate fields shared by metric and hand-designed scenarios."""

    bounds = [float(v) for v in scenario["bounds"]]
    if len(bounds) != 4 or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
        raise ValueError("bounds must be [x_min, y_min, x_max, y_max]")
    devices = list(scenario["device_catalog"]["devices"])
    if len(devices) != int(scenario["device_count"]):
        raise ValueError("device_count does not match device_catalog")
    ids = [str(item["id"]) for item in devices]
    if len(ids) != len(set(ids)):
        raise ValueError("device ids must be unique")
    for point in (
        [
            scenario["device_catalog"]["ground_station"]["position"],
            scenario["device_catalog"]["ground_station"]["los_anchor"],
        ]
        + [item["position"] for item in devices]
        + [item["observation_anchor"] for item in devices]
    ):
        x, y = float(point[0]), float(point[1])
        if not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
            raise ValueError(f"catalog point outside bounds: {point}")
    for obstacle in scenario["obstacles"]:
        obstacle_type = str(obstacle.get("type", "circle"))
        if obstacle_type == "circle":
            x = float(obstacle["x"])
            y = float(obstacle["y"])
            radius = float(obstacle["radius"])
            if radius <= 0.0:
                raise ValueError("obstacle radius must be positive")
            inside = (
                bounds[0] <= x - radius
                and x + radius <= bounds[2]
                and bounds[1] <= y - radius
                and y + radius <= bounds[3]
            )
        elif obstacle_type == "rectangle":
            x_min = float(obstacle["x_min"])
            x_max = float(obstacle["x_max"])
            y_min = float(obstacle["y_min"])
            y_max = float(obstacle["y_max"])
            if not (x_min < x_max and y_min < y_max):
                raise ValueError("rectangle obstacle must have positive width and height")
            inside = (
                bounds[0] <= x_min < x_max <= bounds[2]
                and bounds[1] <= y_min < y_max <= bounds[3]
            )
        else:
            raise ValueError(f"unsupported obstacle type: {obstacle_type}")
        if not inside:
            raise ValueError(f"obstacle outside bounds: {obstacle}")


def validate_metric_scenario(scenario: Mapping[str, Any]) -> None:
    _validate_common_scenario(scenario)
    side_m = float(scenario["physical_side_m"])
    area_m2 = float(scenario["physical_area_m2"])
    if not math.isclose(area_m2, side_m * side_m, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("physical_area_m2 must equal physical_side_m squared")
    bounds = [float(v) for v in scenario["bounds"]]
    expected_side = side_m / float(scenario["meters_per_env_unit"])
    if bounds != [0.0, 0.0, expected_side, expected_side]:
        raise ValueError("bounds do not match the metric coordinate mapping")
    devices = list(scenario["device_catalog"]["devices"])
    ids = [str(item["id"]) for item in devices]
    metadata = dict(scenario.get("metadata", {}))
    if metadata.get("layout_kind") == "density_preserving_tiled":
        grid = int(metadata["tile_grid_size"])
        expected_ids = [
            tiled_device_id(grid, row, col, device_id)
            for row in range(grid)
            for col in range(grid)
            for device_id in DEVICE_ORDER
        ]
        if ids != expected_ids:
            raise ValueError("tiled device catalog does not follow the deterministic tile order")
        if len(devices) != BASE_DEVICE_COUNT * grid * grid:
            raise ValueError("tiled device count does not match tile density")
    elif ids != list(DEVICE_ORDER[: len(ids)]):
        raise ValueError("device catalog does not follow the nested device order")


def write_scalability_scenarios(
    directory: Path,
    *,
    area_sides: Sequence[int] = DEFAULT_AREA_SIDE_METRES,
    device_counts: Sequence[int] = DEVICE_COUNTS,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for scenario in build_scalability_scenarios(
        area_sides=area_sides,
        device_counts=device_counts,
    ):
        path = directory / f"{scenario['scenario_id']}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(scenario, handle, ensure_ascii=False, indent=2)
        paths.append(path)
    return paths


def load_metric_scenario(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        scenario = json.load(handle)
    if not isinstance(scenario, dict):
        raise ValueError("scenario config root must be a JSON object")
    validate_metric_scenario(scenario)
    return scenario


def load_scenario_config(path: str | Path) -> dict[str, Any]:
    """Load either a controlled metric scenario or a diagnostic scenario."""

    with Path(path).open("r", encoding="utf-8") as handle:
        scenario = json.load(handle)
    if not isinstance(scenario, dict):
        raise ValueError("scenario config root must be a JSON object")
    if str(scenario.get("scenario_family", "metric")) == "metric":
        validate_metric_scenario(scenario)
    else:
        _validate_common_scenario(scenario)
    return scenario


def scenario_to_env_kwargs(scenario: Mapping[str, Any]) -> dict[str, Any]:
    if str(scenario.get("scenario_family", "metric")) == "metric":
        validate_metric_scenario(scenario)
    else:
        _validate_common_scenario(scenario)
    obstacles = []
    for item in scenario["obstacles"]:
        obstacle_type = str(item.get("type", "circle"))
        if obstacle_type == "circle":
            obstacles.append(
                CircleObstacle(
                    x=float(item["x"]),
                    y=float(item["y"]),
                    radius=float(item["radius"]),
                )
            )
        elif obstacle_type == "rectangle":
            obstacles.append(
                Obstacle(
                    x_min=float(item["x_min"]),
                    x_max=float(item["x_max"]),
                    y_min=float(item["y_min"]),
                    y_max=float(item["y_max"]),
                )
            )
        else:  # guarded by validation, retained for defensive clarity
            raise ValueError(f"unsupported obstacle type: {obstacle_type}")
    return {
        "device_catalog": copy.deepcopy(scenario["device_catalog"]),
        "bounds": tuple(float(v) for v in scenario["bounds"]),
        "omega_max": float(scenario["omega_max"]),
        "v": float(scenario["v"]),
        "dt": float(scenario["dt"]),
        "max_steps": int(scenario["max_episode_steps"]),
        "obstacles": obstacles,
        "comm_alpha": float(scenario["comm_alpha"]),
        "comm_bias": float(scenario["comm_bias"]),
        "comm_occlusion_penalty": float(scenario["comm_occlusion_penalty"]),
        "comm_threshold": float(scenario["comm_threshold"]),
        "require_ground_station_los": bool(scenario["require_ground_station_los"]),
        "collision_cost": float(scenario["collision_cost"]),
        "out_of_bounds_cost": float(scenario["out_of_bounds_cost"]),
        "communication_break_cost": float(scenario["communication_break_cost"]),
        "observation_violation_cost_weight": float(scenario["observation_violation_cost_weight"]),
        "communication_violation_cost_weight": float(scenario["communication_violation_cost_weight"]),
        "observation_failure_cost": float(scenario["observation_failure_cost"]),
        "taskscore_beta_obs": float(scenario["taskscore_beta_obs"]),
        "taskscore_beta_comm": float(scenario["taskscore_beta_comm"]),
        "taskscore_beta_feas": float(scenario["taskscore_beta_feas"]),
        "taskscore_margin_clip": float(scenario["taskscore_margin_clip"]),
        "min_start_target_distance": float(scenario.get("min_start_target_distance", 0.5)),
    }
