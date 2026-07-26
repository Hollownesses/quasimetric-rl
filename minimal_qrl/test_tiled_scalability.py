import math
from pathlib import Path

import numpy as np

from minimal_qrl.dataset import create_dataset
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.scalability import generate_scenario_task_bank
from minimal_qrl.industry_exp.scalability_scenarios import (
    BASE_OBSTACLES,
    DEVICE_ORDER,
    build_tiled_metric_scenario,
    scenario_to_env_kwargs,
    tiled_device_id,
)
from minimal_qrl.industry_exp.tiled_scalability import build_tiled_manifest


def test_two_by_two_tiled_scenario_preserves_physical_density_and_sizes():
    scenario = build_tiled_metric_scenario(2)
    assert scenario["scenario_id"] == "tiled_g2_l200_k96"
    assert scenario["bounds"] == [0.0, 0.0, 20.0, 20.0]
    assert scenario["physical_area_m2"] == 40_000.0
    assert scenario["max_episode_steps"] == 360
    assert scenario["device_count"] == 96
    assert len(scenario["obstacles"]) == 12
    assert scenario["device_catalog"]["ground_station"] == {
        "position": [10.0, 10.0],
        "los_anchor": [10.0, 10.0],
    }

    ids = [item["id"] for item in scenario["device_catalog"]["devices"]]
    assert len(set(ids)) == 96
    assert ids[0] == tiled_device_id(2, 0, 0, DEVICE_ORDER[0])
    assert ids[-1] == tiled_device_id(2, 1, 1, DEVICE_ORDER[-1])
    radii = sorted(float(item["radius"]) for item in scenario["obstacles"])
    assert radii == sorted(
        float(item["radius"]) for item in BASE_OBSTACLES for _ in range(4)
    )
    inspection_ranges = {
        (
            float(item["observation"]["min_distance"]),
            float(item["observation"]["max_distance"]),
        )
        for item in scenario["device_catalog"]["devices"]
    }
    assert inspection_ranges == {(0.35, 0.85)}
    assert math.isclose(96 / 40_000.0, 24 / 10_000.0)


def test_two_by_two_central_station_has_engineered_whole_map_coverage():
    scenario = build_tiled_metric_scenario(2)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    for x, y in ((0.0, 0.0), (0.0, 20.0), (20.0, 0.0), (20.0, 20.0)):
        comm = env.compute_comm_quality(np.asarray([x, y, 0.0], dtype=np.float32))
        assert float(comm["margin"]) >= float(scenario["comm_threshold"])


def test_tiled_scenario_all_devices_have_nonempty_terminal_sets():
    scenario = build_tiled_metric_scenario(2)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    for index, device_id in enumerate(env.device_ids):
        env.reset(seed=1000 + index, options={"device_id": device_id})
        terminal = env.sample_task_terminal_state(seed=2000 + index)
        assert env.is_terminal_goal_state(terminal)


def test_scenario_task_bank_samples_valid_starts_from_complete_tiled_park(tmp_path: Path):
    scenario = build_tiled_metric_scenario(2)
    bank = generate_scenario_task_bank(
        tmp_path / "task_bank.json",
        [scenario],
        validation_per_device=1,
        test_per_device=1,
    )
    assert bank["generation_mode"] == "independent_full_scenario_random_starts"
    assert len(bank["records"]) == 96 * 2
    assert {row["scenario_id"] for row in bank["records"]} == {
        scenario["scenario_id"]
    }

    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    normalized_xy = []
    for row in bank["records"]:
        normalized = row["start_normalized"]
        start = np.asarray(
            [normalized[0] * 20.0, normalized[1] * 20.0, normalized[2]],
            dtype=np.float32,
        )
        env.reset(
            seed=row["seed"],
            options={"device_id": row["device_id"], "start": start},
        )
        assert env.is_valid_state(start)
        assert not env.is_terminal_goal_state(start)
        normalized_xy.append(normalized[:2])

    points = np.asarray(normalized_xy)
    assert float(points[:, 0].min()) < 0.25
    assert float(points[:, 0].max()) > 0.75
    assert float(points[:, 1].min()) < 0.25
    assert float(points[:, 1].max()) > 0.75


def test_tiled_dataset_budget_is_balanced_across_96_devices():
    scenario = build_tiled_metric_scenario(2)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    stats = {}
    list(
        create_dataset(
            env,
            max_steps_per_episode=5,
            seed=4,
            task_aware_teacher_ratio=0.0,
            target_env_transitions=192,
            collection_stats=stats,
        )
    )
    assert stats["stored_real_transitions"] == 192
    assert set(stats["per_device_real_transitions"].values()) == {2}


def test_tiled_manifest_scales_data_updates_and_checkpoints_by_tile_count(tmp_path: Path):
    manifest = build_tiled_manifest(
        tmp_path,
        tile_grids=[2],
        seeds=[0],
        base_target_env_transitions=1_000,
        base_total_steps=1_000,
        base_checkpoints=[200, 600, 1_000],
        save_interval=200,
        validation_per_device=1,
        test_per_device=1,
    )
    assert len(manifest["jobs"]) == 1
    job = manifest["jobs"][0]
    assert job["scenario_id"] == "tiled_g2_l200_k96"
    assert job["density_multiplier"] == 4
    assert job["target_env_transitions"] == 4_000
    assert job["total_steps"] == 4_000
    assert job["checkpoints"] == [800, 2_400, 4_000]
    assert len(manifest["scenarios"]) == 1
