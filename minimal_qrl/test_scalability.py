import math
from pathlib import Path

import numpy as np

from minimal_qrl.dataset import create_dataset
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp import scalability
from minimal_qrl.industry_exp.scalability_scenarios import (
    DEVICE_ORDER,
    build_metric_scenario,
    build_scalability_scenarios,
    scenario_to_env_kwargs,
)


def _obstacle_fraction(scenario):
    bounds = scenario["bounds"]
    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    return sum(math.pi * item["radius"] ** 2 for item in scenario["obstacles"]) / area


def test_metric_area_scenarios_have_expected_physical_mapping_and_geometry():
    expected = {
        100: (10.0, 10_000.0, 180),
        200: (20.0, 40_000.0, 360),
        500: (50.0, 250_000.0, 900),
        1000: (100.0, 1_000_000.0, 1800),
    }
    fractions = []
    for side_m, (env_side, area_m2, horizon) in expected.items():
        scenario = build_metric_scenario(side_m, 24)
        assert scenario["meters_per_env_unit"] == 10.0
        assert scenario["bounds"] == [0.0, 0.0, env_side, env_side]
        assert scenario["physical_area_m2"] == area_m2
        assert scenario["max_episode_steps"] == horizon
        fractions.append(_obstacle_fraction(scenario))
        distances = {
            (
                item["observation"]["min_distance"],
                item["observation"]["max_distance"],
            )
            for item in scenario["device_catalog"]["devices"]
        }
        assert distances == {(0.35, 0.85)}
    assert max(fractions) - min(fractions) < 1e-12


def test_communication_margin_is_invariant_at_corresponding_positions():
    qualities = []
    los_flags = []
    for side_m in (100, 200, 500, 1000):
        scenario = build_metric_scenario(side_m, 24)
        env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
        scale = scenario["scale_factor"]
        state = np.asarray([8.0 * scale, 7.0 * scale, 0.0], dtype=np.float32)
        comm = env.compute_comm_quality(state)
        qualities.append(float(comm["quality"]))
        los_flags.append(bool(comm["has_los"]))
    assert max(qualities) - min(qualities) < 1e-5
    assert len(set(los_flags)) == 1


def test_device_sets_are_nested_and_all_scenarios_are_feasible():
    ids_by_count = {}
    for count in (4, 12, 24):
        scenario = build_metric_scenario(100, count)
        ids = [item["id"] for item in scenario["device_catalog"]["devices"]]
        ids_by_count[count] = ids
        assert ids == list(DEVICE_ORDER[:count])
    assert set(ids_by_count[4]) < set(ids_by_count[12]) < set(ids_by_count[24])
    for scenario in build_scalability_scenarios():
        env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
        assert len(env.device_ids) == scenario["device_count"]


def test_budgeted_dataset_is_exact_balanced_and_excludes_abstract_edges():
    scenario = build_metric_scenario(100, 4)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    stats = {}
    episodes = list(
        create_dataset(
            env,
            max_steps_per_episode=20,
            seed=7,
            task_aware_teacher_ratio=0.0,
            target_env_transitions=41,
            collection_stats=stats,
        )
    )
    assert stats["stored_real_transitions"] == 41
    assert sorted(stats["per_device_real_transitions"].values()) == [10, 10, 10, 11]
    abstract_count = sum(
        episode.num_transitions
        for episode in episodes
        if bool(episode.transition_infos["abstract_goal_edge"][0])
    )
    real_count = sum(episode.num_transitions for episode in episodes) - abstract_count
    assert real_count == 41
    assert abstract_count == stats["synthetic_abstract_edges"]


def test_task_bank_maps_to_valid_paired_starts_at_every_area(tmp_path: Path):
    bank = scalability.generate_task_bank(
        tmp_path / "task_bank.json", validation_per_device=1, test_per_device=1
    )
    validation = [row for row in bank["records"] if row["split"] == "validation"]
    for side_m in (100, 200, 500, 1000):
        scenario = build_metric_scenario(side_m, 24)
        env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
        env_side = float(scenario["bounds"][2])
        for row in validation:
            normalized = row["start_normalized"]
            start = np.asarray(
                [normalized[0] * env_side, normalized[1] * env_side, normalized[2]],
                dtype=np.float32,
            )
            env.reset(seed=row["seed"], options={"device_id": row["device_id"], "start": start})
            assert env.is_valid_state(start)
            assert not env.is_terminal_goal_state(start)


def test_manifest_contains_18_unique_jobs_and_shared_baseline_once_per_seed(tmp_path: Path):
    manifest = scalability.build_manifest(
        tmp_path,
        seeds=[0, 1, 2],
        target_env_transitions=60_000,
        total_steps=60_000,
        checkpoints=[20_000, 40_000, 60_000],
        validation_per_device=1,
        test_per_device=1,
    )
    jobs = manifest["jobs"]
    assert len(jobs) == 18
    assert len({row["job_id"] for row in jobs}) == 18
    baseline = [row for row in jobs if row["scenario_id"] == "metric_l100_k24"]
    assert len(baseline) == 3


def test_completed_job_is_not_overwritten(tmp_path: Path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "COMPLETE").write_text("", encoding="utf-8")
    called = []
    monkeypatch.setattr(scalability, "_run_logged", lambda *args, **kwargs: called.append(args))
    scalability.run_job(
        {"output_dir": str(job_dir)},
        device="cpu",
        batch_size=8,
        num_critics=2,
        teacher_ratio=0.0,
    )
    assert called == []

