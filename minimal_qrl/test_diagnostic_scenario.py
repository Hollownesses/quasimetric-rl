import json
from pathlib import Path

import numpy as np

from minimal_qrl.envs import CommInspectionDubinsUAV2D, Obstacle
from minimal_qrl.eval.comm_inspection_execution_eval import load_task_records
from minimal_qrl.industry_exp.diagnostic_scenario import (
    CHALLENGE_STRATA,
    SCENARIO_ID,
    build_diagnostic_scenario,
    build_diagnostic_task_bank,
    write_diagnostic_bundle,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)


def _start(row, scenario):
    bounds = scenario["bounds"]
    normalized = row["start_normalized"]
    return np.asarray(
        [
            bounds[0] + normalized[0] * (bounds[2] - bounds[0]),
            bounds[1] + normalized[1] * (bounds[3] - bounds[1]),
            normalized[2],
        ],
        dtype=np.float32,
    )


def test_diagnostic_scenario_has_u_shape_rectangles_and_feasible_devices():
    scenario = build_diagnostic_scenario()
    assert scenario["scenario_id"] == SCENARIO_ID
    assert scenario["metadata"]["corridor_safe_route"] == "lower"
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    assert len(env.obstacles) == 4
    assert all(isinstance(obstacle, Obstacle) for obstacle in env.obstacles)
    for index, device_id in enumerate(env.device_ids):
        env.set_task_by_device_id(device_id)
        terminal = env.sample_task_terminal_state(seed=1000 + index)
        assert env.is_task_feasible(terminal)


def test_fixed_task_bank_is_deterministic_balanced_and_semantic():
    scenario = build_diagnostic_scenario()
    first = build_diagnostic_task_bank(scenario)
    second = build_diagnostic_task_bank(scenario)
    assert first["content_digest"] == second["content_digest"]
    assert first["generation_mode"] == "fixed_source_controlled_stratified_starts"
    assert len(first["records"]) == 48

    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    for split, expected in (("validation", 4), ("test", 12)):
        rows = [row for row in first["records"] if row["split"] == split]
        assert {
            stratum: sum(row["stratum"] == stratum for row in rows)
            for stratum in CHALLENGE_STRATA
        } == {stratum: expected for stratum in CHALLENGE_STRATA}

    for row in first["records"]:
        start = _start(row, scenario)
        env.reset(
            seed=row["seed"],
            options={"device_id": row["device_id"], "start": start},
        )
        assert env.is_valid_state(start)
        assert not env.is_task_feasible(start)
        if row["stratum"] == "easy_open":
            assert env._segment_has_los(tuple(start[:2]), env.inspection_target)
        else:
            assert not env._segment_has_los(tuple(start[:2]), env.inspection_target)


def test_bundle_round_trip_and_task_loader_preserve_strata(tmp_path: Path):
    scenario_path, bank_path = write_diagnostic_bundle(tmp_path)
    scenario = load_scenario_config(scenario_path)
    raw_bank = json.loads(bank_path.read_text(encoding="utf-8"))
    loaded_bank, records = load_task_records(
        bank_path,
        split="test",
        bounds=scenario["bounds"],
        scenario_id=scenario["scenario_id"],
    )
    assert loaded_bank["content_digest"] == raw_bank["content_digest"]
    assert len(records) == 36
    assert {row["stratum"] for row in records} == set(CHALLENGE_STRATA)
    assert all(len(row["start"]) == 3 for row in records)
