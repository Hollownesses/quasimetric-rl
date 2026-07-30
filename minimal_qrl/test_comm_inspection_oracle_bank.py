from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import minimal_qrl.eval.comm_inspection_oracle_bank as oracle_bank
from minimal_qrl.baselines.hybrid_astar import HybridAStarConfig
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.comm_inspection_oracle_bank import (
    CommInspectionOracleBankConfig,
    ensure_comm_inspection_oracle_bank,
    evaluate_qrl_on_oracle_bank,
    visualize_qrl_oracle_bank_heatmap,
)


def make_env(*, comm_bias: float = 5.0) -> CommInspectionDubinsUAV2D:
    catalog = {
        "ground_station": {
            "position": [1.0, 1.0],
            "los_anchor": [1.0, 1.0],
        },
        "devices": [
            {
                "id": "device_a",
                "position": [5.0, 4.0],
                "observation_anchor": [5.0, 4.0],
                "observation": {
                    "min_distance": 0.2,
                    "max_distance": 1.2,
                    "preferred_bearing_rad": 0.0,
                    "bearing_tolerance_rad": np.pi,
                    "fov_angle_rad": np.pi,
                    "require_los": False,
                },
            },
            {
                "id": "device_b",
                "position": [3.0, 6.0],
                "observation_anchor": [3.0, 6.0],
                "observation": {
                    "min_distance": 0.2,
                    "max_distance": 1.2,
                    "preferred_bearing_rad": 0.0,
                    "bearing_tolerance_rad": np.pi,
                    "fov_angle_rad": np.pi,
                    "require_los": False,
                },
            },
        ],
    }
    return CommInspectionDubinsUAV2D(
        device_catalog=catalog,
        bounds=(0.0, 0.0, 8.0, 8.0),
        omega_max=2.0,
        v=1.0,
        dt=0.1,
        max_steps=80,
        obstacles=[],
        start=None,
        comm_bias=comm_bias,
        comm_threshold=-100.0,
    )


class FirstCoordinateMetric(torch.nn.Module):
    def forward(self, source, goal):
        del goal
        return source[:, 0]


class IdentityEncoder(torch.nn.Module):
    def forward(self, value):
        return value


class DummyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IdentityEncoder()
        self.quasimetric_model = FirstCoordinateMetric()


class DummyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.critics = torch.nn.ModuleList([DummyCritic()])


def test_oracle_bank_is_stratified_cached_and_evaluated(tmp_path, monkeypatch):
    env = make_env()
    config = CommInspectionOracleBankConfig(
        sample_count=4,
        generation_seed=17,
        candidate_multiplier=4,
        timeout_sec=0.1,
        terminal_samples=0,
    )
    path = tmp_path / "validation.json"
    planned = []

    def fake_plan_record(_env, record, _planner_config):
        planned.append(record["task_id"])
        record.update(
            {
                "status": "solved",
                "planner_success": True,
                "rollout_verified": True,
                "planner_failure_reason": "",
                "planning_time_sec": 0.01,
                "expanded_nodes": 1,
                "generated_nodes": 2,
                "planned_action_count": 3,
                "oracle_cost": float(record["observation"][0]),
            }
        )

    monkeypatch.setattr(oracle_bank, "_plan_record", fake_plan_record)
    bank = ensure_comm_inspection_oracle_bank(
        env,
        path,
        split="validation",
        config=config,
    )

    assert len(bank["records"]) == 4
    assert {row["device_id"] for row in bank["records"]} == {
        "device_a",
        "device_b",
    }
    assert all(
        sum(row["device_id"] == device_id for row in bank["records"]) == 2
        for device_id in env.device_ids
    )
    assert bank["summary"]["oracle_coverage"] == 1.0
    assert len(planned) == 4

    metrics = evaluate_qrl_on_oracle_bank(
        DummyAgent(),
        bank,
        device="cpu",
        bootstrap_samples=20,
        bootstrap_seed=5,
    )
    assert metrics["requested_samples"] == 4.0
    assert metrics["solved_samples"] == 4.0
    assert np.isclose(metrics["mae"], 0.0)
    assert np.isclose(metrics["rmse"], 0.0)
    assert "mae_ci95_low" in metrics

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["status"] = "pending"
    path.write_text(json.dumps(payload), encoding="utf-8")
    planned.clear()
    resumed = ensure_comm_inspection_oracle_bank(
        env,
        path,
        split="validation",
        config=config,
    )
    assert resumed["summary"]["oracle_coverage"] == 1.0
    assert planned == [payload["records"][0]["task_id"]]


def test_oracle_bank_rejects_stale_environment_signature(tmp_path, monkeypatch):
    config = CommInspectionOracleBankConfig(
        sample_count=2,
        generation_seed=19,
        timeout_sec=0.1,
        terminal_samples=0,
    )
    path = tmp_path / "validation.json"

    def fake_plan_record(_env, record, _planner_config):
        record.update(
            {
                "status": "failed",
                "planner_success": False,
                "rollout_verified": False,
                "planner_failure_reason": "timeout",
                "planning_time_sec": 0.1,
                "expanded_nodes": 1,
                "generated_nodes": 1,
                "planned_action_count": 0,
            }
        )

    monkeypatch.setattr(oracle_bank, "_plan_record", fake_plan_record)
    ensure_comm_inspection_oracle_bank(
        make_env(comm_bias=5.0),
        path,
        split="validation",
        config=config,
    )

    with pytest.raises(ValueError, match="environment_signature"):
        ensure_comm_inspection_oracle_bank(
            make_env(comm_bias=6.0),
            path,
            split="validation",
            config=config,
        )


def test_oracle_record_replays_and_verifies_hybrid_astar_cost():
    env = make_env()
    start = np.array([2.0, 4.0, 0.0], dtype=np.float32)
    env.reset(
        seed=1,
        options={"device_id": "device_a", "start": start},
    )
    record = {
        "episode_seed": 1,
        "device_id": "device_a",
        "start_state": start.tolist(),
    }
    oracle_bank._plan_record(
        env,
        record,
        HybridAStarConfig(
            position_resolution=0.2,
            heading_bins=24,
            primitive_steps=4,
            max_expansions=10_000,
            timeout_sec=3.0,
            terminal_samples=8,
        ),
    )

    assert record["status"] == "solved"
    assert record["rollout_verified"]
    assert np.isclose(
        record["planned_cost"],
        record["oracle_cost"],
        atol=1e-6,
    )


def test_oracle_bank_heatmap_uses_exact_oracle_costs_and_marks_failures(tmp_path):
    records = [
        {
            "device_id": "device_a",
            "sample_index": 0,
            "status": "solved",
            "observation": [2.0, 0.0],
            "goal_observation": [0.0, 0.0],
            "oracle_cost": 101.0,
        },
        {
            "device_id": "device_a",
            "sample_index": 1,
            "status": "failed",
            "observation": [3.0, 0.0],
            "goal_observation": [0.0, 0.0],
            "oracle_cost": None,
        },
        {
            "device_id": "device_b",
            "sample_index": 0,
            "status": "solved",
            "observation": [4.0, 0.0],
            "goal_observation": [0.0, 0.0],
            "oracle_cost": 303.0,
        },
        {
            "device_id": "device_b",
            "sample_index": 1,
            "status": "solved",
            "observation": [5.0, 0.0],
            "goal_observation": [0.0, 0.0],
            "oracle_cost": 404.0,
        },
    ]
    bank = {"records": records}

    data = oracle_bank._oracle_bank_heatmap_data(
        DummyAgent(),
        bank,
        device="cpu",
        distance_scale=1.0,
    )
    np.testing.assert_allclose(
        data["target"],
        np.array([[101.0, np.nan], [303.0, 404.0]]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        data["prediction"],
        np.array([[2.0, np.nan], [4.0, 5.0]]),
        equal_nan=True,
    )
    assert data["solved_samples"] == 3
    assert data["requested_samples"] == 4

    output_path = visualize_qrl_oracle_bank_heatmap(
        DummyAgent(),
        bank,
        step=50_000,
        output_dir=tmp_path,
        device="cpu",
    )
    assert output_path.endswith("oracle_bank_heatmap_step50000.png")
    assert (tmp_path / "oracle_bank_heatmap" / "oracle_bank_heatmap_step50000.png").is_file()
