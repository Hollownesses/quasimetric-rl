from __future__ import annotations

import json

import numpy as np
import pytest

from minimal_qrl.baselines.hybrid_astar import HybridAStarConfig
from minimal_qrl.eval.u_trap_local_navigability import (
    build_probe_records,
    load_reusable_oracle_records,
    summarize_checkpoint_records,
)
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario


def test_local_navigability_probe_records_are_fixed_and_unique():
    records = build_probe_records(build_diagnostic_scenario())
    assert len(records) == 16
    assert len({record["probe_id"] for record in records}) == 16
    assert {record["position_label"] for record in records} == {
        "deep",
        "middle",
        "mouth",
        "outside",
    }
    assert {record["heading_label"] for record in records} == {
        "west",
        "north",
        "east",
        "south",
    }


def test_local_navigability_summary_detects_perfect_order_and_actions():
    records = []
    for heading in ("west", "east"):
        for position_index, cost in enumerate((8.0, 6.0, 4.0, 2.0)):
            records.append(
                {
                    "position_index": position_index,
                    "heading_label": heading,
                    "oracle_cost": cost,
                    "qrl_value": cost,
                    "qrl_action_exact_match": True,
                    "qrl_action_coarse_match": True,
                }
            )

    summary = summarize_checkpoint_records(records)
    assert summary["solved_probes"] == 8
    assert np.isclose(summary["mae"], 0.0)
    assert np.isclose(summary["pearson_corr"], 1.0)
    assert np.isclose(summary["spearman_corr"], 1.0)
    assert np.isclose(summary["pairwise_ranking_accuracy"], 1.0)
    assert np.isclose(summary["exit_progress_ordering_accuracy"], 1.0)
    assert np.isclose(summary["first_action_exact_accuracy"], 1.0)
    assert np.isclose(summary["first_action_coarse_accuracy"], 1.0)


def _cached_oracle_payload(scenario, probes, *, seed=20260802):
    import hashlib

    canonical = json.dumps(
        scenario, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    records = []
    for probe in probes:
        records.append(
            {
                **probe,
                "oracle_solved": True,
                "oracle_cost": 10.0,
                "oracle_first_action": 0.0,
                "oracle_first_action_coarse": "straight",
            }
        )
    return {
        "scenario_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "oracle_config": {
            "seed": seed,
            "astar_position_resolution": 0.25,
            "astar_heading_bins": 24,
            "astar_primitive_steps": 5,
            "astar_heuristic_weight": 1.0,
            "astar_max_expansions": 200_000,
            "astar_timeout_sec": 120.0,
            "astar_terminal_samples": 128,
        },
        "oracle_records": records,
    }


def test_reused_oracle_records_are_validated_and_reordered(tmp_path):
    scenario = build_diagnostic_scenario()
    probes = build_probe_records(scenario)
    payload = _cached_oracle_payload(scenario, probes)
    payload["oracle_records"].reverse()
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_reusable_oracle_records(
        path,
        scenario,
        probes,
        HybridAStarConfig(max_expansions=200_000, timeout_sec=120.0),
        seed=20260802,
    )

    assert [record["probe_id"] for record in records] == [
        probe["probe_id"] for probe in probes
    ]


def test_reused_oracle_records_reject_incompatible_config(tmp_path):
    scenario = build_diagnostic_scenario()
    probes = build_probe_records(scenario)
    payload = _cached_oracle_payload(scenario, probes, seed=7)
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatch for seed"):
        load_reusable_oracle_records(
            path,
            scenario,
            probes,
            HybridAStarConfig(max_expansions=200_000, timeout_sec=120.0),
            seed=20260802,
        )
