from __future__ import annotations

import numpy as np

from minimal_qrl.eval.u_trap_local_navigability import (
    build_probe_records,
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
