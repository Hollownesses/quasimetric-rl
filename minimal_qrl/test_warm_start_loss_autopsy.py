from __future__ import annotations

from pathlib import Path

import numpy as np

from minimal_qrl.baselines import MPPIConfig
from minimal_qrl.industry_exp.joint_feasible_iqe import (
    JointFeasibleProblem,
    lattice_successor_ranking,
)
from minimal_qrl.industry_exp.qrl_pipeline_metrics import (
    MPPIRankingContractCase,
    evaluate_mppi_ranking_contract,
    u_region_constraint_tails,
)
from minimal_qrl.industry_exp.warm_start_loss_autopsy import (
    ARM_COMPONENTS,
    normalize_arms,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_problem() -> JointFeasibleProblem:
    return JointFeasibleProblem(
        states=np.asarray(
            [
                [4.8, 3.6, np.pi],
                [4.0, 3.6, np.pi],
                [3.0, 3.6, 0.0],
            ],
            dtype=np.float32,
        ),
        observations=np.zeros((3, 4), dtype=np.float32),
        goal_observation=np.zeros(4, dtype=np.float32),
        reference_values=np.asarray([3.0, 2.0, 0.0], dtype=np.float32),
        sources=np.asarray([0, 0], dtype=np.int64),
        destinations=np.asarray([1, 2], dtype=np.int64),
        costs=np.asarray([1.0, 5.0], dtype=np.float32),
        families=np.asarray([0, 0], dtype=np.int8),
        terminal_indices=np.asarray([2], dtype=np.int64),
        u_trap_mask=np.asarray([True, False, False]),
        graph_stats={},
    )


def test_successor_topology_reports_oracle_regret_and_breakdowns():
    problem = _small_problem()
    exact = lattice_successor_ranking(
        problem, problem.reference_values, source_mask=problem.u_trap_mask
    )
    assert exact["top1_accuracy"] == 1.0
    assert exact["oracle_regret"]["max"] == 0.0
    assert exact["breakdown"]["depth"]["mouth"]["source_states"] == 1
    wrong = lattice_successor_ranking(
        problem,
        np.asarray([3.0, 10.0, 0.0]),
        source_mask=problem.u_trap_mask,
    )
    assert wrong["top1_accuracy"] == 0.0
    assert wrong["oracle_regret"]["max"] == 2.0


def test_u_region_constraint_tails_separate_oracle_optimal_edges():
    problem = _small_problem()
    result = u_region_constraint_tails(
        problem,
        edge_distances=np.asarray([1.5, 6.0]),
        values=np.asarray([3.0, 1.0, 0.0]),
        epsilons={"ordinary": 0.25, "direct_goal": 0.25, "terminal_goal": 0.0},
    )
    all_edges = result["groups"]["all_outgoing"]["pairwise_quasimetric_edge"]["ordinary"]
    optimal = result["groups"]["oracle_optimal_outgoing"]["pairwise_quasimetric_edge"]["ordinary"]
    nonoptimal = result["groups"]["non_optimal_outgoing"]["pairwise_quasimetric_edge"]["ordinary"]
    assert all_edges["count"] == 2
    assert np.isclose(all_edges["positive_excess_max"], 1.0)
    assert optimal["count"] == 1
    assert np.isclose(optimal["positive_excess_max"], 0.5)
    assert nonoptimal["count"] == 1


class _FirstColumnValue:
    def batch_value(self, observations, goals):
        del goals
        return np.asarray(observations)[:, 0]


def test_mppi_contract_scores_fixed_candidate_ordering():
    case = MPPIRankingContractCase(
        task_id="u:0",
        repeat=0,
        episode_seed=7,
        candidate_actions=np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        final_observations=np.asarray([[0.0], [2.0], [1.0]], dtype=np.float32),
        goal_observation=np.asarray([0.0], dtype=np.float32),
        running_costs=np.zeros(3, dtype=np.float64),
        success=np.zeros(3, dtype=bool),
        invalid=np.zeros(3, dtype=bool),
        oracle_scores=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
    )
    result = evaluate_mppi_ranking_contract(
        _FirstColumnValue(),
        [case],
        config=MPPIConfig(horizon=2, num_samples=3, temperature=1.0),
        top_k=2,
    )
    assert result["cases"] == 1
    assert np.isclose(result["pairwise_concordance_mean"], 2.0 / 3.0)
    assert result["learned_selected_oracle_regret"]["max"] == 0.0
    assert result["records"][0]["top_k_overlap"] == 0.5


def test_autopsy_arms_and_shell_phase_are_exposed():
    assert normalize_arms(["A0", "a6", "a0_no_update"]) == [
        "a0_no_update",
        "a6_full_qrl",
    ]
    assert ARM_COMPONENTS["a5_push_plus_local"] == (
        "global_push",
        "local_constraint",
    )
    script = (ROOT / "minimal_qrl/run_comm_inspection_diagnostic.sh").read_text()
    assert "warm_start_loss_autopsy()" in script
    assert "warm_start_loss_autopsy)" in script
    assert "AUTOPSY_EVAL_STEPS" in script
    assert "--contract-repeats" in script
