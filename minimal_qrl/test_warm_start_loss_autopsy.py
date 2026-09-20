from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
from minimal_qrl.industry_exp.dual_dynamics_autopsy import (
    _latent_dynamics_result,
    compute_2x2_effects,
    parse_dual_scheme,
    select_best_dual_scheme,
)
from quasimetric_rl.modules.quasimetric_critic.losses import CriticBatchInfo
from quasimetric_rl.modules.utils import LossResult


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


def test_dual_scheme_parser_covers_screening_families():
    baseline = parse_dual_scheme("baseline")
    fixed = parse_dual_scheme("fixed:1")
    projected = parse_dual_scheme("projected:0.0001:5")
    assert baseline.mode == "softplus_adam"
    assert not baseline.selectable
    assert fixed.mode == "fixed"
    assert fixed.fixed_lambda == 1.0
    assert projected.mode == "projected"
    assert projected.projected_lr == 0.0001
    assert projected.updates_per_primal == 5


def _candidate(name, *, selectable, catastrophes, shift=0.0):
    return {
        "scheme": {"name": name, "mode": "fixed"},
        "selectable": selectable,
        "catastrophe_count": catastrophes,
        "final_u_successor_top1": 0.7 + shift,
        "worst_u_successor_top1": 0.6 + shift,
        "final_mppi_spearman": 0.8 + shift,
        "worst_mppi_spearman": 0.7 + shift,
        "final_u_value_mae": 10.0 - shift,
        "worst_u_value_mae": 20.0 - shift,
        "final_u_successor_regret": 2.0 - shift,
        "final_mppi_oracle_regret": 5.0 - shift,
        "final_u_bellman_excess_p99": 1.0 - shift,
    }


def test_dual_selector_excludes_baseline_and_prioritizes_stability():
    result = select_best_dual_scheme(
        [
            _candidate("baseline", selectable=False, catastrophes=0, shift=1.0),
            _candidate("unstable", selectable=True, catastrophes=1, shift=2.0),
            _candidate("stable", selectable=True, catastrophes=0, shift=0.0),
        ]
    )
    assert result["best_scheme"]["name"] == "stable"
    assert result["selection_rule"]["baseline_is_reference_only"]


def test_dual_experiment_shell_phases_are_exposed():
    script = (ROOT / "minimal_qrl/run_comm_inspection_diagnostic.sh").read_text()
    assert "warm_start_dual_screen)" in script
    assert "warm_start_dual_dynamics_2x2)" in script
    assert "best_dual_scheme.json" in script


def test_head_only_dynamics_blocks_encoder_and_quasimetric_gradients():
    class ToyCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.quasimetric_model = nn.Linear(2, 2, bias=False)
            self.latent_dynamics = nn.Linear(2, 2, bias=False)

    class ToyLosses:
        def latent_dynamics(self, _batch, info):
            prediction = info.critic.latent_dynamics(info.zx)
            transformed = info.critic.quasimetric_model(prediction)
            loss = (transformed - info.zy).square().mean()
            return LossResult(loss=loss, info={})

    critic = ToyCritic()
    zx = torch.randn(4, 2, requires_grad=True)
    zy = torch.randn(4, 2, requires_grad=True)
    result = _latent_dynamics_result(
        ToyLosses(),
        None,
        CriticBatchInfo(critic=critic, zx=zx, zy=zy),
        gradient_mode="head_only",
    )
    result.loss.backward()
    assert critic.latent_dynamics.weight.grad is not None
    assert critic.quasimetric_model.weight.grad is None
    assert zx.grad is None
    assert zy.grad is None


def test_2x2_effect_decomposition():
    rows = [
        {"variant": "baseline_dual__shared_dynamics", "step": 10, "score": 1.0},
        {"variant": "baseline_dual__head_only_dynamics", "step": 10, "score": 3.0},
        {"variant": "selected_dual__shared_dynamics", "step": 10, "score": 5.0},
        {"variant": "selected_dual__head_only_dynamics", "step": 10, "score": 11.0},
    ]
    effect = compute_2x2_effects(rows)[0]
    assert effect["selected_dual_main_effect"] == 6.0
    assert effect["head_only_main_effect"] == 4.0
    assert effect["dual_x_head_only_interaction"] == 4.0
