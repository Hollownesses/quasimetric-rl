from pathlib import Path

import numpy as np
import torch

from minimal_qrl.baselines import HybridAStarConfig
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.comm_inspection_baseline_eval import METHODS
from minimal_qrl.industry_exp.joint_feasible_iqe import (
    JointFeasibleProblem,
    build_joint_feasible_problem,
    constraint_warmup_multiplier,
    lattice_successor_ranking,
    maxlike_squared_excess,
    select_fixed_u_trap_anchors,
    select_topk_active_edges,
    update_active_replay,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "results/diagnostic_u_shadow_corridors/config/diagnostic_scenario.json"


def test_topk_active_edges_uses_worst_violation_across_critics():
    excess = np.asarray(
        [
            [0.0, 2.0, 3.0, 50.0, 1.0],
            [0.0, 7.0, 1.0, 60.0, 4.0],
        ]
    )
    families = np.asarray([0, 0, 0, 1, 0])
    selected = select_topk_active_edges(excess, families, topk=2, family=0)
    assert selected.tolist() == [1, 4]


def test_cumulative_active_replay_never_forgets_historical_edges():
    replay, new_count = update_active_replay(
        np.asarray([7, 2]), np.asarray([2, 9]), mode="cumulative"
    )
    assert replay.tolist() == [2, 7, 9]
    assert new_count == 1
    replaced, new_count = update_active_replay(
        replay, np.asarray([9, 11]), mode="replace"
    )
    assert replaced.tolist() == [9, 11]
    assert new_count == 1


def test_fixed_u_trap_anchors_support_all_or_even_subsets():
    mask = np.asarray([False, True, True, False, True, True])
    assert select_fixed_u_trap_anchors(mask, -1).tolist() == [1, 2, 4, 5]
    assert select_fixed_u_trap_anchors(mask, 2).tolist() == [1, 5]
    assert select_fixed_u_trap_anchors(mask, 0).size == 0


def test_quadratic_constraint_warmup_and_maxlike_loss():
    assert constraint_warmup_multiplier(0, 100, 2.0) == 0.0
    assert constraint_warmup_multiplier(50, 100, 2.0) == 0.25
    assert constraint_warmup_multiplier(100, 100, 2.0) == 1.0
    assert constraint_warmup_multiplier(200, 100, 2.0) == 1.0
    excess = torch.tensor([1.0, 3.0])
    assert torch.isclose(
        maxlike_squared_excess(excess, 2.0), excess.square().mean()
    )
    assert maxlike_squared_excess(excess, 8.0) > excess.square().mean()


def test_lattice_successor_ranking_is_exact_for_reference_values():
    problem = JointFeasibleProblem(
        states=np.zeros((3, 3), dtype=np.float32),
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
    metrics = lattice_successor_ranking(
        problem,
        problem.reference_values,
        source_mask=problem.u_trap_mask,
    )
    assert metrics["source_states"] == 1
    assert metrics["pair_count"] == 1
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["pairwise_accuracy"] == 1.0


def test_small_constructive_problem_has_labels_and_all_constraint_families():
    scenario = load_scenario_config(SCENARIO)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=0, options={"device_id": "u_trap_target"})
    problem = build_joint_feasible_problem(
        env,
        scenario,
        HybridAStarConfig(
            position_resolution=0.5,
            heading_bins=12,
            primitive_steps=5,
            primitive_scales=(-1.0, -0.5, 0.0, 0.5, 1.0),
        ),
    )
    assert problem.num_states == len(problem.reference_values)
    assert np.all(np.isfinite(problem.reference_values))
    assert set(np.unique(problem.families)) == {0, 1, 2}
    assert int(np.sum(problem.u_trap_mask)) > 1
    assert problem.graph_stats["topology_only_reachability"] is True


def test_shell_and_mppi_evaluator_expose_joint_certificate_phase():
    script = (ROOT / "minimal_qrl/run_comm_inspection_diagnostic.sh").read_text()
    assert "joint_feasible_iqe()" in script
    assert "joint_feasible_iqe)" in script
    assert "joint_feasible_iqe_strong()" in script
    assert "joint_feasible_iqe_strong)" in script
    assert "joint_feasible_iqe_capacity_2x()" in script
    assert "joint_feasible_iqe_capacity_2x)" in script
    assert "JOINT_FEASIBLE_STRONG_REPLAY_MODE:-cumulative" in script
    assert "JOINT_FEASIBLE_STRONG_U_GOAL_ANCHORS:--1" in script
    assert "JOINT_FEASIBLE_CAPACITY_IQE_DIM:-4096" in script
    assert "JOINT_FEASIBLE_CAPACITY_IQE_COMPONENTS:-128" in script
    assert '--reuse-dataset "$source_dataset"' in script
    assert "JOINT_FEASIBLE_RUN_MPPI" in script
    assert "joint_feasible_iqe_mppi" in METHODS
