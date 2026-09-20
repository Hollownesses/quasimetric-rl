"""Reusable topology, constraint-tail, and MPPI-ranking diagnostics for QRL.

These metrics are evaluation-only.  Reverse-Dijkstra values are used to score
the learned critic, never to update it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from minimal_qrl.baselines import HybridAStarValueOracle, MPPIConfig
from minimal_qrl.baselines.mppi import simulate_action_sequences
from minimal_qrl.eval.comm_inspection_execution_eval import load_task_records
from minimal_qrl.industry_exp.full_graph_checkpoint_audit import (
    summarize_constraint_family,
)
from minimal_qrl.industry_exp.joint_feasible_iqe import JointFeasibleProblem


def _empty_constraint_summary(epsilon: float) -> dict[str, Any]:
    return {
        "count": 0,
        "epsilon": float(epsilon),
        "violation_count": 0,
        "violation_fraction": None,
        "positive_excess_mean": None,
        "positive_excess_p99": None,
        "positive_excess_p999": None,
        "positive_excess_max": None,
        "squared_excess_mean": None,
        "all_constraints_satisfied": None,
    }


def _summarize_masked_constraint(
    distances: np.ndarray,
    costs: np.ndarray,
    mask: np.ndarray,
    *,
    epsilon: float,
    numerical_tolerance: float,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if not bool(np.any(mask)):
        return _empty_constraint_summary(epsilon)
    return summarize_constraint_family(
        np.asarray(distances)[mask],
        np.asarray(costs)[mask],
        epsilon=float(epsilon),
        numerical_tolerance=float(numerical_tolerance),
    )


def _oracle_optimal_edge_mask(
    problem: JointFeasibleProblem,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    reference = np.asarray(problem.reference_values, dtype=np.float64)
    successor = np.zeros(problem.num_edges, dtype=np.float64)
    physical = problem.destinations >= 0
    successor[physical] = reference[problem.destinations[physical]]
    scores = problem.costs.astype(np.float64) + successor
    optimal = np.zeros(problem.num_edges, dtype=bool)
    order = np.argsort(problem.sources, kind="stable")
    ordered_sources = problem.sources[order]
    unique_sources, begins = np.unique(ordered_sources, return_index=True)
    ends = np.r_[begins[1:], len(order)]
    for _source, begin, end in zip(unique_sources, begins, ends):
        indices = order[begin:end]
        optimal[indices] = scores[indices] <= float(np.min(scores[indices])) + float(
            tolerance
        )
    return optimal


def u_region_constraint_tails(
    problem: JointFeasibleProblem,
    edge_distances: np.ndarray,
    values: np.ndarray,
    *,
    epsilons: Mapping[str, float] | None = None,
    numerical_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Audit constraint tails on every edge whose source lies inside the U."""

    epsilons = dict(
        epsilons
        or {"ordinary": 0.25, "direct_goal": 0.25, "terminal_goal": 0.0}
    )
    edge_distances = np.asarray(edge_distances, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    costs = problem.costs.astype(np.float64)
    successor_values = np.zeros(problem.num_edges, dtype=np.float64)
    physical = problem.destinations >= 0
    successor_values[physical] = values[problem.destinations[physical]]
    bellman_lhs = values[problem.sources] - successor_values
    u_source = problem.u_trap_mask[problem.sources]
    oracle_optimal = _oracle_optimal_edge_mask(problem)
    family_names = ("ordinary", "direct_goal", "terminal_goal")
    groups = {
        "all_outgoing": u_source,
        "oracle_optimal_outgoing": u_source & oracle_optimal,
        "non_optimal_outgoing": u_source & ~oracle_optimal,
    }
    result: dict[str, Any] = {
        "u_source_states": int(np.sum(problem.u_trap_mask)),
        "u_outgoing_edges": int(np.sum(u_source)),
        "groups": {},
    }
    for group_name, group_mask in groups.items():
        group_payload: dict[str, Any] = {
            "edge_count": int(np.sum(group_mask)),
            "pairwise_quasimetric_edge": {},
            "goal_slice_bellman": {},
        }
        for family_index, family_name in enumerate(family_names):
            mask = group_mask & (problem.families == family_index)
            epsilon = float(epsilons[family_name])
            group_payload["pairwise_quasimetric_edge"][family_name] = (
                _summarize_masked_constraint(
                    edge_distances,
                    costs,
                    mask,
                    epsilon=epsilon,
                    numerical_tolerance=numerical_tolerance,
                )
            )
            group_payload["goal_slice_bellman"][family_name] = (
                _summarize_masked_constraint(
                    bellman_lhs,
                    costs,
                    mask,
                    epsilon=epsilon,
                    numerical_tolerance=numerical_tolerance,
                )
            )
        result["groups"][group_name] = group_payload
    return result


@dataclass(frozen=True)
class MPPIRankingContractCase:
    task_id: str
    repeat: int
    episode_seed: int
    candidate_actions: np.ndarray
    final_observations: np.ndarray
    goal_observation: np.ndarray
    running_costs: np.ndarray
    success: np.ndarray
    invalid: np.ndarray
    oracle_scores: np.ndarray


def build_mppi_ranking_contract(
    env,
    oracle: HybridAStarValueOracle,
    *,
    task_bank_path: str,
    scenario_id: str,
    split: str,
    stratum: str,
    config: MPPIConfig,
    repeats: int,
) -> tuple[list[MPPIRankingContractCase], dict[str, Any]]:
    """Build fixed first-decision MPPI candidate banks with Oracle scores."""

    _task_bank, records = load_task_records(
        task_bank_path,
        split=str(split),
        bounds=[float(env.x_min), float(env.y_min), float(env.x_max), float(env.y_max)],
        scenario_id=str(scenario_id),
    )
    records = [row for row in records if str(row.get("stratum")) == str(stratum)]
    if not records:
        raise ValueError(f"task bank contains no {split}/{stratum} tasks")
    cases: list[MPPIRankingContractCase] = []
    oracle_diagnostics: dict[str, Any] = {}
    for record in records:
        for repeat in range(int(repeats)):
            episode_seed = int(record["seed"]) + repeat * 1_000_003
            env.reset(
                seed=episode_seed,
                options={
                    "device_id": str(record["device_id"]),
                    "start": np.asarray(record["start"], dtype=np.float32),
                },
            )
            oracle_diagnostics = oracle.begin_episode(env, seed=episode_seed)
            rng = np.random.default_rng(episode_seed + 7001)
            candidates = rng.normal(
                0.0,
                float(config.noise_sigma),
                size=(int(config.num_samples), int(config.horizon)),
            ).astype(np.float32)
            candidates = np.clip(
                candidates,
                -float(env.omega_max),
                float(env.omega_max),
            )
            candidates[0] = 0.0
            rollout = simulate_action_sequences(env, env.state, candidates)
            oracle_scores = rollout["costs"].astype(np.float64)
            unfinished = ~(rollout["success"] | rollout["invalid"])
            if np.any(unfinished):
                oracle_scores[unfinished] += float(config.terminal_weight) * oracle.batch_value(
                    env,
                    rollout["final_states"][unfinished],
                ).astype(np.float64)
            oracle_scores[rollout["invalid"]] += float(config.invalid_penalty)
            final_observations = np.stack(
                [env.state_to_observation(state) for state in rollout["final_states"]]
            ).astype(np.float32)
            cases.append(
                MPPIRankingContractCase(
                    task_id=str(record["task_id"]),
                    repeat=int(repeat),
                    episode_seed=episode_seed,
                    candidate_actions=candidates,
                    final_observations=final_observations,
                    goal_observation=env.abstract_goal_observation().astype(np.float32),
                    running_costs=rollout["costs"].astype(np.float64),
                    success=rollout["success"].astype(bool),
                    invalid=rollout["invalid"].astype(bool),
                    oracle_scores=oracle_scores,
                )
            )
            oracle.end_episode()
    return cases, {
        "split": str(split),
        "stratum": str(stratum),
        "tasks": int(len(records)),
        "repeats": int(repeats),
        "cases": int(len(cases)),
        "horizon": int(config.horizon),
        "num_samples": int(config.num_samples),
        "noise_sigma": float(config.noise_sigma),
        "temperature": float(config.temperature),
        "terminal_weight": float(config.terminal_weight),
        "decision": "episode_start_first_decision",
        "oracle": oracle_diagnostics,
    }


def _pairwise_concordance(predicted: np.ndarray, target: np.ndarray) -> tuple[float | None, int]:
    agreements: list[float] = []
    for left in range(len(target)):
        target_delta = target[left] - target[left + 1 :]
        predicted_delta = predicted[left] - predicted[left + 1 :]
        keep = np.abs(target_delta) > 1e-9
        if np.any(keep):
            agreements.extend((predicted_delta[keep] * target_delta[keep] > 0.0).astype(float))
    return (float(np.mean(agreements)) if agreements else None, len(agreements))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        begin = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _mppi_weighted_action(scores: np.ndarray, actions: np.ndarray, temperature: float) -> float:
    shifted = scores - float(np.min(scores))
    weights = np.exp(-shifted / max(float(temperature), 1e-6))
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
        return float(actions[int(np.argmin(scores)), 0])
    weights /= weight_sum
    return float(np.sum(weights * actions[:, 0]))


def evaluate_mppi_ranking_contract(
    value_adapter,
    cases: Sequence[MPPIRankingContractCase],
    *,
    config: MPPIConfig,
    top_k: int = 8,
) -> dict[str, Any]:
    """Compare learned and Oracle ordering over identical horizon-H rollouts."""

    records: list[dict[str, Any]] = []
    for case in cases:
        learned_scores = case.running_costs.copy()
        unfinished = ~(case.success | case.invalid)
        if np.any(unfinished):
            goals = np.repeat(
                case.goal_observation[None, :],
                int(np.sum(unfinished)),
                axis=0,
            )
            learned_scores[unfinished] += float(config.terminal_weight) * value_adapter.batch_value(
                case.final_observations[unfinished], goals
            ).astype(np.float64)
        learned_scores[case.invalid] += float(config.invalid_penalty)
        eligible = ~case.invalid
        learned = learned_scores[eligible]
        oracle = case.oracle_scores[eligible]
        pairwise, pair_count = _pairwise_concordance(learned, oracle)
        learned_ranks = _average_ranks(learned)
        oracle_ranks = _average_ranks(oracle)
        spearman = _correlation(learned_ranks, oracle_ranks)
        count = len(oracle)
        k = min(max(1, int(top_k)), count) if count else 0
        learned_top = set(np.argsort(learned)[:k].tolist())
        oracle_top = set(np.argsort(oracle)[:k].tolist())
        selected = int(np.argmin(learned)) if count else None
        regret = (
            max(0.0, float(oracle[selected] - np.min(oracle)))
            if selected is not None
            else None
        )
        oracle_action = _mppi_weighted_action(
            case.oracle_scores,
            case.candidate_actions,
            float(config.temperature),
        )
        learned_action = _mppi_weighted_action(
            learned_scores,
            case.candidate_actions,
            float(config.temperature),
        )
        records.append(
            {
                "task_id": case.task_id,
                "repeat": int(case.repeat),
                "episode_seed": int(case.episode_seed),
                "eligible_candidates": int(count),
                "pairwise_concordance": pairwise,
                "pair_count": int(pair_count),
                "spearman": spearman,
                "top_k": int(k),
                "top_k_overlap": (
                    float(len(learned_top & oracle_top) / k) if k else None
                ),
                "learned_selected_oracle_regret": regret,
                "oracle_weighted_action": oracle_action,
                "learned_weighted_action": learned_action,
                "weighted_action_abs_error": abs(learned_action - oracle_action),
                "weighted_action_sign_agreement": float(
                    abs(oracle_action) <= 1e-8
                    or np.sign(learned_action) == np.sign(oracle_action)
                ),
            }
        )

    def mean_field(name: str) -> float | None:
        values = [float(row[name]) for row in records if row[name] is not None]
        return float(np.mean(values)) if values else None

    regrets = np.asarray(
        [
            row["learned_selected_oracle_regret"]
            for row in records
            if row["learned_selected_oracle_regret"] is not None
        ],
        dtype=np.float64,
    )
    return {
        "cases": int(len(records)),
        "pairwise_concordance_mean": mean_field("pairwise_concordance"),
        "spearman_mean": mean_field("spearman"),
        "top_k_overlap_mean": mean_field("top_k_overlap"),
        "learned_selected_oracle_regret": {
            "mean": float(np.mean(regrets)) if len(regrets) else None,
            "p50": float(np.quantile(regrets, 0.50)) if len(regrets) else None,
            "p95": float(np.quantile(regrets, 0.95)) if len(regrets) else None,
            "max": float(np.max(regrets)) if len(regrets) else None,
            "zero_fraction": float(np.mean(regrets <= 1e-6)) if len(regrets) else None,
        },
        "weighted_action_abs_error_mean": mean_field("weighted_action_abs_error"),
        "weighted_action_sign_agreement": mean_field(
            "weighted_action_sign_agreement"
        ),
        "records": records,
    }
