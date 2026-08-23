#!/usr/bin/env python3
"""Exact shortest-path LP audit for the dense-transition U-trap experiment.

The dense QRL dataset starts motion primitives from a local 4,368-state bank,
but unfolds each primitive into continuous one-step transitions.  This script
first audits whether that local seed bank is itself a closed terminal-reaching
graph.  It then solves the well-posed LP on the complete goal-reachable Hybrid
A* lattice and compares the result with reverse Dijkstra on exactly the same
finite graph.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr, spearmanr

from minimal_qrl.baselines import HybridAStarConfig, HybridAStarValueOracle
from minimal_qrl.baselines.mppi import _state_terms
from minimal_qrl.dataset import (
    DENSE_U_TRAP_STRATA,
    DenseUTrapTransitionConfig,
    build_dense_u_trap_state_bank,
    create_dense_u_trap_transition_dataset,
    _full_graph_digest,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)


DIRECT_GOAL = -1


@dataclass(frozen=True)
class DiscreteValueGraph:
    states: np.ndarray
    valid: np.ndarray
    terminal: np.ndarray
    sources: np.ndarray
    destinations: np.ndarray
    costs: np.ndarray
    action_indices: np.ndarray
    primitive_attempts: int


def _load_failed_starts(path: str | Path) -> tuple[tuple[float, float, float], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("episode_results", payload) if isinstance(payload, dict) else payload
    failed = [
        tuple(float(value) for value in record["start"])
        for record in records
        if str(record.get("stratum", "")) == "u_trap"
        and not bool(record.get("success", False))
        and record.get("start") is not None
    ]
    if not failed:
        raise ValueError(f"no failed U-trap starts found in {path}")
    return tuple(failed)


def _dense_config(
    scenario: Mapping[str, Any],
    args: argparse.Namespace,
) -> DenseUTrapTransitionConfig:
    regions = scenario.get("metadata", {}).get("exploration_diagnostic_regions")
    if not isinstance(regions, Mapping):
        raise ValueError("scenario is missing metadata.exploration_diagnostic_regions")
    return DenseUTrapTransitionConfig(
        device_id=str(args.device_id),
        position_resolution=float(args.position_resolution),
        heading_bins=int(args.heading_bins),
        primitive_steps=int(args.primitive_steps),
        primitive_scales=tuple(float(value) for value in args.primitive_scales),
        diagnostic_regions=regions,
        failed_start_states=_load_failed_starts(args.failure_results),
        failure_position_radius=float(args.failure_position_radius),
        failure_heading_radius=float(args.failure_heading_radius),
    )


def build_discrete_value_graph(
    env: CommInspectionDubinsUAV2D,
    config: HybridAStarConfig,
) -> DiscreteValueGraph:
    """Materialize the finite primitive graph used by the value Oracle."""

    helper = HybridAStarValueOracle(config)
    states, shape = helper._grid(env)
    helper._grid_shape = shape
    valid = helper._valid_grid_states(env, states)
    zeros = np.zeros(len(states), dtype=bool)
    _unused_cost, terminal = _state_terms(env, states, zeros, zeros)
    terminal &= valid

    edge_sources: list[np.ndarray] = []
    edge_destinations: list[np.ndarray] = []
    edge_costs: list[np.ndarray] = []
    edge_actions: list[np.ndarray] = []
    for action_index, scale in enumerate(config.primitive_scales):
        sources, destinations, costs, success = helper._primitive_edges(
            env,
            states,
            valid,
            terminal,
            float(scale) * float(env.omega_max),
        )
        usable = success.copy()
        unfinished = ~success & (destinations >= 0)
        if np.any(unfinished):
            unfinished_indices = np.flatnonzero(unfinished)
            destination_valid = valid[destinations[unfinished_indices]]
            usable[unfinished_indices[~destination_valid]] = False
        selected = np.flatnonzero(usable | (unfinished & valid[np.maximum(destinations, 0)]))
        if not len(selected):
            continue
        selected_destinations = destinations[selected].copy()
        selected_destinations[success[selected]] = DIRECT_GOAL
        edge_sources.append(sources[selected].astype(np.int64, copy=False))
        edge_destinations.append(selected_destinations.astype(np.int64, copy=False))
        edge_costs.append(costs[selected].astype(np.float64, copy=False))
        edge_actions.append(
            np.full(len(selected), int(action_index), dtype=np.int16)
        )

    return DiscreteValueGraph(
        states=states,
        valid=valid,
        terminal=terminal,
        sources=np.concatenate(edge_sources),
        destinations=np.concatenate(edge_destinations),
        costs=np.concatenate(edge_costs),
        action_indices=np.concatenate(edge_actions),
        primitive_attempts=int(np.sum(valid & ~terminal) * len(config.primitive_scales)),
    )


def reverse_dijkstra(graph: DiscreteValueGraph) -> np.ndarray:
    """Exact shortest-path values on ``graph``; infinity means no path to G."""

    values = np.full(len(graph.states), np.inf, dtype=np.float64)
    values[graph.terminal] = 0.0
    direct = graph.destinations == DIRECT_GOAL
    if np.any(direct):
        np.minimum.at(values, graph.sources[direct], graph.costs[direct])

    normal = ~direct
    sources = graph.sources[normal]
    destinations = graph.destinations[normal]
    costs = graph.costs[normal]
    order = np.argsort(destinations, kind="stable")
    sources = sources[order]
    destinations = destinations[order]
    costs = costs[order]
    offsets = np.zeros(len(graph.states) + 1, dtype=np.int64)
    if len(destinations):
        offsets[1:] = np.cumsum(
            np.bincount(destinations, minlength=len(graph.states))
        )

    queue = [
        (float(values[index]), int(index))
        for index in np.flatnonzero(np.isfinite(values))
    ]
    heapq.heapify(queue)
    while queue:
        current_cost, destination = heapq.heappop(queue)
        if current_cost > float(values[destination]) + 1e-12:
            continue
        for edge_index in range(
            int(offsets[destination]), int(offsets[destination + 1])
        ):
            source = int(sources[edge_index])
            candidate = current_cost + float(costs[edge_index])
            if candidate + 1e-12 < float(values[source]):
                values[source] = candidate
                heapq.heappush(queue, (candidate, source))
    return values


def solve_exact_value_lp(
    *,
    terminal: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    costs: np.ndarray,
    included: np.ndarray,
    time_limit_sec: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maximize sum D on a terminal-reachable subgraph using HiGHS."""

    terminal = np.asarray(terminal, dtype=bool)
    included = np.asarray(included, dtype=bool)
    kept = np.flatnonzero(included)
    global_to_lp = np.full(len(included), -1, dtype=np.int64)
    global_to_lp[kept] = np.arange(len(kept), dtype=np.int64)

    direct = destinations == DIRECT_GOAL
    edge_keep = included[sources] & (
        direct | ((destinations >= 0) & included[np.maximum(destinations, 0)])
    )
    selected_sources = global_to_lp[sources[edge_keep]]
    selected_destinations_global = destinations[edge_keep]
    selected_costs = np.asarray(costs[edge_keep], dtype=np.float64)
    selected_direct = selected_destinations_global == DIRECT_GOAL
    selected_destinations = np.full(
        len(selected_destinations_global), -1, dtype=np.int64
    )
    selected_destinations[~selected_direct] = global_to_lp[
        selected_destinations_global[~selected_direct]
    ]

    constraint_count = len(selected_sources)
    normal_rows = np.flatnonzero(~selected_direct)
    direct_rows = np.flatnonzero(selected_direct)
    rows = np.concatenate([normal_rows, normal_rows, direct_rows])
    columns = np.concatenate(
        [
            selected_sources[normal_rows],
            selected_destinations[normal_rows],
            selected_sources[direct_rows],
        ]
    )
    data = np.concatenate(
        [
            np.ones(len(normal_rows), dtype=np.float64),
            -np.ones(len(normal_rows), dtype=np.float64),
            np.ones(len(direct_rows), dtype=np.float64),
        ]
    )
    inequalities = coo_matrix(
        (data, (rows, columns)),
        shape=(constraint_count, len(kept)),
    ).tocsr()
    bounds = [
        (0.0, 0.0) if bool(terminal[index]) else (0.0, None)
        for index in kept
    ]

    started = perf_counter()
    result = linprog(
        -np.ones(len(kept), dtype=np.float64),
        A_ub=inequalities,
        b_ub=selected_costs,
        bounds=bounds,
        method="highs",
        options={"time_limit": float(time_limit_sec)},
    )
    solve_time = perf_counter() - started
    values = np.full(len(included), np.inf, dtype=np.float64)
    if result.success:
        values[kept] = result.x
        lhs = result.x[selected_sources]
        normal = ~selected_direct
        lhs[normal] -= result.x[selected_destinations[normal]]
        max_violation = float(np.max(lhs - selected_costs, initial=0.0))
    else:
        max_violation = float("nan")
    diagnostics = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "variables": int(len(kept)),
        "constraints": int(constraint_count),
        "iterations": int(getattr(result, "nit", 0)),
        "solve_time_sec": float(solve_time),
        "objective_sum": float(-result.fun) if result.success else None,
        "max_constraint_violation": max_violation,
        "terminal_max_abs": (
            float(np.max(np.abs(values[terminal & included]), initial=0.0))
            if result.success
            else None
        ),
    }
    return values, diagnostics


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    keep = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[keep]
    target = target[keep]
    error = prediction - target
    if len(prediction) >= 2 and np.std(prediction) > 0 and np.std(target) > 0:
        pearson = float(pearsonr(prediction, target).statistic)
        spearman = float(spearmanr(prediction, target).statistic)
    else:
        pearson = float("nan")
        spearman = float("nan")
    return {
        "count": int(len(prediction)),
        "mae": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else float("nan"),
        "max_abs_error": float(np.max(np.abs(error), initial=0.0)),
        "pearson": pearson,
        "spearman": spearman,
    }


def audit_dense_local_graph(
    graph: DiscreteValueGraph,
    helper: HybridAStarValueOracle,
    env: CommInspectionDubinsUAV2D,
    dense_config: DenseUTrapTransitionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    local_states, strata = build_dense_u_trap_state_bank(env, dense_config)
    local_indices = helper._state_indices(env, local_states)
    local_mask = np.zeros(len(graph.states), dtype=bool)
    local_mask[local_indices] = True
    from_local = local_mask[graph.sources]
    destinations = graph.destinations[from_local]
    direct = destinations == DIRECT_GOAL
    inside = (~direct) & local_mask[np.maximum(destinations, 0)]
    leave = (~direct) & ~inside
    attempts = int(len(local_indices) * len(dense_config.primitive_scales))
    represented = int(np.sum(from_local))
    local_goal_count = int(np.sum(graph.terminal[local_indices]))
    audit = {
        "seed_lattice_states": int(len(local_states)),
        "state_bounds": {
            "x_min": float(np.min(local_states[:, 0])),
            "x_max": float(np.max(local_states[:, 0])),
            "y_min": float(np.min(local_states[:, 1])),
            "y_max": float(np.max(local_states[:, 1])),
        },
        "terminal_goal_states": local_goal_count,
        "primitive_attempts": attempts,
        "valid_edges_within_local_nodes": int(np.sum(inside)),
        "valid_edges_leaving_local_nodes": int(np.sum(leave)),
        "direct_goal_edges": int(np.sum(direct)),
        "unusable_collision_oob_or_invalid_projection_primitives": int(
            attempts - represented
        ),
        "successor_closed": bool(np.sum(leave) == 0),
        "contains_terminal_set": bool(local_goal_count > 0),
        "lp_well_posed": bool(local_goal_count > 0 and np.sum(leave) == 0),
        "lp_status": (
            "not_solved_unbounded_no_terminal_set"
            if local_goal_count == 0
            else "not_solved_graph_not_successor_closed"
            if np.sum(leave) > 0
            else "well_posed"
        ),
        "reason": (
            "Adding a constant to every local D(s) preserves all local edge inequalities; "
            "without D(g)=0 in this node set, max sum_s D(s) is unbounded."
            if local_goal_count == 0
            else "Some usable primitive successors have no variable in the local LP."
            if np.sum(leave) > 0
            else ""
        ),
        "per_stratum_states": {
            name: int(np.sum(strata == index))
            for index, name in enumerate(DENSE_U_TRAP_STRATA)
        },
    }
    return local_indices, strata, audit


def validate_dense_real_dynamics(
    graph: DiscreteValueGraph,
    helper: HybridAStarValueOracle,
    env: CommInspectionDubinsUAV2D,
    dense_config: DenseUTrapTransitionConfig,
    local_indices: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Re-run the 105k local env steps and compare primitive edges/costs."""

    lookup = {
        (int(source), int(action)): (int(destination), float(cost))
        for source, destination, cost, action in zip(
            graph.sources,
            graph.destinations,
            graph.costs,
            graph.action_indices,
        )
    }
    stats: dict[str, Any] = {}
    episodes = create_dense_u_trap_transition_dataset(
        env,
        dense_config,
        seed=int(seed),
        collection_stats=stats,
    )
    checked = 0
    matched_validity = 0
    matched_destination = 0
    expected_valid_actual_invalid = 0
    expected_invalid_actual_valid = 0
    actual_valid_projected_to_invalid_lattice = 0
    mismatch_examples: list[dict[str, Any]] = []
    cost_errors: list[float] = []
    for episode_index, episode in enumerate(episodes):
        state_offset, action_index = divmod(
            episode_index, len(dense_config.primitive_scales)
        )
        source = int(local_indices[state_offset])
        actual_cost = -float(episode.rewards.sum().item())
        final_observation = episode.all_observations[-1].numpy()
        final_state = env.observation_to_state(final_observation)
        actual_success = bool(env.is_task_feasible(final_state))
        actual_invalid = bool(episode.terminals[-1].item()) and not actual_success
        actual_destination = DIRECT_GOAL
        if not actual_invalid and not actual_success:
            actual_destination = int(
                helper._state_indices(env, final_state[None, :])[0]
            )
            actual_valid_projected_to_invalid_lattice += int(
                actual_destination < 0 or not bool(graph.valid[actual_destination])
            )
        expected = lookup.get((source, action_index))
        expected_valid = expected is not None
        actual_valid = not actual_invalid
        matched_validity += int(expected_valid == actual_valid)
        expected_valid_actual_invalid += int(expected_valid and not actual_valid)
        expected_invalid_actual_valid += int(not expected_valid and actual_valid)
        if expected_valid != actual_valid and len(mismatch_examples) < 10:
            mismatch_examples.append(
                {
                    "source_state": [
                        float(value) for value in graph.states[source]
                    ],
                    "primitive_scale": float(
                        dense_config.primitive_scales[action_index]
                    ),
                    "expected_valid": bool(expected_valid),
                    "actual_valid": bool(actual_valid),
                    "actual_cost": float(actual_cost),
                    "actual_destination_index": int(actual_destination),
                    "actual_destination_lattice_valid": bool(
                        actual_destination == DIRECT_GOAL
                        or (
                            actual_destination >= 0
                            and graph.valid[actual_destination]
                        )
                    ),
                }
            )
        if expected_valid and actual_valid:
            matched_destination += int(actual_destination == expected[0])
            cost_errors.append(abs(actual_cost - expected[1]))
        checked += 1
    dense_stats = stats["dense_u_trap"]
    return {
        "primitive_rollouts_checked": int(checked),
        "expanded_one_step_transitions": int(dense_stats["transitions"]),
        "validity_match_rate": float(matched_validity / max(checked, 1)),
        "validity_mismatch_count": int(checked - matched_validity),
        "oracle_valid_actual_invalid": int(expected_valid_actual_invalid),
        "oracle_invalid_actual_valid": int(expected_invalid_actual_valid),
        "actual_valid_projected_to_invalid_lattice": int(
            actual_valid_projected_to_invalid_lattice
        ),
        "validity_mismatch_examples": mismatch_examples,
        "destination_match_rate_on_valid": float(
            matched_destination / max(len(cost_errors), 1)
        ),
        "cost_mae_on_valid_primitives": float(np.mean(cost_errors)),
        "cost_max_abs_error_on_valid_primitives": float(
            np.max(cost_errors, initial=0.0)
        ),
        "mean_one_step_cost": float(dense_stats["mean_one_step_cost"]),
    }


def _write_local_csv(
    path: Path,
    *,
    graph: DiscreteValueGraph,
    local_indices: np.ndarray,
    strata: np.ndarray,
    lp_values: np.ndarray,
    oracle_values: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "state_index",
                "x",
                "y",
                "heading",
                "stratum",
                "lp_value",
                "oracle_value",
                "abs_error",
            ),
        )
        writer.writeheader()
        for state_index, stratum in zip(local_indices, strata):
            state = graph.states[state_index]
            lp_value = float(lp_values[state_index])
            oracle_value = float(oracle_values[state_index])
            writer.writerow(
                {
                    "state_index": int(state_index),
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "heading": float(state[2]),
                    "stratum": DENSE_U_TRAP_STRATA[int(stratum)],
                    "lp_value": lp_value,
                    "oracle_value": oracle_value,
                    "abs_error": abs(lp_value - oracle_value),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--failure-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-id", default="u_trap_target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-resolution", type=float, default=0.25)
    parser.add_argument("--heading-bins", type=int, default=24)
    parser.add_argument("--primitive-steps", type=int, default=5)
    parser.add_argument(
        "--primitive-scales",
        type=float,
        nargs="+",
        default=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    parser.add_argument("--failure-position-radius", type=float, default=0.75)
    parser.add_argument("--failure-heading-radius", type=float, default=0.65)
    parser.add_argument("--lp-time-limit-sec", type=float, default=300.0)
    parser.add_argument("--oracle-value-cache-dir", default=None)
    parser.add_argument(
        "--skip-real-dynamics-validation",
        action="store_true",
        help="Skip replaying all local primitives through env.step().",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_config(args.scenario_config)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=int(args.seed), options={"device_id": str(args.device_id)})
    dense_config = _dense_config(scenario, args)
    astar_config = HybridAStarConfig(
        position_resolution=float(args.position_resolution),
        heading_bins=int(args.heading_bins),
        primitive_steps=int(args.primitive_steps),
        primitive_scales=tuple(float(value) for value in args.primitive_scales),
    )

    graph_started = perf_counter()
    graph = build_discrete_value_graph(env, astar_config)
    graph_time = perf_counter() - graph_started
    helper = HybridAStarValueOracle(astar_config)
    _unused_grid, helper._grid_shape = helper._grid(env)
    local_indices, strata, local_audit = audit_dense_local_graph(
        graph, helper, env, dense_config
    )
    dynamics = None
    if not args.skip_real_dynamics_validation:
        dynamics = validate_dense_real_dynamics(
            graph,
            helper,
            env,
            dense_config,
            local_indices,
            seed=int(args.seed),
        )

    dijkstra_started = perf_counter()
    dijkstra_values = reverse_dijkstra(graph)
    dijkstra_time = perf_counter() - dijkstra_started
    reachable = graph.valid & np.isfinite(dijkstra_values)
    direct = graph.destinations == DIRECT_GOAL
    lp_edge_keep = reachable[graph.sources] & (
        direct
        | (
            (graph.destinations >= 0)
            & reachable[np.maximum(graph.destinations, 0)]
        )
    )
    lp_values, lp_diagnostics = solve_exact_value_lp(
        terminal=graph.terminal,
        sources=graph.sources,
        destinations=graph.destinations,
        costs=graph.costs,
        included=reachable,
        time_limit_sec=float(args.lp_time_limit_sec),
    )
    if not lp_diagnostics["success"]:
        raise RuntimeError(f"exact LP failed: {lp_diagnostics['message']}")

    oracle = HybridAStarValueOracle(
        astar_config,
        cache_dir=args.oracle_value_cache_dir,
    )
    oracle_diagnostics = oracle.begin_episode(env, seed=int(args.seed))
    _oracle_states, oracle_values = oracle.lattice_dataset(env, reachable_only=False)
    lp_vs_dijkstra = _metrics(lp_values[reachable], dijkstra_values[reachable])
    lp_vs_oracle = _metrics(lp_values[reachable], oracle_values[reachable])
    local_reachable = reachable[local_indices]
    local_lp_vs_oracle = _metrics(
        lp_values[local_indices][local_reachable],
        oracle_values[local_indices][local_reachable],
    )
    _write_local_csv(
        output_dir / "exact_lp_u_trap_local_values.csv",
        graph=graph,
        local_indices=local_indices,
        strata=strata,
        lp_values=lp_values,
        oracle_values=oracle_values,
    )

    payload = {
        "experiment": "exact_discrete_value_lp",
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "failure_results": str(Path(args.failure_results).resolve()),
        "mathematical_problem": {
            "terminal_condition": "D(g)=0 for g in G",
            "edge_constraints": "D(s) <= c(s,s_prime) + D(s_prime)",
            "objective": "maximize sum_s D(s)",
        },
        "dense_local_graph_audit": local_audit,
        "dense_real_dynamics_validation": dynamics,
        "full_lattice_graph": {
            "grid_states": int(len(graph.states)),
            "valid_states": int(np.sum(graph.valid)),
            "terminal_goal_states": int(np.sum(graph.terminal)),
            "goal_reachable_states": int(np.sum(reachable)),
            "valid_but_goal_unreachable_states": int(np.sum(graph.valid & ~reachable)),
            "edge_constraints": int(len(graph.sources)),
            "ordinary_transition_constraints": int(
                np.sum(graph.destinations != DIRECT_GOAL)
            ),
            "direct_goal_constraints": int(
                np.sum(graph.destinations == DIRECT_GOAL)
            ),
            "primitive_attempts": int(graph.primitive_attempts),
            "graph_build_time_sec": float(graph_time),
            "reverse_dijkstra_time_sec": float(dijkstra_time),
            "raw_all_valid_lp_status": (
                "unbounded_due_to_goal_unreachable_states"
                if np.any(graph.valid & ~reachable)
                else "bounded"
            ),
            "lp_domain": "goal-reachable valid states",
            "lp_constraint_graph_digest": _full_graph_digest(
                sources=graph.sources[lp_edge_keep].astype(np.int64, copy=False),
                destinations=graph.destinations[lp_edge_keep].astype(
                    np.int64, copy=False
                ),
                costs=graph.costs[lp_edge_keep].astype(np.float32, copy=False),
                terminal=graph.terminal,
            ),
        },
        "lp_solver": lp_diagnostics,
        "comparison": {
            "lp_vs_independent_reverse_dijkstra": lp_vs_dijkstra,
            "lp_vs_cached_oracle": lp_vs_oracle,
            "u_trap_local_lp_vs_oracle": local_lp_vs_oracle,
            "u_trap_local_reachable_states": int(np.sum(local_reachable)),
        },
        "oracle_diagnostics": oracle_diagnostics,
        "interpretation": {
            "dense_local_4368_variable_lp_valid": bool(local_audit["lp_well_posed"]),
            "full_reachable_lattice_lp_matches_oracle": bool(
                lp_vs_oracle["max_abs_error"] <= 1e-3
            ),
            "dense_real_transitions_match_oracle_lattice_edges": bool(
                lp_vs_oracle["max_abs_error"] <= 1e-3
                and (
                    dynamics is None
                    or (
                        dynamics["validity_match_rate"] == 1.0
                        and dynamics["destination_match_rate_on_valid"] == 1.0
                        and dynamics["cost_max_abs_error_on_valid_primitives"] <= 1e-4
                    )
                )
            ),
        },
        "total_runtime_sec": float(perf_counter() - started),
    }
    metrics_path = output_dir / "exact_discrete_value_lp_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved exact LP audit: {metrics_path}")


if __name__ == "__main__":
    main()
