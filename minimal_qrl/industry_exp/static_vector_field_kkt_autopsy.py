#!/usr/bin/env python3
"""Static vector-field and KKT autopsy at the exact U-trap value function.

This is a measurement experiment, not a training or repair experiment.  It
uses reverse-Dijkstra values only as the point at which the QRL update field is
probed.  No parameter is optimized toward those values.

The experiment answers two separate questions on the same finite graph:

1. Does the exact per-edge shortest-path LP satisfy its KKT system?
2. Do the full-graph and current-style minibatch QRL surrogate gradients point
   in the same direction at that exact solution?

Both the implemented squared hinge and a linear-hinge right subgradient at
active constraints are audited.  The latter is a diagnostic subgradient, not
a proposed training change.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog, nnls
from scipy.sparse import coo_matrix

from minimal_qrl.baselines import HybridAStarConfig
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.joint_feasible_iqe import (
    JointFeasibleProblem,
    build_joint_feasible_problem,
    lattice_successor_ranking,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)
from minimal_qrl.industry_exp.tabular_potential_qrl import FAMILY_NAMES


SURROGATES = ("squared_hinge", "linear_hinge_right_subgradient")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "std": None, "p05": None,
                "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator <= 1e-30:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _project_fixed_terminals(
    gradient: np.ndarray, problem: JointFeasibleProblem
) -> np.ndarray:
    result = np.asarray(gradient, dtype=np.float64).copy()
    result[problem.terminal_indices] = 0.0
    return result


def compact_reverse_dijkstra(problem: JointFeasibleProblem) -> np.ndarray:
    """Shortest-path values using exactly the compact graph's stored costs."""

    values = np.full(problem.num_states, np.inf, dtype=np.float64)
    values[problem.terminal_indices] = 0.0
    direct = problem.destinations < 0
    if np.any(direct):
        np.minimum.at(
            values,
            problem.sources[direct],
            problem.costs[direct].astype(np.float64),
        )

    normal = ~direct
    sources = problem.sources[normal]
    destinations = problem.destinations[normal]
    costs = problem.costs[normal].astype(np.float64)
    order = np.argsort(destinations, kind="stable")
    sources = sources[order]
    destinations = destinations[order]
    costs = costs[order]
    offsets = np.zeros(problem.num_states + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        np.bincount(destinations, minlength=problem.num_states)
    )
    queue = [
        (float(values[index]), int(index))
        for index in np.flatnonzero(np.isfinite(values))
    ]
    heapq.heapify(queue)
    while queue:
        current, destination = heapq.heappop(queue)
        if current > float(values[destination]) + 1e-12:
            continue
        for edge in range(
            int(offsets[destination]), int(offsets[destination + 1])
        ):
            source = int(sources[edge])
            candidate = current + float(costs[edge])
            if candidate + 1e-12 < float(values[source]):
                values[source] = candidate
                heapq.heappush(queue, (candidate, source))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("compact training graph contains goal-unreachable states")
    return values


def _vector_stats(
    vector: np.ndarray,
    problem: JointFeasibleProblem,
) -> dict[str, Any]:
    vector = np.asarray(vector, dtype=np.float64)
    non_u = ~np.asarray(problem.u_trap_mask, dtype=bool)

    def summarize(selected: np.ndarray) -> dict[str, Any]:
        values = vector[selected]
        return {
            "coordinates": int(len(values)),
            "l2_norm": _norm(values),
            "mean_abs": float(np.mean(np.abs(values))) if len(values) else None,
            "max_abs": float(np.max(np.abs(values), initial=0.0)),
            "nonzero_fraction": (
                float(np.mean(np.abs(values) > 1e-12)) if len(values) else None
            ),
        }

    return {
        "all": summarize(np.ones(problem.num_states, dtype=bool)),
        "u_region": summarize(problem.u_trap_mask),
        "non_u_region": summarize(non_u),
    }


def solve_exact_lp_kkt(
    problem: JointFeasibleProblem,
    *,
    time_limit_sec: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve the mean-push LP and return values, per-edge duals, and KKT audit."""

    rows = [np.arange(problem.num_edges, dtype=np.int64)]
    columns = [problem.sources.astype(np.int64, copy=False)]
    data = [np.ones(problem.num_edges, dtype=np.float64)]
    physical_rows = np.flatnonzero(problem.destinations >= 0)
    rows.append(physical_rows)
    columns.append(problem.destinations[physical_rows].astype(np.int64, copy=False))
    data.append(-np.ones(len(physical_rows), dtype=np.float64))
    inequalities = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(problem.num_edges, problem.num_states),
    ).tocsr()

    equality_rows = np.arange(len(problem.terminal_indices), dtype=np.int64)
    equalities = coo_matrix(
        (
            np.ones(len(equality_rows), dtype=np.float64),
            (equality_rows, problem.terminal_indices),
        ),
        shape=(len(equality_rows), problem.num_states),
    ).tocsr()
    objective = np.zeros(problem.num_states, dtype=np.float64)
    objective_indices = np.ones(problem.num_states, dtype=bool)
    objective_indices[problem.terminal_indices] = False
    objective[objective_indices] = -1.0 / float(np.sum(objective_indices))
    costs = problem.costs.astype(np.float64)

    started = perf_counter()
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=costs,
        A_eq=equalities,
        b_eq=np.zeros(len(equality_rows), dtype=np.float64),
        bounds=[(0.0, None)] * problem.num_states,
        method="highs",
        options={"time_limit": float(time_limit_sec)},
    )
    elapsed = perf_counter() - started
    if not result.success:
        raise RuntimeError(f"exact LP failed: {result.message}")

    values = np.asarray(result.x, dtype=np.float64)
    edge_duals = -np.asarray(result.ineqlin.marginals, dtype=np.float64)
    equality_duals = -np.asarray(result.eqlin.marginals, dtype=np.float64)
    lower_duals = np.asarray(result.lower.marginals, dtype=np.float64)
    slack = costs - inequalities @ values
    equality_residual = equalities @ values
    stationarity = (
        objective
        + inequalities.T @ edge_duals
        + equalities.T @ equality_duals
        - lower_duals
    )

    per_family: dict[str, Any] = {}
    active = slack <= 1e-7
    for family_index, family_name in enumerate(FAMILY_NAMES):
        selected = problem.families == family_index
        duals = edge_duals[selected]
        per_family[family_name] = {
            "edges": int(np.sum(selected)),
            "active_edges": int(np.sum(active & selected)),
            "positive_dual_edges": int(np.sum(duals > 1e-12)),
            "dual": _quantiles(duals),
            "active_dual": _quantiles(edge_duals[active & selected]),
        }

    diagnostics = {
        "success": True,
        "status": int(result.status),
        "message": str(result.message),
        "solve_time_sec": float(elapsed),
        "iterations": int(getattr(result, "nit", 0)),
        "variables": int(problem.num_states),
        "edge_constraints": int(problem.num_edges),
        "terminal_equalities": int(len(problem.terminal_indices)),
        "objective": "minimize -mean(nonterminal u_s)",
        "primal": {
            "max_edge_violation": float(np.max(np.maximum(-slack, 0.0), initial=0.0)),
            "max_terminal_abs": float(np.max(np.abs(equality_residual), initial=0.0)),
            "min_value": float(np.min(values, initial=0.0)),
        },
        "dual": {
            "min_edge_multiplier": float(np.min(edge_duals, initial=0.0)),
            "min_lower_bound_multiplier": float(np.min(lower_duals, initial=0.0)),
            "per_family": per_family,
        },
        "complementarity": {
            "edge_linf": float(np.max(np.abs(edge_duals * slack), initial=0.0)),
            "edge_l2": _norm(edge_duals * slack),
            "lower_bound_linf": float(
                np.max(np.abs(lower_duals * values), initial=0.0)
            ),
        },
        "stationarity": {
            "linf": float(np.max(np.abs(stationarity), initial=0.0)),
            "l2": _norm(stationarity),
        },
    }
    return values, edge_duals, diagnostics


def family_constraint_gradient(
    problem: JointFeasibleProblem,
    values: np.ndarray,
    edge_indices: np.ndarray,
    *,
    surrogate: str,
    active_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gradient of one family's mean constraint statistic with fixed terminals."""

    indices = np.asarray(edge_indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("constraint family sample must not be empty")
    sources = problem.sources[indices]
    destinations = problem.destinations[indices]
    successor = np.zeros(len(indices), dtype=np.float64)
    physical = destinations >= 0
    successor[physical] = values[destinations[physical]]
    delta = values[sources] - successor
    distance = np.maximum(delta, 0.0)
    raw_excess = distance - problem.costs[indices].astype(np.float64)

    if surrogate == "squared_hinge":
        positive = raw_excess > 0.0
        coefficient = 2.0 * np.maximum(raw_excess, 0.0) / float(len(indices))
    elif surrogate == "linear_hinge_right_subgradient":
        # The outer hinge uses its right derivative on numerically active
        # constraints.  The inner potential-distance ReLU keeps its standard
        # zero derivative at delta=0, matching the fixed terminal treatment.
        positive = (raw_excess >= -float(active_tolerance)) & (delta > 0.0)
        coefficient = positive.astype(np.float64) / float(len(indices))
    else:
        raise ValueError(f"unknown surrogate: {surrogate}")

    gradient = np.zeros(problem.num_states, dtype=np.float64)
    np.add.at(gradient, sources, coefficient)
    if np.any(physical):
        np.add.at(gradient, destinations[physical], -coefficient[physical])
    gradient = _project_fixed_terminals(gradient, problem)
    return gradient, {
        "sampled_edges": int(len(indices)),
        "strict_positive_excess_edges": int(np.sum(raw_excess > 0.0)),
        "active_subgradient_edges": int(np.sum(positive)),
        "squared_hinge_mean": float(
            np.mean(np.square(np.maximum(raw_excess, 0.0)))
        ),
        "linear_hinge_mean": float(np.mean(np.maximum(raw_excess, 0.0))),
        "max_raw_excess": float(np.max(raw_excess, initial=0.0)),
        "min_raw_excess": float(np.min(raw_excess, initial=0.0)),
    }


def push_gradient(
    problem: JointFeasibleProblem,
    sampled_states: np.ndarray | None = None,
) -> np.ndarray:
    indices = (
        _nonterminal_indices(problem)
        if sampled_states is None
        else np.asarray(sampled_states, dtype=np.int64)
    )
    gradient = np.zeros(problem.num_states, dtype=np.float64)
    np.add.at(gradient, indices, -1.0 / float(len(indices)))
    return _project_fixed_terminals(gradient, problem)


def _nonterminal_indices(problem: JointFeasibleProblem) -> np.ndarray:
    mask = np.ones(problem.num_states, dtype=bool)
    mask[problem.terminal_indices] = False
    return np.flatnonzero(mask)


def full_surrogate_gradients(
    problem: JointFeasibleProblem,
    values: np.ndarray,
    *,
    surrogate: str,
    lambdas: Mapping[str, float],
    active_tolerance: float,
) -> dict[str, Any]:
    push = push_gradient(problem, _nonterminal_indices(problem))
    family_gradients: dict[str, np.ndarray] = {}
    family_audits: dict[str, Any] = {}
    for family_index, family_name in enumerate(FAMILY_NAMES):
        indices = np.flatnonzero(problem.families == family_index)
        family_gradients[family_name], family_audits[family_name] = (
            family_constraint_gradient(
                problem,
                values,
                indices,
                surrogate=surrogate,
                active_tolerance=active_tolerance,
            )
        )
    constraint = sum(
        float(lambdas[name]) * family_gradients[name] for name in FAMILY_NAMES
    )
    total = push + constraint

    free = _nonterminal_indices(problem)
    matrix = np.column_stack([family_gradients[name][free] for name in FAMILY_NAMES])
    if _norm(matrix) <= 1e-30:
        fitted = np.zeros(len(FAMILY_NAMES), dtype=np.float64)
        residual_norm = _norm(push[free])
    else:
        fitted, residual_norm = nnls(matrix, -push[free])
    fitted_constraint = sum(
        float(fitted[index]) * family_gradients[name]
        for index, name in enumerate(FAMILY_NAMES)
    )
    fitted_total = push + fitted_constraint
    return {
        "push": push,
        "family_gradients": family_gradients,
        "family_audits": family_audits,
        "constraint": constraint,
        "total": total,
        "fitted_lambdas": {
            name: float(fitted[index]) for index, name in enumerate(FAMILY_NAMES)
        },
        "fitted_constraint": fitted_constraint,
        "fitted_total": fitted_total,
        "fitted_residual_norm": float(residual_norm),
    }


def _u_margin_pairs(
    problem: JointFeasibleProblem,
    *,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return destination pairs for U-region optimal/nonoptimal margins."""

    reference = np.asarray(problem.reference_values, dtype=np.float64)
    successor = np.zeros(problem.num_edges, dtype=np.float64)
    physical = problem.destinations >= 0
    successor[physical] = reference[problem.destinations[physical]]
    scores = problem.costs.astype(np.float64) + successor
    order = np.argsort(problem.sources, kind="stable")
    ordered_sources = problem.sources[order]
    optimal_destinations: list[int] = []
    nonoptimal_destinations: list[int] = []
    for source in np.flatnonzero(problem.u_trap_mask):
        begin = int(np.searchsorted(ordered_sources, source, side="left"))
        end = int(np.searchsorted(ordered_sources, source, side="right"))
        indices = order[begin:end]
        if len(indices) < 2:
            continue
        optimum = float(np.min(scores[indices]))
        optimal = indices[scores[indices] <= optimum + float(tolerance)]
        nonoptimal = indices[scores[indices] > optimum + float(tolerance)]
        for optimal_edge in optimal:
            optimal_destination = int(problem.destinations[optimal_edge])
            for nonoptimal_edge in nonoptimal:
                nonoptimal_destination = int(problem.destinations[nonoptimal_edge])
                optimal_destinations.append(optimal_destination)
                nonoptimal_destinations.append(nonoptimal_destination)
    return (
        np.asarray(optimal_destinations, dtype=np.int64),
        np.asarray(nonoptimal_destinations, dtype=np.int64),
    )


def _u_margin_derivatives(
    gradient: np.ndarray,
    pairs: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Derivative under gradient descent of nonoptimal-minus-optimal margins."""

    optimal, nonoptimal = pairs
    padded = np.concatenate([np.asarray(gradient, dtype=np.float64), [0.0]])
    optimal_safe = np.where(optimal >= 0, optimal, len(padded) - 1)
    nonoptimal_safe = np.where(nonoptimal >= 0, nonoptimal, len(padded) - 1)
    # u_new = u - eta*g.  The derivative of
    # (score_nonoptimal-score_optimal) is g_optimal-g_nonoptimal.
    return padded[optimal_safe] - padded[nonoptimal_safe]


def _margin_summary(
    problem: JointFeasibleProblem,
    gradient: np.ndarray,
    pairs: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    derivatives = _u_margin_derivatives(
        gradient, pairs if pairs is not None else _u_margin_pairs(problem)
    )
    result = _quantiles(derivatives)
    result["negative_fraction"] = (
        float(np.mean(derivatives < -1e-12)) if len(derivatives) else None
    )
    result["positive_fraction"] = (
        float(np.mean(derivatives > 1e-12)) if len(derivatives) else None
    )
    return result


def _one_step_topology(
    problem: JointFeasibleProblem,
    values: np.ndarray,
    gradient: np.ndarray,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    updated = np.maximum(
        np.asarray(values, dtype=np.float64)
        - float(learning_rate) * np.asarray(gradient, dtype=np.float64),
        0.0,
    )
    updated[problem.terminal_indices] = 0.0
    return lattice_successor_ranking(
        problem, updated, source_mask=problem.u_trap_mask
    )


class _EstimatorAccumulator:
    def __init__(self, size: int):
        self.count = 0
        self.sum = np.zeros(size, dtype=np.float64)
        self.sum_square = np.zeros(size, dtype=np.float64)
        self.cosines: list[float] = []
        self.norms: list[float] = []
        self.u_norms: list[float] = []
        self.non_u_norms: list[float] = []
        self.margin_negative_fractions: list[float] = []

    def add(
        self,
        vector: np.ndarray,
        reference: np.ndarray,
        problem: JointFeasibleProblem,
        margin_pairs: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, float]:
        vector = np.asarray(vector, dtype=np.float64)
        self.count += 1
        self.sum += vector
        self.sum_square += np.square(vector)
        cosine = _cosine(vector, reference)
        norm = _norm(vector)
        u_norm = _norm(vector[problem.u_trap_mask])
        non_u_norm = _norm(vector[~problem.u_trap_mask])
        margins = (
            _u_margin_derivatives(vector, margin_pairs)
            if margin_pairs is not None
            else np.empty(0, dtype=np.float64)
        )
        margin_negative = (
            float(np.mean(margins < -1e-12)) if len(margins) else float("nan")
        )
        self.cosines.append(cosine)
        self.norms.append(norm)
        self.u_norms.append(u_norm)
        self.non_u_norms.append(non_u_norm)
        self.margin_negative_fractions.append(margin_negative)
        return {
            "cosine_to_full": cosine,
            "l2_norm": norm,
            "u_l2_norm": u_norm,
            "non_u_l2_norm": non_u_norm,
            "u_margin_negative_fraction": margin_negative,
        }

    def finish(
        self,
        reference: np.ndarray,
        problem: JointFeasibleProblem,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        mean = self.sum / float(self.count)
        variance = np.maximum(self.sum_square / float(self.count) - np.square(mean), 0.0)
        bias = mean - reference

        def subspace(mask: np.ndarray) -> dict[str, Any]:
            trace_variance = float(np.sum(variance[mask]))
            mean_norm = _norm(mean[mask])
            reference_norm = _norm(reference[mask])
            bias_norm = _norm(bias[mask])
            mean_standard_error_norm = math.sqrt(
                trace_variance / float(self.count)
            )
            return {
                "mean_norm": mean_norm,
                "reference_norm": reference_norm,
                "bias_norm": bias_norm,
                "relative_bias": (
                    bias_norm / reference_norm if reference_norm > 1e-30 else None
                ),
                "variance_trace": trace_variance,
                "monte_carlo_mean_standard_error_norm": mean_standard_error_norm,
                "empirical_bias_to_mc_standard_error": (
                    bias_norm / mean_standard_error_norm
                    if mean_standard_error_norm > 1e-30
                    else None
                ),
                "snr": (
                    mean_norm / math.sqrt(trace_variance)
                    if trace_variance > 1e-30
                    else None
                ),
                "mean_to_full_cosine": _cosine(mean[mask], reference[mask]),
            }

        nonzero = np.abs(reference) > 1e-12
        sign_agreement = (
            float(np.mean(np.sign(mean[nonzero]) == np.sign(reference[nonzero])))
            if np.any(nonzero)
            else None
        )
        summary = {
            "samples": int(self.count),
            "all": subspace(np.ones(problem.num_states, dtype=bool)),
            "u_region": subspace(problem.u_trap_mask),
            "non_u_region": subspace(~problem.u_trap_mask),
            "sign_agreement_on_nonzero_full_coordinates": sign_agreement,
            "sample_cosine_to_full": _quantiles(self.cosines),
            "sample_l2_norm": _quantiles(self.norms),
            "sample_u_l2_norm": _quantiles(self.u_norms),
            "sample_non_u_l2_norm": _quantiles(self.non_u_norms),
            "sample_u_margin_negative_fraction": _quantiles(
                self.margin_negative_fractions
            ),
            "mean_vector": {
                "stats": _vector_stats(mean, problem),
                "u_margin_derivative": _margin_summary(problem, mean),
            },
        }
        return mean, summary


def minibatch_autopsy(
    problem: JointFeasibleProblem,
    values: np.ndarray,
    full: Mapping[str, Mapping[str, Any]],
    *,
    lambdas: Mapping[str, float],
    active_tolerance: float,
    samples: int,
    ordinary_batch_size: int,
    push_batch_size: int,
    seed: int,
    topology_sample_count: int,
    learning_rate: float,
    step_multipliers: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    family_indices = {
        name: np.flatnonzero(problem.families == index)
        for index, name in enumerate(FAMILY_NAMES)
    }
    push_pool = _nonterminal_indices(problem)
    accumulators = {
        surrogate: {
            component: _EstimatorAccumulator(problem.num_states)
            for component in ("push", "constraint", "total")
        }
        for surrogate in SURROGATES
    }
    rows: list[dict[str, Any]] = []
    sampled_topologies: dict[str, dict[str, list[dict[str, Any]]]] = {
        surrogate: {str(float(multiplier)): [] for multiplier in step_multipliers}
        for surrogate in SURROGATES
    }
    margin_pairs = _u_margin_pairs(problem)

    for sample_index in range(int(samples)):
        ordinary_pool = family_indices["ordinary"]
        replace = int(ordinary_batch_size) > len(ordinary_pool)
        ordinary = rng.choice(
            ordinary_pool, size=int(ordinary_batch_size), replace=replace
        )
        sampled_push = rng.choice(
            push_pool, size=int(push_batch_size), replace=True
        )
        sampled_push_gradient = push_gradient(problem, sampled_push)

        for surrogate in SURROGATES:
            sampled_family: dict[str, np.ndarray] = {}
            for name in FAMILY_NAMES:
                indices = ordinary if name == "ordinary" else family_indices[name]
                sampled_family[name], _audit = family_constraint_gradient(
                    problem,
                    values,
                    indices,
                    surrogate=surrogate,
                    active_tolerance=active_tolerance,
                )
            sampled_constraint = sum(
                float(lambdas[name]) * sampled_family[name] for name in FAMILY_NAMES
            )
            components = {
                "push": sampled_push_gradient,
                "constraint": sampled_constraint,
                "total": sampled_push_gradient + sampled_constraint,
            }
            for component, vector in components.items():
                scalar = accumulators[surrogate][component].add(
                    vector,
                    full[surrogate][component],
                    problem,
                    margin_pairs if component == "total" else None,
                )
                rows.append(
                    {
                        "sample": int(sample_index),
                        "surrogate": surrogate,
                        "component": component,
                        **scalar,
                    }
                )
            if sample_index < int(topology_sample_count):
                for multiplier in step_multipliers:
                    topology = _one_step_topology(
                        problem,
                        values,
                        components["total"],
                        learning_rate=float(learning_rate) * float(multiplier),
                    )
                    sampled_topologies[surrogate][str(float(multiplier))].append(
                        topology
                    )

    summaries: dict[str, Any] = {}
    mean_vectors: dict[str, np.ndarray] = {}
    for surrogate in SURROGATES:
        summaries[surrogate] = {}
        for component in ("push", "constraint", "total"):
            mean, summary = accumulators[surrogate][component].finish(
                full[surrogate][component], problem
            )
            summaries[surrogate][component] = summary
            mean_vectors[f"{surrogate}.{component}"] = mean

        topology_summary: dict[str, Any] = {}
        for multiplier in step_multipliers:
            key = str(float(multiplier))
            records = sampled_topologies[surrogate][key]
            topology_summary[key] = {
                "learning_rate": float(learning_rate) * float(multiplier),
                "samples": int(len(records)),
                "top1_accuracy": _quantiles(
                    [record["top1_accuracy"] for record in records]
                ),
                "pairwise_accuracy": _quantiles(
                    [record["pairwise_accuracy"] for record in records]
                ),
                "oracle_regret_mean": _quantiles(
                    [record["oracle_regret"]["mean"] for record in records]
                ),
                "oracle_regret_p95": _quantiles(
                    [record["oracle_regret"]["p95"] for record in records]
                ),
            }
        summaries[surrogate]["sampled_one_step_topology"] = topology_summary
    return summaries, rows, mean_vectors


def _compact_topology(topology: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_states": topology["source_states"],
        "top1_accuracy": topology["top1_accuracy"],
        "pairwise_accuracy": topology["pairwise_accuracy"],
        "oracle_regret": topology["oracle_regret"],
        "oracle_action_gap": topology["oracle_action_gap"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-id", default="u_trap_target")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--position-resolution", type=float, default=0.25)
    parser.add_argument("--heading-bins", type=int, default=24)
    parser.add_argument("--primitive-steps", type=int, default=5)
    parser.add_argument(
        "--primitive-scales", type=float, nargs="+",
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    parser.add_argument("--lp-time-limit-sec", type=float, default=600.0)
    parser.add_argument("--active-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--family-lambdas", type=float, nargs=3, metavar=("ORDINARY", "DIRECT", "TERMINAL"),
        default=[10.0, 10.0, 10.0],
    )
    parser.add_argument(
        "--family-epsilons", type=float, nargs=3, metavar=("ORDINARY", "DIRECT", "TERMINAL"),
        default=[0.25, 0.25, 0.0],
    )
    parser.add_argument("--minibatch-samples", type=int, default=2000)
    parser.add_argument("--ordinary-batch-size", type=int, default=512)
    parser.add_argument(
        "--push-batch-size", type=int, default=0,
        help="0 reproduces ordinary_batch_size + number of direct-goal edges",
    )
    parser.add_argument("--topology-sample-count", type=int, default=200)
    parser.add_argument("--base-learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--step-multipliers", type=float, nargs="+", default=[0.1, 1.0, 10.0]
    )
    return parser


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if int(args.minibatch_samples) <= 0:
        raise ValueError("minibatch-samples must be positive")
    if int(args.ordinary_batch_size) <= 0:
        raise ValueError("ordinary-batch-size must be positive")
    if int(args.topology_sample_count) < 0:
        raise ValueError("topology-sample-count must be non-negative")
    started = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_config(args.scenario_config)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=0, options={"device_id": str(args.device_id)})
    graph_config = HybridAStarConfig(
        position_resolution=float(args.position_resolution),
        heading_bins=int(args.heading_bins),
        primitive_steps=int(args.primitive_steps),
        primitive_scales=tuple(float(value) for value in args.primitive_scales),
    )
    problem = build_joint_feasible_problem(env, scenario, graph_config)
    inherited_reference = np.asarray(problem.reference_values, dtype=np.float64)
    dijkstra = compact_reverse_dijkstra(problem)
    # All topology helpers should use the same exact-cost Dijkstra point as the
    # vector-field probe, not the float32 neural-evaluation copy constructed by
    # build_joint_feasible_problem.
    problem = replace(problem, reference_values=dijkstra)
    lp_values, edge_duals, kkt = solve_exact_lp_kkt(
        problem, time_limit_sec=float(args.lp_time_limit_sec)
    )
    lambdas = {
        name: float(args.family_lambdas[index])
        for index, name in enumerate(FAMILY_NAMES)
    }
    epsilons = {
        name: float(args.family_epsilons[index])
        for index, name in enumerate(FAMILY_NAMES)
    }
    full_raw = {
        surrogate: full_surrogate_gradients(
            problem,
            dijkstra,
            surrogate=surrogate,
            lambdas=lambdas,
            active_tolerance=float(args.active_tolerance),
        )
        for surrogate in SURROGATES
    }
    full_for_sampling = {
        surrogate: {
            component: full_raw[surrogate][component]
            for component in ("push", "constraint", "total")
        }
        for surrogate in SURROGATES
    }
    direct_count = int(np.sum(problem.families == FAMILY_NAMES.index("direct_goal")))
    push_batch_size = int(args.push_batch_size)
    if push_batch_size <= 0:
        push_batch_size = int(args.ordinary_batch_size) + direct_count
    minibatch, sample_rows, mean_vectors = minibatch_autopsy(
        problem,
        dijkstra,
        full_for_sampling,
        lambdas=lambdas,
        active_tolerance=float(args.active_tolerance),
        samples=int(args.minibatch_samples),
        ordinary_batch_size=int(args.ordinary_batch_size),
        push_batch_size=push_batch_size,
        seed=int(args.seed),
        topology_sample_count=min(
            int(args.topology_sample_count), int(args.minibatch_samples)
        ),
        learning_rate=float(args.base_learning_rate),
        step_multipliers=args.step_multipliers,
    )

    baseline_topology = lattice_successor_ranking(
        problem, dijkstra, source_mask=problem.u_trap_mask
    )
    full_payload: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "states": problem.states,
        "dijkstra_values": dijkstra,
        "lp_values": lp_values,
        "edge_sources": problem.sources,
        "edge_destinations": problem.destinations,
        "edge_costs": problem.costs,
        "edge_families": problem.families,
        "edge_duals": edge_duals,
        "u_trap_mask": problem.u_trap_mask,
    }
    for name, vector in mean_vectors.items():
        arrays[f"minibatch_mean.{name}"] = vector
    for surrogate in SURROGATES:
        result = full_raw[surrogate]
        component_payload: dict[str, Any] = {}
        for component in ("push", "constraint", "total", "fitted_constraint", "fitted_total"):
            vector = result[component]
            arrays[f"{surrogate}.{component}"] = vector
            component_payload[component] = {
                "vector": _vector_stats(vector, problem),
                "u_margin_derivative": _margin_summary(problem, vector),
            }
        component_payload["push_constraint_cosine"] = _cosine(
            result["push"], result["constraint"]
        )
        component_payload["configured_lambdas"] = lambdas
        component_payload["fitted_lambdas"] = result["fitted_lambdas"]
        component_payload["fitted_stationarity_residual_norm"] = result[
            "fitted_residual_norm"
        ]
        component_payload["family_unweighted_gradients"] = {
            name: {
                "vector": _vector_stats(result["family_gradients"][name], problem),
                "edge_activity": {
                    **result["family_audits"][name],
                    "epsilon": epsilons[name],
                    "dual_residual": (
                        result["family_audits"][name]["squared_hinge_mean"]
                        - epsilons[name] ** 2
                        if surrogate == "squared_hinge"
                        else result["family_audits"][name]["linear_hinge_mean"]
                        - epsilons[name]
                    ),
                },
            }
            for name in FAMILY_NAMES
        }
        one_step: dict[str, Any] = {}
        for multiplier in args.step_multipliers:
            key = str(float(multiplier))
            one_step[key] = {
                "learning_rate": float(args.base_learning_rate) * float(multiplier),
                "push": _compact_topology(
                    _one_step_topology(
                        problem, dijkstra, result["push"],
                        learning_rate=float(args.base_learning_rate) * float(multiplier),
                    )
                ),
                "constraint": _compact_topology(
                    _one_step_topology(
                        problem, dijkstra, result["constraint"],
                        learning_rate=float(args.base_learning_rate) * float(multiplier),
                    )
                ),
                "total": _compact_topology(
                    _one_step_topology(
                        problem, dijkstra, result["total"],
                        learning_rate=float(args.base_learning_rate) * float(multiplier),
                    )
                ),
                "fitted_total": _compact_topology(
                    _one_step_topology(
                        problem, dijkstra, result["fitted_total"],
                        learning_rate=float(args.base_learning_rate) * float(multiplier),
                    )
                ),
                "minibatch_mean_total": _compact_topology(
                    _one_step_topology(
                        problem,
                        dijkstra,
                        mean_vectors[f"{surrogate}.total"],
                        learning_rate=float(args.base_learning_rate) * float(multiplier),
                    )
                ),
            }
        component_payload["one_step_u_topology"] = one_step
        full_payload[surrogate] = component_payload

    lp_difference = lp_values - dijkstra
    payload = {
        "experiment": "static_vector_field_kkt_autopsy",
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "graph_config": asdict(graph_config),
        "graph": problem.graph_stats,
        "design": {
            "measurement_point": "reverse-Dijkstra potential; never used as a training target",
            "optimization_performed": False,
            "surrogates": {
                "squared_hinge": "current mean(relu(d-c)^2) primal gradient",
                "linear_hinge_right_subgradient": (
                    "mean relu(d-c), choosing the outer hinge right derivative "
                    "for residual >= -active_tolerance"
                ),
            },
            "gradient_sign": (
                "reported vectors are gradients of the minimized loss; a primal "
                "step is u_new = projection(u - learning_rate * gradient)"
            ),
            "minibatch": {
                "samples": int(args.minibatch_samples),
                "ordinary_edges": "independent uniform samples without replacement when possible",
                "ordinary_batch_size": int(args.ordinary_batch_size),
                "direct_and_terminal_edges": "all edges in every sample",
                "push_states": "independent uniform samples with replacement",
                "push_batch_size": int(push_batch_size),
                "seed": int(args.seed),
            },
            "configured_family_lambdas": lambdas,
            "family_epsilons": epsilons,
            "active_tolerance": float(args.active_tolerance),
            "step_learning_rates": [
                float(args.base_learning_rate) * float(value)
                for value in args.step_multipliers
            ],
        },
        "exact_lp_kkt": kkt,
        "lp_vs_dijkstra": {
            "max_abs_error": float(np.max(np.abs(lp_difference), initial=0.0)),
            "mae": float(np.mean(np.abs(lp_difference))),
            "rmse": float(np.sqrt(np.mean(np.square(lp_difference)))),
        },
        "inherited_float32_reference_vs_exact_cost_dijkstra": {
            "max_abs_error": float(
                np.max(np.abs(inherited_reference - dijkstra), initial=0.0)
            ),
            "mae": float(np.mean(np.abs(inherited_reference - dijkstra))),
        },
        "baseline_u_topology": _compact_topology(baseline_topology),
        "full_graph_vector_field": full_payload,
        "minibatch_estimator": minibatch,
        "artifacts": {
            "arrays": "static_vector_field_arrays.npz",
            "minibatch_samples": "minibatch_gradient_samples.csv",
        },
        "runtime_sec": float(perf_counter() - started),
    }
    metrics_path = output_dir / "static_vector_field_kkt_autopsy.json"
    metrics_path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "static_vector_field_arrays.npz", **arrays)
    _write_csv(output_dir / "minibatch_gradient_samples.csv", sample_rows)
    print(json.dumps(_jsonable({
        "metrics": str(metrics_path.resolve()),
        "lp_kkt": kkt,
        "lp_vs_dijkstra": payload["lp_vs_dijkstra"],
        "baseline_u_topology": payload["baseline_u_topology"],
        "runtime_sec": payload["runtime_sec"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
