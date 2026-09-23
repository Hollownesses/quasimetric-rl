#!/usr/bin/env python3
"""Validate the exact per-edge KKT reaction stored by the static autopsy.

The static autopsy stores LP multipliers in sum form.  Neural training uses an
edge expectation, so this audit also reports the equivalent dual-density scale
``lambda_density(e) = lambda_lp(e) / q(e)`` for uniform edge sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def exact_dual_reaction_audit(
    *,
    values: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    costs: np.ndarray,
    edge_duals: np.ndarray,
    u_trap_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.int64)
    destinations = np.asarray(destinations, dtype=np.int64)
    costs = np.asarray(costs, dtype=np.float64)
    edge_duals = np.asarray(edge_duals, dtype=np.float64)
    if not (
        len(sources) == len(destinations) == len(costs) == len(edge_duals)
    ):
        raise ValueError("edge arrays must have equal lengths")

    terminal_edges = (destinations < 0) & (np.abs(costs) <= 1e-12)
    terminal_indices = np.unique(sources[terminal_edges])
    free = np.ones(len(values), dtype=bool)
    free[terminal_indices] = False

    successor_values = np.zeros(len(sources), dtype=np.float64)
    physical = destinations >= 0
    successor_values[physical] = values[destinations[physical]]
    delta = values[sources] - successor_values
    distances = np.maximum(delta, 0.0)
    residual = distances - costs

    # Positive-dual nonterminal constraints at the shortest-path solution have
    # delta>0.  The delta=0 terminal coordinates are fixed and projected out.
    coefficients = edge_duals * (delta > 0.0)
    constraint_gradient = np.zeros(len(values), dtype=np.float64)
    np.add.at(constraint_gradient, sources, coefficients)
    np.add.at(
        constraint_gradient,
        destinations[physical],
        -coefficients[physical],
    )
    constraint_gradient[~free] = 0.0

    push_gradient = np.zeros(len(values), dtype=np.float64)
    push_gradient[free] = -1.0 / float(np.sum(free))
    stationarity = push_gradient + constraint_gradient

    positive = edge_duals > 1e-12
    uniform_dual_density = edge_duals * float(len(edge_duals))
    result: dict[str, Any] = {
        "states": int(len(values)),
        "free_states": int(np.sum(free)),
        "terminal_states": int(len(terminal_indices)),
        "edges": int(len(sources)),
        "primal": {
            "max_constraint_violation": float(
                np.max(np.maximum(residual, 0.0), initial=0.0)
            ),
        },
        "complementarity": {
            "linf": float(
                np.max(np.abs(edge_duals * residual), initial=0.0)
            ),
            "l2": _norm(edge_duals * residual),
        },
        "stationarity": {
            "push_l2": _norm(push_gradient),
            "exact_edge_reaction_l2": _norm(constraint_gradient),
            "residual_l2": _norm(stationarity),
            "residual_linf": float(
                np.max(np.abs(stationarity), initial=0.0)
            ),
        },
        "dual_scaling": {
            "lp_sum_form_positive": _quantiles(edge_duals[positive]),
            "uniform_expectation_density_positive": _quantiles(
                uniform_dual_density[positive]
            ),
            "identity": "lambda_density(e)=num_edges*lambda_lp(e)",
        },
    }
    if u_trap_mask is not None:
        u_trap_mask = np.asarray(u_trap_mask, dtype=bool)
        if len(u_trap_mask) != len(values):
            raise ValueError("u_trap_mask must match the value vector")
        result["stationarity"]["u_region_residual_l2"] = _norm(
            stationarity[u_trap_mask]
        )
    return result


def audit_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path) as arrays:
        required = (
            "dijkstra_values",
            "edge_sources",
            "edge_destinations",
            "edge_costs",
            "edge_duals",
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"autopsy arrays are missing: {', '.join(missing)}")
        return exact_dual_reaction_audit(
            values=arrays["dijkstra_values"],
            sources=arrays["edge_sources"],
            destinations=arrays["edge_destinations"],
            costs=arrays["edge_costs"],
            edge_duals=arrays["edge_duals"],
            u_trap_mask=(arrays["u_trap_mask"] if "u_trap_mask" in arrays else None),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = audit_npz(args.arrays)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
