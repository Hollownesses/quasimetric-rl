"""Shared command-line and reporting helpers for communication-inspection experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from minimal_qrl.eval.comm_inspection_execution_eval import make_comm_inspection_env


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("industrial inspection environment")
    group.add_argument("--bounds", type=float, nargs=4, default=[0.0, 0.0, 10.0, 10.0])
    group.add_argument("--omega-max", type=float, default=3.0)
    group.add_argument("--v", type=float, default=1.0)
    group.add_argument("--dt", type=float, default=0.1)
    group.add_argument("--max-episode-steps", type=int, default=180)
    group.add_argument(
        "--obstacle-config",
        choices=["none", "simple", "medium", "hard"],
        default="medium",
    )
    group.add_argument("--obstacles", type=float, nargs="*", default=None)
    group.add_argument("--device-catalog", required=True)
    group.add_argument("--comm-alpha", type=float, default=2.0)
    group.add_argument("--comm-bias", type=float, default=5.0)
    group.add_argument("--comm-occlusion-penalty", type=float, default=6.0)
    group.add_argument("--comm-threshold", type=float, default=0.5)
    group.add_argument("--require-ground-station-los", action="store_true")
    group.add_argument("--collision-cost", type=float, default=10.0)
    group.add_argument("--out-of-bounds-cost", type=float, default=10.0)
    group.add_argument("--communication-break-cost", type=float, default=1.0)
    group.add_argument("--observation-violation-cost-weight", type=float, default=1.0)
    group.add_argument("--communication-violation-cost-weight", type=float, default=0.5)
    group.add_argument("--observation-failure-cost", type=float, default=0.25)
    group.add_argument("--taskscore-beta-obs", type=float, default=1.0)
    group.add_argument("--taskscore-beta-comm", type=float, default=1.0)
    group.add_argument("--taskscore-beta-feas", type=float, default=0.5)
    group.add_argument("--taskscore-margin-clip", type=float, default=2.0)


def add_mppi_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("MPPI")
    group.add_argument("--mppi-horizon", type=int, default=10)
    group.add_argument("--mppi-num-samples", type=int, default=128)
    group.add_argument("--mppi-noise-sigma", type=float, default=0.8)
    group.add_argument("--mppi-temperature", type=float, default=1.0)
    group.add_argument("--mppi-terminal-weight", type=float, default=1.0)
    group.add_argument("--mppi-terminal-samples", type=int, default=128)


def add_astar_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Hybrid A*")
    group.add_argument("--astar-position-resolution", type=float, default=0.25)
    group.add_argument("--astar-heading-bins", type=int, default=24)
    group.add_argument("--astar-primitive-steps", type=int, default=5)
    group.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    group.add_argument("--astar-max-expansions", type=int, default=50_000)
    group.add_argument("--astar-timeout-sec", type=float, default=5.0)
    group.add_argument("--astar-terminal-samples", type=int, default=128)


def make_env(args):
    return make_comm_inspection_env(args)


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("expected positive comma-separated integers")
    return values


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int,
    samples: int = 2000,
) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(max(1, int(samples)), array.size), replace=True)
    means = np.mean(draws, axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metric_summary(
    values: Iterable[float],
    *,
    seed: int,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "bootstrap_95_ci": [None, None]}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "bootstrap_95_ci": bootstrap_mean_ci(
            array,
            seed=seed,
            samples=bootstrap_samples,
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

