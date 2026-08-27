"""Diagnose QRL optimization on the finite U-trap graph with tabular potentials.

This experiment deliberately removes the encoder/projector/IQE coupling.  Every
goal-reachable lattice state owns one non-negative scalar ``u_s`` and

    d_u(s, s') = relu(u_s - u_s'),    d_u(s, G) = u_s.

The four pre-registered cells cross full-edge versus current-style minibatch
constraint sampling with exact (epsilon=0) versus practical relaxed epsilons.
Shortest-path values are never passed to the trainer; they are constructed only
by the evaluator after optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from minimal_qrl.baselines import HybridAStarConfig
from minimal_qrl.dataset import _full_graph_digest, _goal_reachable_lattice_mask
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.exact_discrete_value_lp import (
    DIRECT_GOAL,
    build_discrete_value_graph,
    reverse_dijkstra,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)


FAMILY_NAMES = ("ordinary", "direct_goal", "terminal_goal")
CELL_SPECS: Mapping[str, tuple[str, str]] = {
    "full_zero": ("full", "zero"),
    "minibatch_zero": ("minibatch", "zero"),
    "full_practical": ("full", "practical"),
    "minibatch_practical": ("minibatch", "practical"),
}


@dataclass(frozen=True)
class TabularPotentialProblem:
    """Compact goal-reachable training graph; contains no reference values."""

    global_state_indices: np.ndarray
    states: np.ndarray
    sources: np.ndarray
    destinations: np.ndarray
    costs: np.ndarray
    families: np.ndarray
    terminal_indices: np.ndarray
    push_indices: np.ndarray
    graph_stats: Mapping[str, Any]

    @property
    def num_states(self) -> int:
        return int(len(self.global_state_indices))

    @property
    def num_edges(self) -> int:
        return int(len(self.sources))


@dataclass(frozen=True)
class TabularTrainConfig:
    sampling_mode: str
    epsilon_mode: str
    total_steps: int
    ordinary_batch_size: int = 512
    primal_lr: float = 0.1
    dual_lr: float = 5e-3
    init_lagrange_multiplier: float = 0.01
    practical_ordinary_epsilon: float = 0.25
    practical_direct_epsilon: float = 0.25
    practical_terminal_epsilon: float = 0.0
    init_scale: float = 0.0
    eval_interval: int = 500

    def epsilons(self) -> dict[str, float]:
        if self.epsilon_mode == "zero":
            return {name: 0.0 for name in FAMILY_NAMES}
        if self.epsilon_mode != "practical":
            raise ValueError(f"unknown epsilon mode: {self.epsilon_mode}")
        return {
            "ordinary": float(self.practical_ordinary_epsilon),
            "direct_goal": float(self.practical_direct_epsilon),
            "terminal_goal": float(self.practical_terminal_epsilon),
        }


@dataclass
class TabularTrainResult:
    values: np.ndarray
    history: list[dict[str, Any]]
    elapsed_sec: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_tabular_problem(
    env: CommInspectionDubinsUAV2D,
    config: HybridAStarConfig,
) -> tuple[TabularPotentialProblem, Any]:
    """Build the same reachable LP graph plus explicit terminal-to-G edges.

    Reachability is topology-only.  The returned problem contains no Dijkstra or
    Oracle labels.  The raw graph is returned separately for post-training
    evaluation.
    """

    graph = build_discrete_value_graph(env, config)
    reachable = _goal_reachable_lattice_mask(graph)
    direct = graph.destinations == DIRECT_GOAL
    edge_keep = reachable[graph.sources] & (
        direct
        | (
            (graph.destinations >= 0)
            & reachable[np.maximum(graph.destinations, 0)]
        )
    )

    global_state_indices = np.flatnonzero(reachable).astype(np.int64)
    global_to_compact = np.full(len(graph.states), -1, dtype=np.int64)
    global_to_compact[global_state_indices] = np.arange(
        len(global_state_indices), dtype=np.int64
    )
    kept_sources_global = graph.sources[edge_keep].astype(np.int64, copy=False)
    kept_destinations_global = graph.destinations[edge_keep].astype(
        np.int64, copy=False
    )
    kept_costs = graph.costs[edge_keep].astype(np.float32, copy=False)
    sources = global_to_compact[kept_sources_global]
    destinations = kept_destinations_global.copy()
    physical_destination = destinations >= 0
    destinations[physical_destination] = global_to_compact[
        destinations[physical_destination]
    ]

    terminal_global = np.flatnonzero(graph.terminal & reachable).astype(np.int64)
    terminal_indices = global_to_compact[terminal_global]
    original_edge_count = len(sources)
    sources = np.concatenate([sources, terminal_indices]).astype(np.int64)
    destinations = np.concatenate(
        [destinations, np.full(len(terminal_indices), DIRECT_GOAL, dtype=np.int64)]
    )
    costs = np.concatenate(
        [kept_costs, np.zeros(len(terminal_indices), dtype=np.float32)]
    )
    families = np.empty(len(sources), dtype=np.int8)
    families[:original_edge_count] = np.where(
        kept_destinations_global >= 0, 0, 1
    ).astype(np.int8)
    families[original_edge_count:] = 2

    terminal_mask = np.zeros(len(global_state_indices), dtype=bool)
    terminal_mask[terminal_indices] = True
    push_indices = np.flatnonzero(~terminal_mask).astype(np.int64)
    family_counts = {
        name: int(np.sum(families == index))
        for index, name in enumerate(FAMILY_NAMES)
    }
    empty_families = [
        name for name, count in family_counts.items() if int(count) == 0
    ]
    if empty_families:
        raise ValueError(
            "tabular 2x2 requires all three constraint families; empty at "
            f"this lattice resolution: {empty_families}"
        )
    lp_digest = _full_graph_digest(
        sources=kept_sources_global,
        destinations=kept_destinations_global,
        costs=kept_costs,
        terminal=graph.terminal,
    )
    training_digest = _full_graph_digest(
        sources=np.concatenate([kept_sources_global, terminal_global]),
        destinations=np.concatenate(
            [
                kept_destinations_global,
                np.full(len(terminal_global), DIRECT_GOAL, dtype=np.int64),
            ]
        ),
        costs=costs,
        terminal=graph.terminal,
    )
    problem = TabularPotentialProblem(
        global_state_indices=global_state_indices,
        states=graph.states[global_state_indices].astype(np.float32, copy=False),
        sources=sources,
        destinations=destinations,
        costs=costs,
        families=families,
        terminal_indices=terminal_indices,
        push_indices=push_indices,
        graph_stats={
            "grid_states": int(len(graph.states)),
            "valid_states": int(np.sum(graph.valid)),
            "goal_reachable_states": int(len(global_state_indices)),
            "goal_reachable_nonterminal_states": int(len(push_indices)),
            "training_transitions": int(len(sources)),
            "family_counts": family_counts,
            "lp_constraint_graph_digest": lp_digest,
            "training_graph_digest": training_digest,
            "state_index_digest": _sha256_array(global_state_indices),
            "topology_only_reachability": True,
            "terminal_condition": "u_t=0 for every physical terminal state",
        },
    )
    return problem, graph


def potential_distances(
    values: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``relu(u_s-u_s')`` with destination -1 denoting G=0."""

    destination_values = torch.zeros(
        destinations.shape, device=values.device, dtype=values.dtype
    )
    physical = destinations >= 0
    if bool(physical.any()):
        destination_values[physical] = values[destinations[physical]]
    return (values[sources] - destination_values).relu()


class _OrdinaryEdgeSampler:
    """Current-style shuffled sweeps over ordinary edges with final padding."""

    def __init__(self, indices: torch.Tensor, batch_size: int, seed: int):
        self.indices = indices.detach().cpu()
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.permutation = self.indices[:0]
        self.cursor = 0

    def next(self, device: torch.device) -> torch.Tensor:
        if self.cursor >= len(self.permutation):
            self.permutation = self.indices[
                torch.randperm(len(self.indices), generator=self.generator)
            ]
            self.cursor = 0
        result = self.permutation[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        if len(result) < self.batch_size:
            padding = self.indices[
                torch.randint(
                    len(self.indices),
                    (self.batch_size - len(result),),
                    generator=self.generator,
                )
            ]
            result = torch.cat([result, padding])
        return result.to(device=device)


def _inverse_softplus(value: float) -> float:
    return float(math.log(math.expm1(float(value))))


def _constraint_terms(
    values: torch.Tensor,
    *,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    costs: torch.Tensor,
    family_indices: Mapping[str, torch.Tensor],
    epsilons: Mapping[str, float],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    violations: dict[str, torch.Tensor] = {}
    excesses: dict[str, torch.Tensor] = {}
    for name in FAMILY_NAMES:
        indices = family_indices[name]
        distance = potential_distances(
            values, sources[indices], destinations[indices]
        )
        excess = (distance - costs[indices]).relu()
        excesses[name] = excess
        violations[name] = excess.square().mean() - float(epsilons[name]) ** 2
    return violations, excesses


def _regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    keep = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[keep]
    target = target[keep]
    error = prediction - target
    if len(error) >= 2 and np.std(prediction) > 0 and np.std(target) > 0:
        pearson = float(pearsonr(prediction, target).statistic)
        spearman = float(spearmanr(prediction, target).statistic)
    else:
        pearson = float("nan")
        spearman = float("nan")
    target_range = float(np.max(target) - np.min(target)) if len(target) else 0.0
    return {
        "count": int(len(error)),
        "mae": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(error))))
        if len(error)
        else float("nan"),
        "nrmse_range": (
            float(np.sqrt(np.mean(np.square(error))) / target_range)
            if len(error) and target_range > 0
            else float("nan")
        ),
        "max_abs_error": float(np.max(np.abs(error), initial=0.0)),
        "mean_prediction": float(np.mean(prediction)) if len(error) else float("nan"),
        "mean_target": float(np.mean(target)) if len(error) else float("nan"),
        "pearson": pearson,
        "spearman": spearman,
    }


def _family_audit(
    problem: TabularPotentialProblem,
    values: np.ndarray,
    epsilons: Mapping[str, float],
) -> dict[str, dict[str, float | int | bool]]:
    values = np.asarray(values, dtype=np.float64)
    destination_values = np.zeros(problem.num_edges, dtype=np.float64)
    physical = problem.destinations >= 0
    destination_values[physical] = values[problem.destinations[physical]]
    distance = np.maximum(values[problem.sources] - destination_values, 0.0)
    excess = np.maximum(distance - problem.costs.astype(np.float64), 0.0)
    result: dict[str, dict[str, float | int | bool]] = {}
    for family_index, name in enumerate(FAMILY_NAMES):
        selected = excess[problem.families == family_index]
        epsilon = float(epsilons[name])
        squared_mean = float(np.mean(np.square(selected)))
        result[name] = {
            "count": int(len(selected)),
            "violation_count": int(np.sum(selected > 0.0)),
            "violation_fraction": float(np.mean(selected > 0.0)),
            "epsilon_violation_count": int(np.sum(selected > epsilon)),
            "epsilon_violation_fraction": float(np.mean(selected > epsilon)),
            "mean_excess": float(np.mean(selected)),
            "rms_excess": float(np.sqrt(squared_mean)),
            "max_excess": float(np.max(selected, initial=0.0)),
            "p99_excess": float(np.quantile(selected, 0.99)),
            "p999_excess": float(np.quantile(selected, 0.999)),
            "squared_excess_mean": squared_mean,
            "dual_residual": squared_mean - epsilon**2,
            "epsilon": epsilon,
            "expectation_constraint_satisfied": bool(
                squared_mean <= epsilon**2 + 1e-8
            ),
        }
    return result


def evaluate_values(
    problem: TabularPotentialProblem,
    values: np.ndarray,
    reference_values: np.ndarray,
    epsilons: Mapping[str, float],
) -> dict[str, Any]:
    """Post-training evaluation; this is the only API that receives Dijkstra."""

    values = np.asarray(values, dtype=np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    terminal_abs_max = float(
        np.max(np.abs(values[problem.terminal_indices]), initial=0.0)
    )
    return {
        "dijkstra_regression": _regression_metrics(values, reference_values),
        "constraint_families": _family_audit(problem, values, epsilons),
        "linear_global_push": {
            "learned_mean": float(np.mean(values[problem.push_indices])),
            "reference_mean": float(np.mean(reference_values[problem.push_indices])),
            "objective_gap": float(
                np.mean(reference_values[problem.push_indices])
                - np.mean(values[problem.push_indices])
            ),
        },
        "nonnegative_min": float(np.min(values, initial=0.0)),
        "terminal_max_abs": terminal_abs_max,
    }


def _flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flattened.update(_flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[name] = value
    return flattened


def train_tabular_potential(
    problem: TabularPotentialProblem,
    config: TabularTrainConfig,
    *,
    seed: int,
    device: torch.device,
    evaluation_callback=None,
) -> TabularTrainResult:
    """Train without access to Oracle/Dijkstra labels."""

    if config.sampling_mode not in ("full", "minibatch"):
        raise ValueError(f"unknown sampling mode: {config.sampling_mode}")
    if int(config.total_steps) < 0:
        raise ValueError("total_steps must be non-negative")
    if int(config.ordinary_batch_size) <= 0:
        raise ValueError("ordinary_batch_size must be positive")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    sources = torch.as_tensor(problem.sources, device=device, dtype=torch.long)
    destinations = torch.as_tensor(
        problem.destinations, device=device, dtype=torch.long
    )
    costs = torch.as_tensor(problem.costs, device=device, dtype=torch.float32)
    families = torch.as_tensor(problem.families, device=device, dtype=torch.long)
    terminal_indices = torch.as_tensor(
        problem.terminal_indices, device=device, dtype=torch.long
    )
    push_indices = torch.as_tensor(
        problem.push_indices, device=device, dtype=torch.long
    )
    full_family_indices = {
        name: torch.nonzero(families == index, as_tuple=False).flatten()
        for index, name in enumerate(FAMILY_NAMES)
    }
    ordinary_sampler = _OrdinaryEdgeSampler(
        full_family_indices["ordinary"],
        int(config.ordinary_batch_size),
        seed=int(seed) + 17_171,
    )
    push_generator = torch.Generator(device="cpu")
    push_generator.manual_seed(int(seed) + 91_981)

    rng = np.random.default_rng(int(seed) + 104_729)
    initial = rng.uniform(
        0.0, max(0.0, float(config.init_scale)), size=problem.num_states
    ).astype(np.float32)
    initial[problem.terminal_indices] = 0.0
    values = torch.nn.Parameter(torch.as_tensor(initial, device=device))
    raw_lambdas = torch.nn.Parameter(
        torch.full(
            (len(FAMILY_NAMES),),
            _inverse_softplus(float(config.init_lagrange_multiplier)),
            device=device,
            dtype=torch.float32,
        )
    )
    primal_optimizer = torch.optim.AdamW(
        [values], lr=float(config.primal_lr), weight_decay=0.0
    )
    dual_optimizer = torch.optim.AdamW(
        [raw_lambdas], lr=float(config.dual_lr), weight_decay=0.0
    )
    epsilons = config.epsilons()
    history: list[dict[str, Any]] = []

    def record(step: int) -> None:
        row: dict[str, Any] = {
            "step": int(step),
            "push_mean": float(values[push_indices].mean().detach().cpu()),
            "value_min": float(values.min().detach().cpu()),
            "value_max": float(values.max().detach().cpu()),
        }
        lambdas = F.softplus(raw_lambdas).detach().cpu().numpy()
        for index, name in enumerate(FAMILY_NAMES):
            row[f"lambda.{name}"] = float(lambdas[index])
        if evaluation_callback is not None:
            evaluated = evaluation_callback(values.detach().cpu().numpy())
            row.update(_flatten_metrics(evaluated))
        history.append(row)

    record(0)
    started = perf_counter()
    for step in range(1, int(config.total_steps) + 1):
        if config.sampling_mode == "full":
            family_indices = full_family_indices
            selected_push = push_indices
        else:
            family_indices = {
                "ordinary": ordinary_sampler.next(device),
                "direct_goal": full_family_indices["direct_goal"],
                "terminal_goal": full_family_indices["terminal_goal"],
            }
            push_count = int(config.ordinary_batch_size) + len(
                full_family_indices["direct_goal"]
            )
            sampled_positions = torch.randint(
                len(problem.push_indices),
                (push_count,),
                generator=push_generator,
            )
            selected_push = push_indices[sampled_positions.to(device=device)]

        primal_optimizer.zero_grad(set_to_none=True)
        dual_optimizer.zero_grad(set_to_none=True)
        violations, _excesses = _constraint_terms(
            values,
            sources=sources,
            destinations=destinations,
            costs=costs,
            family_indices=family_indices,
            epsilons=epsilons,
        )
        lambdas = F.softplus(raw_lambdas)
        loss = -values[selected_push].mean()
        for index, name in enumerate(FAMILY_NAMES):
            loss = loss + lambdas[index] * violations[name]
        loss.backward()
        if raw_lambdas.grad is not None:
            raw_lambdas.grad.mul_(-1.0)
        primal_optimizer.step()
        dual_optimizer.step()
        with torch.no_grad():
            values.clamp_(min=0.0)
            values[terminal_indices] = 0.0

        should_record = step == int(config.total_steps) or (
            int(config.eval_interval) > 0 and step % int(config.eval_interval) == 0
        )
        if should_record:
            record(step)

    elapsed = perf_counter() - started
    return TabularTrainResult(
        values=values.detach().cpu().numpy().astype(np.float64),
        history=history,
        elapsed_sec=float(elapsed),
    )


def reference_values_for_problem(
    problem: TabularPotentialProblem,
    raw_graph: Any,
) -> np.ndarray:
    """Construct Dijkstra values strictly for post-training evaluation."""

    global_values = reverse_dijkstra(raw_graph)
    reference = global_values[problem.global_state_indices].astype(np.float64)
    if not np.all(np.isfinite(reference)):
        raise RuntimeError("goal-reachable problem contains non-finite Dijkstra values")
    return reference


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "mae": ("final_metrics", "dijkstra_regression", "mae"),
        "rmse": ("final_metrics", "dijkstra_regression", "rmse"),
        "nrmse_range": ("final_metrics", "dijkstra_regression", "nrmse_range"),
        "max_abs_error": (
            "final_metrics",
            "dijkstra_regression",
            "max_abs_error",
        ),
        "pearson": ("final_metrics", "dijkstra_regression", "pearson"),
        "spearman": ("final_metrics", "dijkstra_regression", "spearman"),
        "push_objective_gap": (
            "final_metrics",
            "linear_global_push",
            "objective_gap",
        ),
        "ordinary_max_excess": (
            "final_metrics",
            "constraint_families",
            "ordinary",
            "max_excess",
        ),
        "direct_goal_max_excess": (
            "final_metrics",
            "constraint_families",
            "direct_goal",
            "max_excess",
        ),
    }
    summary: dict[str, Any] = {"num_runs": int(len(runs))}
    for name, path in metric_paths.items():
        values = []
        for run in runs:
            value: Any = run
            for key in path:
                value = value[key]
            values.append(float(value))
        array = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-id", default="u_trap_target")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--cells", nargs="+", choices=tuple(CELL_SPECS), default=list(CELL_SPECS)
    )
    parser.add_argument("--position-resolution", type=float, default=0.25)
    parser.add_argument("--heading-bins", type=int, default=24)
    parser.add_argument("--primitive-steps", type=int, default=5)
    parser.add_argument(
        "--primitive-scales",
        type=float,
        nargs="+",
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    parser.add_argument("--full-steps", type=int, default=20_000)
    parser.add_argument("--minibatch-steps", type=int, default=120_000)
    parser.add_argument("--ordinary-batch-size", type=int, default=512)
    parser.add_argument("--primal-lr", type=float, default=0.1)
    parser.add_argument("--dual-lr", type=float, default=5e-3)
    parser.add_argument("--init-lagrange-multiplier", type=float, default=0.01)
    parser.add_argument("--ordinary-epsilon", type=float, default=0.25)
    parser.add_argument("--direct-goal-epsilon", type=float, default=0.25)
    parser.add_argument("--terminal-goal-epsilon", type=float, default=0.0)
    parser.add_argument("--init-scale", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def main() -> None:
    args = build_parser().parse_args()
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
    problem, raw_graph = build_tabular_problem(env, graph_config)
    # This is created after the training problem and is passed only to callbacks
    # and final evaluators, never to train_tabular_potential's optimization path.
    reference_values = reference_values_for_problem(problem, raw_graph)
    reference_audit = evaluate_values(
        problem,
        reference_values,
        reference_values,
        {name: 0.0 for name in FAMILY_NAMES},
    )
    device = _resolve_device(str(args.device))

    all_runs: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    for cell_name in args.cells:
        sampling_mode, epsilon_mode = CELL_SPECS[cell_name]
        total_steps = (
            int(args.full_steps)
            if sampling_mode == "full"
            else int(args.minibatch_steps)
        )
        config = TabularTrainConfig(
            sampling_mode=sampling_mode,
            epsilon_mode=epsilon_mode,
            total_steps=total_steps,
            ordinary_batch_size=int(args.ordinary_batch_size),
            primal_lr=float(args.primal_lr),
            dual_lr=float(args.dual_lr),
            init_lagrange_multiplier=float(args.init_lagrange_multiplier),
            practical_ordinary_epsilon=float(args.ordinary_epsilon),
            practical_direct_epsilon=float(args.direct_goal_epsilon),
            practical_terminal_epsilon=float(args.terminal_goal_epsilon),
            init_scale=float(args.init_scale),
            eval_interval=int(args.eval_interval),
        )
        epsilons = config.epsilons()
        for seed in args.seeds:
            run_dir = output_dir / cell_name / f"seed_{int(seed)}"
            run_dir.mkdir(parents=True, exist_ok=True)
            callback = lambda current, p=problem, r=reference_values, e=epsilons: evaluate_values(  # noqa: E731
                p, current, r, e
            )
            trained = train_tabular_potential(
                problem,
                config,
                seed=int(seed),
                device=device,
                evaluation_callback=callback,
            )
            final_metrics = evaluate_values(
                problem, trained.values, reference_values, epsilons
            )
            np.savez_compressed(
                run_dir / "tabular_values.npz",
                global_state_indices=problem.global_state_indices,
                states=problem.states,
                learned_values=trained.values,
                dijkstra_values=reference_values,
            )
            run_payload = {
                "cell": cell_name,
                "seed": int(seed),
                "training_config": asdict(config),
                "training_labels": {
                    "oracle_values_used": False,
                    "dijkstra_values_used": False,
                    "hybrid_astar_trajectories_used": False,
                },
                "elapsed_sec": float(trained.elapsed_sec),
                "final_lagrange_multipliers": {
                    name: float(trained.history[-1][f"lambda.{name}"])
                    for name in FAMILY_NAMES
                },
                "final_metrics": final_metrics,
                "values_path": str((run_dir / "tabular_values.npz").resolve()),
            }
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(_jsonable(run_payload), handle, indent=2)
            all_runs.append(run_payload)
            for row in trained.history:
                all_history.append(
                    {"cell": cell_name, "seed": int(seed), **row}
                )
            print(
                f"{cell_name} seed={seed}: "
                f"RMSE={final_metrics['dijkstra_regression']['rmse']:.6f}, "
                f"Pearson={final_metrics['dijkstra_regression']['pearson']:.6f}, "
                f"ordinary max={final_metrics['constraint_families']['ordinary']['max_excess']:.6f}"
            )

    per_cell = {
        cell: _aggregate_runs([run for run in all_runs if run["cell"] == cell])
        for cell in args.cells
    }
    payload = {
        "experiment": "tabular_potential_qrl_2x2",
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "device": str(device),
        "graph_config": asdict(graph_config),
        "graph": problem.graph_stats,
        "parameterization": {
            "definition": "d_u(s,s_prime)=relu(u_s-u_s_prime), d_u(s,G)=u_s",
            "nonnegative_projection": True,
            "goal_value": 0.0,
            "terminal_values_fixed_to_zero": True,
            "triangle_inequality_guaranteed": True,
        },
        "causal_design": {
            "cells": {name: CELL_SPECS[name] for name in args.cells},
            "reference_values_created_for_evaluation_only": True,
            "training_receives_reference_values": False,
            "full_mode": "all edges and all nonterminal push states per update",
            "minibatch_mode": (
                "shuffled ordinary sample plus every direct/terminal edge; "
                "independent uniform push sources"
            ),
        },
        "reference_dijkstra_audit": reference_audit,
        "runs": all_runs,
        "summary_by_cell": per_cell,
    }
    with (output_dir / "tabular_potential_qrl_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_jsonable(payload), handle, indent=2)
    _write_history(output_dir / "tabular_potential_qrl_history.csv", all_history)
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
