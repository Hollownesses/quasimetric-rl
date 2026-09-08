#!/usr/bin/env python3
"""Constructively search for a jointly correct and edge-feasible IQE.

This is a finite-graph representability experiment, not QRL training.  It may
use every reverse-Dijkstra state label and every validated graph edge.  The
model architecture is unchanged from the practical QRL critic; only the
encoder and IQE quasimetric model are optimized.  Global push, dual variables,
latent dynamics, bootstrapping, and trajectory losses are deliberately absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "joint_feasible_iqe_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "joint_feasible_iqe_xdg")
)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from minimal_qrl.baselines import HybridAStarConfig
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.industry_exp.full_graph_checkpoint_audit import (
    summarize_constraint_family,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)
from minimal_qrl.industry_exp.supervised_iqe_oracle import (
    _make_agent,
    _regression_metrics,
)
from minimal_qrl.industry_exp.tabular_potential_qrl import (
    FAMILY_NAMES,
    build_tabular_problem,
    reference_values_for_problem,
)


@dataclass(frozen=True)
class JointFeasibleProblem:
    states: np.ndarray
    observations: np.ndarray
    goal_observation: np.ndarray
    reference_values: np.ndarray
    sources: np.ndarray
    destinations: np.ndarray
    costs: np.ndarray
    families: np.ndarray
    terminal_indices: np.ndarray
    u_trap_mask: np.ndarray
    graph_stats: Mapping[str, Any]

    @property
    def num_states(self) -> int:
        return int(len(self.states))

    @property
    def num_edges(self) -> int:
        return int(len(self.sources))


@dataclass(frozen=True)
class CertificateThresholds:
    u_local_pearson: float = 0.8
    successor_pairwise: float = 0.8
    ordinary_max_excess: float = 0.25
    direct_goal_max_excess: float = 0.25
    terminal_goal_max_distance: float = 1e-3
    bellman_max_excess: float = 0.25


class CyclingSampler:
    """Seeded shuffled sweeps, with replacement only for final padding."""

    def __init__(self, indices: np.ndarray, batch_size: int, seed: int):
        self.indices = np.asarray(indices, dtype=np.int64)
        if not len(self.indices):
            raise ValueError("sampling pool must be non-empty")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.rng = np.random.default_rng(int(seed))
        self.permutation = np.empty(0, dtype=np.int64)
        self.cursor = 0

    def next(self) -> np.ndarray:
        if self.cursor >= len(self.permutation):
            self.permutation = self.rng.permutation(self.indices)
            self.cursor = 0
        result = self.permutation[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        if len(result) < self.batch_size:
            padding = self.rng.choice(
                self.indices,
                size=self.batch_size - len(result),
                replace=True,
            )
            result = np.concatenate([result, padding])
        return result


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


def build_joint_feasible_problem(
    env: CommInspectionDubinsUAV2D,
    scenario: Mapping[str, Any],
    config: HybridAStarConfig,
) -> JointFeasibleProblem:
    """Build the exact finite graph and all permitted Dijkstra labels."""

    tabular, raw_graph = build_tabular_problem(env, config)
    reference = reference_values_for_problem(tabular, raw_graph)
    observations = np.stack(
        [env.state_to_observation(state) for state in tabular.states]
    ).astype(np.float32)
    goal = env.abstract_goal_observation().astype(np.float32)
    regions = scenario.get("metadata", {}).get(
        "exploration_diagnostic_regions", {}
    )
    bounds = regions.get("u_trap_interior")
    if not isinstance(bounds, Sequence) or len(bounds) != 4:
        raise ValueError("scenario must define the u_trap_interior region")
    x_min, y_min, x_max, y_max = (float(value) for value in bounds)
    u_mask = (
        (tabular.states[:, 0] >= x_min)
        & (tabular.states[:, 0] <= x_max)
        & (tabular.states[:, 1] >= y_min)
        & (tabular.states[:, 1] <= y_max)
    )
    if int(np.sum(u_mask)) < 2:
        raise ValueError("U-trap interior contains fewer than two graph states")
    return JointFeasibleProblem(
        states=tabular.states,
        observations=observations,
        goal_observation=goal,
        reference_values=reference.astype(np.float32),
        sources=tabular.sources,
        destinations=tabular.destinations,
        costs=tabular.costs,
        families=tabular.families,
        terminal_indices=tabular.terminal_indices,
        u_trap_mask=u_mask,
        graph_stats={**dict(tabular.graph_stats), "u_trap_states": int(np.sum(u_mask))},
    )


def select_topk_active_edges(
    excess_by_critic: np.ndarray,
    families: np.ndarray,
    *,
    topk: int,
    family: int = 0,
) -> np.ndarray:
    """Select the worst union-of-critics edges in one constraint family."""

    excess = np.asarray(excess_by_critic, dtype=np.float64)
    if excess.ndim == 1:
        excess = excess[None, :]
    families = np.asarray(families)
    if excess.shape[1] != len(families):
        raise ValueError("edge excess and family arrays have incompatible sizes")
    candidates = np.flatnonzero(families == int(family))
    count = min(max(int(topk), 0), len(candidates))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    severity = np.max(excess[:, candidates], axis=0)
    if count == len(candidates):
        selected = np.arange(len(candidates))
    else:
        selected = np.argpartition(severity, -count)[-count:]
    selected = selected[np.argsort(severity[selected])[::-1]]
    return candidates[selected].astype(np.int64)


def update_active_replay(
    replay: np.ndarray,
    current: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, int]:
    """Update the worst-edge replay while reporting genuinely new edges."""

    replay = np.asarray(replay, dtype=np.int64)
    current = np.asarray(current, dtype=np.int64)
    if mode == "replace":
        return current.copy(), int(len(np.setdiff1d(current, replay)))
    if mode != "cumulative":
        raise ValueError(f"unsupported active replay mode: {mode}")
    updated = np.union1d(replay, current).astype(np.int64)
    return updated, int(len(updated) - len(np.unique(replay)))


def select_fixed_u_trap_anchors(mask: np.ndarray, size: int) -> np.ndarray:
    """Choose a deterministic, fixed subset of U-trap state indices.

    A negative size means every U-trap state; zero disables the extra local
    supervision.  Even spacing avoids making the subset depend on an RNG draw.
    """

    candidates = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)
    requested = int(size)
    if requested == 0:
        return np.empty(0, dtype=np.int64)
    if requested < 0 or requested >= len(candidates):
        return candidates
    positions = np.linspace(0, len(candidates) - 1, num=requested, dtype=np.int64)
    return candidates[positions]


def constraint_warmup_multiplier(
    step: int,
    warmup_steps: int,
    power: float = 1.0,
) -> float:
    """Polynomial constraint ramp, equal to one after the warm-up horizon."""

    if int(warmup_steps) <= 0:
        return 1.0
    if float(power) <= 0.0:
        raise ValueError("constraint warm-up power must be positive")
    progress = float(np.clip(int(step) / int(warmup_steps), 0.0, 1.0))
    return progress ** float(power)


def maxlike_squared_excess(excess: torch.Tensor, p: float) -> torch.Tensor:
    """Stable squared p-mean; approaches squared max as ``p`` increases."""

    if excess.numel() == 0:
        return torch.zeros((), device=excess.device, dtype=excess.dtype)
    exponent = float(p)
    if exponent <= 0.0:
        raise ValueError("tail p-norm exponent must be positive")
    value = torch.linalg.vector_norm(excess, ord=exponent)
    value = value / (float(excess.numel()) ** (1.0 / exponent))
    return value.square()


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            result.update(_flatten(value, prefix=f"{name}."))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
    return result


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _trained_parameters(agent) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    for critic in agent.critics:
        parameters.extend(critic.encoder.parameters())
        parameters.extend(critic.quasimetric_model.parameters())
    return parameters


def _edge_distances(
    critic,
    observations: torch.Tensor,
    goal: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    source = observations[sources[indices]]
    selected_destinations = destinations[indices]
    successor = goal.expand(len(indices), -1).clone()
    physical = selected_destinations >= 0
    if bool(physical.any()):
        successor[physical] = observations[selected_destinations[physical]]
    return critic(source, successor).reshape(-1)


@torch.no_grad()
def _predict_values(
    critic,
    observations: torch.Tensor,
    goal: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    critic.eval()
    outputs: list[np.ndarray] = []
    for begin in range(0, len(observations), int(batch_size)):
        source = observations[begin : begin + int(batch_size)]
        outputs.append(
            critic(source, goal.expand(len(source), -1))
            .reshape(-1)
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs).astype(np.float64)


@torch.no_grad()
def _predict_edges(
    critic,
    observations: torch.Tensor,
    goal: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    *,
    batch_size: int,
) -> np.ndarray:
    critic.eval()
    outputs: list[np.ndarray] = []
    all_indices = torch.arange(len(sources), device=sources.device)
    for begin in range(0, len(sources), int(batch_size)):
        indices = all_indices[begin : begin + int(batch_size)]
        outputs.append(
            _edge_distances(
                critic,
                observations,
                goal,
                sources,
                destinations,
                indices,
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs).astype(np.float64)


def lattice_successor_ranking(
    problem: JointFeasibleProblem,
    predictions: np.ndarray,
    *,
    source_mask: np.ndarray,
) -> dict[str, Any]:
    """Compare c+d(s',G) ordering against finite-graph Dijkstra ordering."""

    predictions = np.asarray(predictions, dtype=np.float64)
    reference = np.asarray(problem.reference_values, dtype=np.float64)
    destination_prediction = np.zeros(problem.num_edges, dtype=np.float64)
    destination_reference = np.zeros(problem.num_edges, dtype=np.float64)
    physical = problem.destinations >= 0
    destination_prediction[physical] = predictions[problem.destinations[physical]]
    destination_reference[physical] = reference[problem.destinations[physical]]
    predicted_scores = problem.costs.astype(np.float64) + destination_prediction
    reference_scores = problem.costs.astype(np.float64) + destination_reference
    order = np.argsort(problem.sources, kind="stable")
    ordered_sources = problem.sources[order]
    allowed = np.flatnonzero(np.asarray(source_mask, dtype=bool))
    begins = np.searchsorted(ordered_sources, allowed, side="left")
    ends = np.searchsorted(ordered_sources, allowed, side="right")
    top1: list[float] = []
    pairwise: list[float] = []
    for begin, end in zip(begins, ends):
        indices = order[begin:end]
        if len(indices) < 2:
            continue
        predicted = predicted_scores[indices]
        target = reference_scores[indices]
        chosen = int(np.argmin(predicted))
        top1.append(float(target[chosen] <= float(np.min(target)) + 1e-6))
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                target_delta = float(target[left] - target[right])
                if abs(target_delta) <= 1e-9:
                    continue
                prediction_delta = float(predicted[left] - predicted[right])
                pairwise.append(float(prediction_delta * target_delta > 0.0))
    return {
        "source_states": int(len(top1)),
        "top1_accuracy": float(np.mean(top1)) if top1 else None,
        "pairwise_accuracy": float(np.mean(pairwise)) if pairwise else None,
        "pair_count": int(len(pairwise)),
    }


def _family_audits(
    problem: JointFeasibleProblem,
    edge_distances: np.ndarray,
    values: np.ndarray,
    *,
    numerical_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    edge_audit: dict[str, Any] = {}
    bellman_audit: dict[str, Any] = {}
    successor_values = np.zeros(problem.num_edges, dtype=np.float64)
    physical = problem.destinations >= 0
    successor_values[physical] = values[problem.destinations[physical]]
    bellman_lhs = values[problem.sources] - successor_values
    for family_index, name in enumerate(FAMILY_NAMES):
        mask = problem.families == family_index
        epsilon = 0.0
        edge_audit[name] = summarize_constraint_family(
            edge_distances[mask],
            problem.costs[mask],
            epsilon=epsilon,
            numerical_tolerance=float(numerical_tolerance),
        )
        bellman_audit[name] = summarize_constraint_family(
            bellman_lhs[mask],
            problem.costs[mask],
            epsilon=epsilon,
            numerical_tolerance=float(numerical_tolerance),
        )
    return edge_audit, bellman_audit


def evaluate_critic(
    critic,
    problem: JointFeasibleProblem,
    tensors: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    numerical_tolerance: float,
    thresholds: CertificateThresholds,
) -> tuple[dict[str, Any], np.ndarray]:
    values = _predict_values(
        critic,
        tensors["observations"],
        tensors["goal"],
        batch_size=int(batch_size),
    )
    edges = _predict_edges(
        critic,
        tensors["observations"],
        tensors["goal"],
        tensors["sources"],
        tensors["destinations"],
        batch_size=int(batch_size),
    )
    edge_audit, bellman_audit = _family_audits(
        problem,
        edges,
        values,
        numerical_tolerance=float(numerical_tolerance),
    )
    u_metrics = _regression_metrics(
        values[problem.u_trap_mask],
        problem.reference_values[problem.u_trap_mask],
    )
    ranking = lattice_successor_ranking(
        problem,
        values,
        source_mask=problem.u_trap_mask,
    )
    terminal_max = float(
        np.max(values[problem.terminal_indices], initial=0.0)
    )
    max_bellman = float(
        max(audit["positive_excess_max"] for audit in bellman_audit.values())
    )
    checks = {
        "u_local_pearson": bool(
            u_metrics["pearson"] is not None
            and float(u_metrics["pearson"]) >= thresholds.u_local_pearson
        ),
        "successor_pairwise": bool(
            ranking["pairwise_accuracy"] is not None
            and float(ranking["pairwise_accuracy"])
            >= thresholds.successor_pairwise
        ),
        "ordinary_max_excess": bool(
            edge_audit["ordinary"]["positive_excess_max"]
            <= thresholds.ordinary_max_excess
        ),
        "direct_goal_max_excess": bool(
            edge_audit["direct_goal"]["positive_excess_max"]
            <= thresholds.direct_goal_max_excess
        ),
        "terminal_goal_max_distance": bool(
            terminal_max <= thresholds.terminal_goal_max_distance
        ),
        "bellman_max_excess": bool(
            max_bellman <= thresholds.bellman_max_excess
        ),
    }
    metrics = {
        "goal_slice": _regression_metrics(values, problem.reference_values),
        "u_trap_local": u_metrics,
        "u_trap_successor_ranking": ranking,
        "pairwise_edge_audit": edge_audit,
        "goal_slice_bellman_audit": bellman_audit,
        "terminal_goal_max_distance": terminal_max,
        "bellman_max_excess": max_bellman,
        "certificate_checks": checks,
        "certificate_pass": bool(all(checks.values())),
    }
    excess = np.maximum(edges - problem.costs.astype(np.float64), 0.0)
    return metrics, excess


def _certificate_score(
    critic_metrics: Sequence[Mapping[str, Any]],
    thresholds: CertificateThresholds,
) -> float:
    """Fixed checkpoint rule: minimize the worst normalized certificate gap."""

    scores = []
    for metrics in critic_metrics:
        edge = metrics["pairwise_edge_audit"]
        pearson = metrics["u_trap_local"]["pearson"]
        ranking = metrics["u_trap_successor_ranking"]["pairwise_accuracy"]
        scores.append(
            max(
                float(edge["ordinary"]["positive_excess_max"])
                / max(thresholds.ordinary_max_excess, 1e-12),
                float(edge["direct_goal"]["positive_excess_max"])
                / max(thresholds.direct_goal_max_excess, 1e-12),
                float(metrics["terminal_goal_max_distance"])
                / max(thresholds.terminal_goal_max_distance, 1e-12),
                float(metrics["bellman_max_excess"])
                / max(thresholds.bellman_max_excess, 1e-12),
                max(0.0, thresholds.u_local_pearson - float(pearson or -1.0))
                / max(thresholds.u_local_pearson, 1e-12),
                max(0.0, thresholds.successor_pairwise - float(ranking or -1.0))
                / max(thresholds.successor_pairwise, 1e-12),
            )
        )
    return float(max(scores))


def _model_signature(agent) -> dict[str, Any]:
    return {
        "num_critics": int(len(agent.critics)),
        "encoder": repr(agent.critics[0].encoder),
        "quasimetric_model": repr(agent.critics[0].quasimetric_model),
        "quasimetric_head": repr(
            agent.critics[0].quasimetric_model.quasimetric_head
        ),
        "total_parameters": int(sum(p.numel() for p in agent.parameters())),
        "trained_parameters": int(sum(p.numel() for p in _trained_parameters(agent))),
    }


def _checkpoint_payload(
    agent,
    *,
    step: int,
    args: argparse.Namespace,
    problem: JointFeasibleProblem,
    signature: Mapping[str, Any],
    score: float,
) -> dict[str, Any]:
    return {
        "optim_steps": int(step),
        "agent": agent.state_dict(),
        "training_mode": "joint_feasible_iqe_constructive_search",
        "objective": (
            "supervised full-state Dijkstra goal slice + random full-graph edge "
            f"sweeps + {args.active_replay_mode} top-k violation replay"
        ),
        "model_signature": dict(signature),
        "graph": dict(problem.graph_stats),
        "certificate_score": float(score),
        "config": vars(args),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-id", default="u_trap_target")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--position-resolution", type=float, default=0.25)
    parser.add_argument("--heading-bins", type=int, default=24)
    parser.add_argument("--primitive-steps", type=int, default=5)
    parser.add_argument(
        "--primitive-scales",
        type=float,
        nargs="+",
        default=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    parser.add_argument("--goal-pretrain-steps", type=int, default=10_000)
    parser.add_argument("--joint-steps", type=int, default=30_000)
    parser.add_argument("--goal-batch-size", type=int, default=2048)
    parser.add_argument("--ordinary-batch-size", type=int, default=4096)
    parser.add_argument("--active-set-size", type=int, default=4096)
    parser.add_argument("--active-batch-size", type=int, default=4096)
    parser.add_argument("--active-refresh-interval", type=int, default=500)
    parser.add_argument(
        "--active-replay-mode",
        choices=("replace", "cumulative"),
        default="replace",
        help="replace the worst-edge set or permanently retain its historical union",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.0,
        help="fraction of currently worst ordinary edges used by tail losses",
    )
    parser.add_argument("--tail-weight", type=float, default=0.0)
    parser.add_argument("--tail-maxlike-weight", type=float, default=0.0)
    parser.add_argument("--tail-pnorm", type=float, default=8.0)
    parser.add_argument(
        "--constraint-warmup-steps",
        type=int,
        default=0,
        help="polynomially ramp all edge-loss weights over this many joint steps",
    )
    parser.add_argument("--constraint-warmup-power", type=float, default=1.0)
    parser.add_argument(
        "--u-trap-goal-anchor-size",
        type=int,
        default=0,
        help="fixed U-trap label set per update; negative means all U-trap states",
    )
    parser.add_argument("--u-trap-goal-weight", type=float, default=1.0)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--joint-lr", type=float, default=5e-5)
    parser.add_argument("--goal-weight", type=float, default=1.0)
    parser.add_argument("--ordinary-weight", type=float, default=1.0)
    parser.add_argument("--active-weight", type=float, default=10.0)
    parser.add_argument("--direct-goal-weight", type=float, default=10.0)
    parser.add_argument("--terminal-goal-weight", type=float, default=10.0)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--numerical-tolerance", type=float, default=1e-6)
    parser.add_argument("--success-u-local-pearson", type=float, default=0.8)
    parser.add_argument("--success-successor-pairwise", type=float, default=0.8)
    parser.add_argument("--success-ordinary-max-excess", type=float, default=0.25)
    parser.add_argument("--success-direct-max-excess", type=float, default=0.25)
    parser.add_argument("--success-terminal-max-distance", type=float, default=1e-3)
    parser.add_argument("--success-bellman-max-excess", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.goal_pretrain_steps) < 0 or int(args.joint_steps) < 0:
        raise ValueError("training step counts must be non-negative")
    if int(args.active_refresh_interval) <= 0:
        raise ValueError("active refresh interval must be positive")
    if int(args.active_set_size) <= 0 or int(args.active_batch_size) <= 0:
        raise ValueError("active-set sizes must be positive")
    if not 0.0 <= float(args.tail_fraction) <= 1.0:
        raise ValueError("tail fraction must lie in [0, 1]")
    if float(args.tail_pnorm) <= 0.0:
        raise ValueError("tail p-norm exponent must be positive")
    if int(args.constraint_warmup_steps) < 0:
        raise ValueError("constraint warm-up steps must be non-negative")
    if float(args.constraint_warmup_power) <= 0.0:
        raise ValueError("constraint warm-up power must be positive")
    if float(args.u_trap_goal_weight) < 0.0:
        raise ValueError("U-trap goal weight must be non-negative")
    weighted_terms = {
        "goal": args.goal_weight,
        "ordinary": args.ordinary_weight,
        "active": args.active_weight,
        "tail": args.tail_weight,
        "tail max-like": args.tail_maxlike_weight,
        "direct-goal": args.direct_goal_weight,
        "terminal-goal": args.terminal_goal_weight,
    }
    for name, value in weighted_terms.items():
        if float(value) < 0.0:
            raise ValueError(f"{name} weight must be non-negative")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_config(args.scenario_config)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=int(args.seed), options={"device_id": str(args.device_id)})
    graph_config = HybridAStarConfig(
        position_resolution=float(args.position_resolution),
        heading_bins=int(args.heading_bins),
        primitive_steps=int(args.primitive_steps),
        primitive_scales=tuple(float(value) for value in args.primitive_scales),
    )
    problem = build_joint_feasible_problem(env, scenario, graph_config)
    device = auto_device(str(args.device))
    agent = _make_agent(
        env,
        scenario,
        num_critics=int(args.num_critics),
        total_steps=int(args.goal_pretrain_steps) + int(args.joint_steps),
    )
    initialization = "random"
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        state_dict = checkpoint["agent"] if isinstance(checkpoint, dict) else checkpoint
        agent.load_state_dict(state_dict, strict=True)
        initialization = str(Path(args.init_checkpoint).resolve())
    agent.to(device)
    signature = _model_signature(agent)
    parameters = _trained_parameters(agent)
    observations = torch.as_tensor(problem.observations, device=device)
    goal = torch.as_tensor(problem.goal_observation, device=device).reshape(1, -1)
    reference = torch.as_tensor(problem.reference_values, device=device)
    sources = torch.as_tensor(problem.sources, device=device, dtype=torch.long)
    destinations = torch.as_tensor(
        problem.destinations, device=device, dtype=torch.long
    )
    costs = torch.as_tensor(problem.costs, device=device)
    tensors = {
        "observations": observations,
        "goal": goal,
        "sources": sources,
        "destinations": destinations,
    }
    family_indices = {
        name: np.flatnonzero(problem.families == index).astype(np.int64)
        for index, name in enumerate(FAMILY_NAMES)
    }
    state_sampler = CyclingSampler(
        np.arange(problem.num_states),
        int(args.goal_batch_size),
        int(args.seed) + 101,
    )
    ordinary_sampler = CyclingSampler(
        family_indices["ordinary"],
        int(args.ordinary_batch_size),
        int(args.seed) + 211,
    )
    u_trap_anchor_array = select_fixed_u_trap_anchors(
        problem.u_trap_mask,
        int(args.u_trap_goal_anchor_size),
    )
    u_trap_anchor_indices = torch.as_tensor(
        u_trap_anchor_array, device=device, dtype=torch.long
    )
    thresholds = CertificateThresholds(
        u_local_pearson=float(args.success_u_local_pearson),
        successor_pairwise=float(args.success_successor_pairwise),
        ordinary_max_excess=float(args.success_ordinary_max_excess),
        direct_goal_max_excess=float(args.success_direct_max_excess),
        terminal_goal_max_distance=float(args.success_terminal_max_distance),
        bellman_max_excess=float(args.success_bellman_max_excess),
    )
    goal_scale = max(float(np.std(problem.reference_values)), 1.0)
    ordinary_scale = max(
        float(np.mean(problem.costs[problem.families == 0])), 1.0
    )
    direct_scale = max(
        float(np.mean(problem.costs[problem.families == 1])), 1.0
    )
    terminal_scale = 1.0
    history: list[dict[str, Any]] = []
    started = perf_counter()

    def supervised_goal_losses(
        critic,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = critic(
            observations[indices], goal.expand(len(indices), -1)
        ).reshape(-1)
        global_loss = F.mse_loss(
            prediction / goal_scale,
            reference[indices] / goal_scale,
        )
        if len(u_trap_anchor_indices):
            u_prediction = critic(
                observations[u_trap_anchor_indices],
                goal.expand(len(u_trap_anchor_indices), -1),
            ).reshape(-1)
            u_loss = F.mse_loss(
                u_prediction / goal_scale,
                reference[u_trap_anchor_indices] / goal_scale,
            )
        else:
            u_loss = torch.zeros((), device=device, dtype=global_loss.dtype)
        combined = global_loss + float(args.u_trap_goal_weight) * u_loss
        return combined, global_loss, u_loss

    if int(args.goal_pretrain_steps) > 0:
        optimizer = torch.optim.AdamW(
            parameters, lr=float(args.pretrain_lr), weight_decay=0.0
        )
        running: list[float] = []
        progress = tqdm(
            range(1, int(args.goal_pretrain_steps) + 1),
            desc="joint certificate goal pretrain",
        )
        for step in progress:
            indices = torch.as_tensor(
                state_sampler.next(), device=device, dtype=torch.long
            )
            losses = []
            for critic in agent.critics:
                combined, _, _ = supervised_goal_losses(critic, indices)
                losses.append(combined)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(args.gradient_clip))
            optimizer.step()
            running.append(float(loss.detach().cpu()))
            if (
                step == 1
                or step % max(1, int(args.log_interval)) == 0
                or step == int(args.goal_pretrain_steps)
            ):
                mean_loss = float(np.mean(running))
                history.append(
                    {
                        "phase": "goal_pretrain",
                        "step": int(step),
                        "total_step": int(step),
                        "loss": mean_loss,
                    }
                )
                progress.set_postfix(loss=f"{mean_loss:.5f}")
                running.clear()

    direct_indices = torch.as_tensor(
        family_indices["direct_goal"], device=device, dtype=torch.long
    )
    terminal_indices = torch.as_tensor(
        family_indices["terminal_goal"], device=device, dtype=torch.long
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=float(args.joint_lr), weight_decay=0.0
    )
    active_rng = np.random.default_rng(int(args.seed) + 307)
    active_edges = np.empty(0, dtype=np.int64)
    current_worst_edges = np.empty(0, dtype=np.int64)
    current_tail_edges = np.empty(0, dtype=np.int64)
    ordinary_edge_count = int(len(family_indices["ordinary"]))
    tail_edge_count = (
        max(1, int(math.ceil(ordinary_edge_count * float(args.tail_fraction))))
        if float(args.tail_fraction) > 0.0
        else 0
    )
    best_score = float("inf")
    best_step = 0
    best_metrics: list[dict[str, Any]] | None = None

    def evaluate_and_refresh(joint_step: int) -> list[dict[str, Any]]:
        nonlocal active_edges, current_worst_edges, current_tail_edges
        nonlocal best_score, best_step, best_metrics
        critic_metrics: list[dict[str, Any]] = []
        excesses: list[np.ndarray] = []
        for critic in agent.critics:
            metrics, excess = evaluate_critic(
                critic,
                problem,
                tensors,
                batch_size=int(args.eval_batch_size),
                numerical_tolerance=float(args.numerical_tolerance),
                thresholds=thresholds,
            )
            critic_metrics.append(metrics)
            excesses.append(excess)
        current_worst_edges = select_topk_active_edges(
            np.stack(excesses),
            problem.families,
            topk=int(args.active_set_size),
            family=0,
        )
        active_edges, newly_added = update_active_replay(
            active_edges,
            current_worst_edges,
            mode=str(args.active_replay_mode),
        )
        current_tail_edges = select_topk_active_edges(
            np.stack(excesses),
            problem.families,
            topk=tail_edge_count,
            family=0,
        )
        score = _certificate_score(critic_metrics, thresholds)
        total_step = int(args.goal_pretrain_steps) + int(joint_step)
        row: dict[str, Any] = {
            "phase": "joint",
            "step": int(joint_step),
            "total_step": total_step,
            "certificate_score": score,
            "all_critics_pass": bool(
                all(metrics["certificate_pass"] for metrics in critic_metrics)
            ),
            "active_set_size": int(len(active_edges)),
            "current_worst_set_size": int(len(current_worst_edges)),
            "new_active_edges": int(newly_added),
            "current_tail_set_size": int(len(current_tail_edges)),
            "constraint_weight_multiplier": constraint_warmup_multiplier(
                int(joint_step),
                int(args.constraint_warmup_steps),
                float(args.constraint_warmup_power),
            ),
        }
        for critic_index, metrics in enumerate(critic_metrics):
            row.update(_flatten(metrics, prefix=f"critic_{critic_index}."))
        history.append(row)
        if score < best_score:
            best_score = score
            best_step = total_step
            best_metrics = critic_metrics
            torch.save(
                _checkpoint_payload(
                    agent,
                    step=total_step,
                    args=args,
                    problem=problem,
                    signature=signature,
                    score=score,
                ),
                output_dir / "checkpoint_best.pth",
            )
        agent.train()
        return critic_metrics

    current_metrics = evaluate_and_refresh(0)
    progress = tqdm(
        range(1, int(args.joint_steps) + 1),
        desc="joint feasible IQE",
    )
    running_terms: list[dict[str, float]] = []
    for step in progress:
        if step > 1 and (step - 1) % int(args.active_refresh_interval) == 0:
            current_metrics = evaluate_and_refresh(step - 1)
        state_indices = torch.as_tensor(
            state_sampler.next(), device=device, dtype=torch.long
        )
        ordinary_indices = torch.as_tensor(
            ordinary_sampler.next(), device=device, dtype=torch.long
        )
        if len(active_edges) > int(args.active_batch_size):
            selected_active = active_rng.choice(
                active_edges,
                size=int(args.active_batch_size),
                replace=False,
            )
        else:
            selected_active = active_edges
        active_indices = torch.as_tensor(
            selected_active, device=device, dtype=torch.long
        )
        tail_indices = torch.as_tensor(
            current_tail_edges, device=device, dtype=torch.long
        )
        constraint_multiplier = constraint_warmup_multiplier(
            step,
            int(args.constraint_warmup_steps),
            float(args.constraint_warmup_power),
        )
        critic_losses = []
        detached_terms: list[dict[str, float]] = []
        for critic in agent.critics:
            goal_loss, global_goal_loss, u_goal_loss = supervised_goal_losses(
                critic, state_indices
            )

            def constraint_excess(
                indices: torch.Tensor, scale: float
            ) -> torch.Tensor:
                if indices.numel() == 0:
                    return torch.empty(0, device=device, dtype=costs.dtype)
                distance = _edge_distances(
                    critic,
                    observations,
                    goal,
                    sources,
                    destinations,
                    indices,
                )
                return (distance - costs[indices]).relu() / scale

            def constraint_loss(indices: torch.Tensor, scale: float) -> torch.Tensor:
                excess = constraint_excess(indices, scale)
                if excess.numel() == 0:
                    return torch.zeros((), device=device, dtype=costs.dtype)
                return excess.square().mean()

            ordinary_loss = constraint_loss(ordinary_indices, ordinary_scale)
            active_loss = constraint_loss(active_indices, ordinary_scale)
            tail_excess = constraint_excess(tail_indices, ordinary_scale)
            tail_loss = (
                tail_excess.square().mean()
                if tail_excess.numel()
                else torch.zeros((), device=device, dtype=costs.dtype)
            )
            tail_maxlike_loss = maxlike_squared_excess(
                tail_excess, float(args.tail_pnorm)
            )
            direct_loss = constraint_loss(direct_indices, direct_scale)
            terminal_loss = constraint_loss(terminal_indices, terminal_scale)
            constraint_total = (
                float(args.ordinary_weight) * ordinary_loss
                + float(args.active_weight) * active_loss
                + float(args.tail_weight) * tail_loss
                + float(args.tail_maxlike_weight) * tail_maxlike_loss
                + float(args.direct_goal_weight) * direct_loss
                + float(args.terminal_goal_weight) * terminal_loss
            )
            total = (
                float(args.goal_weight) * goal_loss
                + constraint_multiplier * constraint_total
            )
            critic_losses.append(total)
            detached_terms.append(
                {
                    "goal": float(goal_loss.detach().cpu()),
                    "goal_global": float(global_goal_loss.detach().cpu()),
                    "goal_u_trap": float(u_goal_loss.detach().cpu()),
                    "ordinary": float(ordinary_loss.detach().cpu()),
                    "active": float(active_loss.detach().cpu()),
                    "tail": float(tail_loss.detach().cpu()),
                    "tail_maxlike": float(tail_maxlike_loss.detach().cpu()),
                    "direct_goal": float(direct_loss.detach().cpu()),
                    "terminal_goal": float(terminal_loss.detach().cpu()),
                    "constraint_multiplier": float(constraint_multiplier),
                    "constraint_total": float(constraint_total.detach().cpu()),
                    "total": float(total.detach().cpu()),
                }
            )
        loss = torch.stack(critic_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(args.gradient_clip))
        optimizer.step()
        running_terms.append(
            {
                key: float(np.mean([terms[key] for terms in detached_terms]))
                for key in detached_terms[0]
            }
        )
        if step % max(1, int(args.log_interval)) == 0 or step == int(args.joint_steps):
            means = {
                key: float(np.mean([terms[key] for terms in running_terms]))
                for key in running_terms[0]
            }
            progress.set_postfix(
                loss=f"{means['total']:.4f}",
                active=f"{means['active']:.4f}",
            )
            history.append(
                {
                    "phase": "joint_loss",
                    "step": int(step),
                    "total_step": int(args.goal_pretrain_steps) + int(step),
                    **{f"loss.{key}": value for key, value in means.items()},
                }
            )
            running_terms.clear()

    if int(args.joint_steps) > 0:
        current_metrics = evaluate_and_refresh(int(args.joint_steps))
    final_step = int(args.goal_pretrain_steps) + int(args.joint_steps)
    torch.save(
        _checkpoint_payload(
            agent,
            step=final_step,
            args=args,
            problem=problem,
            signature=signature,
            score=_certificate_score(current_metrics, thresholds),
        ),
        output_dir / "checkpoint_final.pth",
    )
    values_payload: dict[str, np.ndarray] = {
        "states": problem.states,
        "dijkstra_values": problem.reference_values,
        "u_trap_mask": problem.u_trap_mask,
        "u_trap_goal_anchor_indices": u_trap_anchor_array,
        "final_active_replay_edges": active_edges,
        "final_current_tail_edges": current_tail_edges,
    }
    for critic_index, critic in enumerate(agent.critics):
        values_payload[f"critic_{critic_index}_values"] = _predict_values(
            critic,
            observations,
            goal,
            batch_size=int(args.eval_batch_size),
        )
    np.savez_compressed(output_dir / "joint_feasible_values.npz", **values_payload)
    _write_rows(output_dir / "joint_feasible_history.csv", history)
    payload = {
        "experiment": "joint_feasible_iqe_constructive_search",
        "optimizer_variant": (
            "persistent_tail_constructive"
            if str(args.active_replay_mode) == "cumulative"
            or float(args.tail_fraction) > 0.0
            or int(args.constraint_warmup_steps) > 0
            or len(u_trap_anchor_array) > 0
            else "original_replace_mean"
        ),
        "interpretation": (
            "constructive finite-graph representability search; not QRL and not "
            "a label-free learning result"
        ),
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "device": str(device),
        "initialization": initialization,
        "graph_config": asdict(graph_config),
        "graph": dict(problem.graph_stats),
        "model_signature": signature,
        "objective": {
            "goal": "MSE(d_theta(s,G), Dijkstra(s,G)) over shuffled full-state sweeps",
            "ordinary": "mean relu(d_theta(s,s_prime)-c)^2 over shuffled edge sweeps",
            "u_trap_goal": (
                "separate MSE on a fixed set of U-trap states in every update"
            ),
            "u_trap_goal_anchor_count": int(len(u_trap_anchor_array)),
            "active": (
                "mean squared excess on sampled edges from the replace/cumulative "
                "worst-edge replay selected by active_replay_mode"
            ),
            "active_replay_mode": str(args.active_replay_mode),
            "tail": (
                "mean squared excess plus squared p-mean on the currently worst "
                f"{float(args.tail_fraction):.6g} fraction of ordinary edges"
            ),
            "tail_edge_count": int(tail_edge_count),
            "constraint_warmup": (
                "all edge terms multiplied by min(step/warmup_steps,1)^power"
            ),
            "direct_goal": "all direct-to-G edges every update",
            "terminal_goal": "all terminal-to-G zero-cost edges every update",
            "global_push_used": False,
            "dual_optimizer_used": False,
            "latent_dynamics_used": False,
            "trajectory_loss_used": False,
        },
        "normalization": {
            "goal_scale_std": goal_scale,
            "ordinary_cost_mean_scale": ordinary_scale,
            "direct_goal_cost_mean_scale": direct_scale,
            "terminal_scale": terminal_scale,
        },
        "thresholds": asdict(thresholds),
        "checkpoint_selection": (
            "minimum worst normalized certificate gap over scheduled full-graph audits"
        ),
        "best_checkpoint": str((output_dir / "checkpoint_best.pth").resolve()),
        "best_step": int(best_step),
        "best_score": float(best_score),
        "best_metrics_by_critic": best_metrics,
        "final_checkpoint": str((output_dir / "checkpoint_final.pth").resolve()),
        "final_metrics_by_critic": current_metrics,
        "all_critics_final_pass": bool(
            all(metrics["certificate_pass"] for metrics in current_metrics)
        ),
        "elapsed_sec": float(perf_counter() - started),
        "config": vars(args),
    }
    metrics_path = output_dir / "joint_feasible_iqe_metrics.json"
    metrics_path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summaries = []
    for critic_index, metrics in enumerate(current_metrics):
        summaries.append(
            {
                "critic": critic_index,
                "goal_rmse": metrics["goal_slice"]["rmse"],
                "u_local_pearson": metrics["u_trap_local"]["pearson"],
                "successor_pairwise": metrics["u_trap_successor_ranking"][
                    "pairwise_accuracy"
                ],
                "ordinary_max_excess": metrics["pairwise_edge_audit"][
                    "ordinary"
                ]["positive_excess_max"],
                "direct_goal_max_excess": metrics["pairwise_edge_audit"][
                    "direct_goal"
                ]["positive_excess_max"],
                "terminal_goal_max_distance": metrics[
                    "terminal_goal_max_distance"
                ],
                "bellman_max_excess": metrics["bellman_max_excess"],
                "certificate_pass": metrics["certificate_pass"],
            }
        )
    print(json.dumps({
        "best_step": best_step,
        "best_score": best_score,
        "all_critics_final_pass": payload["all_critics_final_pass"],
        "critics": summaries,
    }, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
