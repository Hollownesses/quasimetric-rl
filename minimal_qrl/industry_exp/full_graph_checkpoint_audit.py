#!/usr/bin/env python3
"""Audit a trained IQE checkpoint on the exact full goal-reachable graph.

This is an evaluation-only diagnostic.  It does not use Oracle values or
graph constraints to update the checkpoint.  In particular, it measures both
the pairwise quasimetric edge geometry ``d(s, s') <= c`` and the induced goal
slice Bellman inequalities ``d(s, G) <= c + d(s', G)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Optional, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "full_graph_checkpoint_audit_mpl"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "full_graph_checkpoint_audit_xdg"),
)

import numpy as np
import torch

from minimal_qrl.dataset import (
    FullGraphGoalSetQRLConfig,
    create_full_graph_goal_set_qrl_dataset,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)
from minimal_qrl.industry_exp.supervised_iqe_oracle import _make_agent


FAMILY_ORDER = ("ordinary", "direct_goal", "terminal_goal")


def summarize_constraint_family(
    distances: np.ndarray,
    costs: np.ndarray,
    *,
    epsilon: float,
    numerical_tolerance: float,
) -> dict[str, float | int | bool]:
    """Return exact per-family inequality diagnostics."""

    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    if len(distances) != len(costs) or not len(distances):
        raise ValueError("distances and costs must be equally sized and non-empty")
    residual = distances - costs
    positive_excess = np.maximum(residual, 0.0)
    epsilon_excess = np.maximum(residual - float(epsilon), 0.0)
    squared_excess_mean = float(np.mean(np.square(positive_excess)))
    return {
        "count": int(len(distances)),
        "distance_mean": float(np.mean(distances)),
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "cost_mean": float(np.mean(costs)),
        "cost_min": float(np.min(costs)),
        "cost_max": float(np.max(costs)),
        "residual_mean": float(np.mean(residual)),
        "residual_min": float(np.min(residual)),
        "residual_max": float(np.max(residual)),
        "violation_count": int(np.sum(residual > float(numerical_tolerance))),
        "violation_fraction": float(
            np.mean(residual > float(numerical_tolerance))
        ),
        "positive_excess_mean": float(np.mean(positive_excess)),
        "positive_excess_max": float(np.max(positive_excess)),
        "positive_excess_p99": float(np.quantile(positive_excess, 0.99)),
        "squared_excess_mean": squared_excess_mean,
        "epsilon": float(epsilon),
        "epsilon_violation_count": int(
            np.sum(residual > float(epsilon) + float(numerical_tolerance))
        ),
        "epsilon_violation_fraction": float(
            np.mean(residual > float(epsilon) + float(numerical_tolerance))
        ),
        "epsilon_excess_mean": float(np.mean(epsilon_excess)),
        "epsilon_excess_max": float(np.max(epsilon_excess)),
        "dual_residual_squared": float(
            squared_excess_mean - float(epsilon) ** 2
        ),
        "all_constraints_satisfied": bool(
            np.max(residual) <= float(numerical_tolerance)
        ),
    }


def _family_masks(transition_infos: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    terminal = transition_infos["abstract_goal_edge"].cpu().numpy().astype(bool)
    direct = (
        transition_infos["full_graph_direct_goal_edge"]
        .cpu()
        .numpy()
        .astype(bool)
    ) & ~terminal
    return {
        "ordinary": ~direct & ~terminal,
        "direct_goal": direct,
        "terminal_goal": terminal,
    }


@torch.inference_mode()
def _evaluate_edges(
    critic: torch.nn.Module,
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    goal_observation: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_distances: list[np.ndarray] = []
    source_goal_distances: list[np.ndarray] = []
    successor_goal_distances: list[np.ndarray] = []
    goal = goal_observation.to(device=device, dtype=torch.float32).reshape(1, -1)
    goal_embedding = critic.encoder(goal)
    for begin in range(0, len(observations), int(batch_size)):
        end = min(begin + int(batch_size), len(observations))
        source = observations[begin:end].to(device=device, dtype=torch.float32)
        successor = next_observations[begin:end].to(
            device=device,
            dtype=torch.float32,
        )
        source_embedding, successor_embedding = critic.encoder(
            torch.stack([source, successor])
        ).unbind(0)
        batch_goal_embedding = goal_embedding.expand(len(source), -1)
        edge_distances.append(
            critic.quasimetric_model(
                source_embedding,
                successor_embedding,
            ).reshape(-1).cpu().numpy()
        )
        source_goal_distances.append(
            critic.quasimetric_model(
                source_embedding,
                batch_goal_embedding,
            ).reshape(-1).cpu().numpy()
        )
        successor_goal_distances.append(
            critic.quasimetric_model(
                successor_embedding,
                batch_goal_embedding,
            ).reshape(-1).cpu().numpy()
        )
    return (
        np.concatenate(edge_distances).astype(np.float64),
        np.concatenate(source_goal_distances).astype(np.float64),
        np.concatenate(successor_goal_distances).astype(np.float64),
    )


@torch.inference_mode()
def _evaluate_linear_push(
    critic: torch.nn.Module,
    source_pool: torch.Tensor,
    goal_observation: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    goal = goal_observation.to(device=device, dtype=torch.float32).reshape(1, -1)
    goal_embedding = critic.encoder(goal)
    for begin in range(0, len(source_pool), int(batch_size)):
        source = source_pool[begin : begin + int(batch_size)].to(
            device=device,
            dtype=torch.float32,
        )
        source_embedding = critic.encoder(source)
        outputs.append(
            critic.quasimetric_model(
                source_embedding,
                goal_embedding.expand(len(source), -1),
            ).reshape(-1).cpu().numpy()
        )
    return np.concatenate(outputs).astype(np.float64)


def _assert_expected_count(label: str, actual: int, expected: Optional[int]) -> None:
    if expected is not None and int(actual) != int(expected):
        raise RuntimeError(f"{label}: expected {expected}, found {actual}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-id", default="u_trap_target")
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--critic-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--position-resolution", type=float, default=0.25)
    parser.add_argument("--heading-bins", type=int, default=24)
    parser.add_argument("--primitive-steps", type=int, default=5)
    parser.add_argument(
        "--primitive-scales",
        type=float,
        nargs="+",
        default=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    parser.add_argument("--ordinary-epsilon", type=float, default=0.25)
    parser.add_argument("--direct-goal-epsilon", type=float, default=0.25)
    parser.add_argument("--terminal-goal-epsilon", type=float, default=0.0)
    parser.add_argument("--numerical-tolerance", type=float, default=1e-6)
    parser.add_argument("--expected-transitions", type=int, default=None)
    parser.add_argument("--expected-ordinary", type=int, default=None)
    parser.add_argument("--expected-direct-goal", type=int, default=None)
    parser.add_argument("--expected-terminal-goal", type=int, default=None)
    return parser


def _write_family_csv(
    path: Path,
    edge_audit: Mapping[str, Mapping[str, Any]],
    bellman_audit: Mapping[str, Mapping[str, Any]],
) -> None:
    rows = []
    for audit_name, audit in (
        ("pairwise_quasimetric_edge", edge_audit),
        ("goal_slice_bellman", bellman_audit),
    ):
        for family in FAMILY_ORDER:
            rows.append({"audit": audit_name, "family": family, **audit[family]})
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    started = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_config(args.scenario_config)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    stats: dict[str, Any] = {}
    dataset = create_full_graph_goal_set_qrl_dataset(
        env,
        FullGraphGoalSetQRLConfig(
            device_id=str(args.device_id),
            position_resolution=float(args.position_resolution),
            heading_bins=int(args.heading_bins),
            primitive_steps=int(args.primitive_steps),
            primitive_scales=tuple(float(value) for value in args.primitive_scales),
            stratified_constraints=True,
        ),
        collection_stats=stats,
    )
    graph_stats = stats["full_graph_goal_set_qrl"]
    raw = dataset.raw_data
    masks = _family_masks(raw.transition_infos)
    counts = {name: int(np.sum(mask)) for name, mask in masks.items()}
    _assert_expected_count("training transitions", len(raw.rewards), args.expected_transitions)
    _assert_expected_count("ordinary edges", counts["ordinary"], args.expected_ordinary)
    _assert_expected_count("direct-goal edges", counts["direct_goal"], args.expected_direct_goal)
    _assert_expected_count("terminal-goal edges", counts["terminal_goal"], args.expected_terminal_goal)

    device = auto_device(str(args.device))
    agent = _make_agent(
        env,
        scenario,
        num_critics=int(args.num_critics),
        total_steps=1,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["agent"] if isinstance(checkpoint, dict) else checkpoint
    agent.load_state_dict(state_dict)
    if not 0 <= int(args.critic_index) < len(agent.critics):
        raise ValueError(
            f"critic index {args.critic_index} is outside [0, {len(agent.critics)})"
        )
    agent.to(device).eval()
    critic = agent.critics[int(args.critic_index)]
    goal_observation = torch.as_tensor(env.abstract_goal_observation())
    edge_distance, source_value, successor_value = _evaluate_edges(
        critic,
        raw.observations,
        raw.next_observations,
        goal_observation,
        device=device,
        batch_size=int(args.batch_size),
    )
    costs = (-raw.rewards).cpu().numpy().astype(np.float64).clip(min=0.0)
    epsilons = {
        "ordinary": float(args.ordinary_epsilon),
        "direct_goal": float(args.direct_goal_epsilon),
        "terminal_goal": float(args.terminal_goal_epsilon),
    }
    edge_audit = {
        name: summarize_constraint_family(
            edge_distance[mask],
            costs[mask],
            epsilon=epsilons[name],
            numerical_tolerance=float(args.numerical_tolerance),
        )
        for name, mask in masks.items()
    }
    bellman_lhs = source_value - successor_value
    bellman_audit = {
        name: summarize_constraint_family(
            bellman_lhs[mask],
            costs[mask],
            epsilon=epsilons[name],
            numerical_tolerance=float(args.numerical_tolerance),
        )
        for name, mask in masks.items()
    }
    if dataset.uniform_task_source_observation_pool is None:
        raise RuntimeError("full graph dataset did not expose the uniform push pool")
    linear_values = _evaluate_linear_push(
        critic,
        dataset.uniform_task_source_observation_pool,
        goal_observation,
        device=device,
        batch_size=int(args.batch_size),
    )
    linear_push = {
        "source_distribution": "uniform_goal_reachable_nonterminal_lattice_states",
        "count": int(len(linear_values)),
        "distance_mean": float(np.mean(linear_values)),
        "distance_sum": float(np.sum(linear_values)),
        "distance_min": float(np.min(linear_values)),
        "distance_max": float(np.max(linear_values)),
        "maximization_objective_mean": float(np.mean(linear_values)),
        "maximization_objective_sum": float(np.sum(linear_values)),
        "training_loss_negative_mean": float(-np.mean(linear_values)),
        "oracle_value_labels_used": False,
    }
    goal_self_distance = float(
        _evaluate_linear_push(
            critic,
            goal_observation.reshape(1, -1),
            goal_observation,
            device=device,
            batch_size=1,
        )[0]
    )
    payload = {
        "experiment": "full_graph_checkpoint_quasimetric_capacity_audit",
        "evaluation_only": True,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_training_mode": (
            checkpoint.get("training_mode", "unknown")
            if isinstance(checkpoint, dict)
            else "unknown"
        ),
        "checkpoint_objective": (
            checkpoint.get("objective", "unknown")
            if isinstance(checkpoint, dict)
            else "unknown"
        ),
        "critic_index": int(args.critic_index),
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "device": str(device),
        "graph": graph_stats,
        "constraint_definition": {
            "pairwise_quasimetric_edge": "d_theta(s,s_prime) <= edge_cost",
            "goal_slice_bellman": "d_theta(s,G) <= edge_cost + d_theta(s_prime,G)",
            "numerical_tolerance": float(args.numerical_tolerance),
            "family_epsilons": epsilons,
            "oracle_values_used": False,
        },
        "goal_self_distance": goal_self_distance,
        "pairwise_quasimetric_edge_audit": edge_audit,
        "goal_slice_bellman_audit": bellman_audit,
        "linear_global_push": linear_push,
        "elapsed_sec": float(perf_counter() - started),
        "config": vars(args),
    }
    json_path = output_dir / "full_graph_checkpoint_audit.json"
    csv_path = output_dir / "full_graph_checkpoint_audit.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_family_csv(csv_path, edge_audit, bellman_audit)
    print(json.dumps({
        "pairwise_quasimetric_edge_audit": edge_audit,
        "goal_slice_bellman_audit": bellman_audit,
        "linear_global_push": linear_push,
        "goal_self_distance": goal_self_distance,
        "elapsed_sec": payload["elapsed_sec"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved JSON audit: {json_path}")
    print(f"Saved CSV audit: {csv_path}")


if __name__ == "__main__":
    main()
