#!/usr/bin/env python3
"""Evaluate point-goal QRL against reverse Dijkstra on the same lattice."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qrl_point_goal_diag_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qrl_point_goal_diag_xdg"))

import numpy as np
import torch

from minimal_qrl.baselines import HybridAStarConfig
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.u_trap_local_navigability import build_probe_records
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.gc_agents import QRLGoalValueAdapter
from minimal_qrl.industry_exp.exact_discrete_value_lp import (
    build_discrete_point_goal_graph,
    reverse_dijkstra,
    select_fixed_terminal_lattice_goal,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)
from minimal_qrl.industry_exp.supervised_iqe_oracle import (
    _make_agent,
    _regression_metrics,
)
from minimal_qrl.iqe_capacity import iqe_capacity_from_checkpoint


def _encode_states(env: CommInspectionDubinsUAV2D, states: np.ndarray) -> np.ndarray:
    return np.stack([env.state_to_observation(state) for state in states]).astype(
        np.float32
    )


def _greedy_edge_agreement(
    *,
    sources: np.ndarray,
    destinations: np.ndarray,
    costs: np.ndarray,
    oracle_values: np.ndarray,
    predicted_values: np.ndarray,
    source_mask: np.ndarray,
) -> dict[str, float | int]:
    selected = np.flatnonzero(source_mask[sources])
    groups: dict[int, list[int]] = {}
    for edge_index in selected:
        source = int(sources[edge_index])
        destination = int(destinations[edge_index])
        if destination >= 0 and np.isfinite(oracle_values[destination]):
            groups.setdefault(source, []).append(int(edge_index))

    agreements = []
    strict_agreements = []
    for edge_indices in groups.values():
        edge_indices_array = np.asarray(edge_indices, dtype=np.int64)
        destination_indices = destinations[edge_indices_array]
        oracle_scores = costs[edge_indices_array] + oracle_values[destination_indices]
        predicted_scores = costs[edge_indices_array] + predicted_values[destination_indices]
        predicted_best = int(np.argmin(predicted_scores))
        oracle_best = int(np.argmin(oracle_scores))
        agreements.append(
            float(
                oracle_scores[predicted_best]
                <= float(np.min(oracle_scores)) + 1e-8
            )
        )
        strict_agreements.append(float(predicted_best == oracle_best))
    return {
        "source_count": int(len(groups)),
        "oracle_optimal_action_set_accuracy": (
            float(np.mean(agreements)) if agreements else float("nan")
        ),
        "strict_argmin_accuracy": (
            float(np.mean(strict_agreements)) if strict_agreements else float("nan")
        ),
    }


def _u_trap_mask(states: np.ndarray, scenario: dict[str, Any]) -> np.ndarray:
    regions = scenario.get("metadata", {}).get("exploration_diagnostic_regions", {})
    selected = np.zeros(len(states), dtype=bool)
    for name in ("u_trap_interior", "u_trap_west_exit"):
        bounds = regions.get(name)
        if bounds is None:
            continue
        x0, y0, x1, y1 = (float(value) for value in bounds)
        selected |= (
            (states[:, 0] >= x0)
            & (states[:, 0] <= x1)
            & (states[:, 1] >= y0)
            & (states[:, 1] <= y1)
        )
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--device-id", default="u_trap_target")
    parser.add_argument("--point-goal-candidate-index", type=int, default=0)
    parser.add_argument("--astar-position-resolution", type=float, default=0.25)
    parser.add_argument("--astar-heading-bins", type=int, default=24)
    parser.add_argument("--astar-primitive-steps", type=int, default=5)
    parser.add_argument(
        "--astar-primitive-scales",
        type=float,
        nargs="+",
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_config(args.scenario_config)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=int(args.seed), options={"device_id": str(args.device_id)})
    graph_config = HybridAStarConfig(
        position_resolution=float(args.astar_position_resolution),
        heading_bins=int(args.astar_heading_bins),
        primitive_steps=int(args.astar_primitive_steps),
        primitive_scales=tuple(float(value) for value in args.astar_primitive_scales),
    )
    goal_state, goal_index, candidates = select_fixed_terminal_lattice_goal(
        env,
        graph_config,
        candidate_index=int(args.point_goal_candidate_index),
    )
    graph = build_discrete_point_goal_graph(
        env,
        graph_config,
        goal_state_index=goal_index,
    )
    oracle_values = reverse_dijkstra(graph)
    reachable = np.isfinite(oracle_values) & graph.valid

    device = auto_device(str(args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not isinstance(checkpoint, dict) or "agent" not in checkpoint:
        raise ValueError("point-goal diagnostics require a full training checkpoint")
    if str(checkpoint.get("training_mode")) != "full_graph_point_goal":
        raise ValueError(
            "checkpoint training_mode must be 'full_graph_point_goal', got "
            f"{checkpoint.get('training_mode')!r}"
        )
    saved_goal = checkpoint.get("point_goal", {})
    if saved_goal:
        saved_index = int(saved_goal.get("point_goal_candidate_index", -1))
        saved_state = np.asarray(saved_goal.get("point_goal_state"), dtype=np.float64)
        if saved_index != int(args.point_goal_candidate_index) or not np.allclose(
            saved_state, goal_state, rtol=0.0, atol=1e-7
        ):
            raise ValueError("diagnostic point goal does not match checkpoint provenance")

    capacity = iqe_capacity_from_checkpoint(checkpoint)
    agent = _make_agent(
        env,
        scenario,
        num_critics=int(args.num_critics),
        total_steps=1,
        iqe_dim=capacity.dim,
        iqe_components=capacity.components,
    )
    agent.load_state_dict(checkpoint["agent"])
    agent.to(device).eval()
    value = QRLGoalValueAdapter(agent, env, device, distance_scale=1.0)

    reachable_indices = np.flatnonzero(reachable)
    reachable_observations = _encode_states(env, graph.states[reachable_indices])
    goal_observation = env.state_to_observation(goal_state).astype(np.float32)
    predictions = []
    for begin in range(0, len(reachable_indices), int(args.eval_batch_size)):
        batch = reachable_observations[begin : begin + int(args.eval_batch_size)]
        goals = np.repeat(goal_observation[None, :], len(batch), axis=0)
        predictions.append(value.batch_value(batch, goals))
    reachable_predictions = np.concatenate(predictions)
    predicted_values = np.full(len(graph.states), np.nan, dtype=np.float64)
    predicted_values[reachable_indices] = reachable_predictions

    probe_records = build_probe_records(scenario)
    helper_indices = np.empty(len(probe_records), dtype=np.int64)
    # Use the graph helper's deterministic nearest-lattice projection.
    from minimal_qrl.baselines import HybridAStarValueOracle

    helper = HybridAStarValueOracle(graph_config)
    _states, helper._grid_shape = helper._grid(env)
    helper_indices[:] = helper._state_indices(
        env,
        np.asarray([record["state"] for record in probe_records], dtype=np.float32),
    )
    local_keep = (
        (helper_indices >= 0)
        & reachable[np.maximum(helper_indices, 0)]
    )
    local_indices = helper_indices[local_keep]
    local_metrics = _regression_metrics(
        predicted_values[local_indices],
        oracle_values[local_indices],
    )

    u_mask = _u_trap_mask(graph.states, scenario) & reachable
    all_policy = _greedy_edge_agreement(
        sources=graph.sources,
        destinations=graph.destinations,
        costs=graph.costs,
        oracle_values=oracle_values,
        predicted_values=predicted_values,
        source_mask=reachable,
    )
    u_policy = _greedy_edge_agreement(
        sources=graph.sources,
        destinations=graph.destinations,
        costs=graph.costs,
        oracle_values=oracle_values,
        predicted_values=predicted_values,
        source_mask=u_mask,
    )
    goal_prediction = float(predicted_values[goal_index])
    metrics = {
        "global_value": _regression_metrics(
            reachable_predictions,
            oracle_values[reachable_indices],
        ),
        "u_trap_value": _regression_metrics(
            predicted_values[u_mask],
            oracle_values[u_mask],
        ),
        "fixed_16_probe_value": {
            **local_metrics,
            "requested": int(len(probe_records)),
            "reachable": int(np.sum(local_keep)),
        },
        "global_greedy_edge_topology": all_policy,
        "u_trap_greedy_edge_topology": u_policy,
        "self_distance_at_point_goal": goal_prediction,
    }
    payload = {
        "experiment": "full_graph_point_goal_reverse_dijkstra_evaluation",
        "training_mode": str(checkpoint.get("training_mode")),
        "oracle_role": "post-training evaluation only",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "goal": {
            "selection": "fixed index in original terminal-lattice ordering",
            "candidate_index": int(args.point_goal_candidate_index),
            "original_candidate_count": int(len(candidates)),
            "global_state_index": int(goal_index),
            "state": [float(value) for value in goal_state],
            "observation_abstract_goal_feature": float(
                goal_observation[int(env.task_context_indices["abstract_goal"])]
            ),
        },
        "graph": {
            "grid_states": int(len(graph.states)),
            "valid_states": int(np.sum(graph.valid)),
            "reachable_states": int(np.sum(reachable)),
            "edges": int(len(graph.sources)),
            "reachable_edges": int(
                np.sum(reachable[graph.sources] & reachable[graph.destinations])
            ),
            "synthetic_goal_nodes": 0,
            "synthetic_goal_edges": 0,
        },
        "config": vars(args),
        "metrics": metrics,
    }
    path = output_dir / "point_goal_qrl_metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"goal": payload["goal"], "metrics": metrics}, ensure_ascii=False, indent=2))
    print(f"Saved point-goal diagnostics: {path}")


if __name__ == "__main__":
    main()
