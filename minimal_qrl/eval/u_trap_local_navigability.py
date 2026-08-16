#!/usr/bin/env python3
"""Evaluate whether a QRL value field is locally navigable inside the U trap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qrl_u_nav_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from minimal_qrl.baselines.hybrid_astar import HybridAStarConfig, HybridAStarController
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.comm_inspection_execution_eval import build_qrl_adapter
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)


def build_probe_records(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = scenario.get("metadata", {})
    config = metadata.get("u_trap_local_navigability_probes")
    if not isinstance(config, Mapping):
        raise ValueError(
            "scenario metadata must define u_trap_local_navigability_probes"
        )
    device_id = str(config["device_id"])
    y = float(config["centerline_y"])
    records = []
    for position_index, position in enumerate(config["positions"]):
        for heading_index, heading in enumerate(config["headings"]):
            records.append(
                {
                    "probe_id": f"{position['label']}__{heading['label']}",
                    "position_label": str(position["label"]),
                    "position_index": int(position_index),
                    "heading_label": str(heading["label"]),
                    "heading_index": int(heading_index),
                    "device_id": device_id,
                    "state": [float(position["x"]), y, float(heading["theta"])],
                }
            )
    if not records:
        raise ValueError("U-trap local probe bank is empty")
    return records


def _coarse_action(omega: float | None, omega_max: float) -> str:
    if omega is None or not math.isfinite(float(omega)):
        return "unavailable"
    if abs(float(omega)) <= 0.25 * max(float(omega_max), 1e-8):
        return "straight"
    return "left" if float(omega) > 0.0 else "right"


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1)
        cursor = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if len(left_array) < 2 or float(np.std(left_array)) <= 1e-12 or float(np.std(right_array)) <= 1e-12:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _pairwise_ranking_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float | None:
    agreements = []
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            target_delta = float(targets[left] - targets[right])
            if abs(target_delta) <= 1e-9:
                continue
            prediction_delta = float(predictions[left] - predictions[right])
            agreements.append(float(prediction_delta * target_delta > 0.0))
    return float(np.mean(agreements)) if agreements else None


def summarize_checkpoint_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    solved = [
        record
        for record in records
        if record.get("oracle_cost") is not None and record.get("qrl_value") is not None
    ]
    predictions = np.asarray([record["qrl_value"] for record in solved], dtype=np.float64)
    targets = np.asarray([record["oracle_cost"] for record in solved], dtype=np.float64)
    exit_progress = []
    by_heading: dict[str, list[Mapping[str, Any]]] = {}
    for record in solved:
        by_heading.setdefault(str(record["heading_label"]), []).append(record)
    for group in by_heading.values():
        ordered = sorted(group, key=lambda item: int(item["position_index"]))
        for current, closer_to_exit in zip(ordered[:-1], ordered[1:]):
            oracle_delta = float(current["oracle_cost"]) - float(closer_to_exit["oracle_cost"])
            if abs(oracle_delta) <= 1e-9:
                continue
            qrl_delta = float(current["qrl_value"]) - float(closer_to_exit["qrl_value"])
            exit_progress.append(float(qrl_delta * oracle_delta > 0.0))
    exact_actions = [
        float(record["qrl_action_exact_match"])
        for record in solved
        if record.get("qrl_action_exact_match") is not None
    ]
    coarse_actions = [
        float(record["qrl_action_coarse_match"])
        for record in solved
        if record.get("qrl_action_coarse_match") is not None
    ]
    errors = np.abs(predictions - targets) if len(solved) else np.asarray([])
    return {
        "requested_probes": int(len(records)),
        "solved_probes": int(len(solved)),
        "oracle_success_rate": float(len(solved) / max(len(records), 1)),
        "mae": float(errors.mean()) if len(errors) else None,
        "normalized_mae": (
            float(errors.mean() / max(float(np.mean(np.abs(targets))), 1e-8))
            if len(errors)
            else None
        ),
        "pearson_corr": _correlation(predictions, targets),
        "spearman_corr": _correlation(_rankdata(predictions), _rankdata(targets)) if len(solved) else None,
        "pairwise_ranking_accuracy": _pairwise_ranking_accuracy(predictions, targets) if len(solved) else None,
        "exit_progress_ordering_accuracy": float(np.mean(exit_progress)) if exit_progress else None,
        "first_action_exact_accuracy": float(np.mean(exact_actions)) if exact_actions else None,
        "first_action_coarse_accuracy": float(np.mean(coarse_actions)) if coarse_actions else None,
        "prediction_mean": float(predictions.mean()) if len(predictions) else None,
        "oracle_mean": float(targets.mean()) if len(targets) else None,
    }


def compute_oracle_records(
    env: CommInspectionDubinsUAV2D,
    probes: Sequence[Mapping[str, Any]],
    config: HybridAStarConfig,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    records = []
    for index, probe in enumerate(probes):
        state = np.asarray(probe["state"], dtype=np.float32)
        env.reset(
            seed=int(seed) + index * 104_729,
            options={"device_id": str(probe["device_id"]), "start": state},
        )
        controller = HybridAStarController(config)
        diagnostics = controller.begin_episode(
            env,
            env.abstract_goal_observation(),
            seed=int(seed) + index * 104_729,
        )
        first_action = (
            float(controller._actions[0][0])
            if diagnostics["planner_success"] and controller._actions
            else None
        )
        records.append(
            {
                **dict(probe),
                "oracle_solved": bool(diagnostics["planner_success"]),
                "oracle_cost": (
                    float(diagnostics["planned_cost"])
                    if diagnostics["planner_success"]
                    else None
                ),
                "oracle_first_action": first_action,
                "oracle_first_action_coarse": _coarse_action(first_action, env.omega_max),
                "oracle_planning_time_sec": float(diagnostics["initial_planning_time_sec"]),
                "oracle_expanded_nodes": int(diagnostics["expanded_nodes"]),
                "oracle_failure_reason": str(diagnostics.get("planner_failure_reason", "")),
            }
        )
    return records


def evaluate_checkpoint(
    checkpoint: str,
    scenario: Mapping[str, Any],
    oracle_records: Sequence[Mapping[str, Any]],
    *,
    device_string: str,
    num_critics: int,
    seed: int,
    checkpoint_index: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    import argparse as _argparse

    device = auto_device(device_string)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    adapter_args = _argparse.Namespace(
        _scenario_data=scenario,
        env_name=f"u_trap_local_nav_{os.getpid()}_{checkpoint_index}",
        checkpoint=str(checkpoint),
        num_critics=int(num_critics),
        seed=int(seed),
    )
    agent, optim_steps = build_qrl_adapter(adapter_args, device, env)
    checkpoint_path = Path(checkpoint)
    checkpoint_id = f"{checkpoint_path.parent.parent.name}:{checkpoint_path.stem}"
    candidate_scales = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    records = []
    for index, oracle_record in enumerate(oracle_records):
        state = np.asarray(oracle_record["state"], dtype=np.float32)
        env.reset(
            seed=int(seed) + index * 104_729,
            options={"device_id": str(oracle_record["device_id"]), "start": state},
        )
        goal = env.abstract_goal_observation().astype(np.float32)
        current_value = float(agent.value(env.state_to_observation(state), goal))
        base_state = env.get_state()
        action_candidates = []
        for scale in candidate_scales:
            env.set_state(base_state)
            omega = float(scale) * float(env.omega_max)
            next_obs, reward, _terminated, _truncated, info = env.step(
                np.asarray([omega], dtype=np.float32)
            )
            invalid = bool(info.get("collision", False) or info.get("out_of_bounds", False))
            successor_value = None if invalid else float(agent.value(next_obs, goal))
            immediate_cost = float(info.get("cost_total", -float(reward)))
            score = None if invalid else immediate_cost + float(successor_value)
            action_candidates.append(
                {
                    "omega": omega,
                    "coarse": _coarse_action(omega, env.omega_max),
                    "immediate_cost": immediate_cost,
                    "successor_value": successor_value,
                    "score": score,
                    "invalid": invalid,
                }
            )
        env.set_state(base_state)
        valid_candidates = [item for item in action_candidates if item["score"] is not None]
        best = min(valid_candidates, key=lambda item: float(item["score"])) if valid_candidates else None
        oracle_action = oracle_record.get("oracle_first_action")
        qrl_action = None if best is None else float(best["omega"])
        exact_match = None
        coarse_match = None
        if oracle_action is not None and qrl_action is not None:
            exact_match = bool(abs(float(oracle_action) - qrl_action) <= 1e-6)
            coarse_match = bool(
                _coarse_action(float(oracle_action), env.omega_max)
                == _coarse_action(qrl_action, env.omega_max)
            )
        records.append(
            {
                **dict(oracle_record),
                "checkpoint_id": checkpoint_id,
                "qrl_value": current_value,
                "qrl_first_action": qrl_action,
                "qrl_first_action_coarse": _coarse_action(qrl_action, env.omega_max),
                "qrl_action_exact_match": exact_match,
                "qrl_action_coarse_match": coarse_match,
                "qrl_action_candidates": action_candidates,
            }
        )
    summary = summarize_checkpoint_records(records)
    summary.update(
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_id": checkpoint_id,
            "optim_steps": None if optim_steps is None else int(optim_steps),
        }
    )
    return checkpoint_id, records, summary


def _write_csv(path: Path, checkpoint_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fieldnames = [
        "checkpoint_id", "probe_id", "position_label", "position_index",
        "heading_label", "state", "oracle_solved", "oracle_cost",
        "qrl_value", "oracle_first_action", "qrl_first_action",
        "oracle_first_action_coarse", "qrl_first_action_coarse",
        "qrl_action_exact_match", "qrl_action_coarse_match",
        "oracle_planning_time_sec", "oracle_expanded_nodes", "oracle_failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for records in checkpoint_records.values():
            for record in records:
                row = {key: record.get(key) for key in fieldnames}
                row["state"] = json.dumps(row["state"])
                writer.writerow(row)


def _plot_results(
    path: Path,
    checkpoint_records: Mapping[str, Sequence[Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    all_oracle = []
    for checkpoint_id, records in checkpoint_records.items():
        solved = [record for record in records if record.get("oracle_cost") is not None]
        oracle = np.asarray([record["oracle_cost"] for record in solved], dtype=np.float64)
        predicted = np.asarray([record["qrl_value"] for record in solved], dtype=np.float64)
        axes[0].scatter(oracle, predicted, s=32, alpha=0.8, label=checkpoint_id)
        all_oracle.extend(oracle.tolist())
        west = sorted(
            [record for record in solved if record["heading_label"] == "west"],
            key=lambda record: int(record["position_index"]),
        )
        axes[1].plot(
            [record["position_label"] for record in west],
            [record["qrl_value"] for record in west],
            marker="o",
            label=checkpoint_id,
        )
    if all_oracle:
        lower, upper = min(all_oracle), max(all_oracle)
        axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1, label="ideal")
    axes[0].set_title("Local QRL value vs Hybrid A* cost")
    axes[0].set_xlabel("Hybrid A* cost-to-go")
    axes[0].set_ylabel("QRL value")
    axes[0].legend(fontsize=8)

    first_records = next(iter(checkpoint_records.values()))
    oracle_west = sorted(
        [record for record in first_records if record["heading_label"] == "west" and record.get("oracle_cost") is not None],
        key=lambda record: int(record["position_index"]),
    )
    axes[1].plot(
        [record["position_label"] for record in oracle_west],
        [record["oracle_cost"] for record in oracle_west],
        "k--o",
        label="Hybrid A*",
    )
    axes[1].set_title("West-heading U centerline")
    axes[1].set_xlabel("deep → exit")
    axes[1].set_ylabel("cost-to-go / value")
    axes[1].legend(fontsize=8)

    labels = list(summaries)
    x = np.arange(len(labels), dtype=np.float64)
    ranking = [summaries[label].get("pairwise_ranking_accuracy") or 0.0 for label in labels]
    actions = [summaries[label].get("first_action_coarse_accuracy") or 0.0 for label in labels]
    axes[2].bar(x - 0.18, ranking, width=0.36, label="pairwise ranking")
    axes[2].bar(x + 0.18, actions, width=0.36, label="coarse first action")
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Local navigability accuracy")
    axes[2].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260802)
    # Keep the benchmark controller's discretization and admissible weight=1,
    # but raise the search/time budgets for this offline diagnostic oracle.
    parser.add_argument("--astar-position-resolution", type=float, default=0.25)
    parser.add_argument("--astar-heading-bins", type=int, default=24)
    parser.add_argument("--astar-primitive-steps", type=int, default=5)
    parser.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    parser.add_argument("--astar-max-expansions", type=int, default=200_000)
    parser.add_argument("--astar-timeout-sec", type=float, default=120.0)
    parser.add_argument("--astar-terminal-samples", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenario = load_scenario_config(args.scenario_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = build_probe_records(scenario)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    oracle_config = HybridAStarConfig(
        position_resolution=float(args.astar_position_resolution),
        heading_bins=int(args.astar_heading_bins),
        primitive_steps=int(args.astar_primitive_steps),
        heuristic_weight=float(args.astar_heuristic_weight),
        max_expansions=int(args.astar_max_expansions),
        timeout_sec=float(args.astar_timeout_sec),
        terminal_samples=int(args.astar_terminal_samples),
    )
    oracle_records = compute_oracle_records(env, probes, oracle_config, seed=int(args.seed))
    checkpoint_records = {}
    summaries = {}
    for checkpoint_index, checkpoint in enumerate(args.checkpoints):
        checkpoint_id, records, summary = evaluate_checkpoint(
            checkpoint,
            scenario,
            oracle_records,
            device_string=str(args.device),
            num_critics=int(args.num_critics),
            seed=int(args.seed),
            checkpoint_index=checkpoint_index,
        )
        checkpoint_records[checkpoint_id] = records
        summaries[checkpoint_id] = summary
    payload = {
        "schema_version": 1,
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "probe_definition": scenario["metadata"]["u_trap_local_navigability_probes"],
        "oracle_config": vars(args),
        "oracle_records": oracle_records,
        "summaries": summaries,
        "checkpoint_records": checkpoint_records,
    }
    json_path = output_dir / "u_trap_local_navigability.json"
    csv_path = output_dir / "u_trap_local_navigability.csv"
    plot_path = output_dir / "u_trap_local_navigability.png"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    _write_csv(csv_path, checkpoint_records)
    _plot_results(plot_path, checkpoint_records, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
