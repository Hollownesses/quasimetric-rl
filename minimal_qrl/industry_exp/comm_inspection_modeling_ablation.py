#!/usr/bin/env python3
"""Run a 2x2 ablation of goal-set/point guidance and communication-aware/unaware planning."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from minimal_qrl.industry_exp.comm_inspection_experiment_common import (
    add_environment_arguments,
    add_mppi_arguments,
    bootstrap_mean_ci,
    make_env,
    metric_summary,
    write_json,
)
from minimal_qrl.baselines import MPPIConfig, MPPIController, rollout_controller_episode
from minimal_qrl.eval.comm_inspection_baseline_eval import _rollout_record


METRICS = (
    "success",
    "num_steps",
    "total_cost",
    "cost_per_step",
    "observation_feasible_ratio",
    "communication_feasible_ratio",
    "first_task_feasible_step",
    "final_obs_margin",
    "final_comm_margin",
    "collision",
    "out_of_bounds",
    "first_decision_time_sec",
    "planning_time_p95_sec",
)


class FactorialMPPIController(MPPIController):
    """Model-MPPI with independently controlled goal samples and planning communication model."""

    def __init__(
        self,
        cfg: MPPIConfig,
        *,
        condition_name: str,
        communication_aware: bool,
    ) -> None:
        super().__init__(cfg, terminal_mode="model")
        self.name = str(condition_name)
        self.communication_aware = bool(communication_aware)

    @contextmanager
    def _planning_model(self, env) -> Iterator[None]:
        if self.communication_aware:
            yield
            return
        names = (
            "comm_threshold",
            "require_ground_station_los",
            "communication_break_cost",
            "communication_violation_cost_weight",
        )
        original = {name: getattr(env, name) for name in names}
        try:
            env.comm_threshold = -1_000_000.0
            env.require_ground_station_los = False
            env.communication_break_cost = 0.0
            env.communication_violation_cost_weight = 0.0
            yield
        finally:
            for name, value in original.items():
                setattr(env, name, value)

    def begin_episode(self, env, goal_obs: np.ndarray, seed: int) -> dict[str, Any]:
        with self._planning_model(env):
            diagnostics = super().begin_episode(env, goal_obs, seed)
        diagnostics.update(
            {
                "communication_aware_planning": self.communication_aware,
                "goal_terminal_sample_count": int(self.cfg.terminal_samples),
            }
        )
        return diagnostics

    def act(self, obs: np.ndarray, env):
        with self._planning_model(env):
            return super().act(obs, env)


def _condition_definitions(args) -> dict[str, tuple[str, str, MPPIConfig]]:
    base = MPPIConfig(
        horizon=int(args.mppi_horizon),
        num_samples=int(args.mppi_num_samples),
        noise_sigma=float(args.mppi_noise_sigma),
        temperature=float(args.mppi_temperature),
        terminal_weight=float(args.mppi_terminal_weight),
        terminal_samples=int(args.mppi_terminal_samples),
    )
    if int(args.mppi_terminal_samples) < 2:
        raise ValueError("--mppi-terminal-samples must be >= 2 for the set condition")
    return {
        "goal_set_comm_aware": ("set", "aware", base),
        "point_goal_comm_aware": ("point", "aware", replace(base, terminal_samples=1)),
        "goal_set_comm_unaware": ("set", "unaware", base),
        "point_goal_comm_unaware": ("point", "unaware", replace(base, terminal_samples=1)),
    }


def _summarize(records: list[dict], args) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    conditions = list(dict.fromkeys(str(row["condition"]) for row in records))
    for condition in conditions:
        rows = [row for row in records if row["condition"] == condition]
        metrics = {
            metric: metric_summary(
                [float(row[metric]) for row in rows],
                seed=int(args.seed) + sum(map(ord, condition + metric)),
                bootstrap_samples=int(args.bootstrap_samples),
            )
            for metric in METRICS
        }
        per_device = {}
        for device_id in sorted({str(row["device_id"]) for row in rows}):
            selected = [row for row in rows if row["device_id"] == device_id]
            per_device[device_id] = {
                "count": len(selected),
                "success_rate": float(np.mean([float(row["success"]) for row in selected])),
                "communication_feasible_ratio": float(
                    np.mean([float(row["communication_feasible_ratio"]) for row in selected])
                ),
                "avg_total_cost": float(np.mean([float(row["total_cost"]) for row in selected])),
            }
        summary[condition] = {
            "goal_representation": rows[0]["goal_representation"],
            "communication_planning": rows[0]["communication_planning"],
            "num_records": len(rows),
            "metrics": metrics,
            "per_device": per_device,
        }

    reference_name = "goal_set_comm_aware"
    reference = {
        (str(row["device_id"]), int(row["episode_seed"])): row
        for row in records
        if row["condition"] == reference_name
    }
    paired = {}
    for condition in conditions:
        if condition == reference_name:
            continue
        selected = {
            (str(row["device_id"]), int(row["episode_seed"])): row
            for row in records
            if row["condition"] == condition
        }
        keys = sorted(set(reference) & set(selected))
        condition_results = {}
        for metric in ("success", "total_cost", "num_steps", "communication_feasible_ratio", "collision"):
            differences = [
                float(selected[key][metric]) - float(reference[key][metric]) for key in keys
            ]
            condition_results[f"{metric}_difference_vs_{reference_name}"] = {
                "mean": float(np.mean(differences)) if differences else None,
                "bootstrap_95_ci": bootstrap_mean_ci(
                    differences,
                    seed=int(args.seed) + sum(map(ord, condition + metric)),
                    samples=int(args.bootstrap_samples),
                )
                if differences
                else [None, None],
                "num_pairs": len(keys),
            }
        paired[condition] = condition_results
    summary["paired_comparisons"] = paired
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: dict[str, Any], path: Path) -> None:
    conditions = [key for key in summary if key != "paired_comparisons"]
    labels = [
        "Set\nComm-aware",
        "Point\nComm-aware",
        "Set\nComm-unaware",
        "Point\nComm-unaware",
    ]
    ordered = [
        "goal_set_comm_aware",
        "point_goal_comm_aware",
        "goal_set_comm_unaware",
        "point_goal_comm_unaware",
    ]
    ordered = [name for name in ordered if name in conditions]
    labels = labels[: len(ordered)]
    success = [100.0 * summary[name]["metrics"]["success"]["mean"] for name in ordered]
    comm = [
        100.0 * summary[name]["metrics"]["communication_feasible_ratio"]["mean"]
        for name in ordered
    ]
    cost = [summary[name]["metrics"]["total_cost"]["mean"] for name in ordered]
    x = np.arange(len(ordered))
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#eeca3b"][: len(ordered)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    axes[0].bar(x, success, color=colors)
    axes[0].set_ylabel("Task success rate (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(x, comm, color=colors)
    axes[1].set_ylabel("Communication-feasible time (%)")
    axes[1].set_ylim(0, 100)
    axes[2].bar(x, cost, color=colors)
    axes[2].set_ylabel("Average total cost")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/experiments/comm_inspection_modeling_ablation",
    )
    parser.add_argument("--starts-per-device", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260712)
    add_environment_arguments(parser)
    add_mppi_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.starts_per_device <= 0:
        raise ValueError("--starts-per-device must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_env = make_env(args)
    episode_specs = [
        (device_id, int(args.seed) + device_index * 1_000_003 + start_index)
        for device_index, device_id in enumerate(spec_env.device_ids)
        for start_index in range(int(args.starts_per_device))
    ]
    conditions = _condition_definitions(args)
    records: list[dict] = []
    partial_path = output_dir / "ablation_results.partial.csv"
    partial_handle = open(partial_path, "w", encoding="utf-8", newline="")
    partial_writer = None
    try:
        for condition, (goal_representation, communication_planning, cfg) in conditions.items():
            env = make_env(args)
            controller = FactorialMPPIController(
                cfg,
                condition_name=condition,
                communication_aware=communication_planning == "aware",
            )
            successes = 0
            for episode_index, (device_id, episode_seed) in enumerate(episode_specs, start=1):
                rollout = rollout_controller_episode(
                    controller,
                    env,
                    episode_seed=episode_seed,
                    device_id=device_id,
                )
                record = _rollout_record(condition, rollout, model_run="factorial_model_mppi")
                record.update(
                    {
                        "condition": condition,
                        "goal_representation": goal_representation,
                        "communication_planning": communication_planning,
                        "terminal_samples": int(cfg.terminal_samples),
                    }
                )
                records.append(record)
                if partial_writer is None:
                    partial_writer = csv.DictWriter(
                        partial_handle,
                        fieldnames=list(record.keys()),
                        extrasaction="ignore",
                    )
                    partial_writer.writeheader()
                partial_writer.writerow(record)
                partial_handle.flush()
                successes += int(bool(record["success"]))
                if episode_index % max(1, len(episode_specs) // 10) == 0:
                    print(
                        f"[{condition}] {episode_index}/{len(episode_specs)} "
                        f"success={successes / episode_index:.3f}",
                        flush=True,
                    )
    finally:
        partial_handle.close()

    summary = _summarize(records, args)
    _write_csv(output_dir / "ablation_results.csv", records)
    write_json(
        output_dir / "ablation_results.json",
        {
            "experiment": "goal_representation_x_communication_planning_ablation",
            "episode_specs": [
                {"device_id": device_id, "episode_seed": episode_seed}
                for device_id, episode_seed in episode_specs
            ],
            "summary": summary,
            "episode_results": records,
            "config": vars(args),
            "notes": {
                "goal_set": "approximated by mppi_terminal_samples feasible terminal samples",
                "point_goal": "the same controller uses exactly one feasible terminal sample",
                "communication_unaware": "communication threshold, LOS requirement, and communication costs are disabled only inside the planning model",
                "execution": "all conditions execute in the original full environment and use the original joint task-success criterion",
            },
        },
    )
    _plot(summary, output_dir / "ablation_summary.png")
    print(f"Saved modeling ablation to {output_dir}")


if __name__ == "__main__":
    main()
