#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minimal_qrl.baselines import (  # noqa: E402
    GoalSetSACConfig,
    HybridAStarConfig,
    HybridAStarController,
    MPPIConfig,
    MPPIController,
    PolicyController,
    load_goal_set_sac_checkpoint,
    rollout_controller_episode,
    train_goal_set_sac,
)
from minimal_qrl.eval.comm_inspection_execution_eval import (  # noqa: E402
    VisualizationConfig,
    _save_rollout_visualization,
    build_qrl_adapter,
    make_comm_inspection_env,
)
from minimal_qrl.eval.utils import auto_device  # noqa: E402


METHODS = {"hybrid_astar", "mppi_no_terminal", "model_mppi", "goal_set_sac", "qrl_greedy", "qrl_mppi"}
SCALAR_METRICS = (
    "success",
    "num_steps",
    "total_cost",
    "cost_per_step",
    "observation_feasible_ratio",
    "communication_feasible_ratio",
    "task_feasible_ratio",
    "collision",
    "out_of_bounds",
    "first_task_feasible_step",
    "final_obs_margin",
    "final_comm_margin",
    "final_task_score",
    "total_planning_time_sec",
    "expanded_nodes",
    "model_rollouts",
    "training_env_steps",
)


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"Unknown baseline methods: {sorted(unknown)}")
    return methods


def _rollout_record(method: str, rollout: Dict[str, Any], model_run: str) -> Dict[str, Any]:
    infos = list(rollout["step_infos"])
    final_info = rollout["final_info"]
    steps = int(rollout["num_steps"])
    total_cost = float(-np.sum(np.asarray(rollout["rewards"], dtype=np.float64)))
    first_step = final_info.get("first_task_feasible_step")
    diagnostics = dict(rollout.get("controller_diagnostics", {}))
    return {
        "method": method,
        "model_run": str(model_run),
        "episode_seed": int(rollout["seed"]),
        "success": float(bool(rollout["success"])),
        "num_steps": float(steps),
        "total_cost": total_cost,
        "cost_per_step": total_cost / max(steps, 1),
        "observation_feasible_ratio": float(np.mean([
            bool(info.get("observation_feasible", False)) for info in infos
        ])) if infos else 0.0,
        "communication_feasible_ratio": float(np.mean([
            bool(info.get("communication_feasible", False)) for info in infos
        ])) if infos else 0.0,
        "task_feasible_ratio": float(np.mean([
            bool(info.get("task_feasible", False)) for info in infos
        ])) if infos else 0.0,
        "collision": float(bool(rollout["collision"])),
        "out_of_bounds": float(bool(rollout["out_of_bounds"])),
        "first_task_feasible_step": float(first_step) if first_step is not None else float(steps),
        "final_obs_margin": float(final_info.get("obs_margin", 0.0)),
        "final_comm_margin": float(final_info.get("comm_margin", 0.0)),
        "final_task_score": float(final_info.get("task_score", 0.0)),
        "total_planning_time_sec": float(diagnostics.get("total_planning_time_sec", 0.0)),
        "expanded_nodes": float(diagnostics.get("expanded_nodes", 0.0)),
        "model_rollouts": float(diagnostics.get("model_rollouts", 0.0)),
        "training_env_steps": float(diagnostics.get("training_env_steps", 0.0)),
        "planner_failure_reason": str(diagnostics.get("planner_failure_reason", "")),
        "start": [float(v) for v in rollout["start"]],
        "inspection_target": [float(v) for v in rollout["inspection_target"]],
        "ground_station": [float(v) for v in rollout["ground_station"]],
    }


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0]
    samples = rng.choice(values, size=(max(1, int(n_boot)), values.size), replace=True)
    means = samples.mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _summarize(records: list[Dict[str, Any]], *, seed: int, n_boot: int) -> Dict[str, Any]:
    by_method: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[str(record["method"])].append(record)
    summary: Dict[str, Any] = {}
    for method, method_records in by_method.items():
        rng = np.random.default_rng(int(seed) + sum(ord(ch) for ch in method))
        metrics: Dict[str, Any] = {"num_records": len(method_records)}
        for key in SCALAR_METRICS:
            values = np.asarray([float(row[key]) for row in method_records], dtype=np.float64)
            metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "bootstrap_95_ci": _bootstrap_ci(values, rng, n_boot),
            }
        failure_reasons: Dict[str, int] = defaultdict(int)
        for row in method_records:
            reason = str(row.get("planner_failure_reason", ""))
            if reason:
                failure_reasons[reason] += 1
        metrics["planner_failure_reasons"] = dict(failure_reasons)
        summary[method] = metrics

    reference = by_method.get("qrl_greedy", [])
    if reference:
        reference_by_seed: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
        for row in reference:
            reference_by_seed[int(row["episode_seed"])].append(row)
        paired: Dict[str, Any] = {}
        for method, method_records in by_method.items():
            if method == "qrl_greedy":
                continue
            method_by_seed: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
            for row in method_records:
                method_by_seed[int(row["episode_seed"])].append(row)
            common = sorted(set(reference_by_seed) & set(method_by_seed))
            if not common:
                continue
            rng = np.random.default_rng(int(seed) + 17 + sum(ord(ch) for ch in method))
            paired[method] = {}
            for key in ("success", "total_cost", "num_steps", "collision"):
                differences = np.asarray([
                    np.mean([float(row[key]) for row in method_by_seed[s]])
                    - np.mean([float(row[key]) for row in reference_by_seed[s]])
                    for s in common
                ])
                paired[method][f"{key}_difference_vs_qrl_greedy"] = {
                    "mean": float(np.mean(differences)),
                    "bootstrap_95_ci": _bootstrap_ci(differences, rng, n_boot),
                    "num_pairs": len(common),
                }
        summary["paired_comparisons"] = paired
    return summary


def _save_visualization(
    env,
    rollout: Dict[str, Any],
    method: str,
    episode_index: int,
    output_dir: Path,
    viz_cfg: VisualizationConfig,
    counters: Dict[str, Dict[str, int]],
) -> None:
    if not viz_cfg.save_visualizations:
        return
    category = "success" if rollout["success"] else "failure"
    limit = viz_cfg.max_successes if category == "success" else viz_cfg.max_failures
    if counters[method][category] >= int(limit):
        return
    category_dir = output_dir / "visualizations" / method / category
    category_dir.mkdir(parents=True, exist_ok=True)
    _save_rollout_visualization(
        env,
        rollout,
        execution_mode=method,
        episode_index=episode_index,
        category=category,
        category_dir=category_dir,
        base_output_dir=output_dir,
        viz_cfg=viz_cfg,
    )
    counters[method][category] += 1


def _evaluate_controller(
    method: str,
    controller,
    env,
    episode_seeds: Iterable[int],
    *,
    model_run: str,
    output_dir: Path,
    viz_cfg: VisualizationConfig,
    counters: Dict[str, Dict[str, int]],
) -> list[Dict[str, Any]]:
    records = []
    for index, episode_seed in enumerate(episode_seeds):
        rollout = rollout_controller_episode(
            controller, env, episode_seed=int(episode_seed)
        )
        records.append(_rollout_record(method, rollout, model_run))
        _save_visualization(
            env, rollout, method, index, output_dir, viz_cfg, counters
        )
    return records


def _write_csv(path: Path, records: list[Dict[str, Any]]) -> None:
    scalar_fields = [
        "method", "model_run", "episode_seed", *SCALAR_METRICS, "planner_failure_reason"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _stage_defaults(stage: str) -> tuple[int, int, list[int]]:
    if stage == "smoke":
        return 3, 200, [0]
    if stage == "pilot":
        return 50, 300_000, [0]
    return 300, 300_000, [0, 1, 2, 3, 4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified goal-set communication-inspection baselines")
    parser.add_argument("--stage", choices=["smoke", "pilot", "final"], default="smoke")
    parser.add_argument("--methods", default=",".join(sorted(METHODS)))
    parser.add_argument("--output-dir", default="results/comm_inspection_baselines")
    parser.add_argument("--qrl-checkpoints", nargs="*", default=[])
    parser.add_argument("--sac-checkpoints", nargs="*", default=[])
    parser.add_argument("--train-sac", action="store_true")
    parser.add_argument("--sac-seeds", default=None)
    parser.add_argument("--sac-total-env-steps", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--viz-max-successes", type=int, default=3)
    parser.add_argument("--viz-max-failures", type=int, default=3)

    parser.add_argument("--bounds", type=float, nargs=4, default=[0.0, 0.0, 10.0, 10.0])
    parser.add_argument("--omega-max", type=float, default=3.0)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--obstacle-config", choices=["none", "simple", "medium", "hard"], default="medium")
    parser.add_argument("--obstacles", type=float, nargs="*", default=None)
    parser.add_argument("--observation-mode", default="task_context", choices=["task_context", "cos_sin", "state"])
    parser.add_argument("--inspection-target", type=float, nargs=2, default=[3.0, 7.5])
    parser.add_argument("--ground-station", type=float, nargs=2, default=[1.5, 2.0])
    parser.add_argument("--randomize-inspection-target", action="store_true", default=False)
    parser.add_argument("--randomize-ground-station", action="store_true", default=False)
    parser.add_argument("--observation-radius", type=float, default=1.8)
    parser.add_argument("--fov-angle", type=float, default=float(np.pi / 2.0))
    parser.add_argument("--require-target-los", action="store_true", default=True)
    parser.add_argument("--no-require-target-los", dest="require_target_los", action="store_false")
    parser.add_argument("--comm-alpha", type=float, default=2.0)
    parser.add_argument("--comm-bias", type=float, default=5.0)
    parser.add_argument("--comm-occlusion-penalty", type=float, default=6.0)
    parser.add_argument("--comm-threshold", type=float, default=0.5)
    parser.add_argument("--require-ground-station-los", action="store_true")
    parser.add_argument("--goal-sampling-mode", default="task_feasible", choices=["task_feasible", "valid"])
    parser.add_argument("--goal-position-tolerance", type=float, default=0.25)
    parser.add_argument("--goal-heading-tolerance", type=float, default=0.3)
    parser.add_argument("--collision-cost", type=float, default=10.0)
    parser.add_argument("--out-of-bounds-cost", type=float, default=10.0)
    parser.add_argument("--communication-break-cost", type=float, default=1.0)
    parser.add_argument("--observation-violation-cost-weight", type=float, default=1.0)
    parser.add_argument("--communication-violation-cost-weight", type=float, default=0.5)
    parser.add_argument("--observation-failure-cost", type=float, default=0.25)
    parser.add_argument("--taskscore-beta-obs", type=float, default=1.0)
    parser.add_argument("--taskscore-beta-comm", type=float, default=1.0)
    parser.add_argument("--taskscore-beta-feas", type=float, default=0.5)
    parser.add_argument("--taskscore-margin-clip", type=float, default=2.0)
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--env-name", default="comm_inspection_unified_baselines")

    parser.add_argument("--astar-position-resolution", type=float, default=0.25)
    parser.add_argument("--astar-heading-bins", type=int, default=24)
    parser.add_argument("--astar-primitive-steps", type=int, default=5)
    parser.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    parser.add_argument("--astar-max-expansions", type=int, default=50_000)
    parser.add_argument("--astar-timeout-sec", type=float, default=30.0)
    parser.add_argument("--astar-terminal-samples", type=int, default=128)

    parser.add_argument("--mppi-horizon", type=int, default=20)
    parser.add_argument("--mppi-num-samples", type=int, default=256)
    parser.add_argument("--mppi-noise-sigma", type=float, default=0.8)
    parser.add_argument("--mppi-temperature", type=float, default=1.0)
    parser.add_argument("--mppi-terminal-weight", type=float, default=1.0)
    parser.add_argument("--mppi-terminal-samples", type=int, default=128)

    parser.add_argument("--sac-batch-size", type=int, default=256)
    parser.add_argument("--sac-start-random-steps", type=int, default=5_000)
    parser.add_argument("--sac-hidden-dim", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(args.methods)
    stage_trials, stage_steps, stage_seeds = _stage_defaults(args.stage)
    n_trials = stage_trials if args.n_trials is None else int(args.n_trials)
    sac_steps = stage_steps if args.sac_total_env_steps is None else int(args.sac_total_env_steps)
    sac_seeds = stage_seeds if args.sac_seeds is None else [
        int(value) for value in str(args.sac_seeds).split(",") if value.strip()
    ]
    device = auto_device(args.device)
    episode_seeds = [int(args.seed) + i for i in range(n_trials)]
    viz_cfg = VisualizationConfig(
        save_visualizations=bool(args.save_visualizations),
        max_successes=int(args.viz_max_successes),
        max_failures=int(args.viz_max_failures),
        save_gif=False,
        gif_fps=8,
    )
    counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    records: list[Dict[str, Any]] = []

    astar_cfg = HybridAStarConfig(
        position_resolution=args.astar_position_resolution,
        heading_bins=args.astar_heading_bins,
        primitive_steps=args.astar_primitive_steps,
        heuristic_weight=args.astar_heuristic_weight,
        max_expansions=args.astar_max_expansions,
        timeout_sec=args.astar_timeout_sec,
        terminal_samples=args.astar_terminal_samples,
    )
    mppi_cfg = MPPIConfig(
        horizon=args.mppi_horizon,
        num_samples=args.mppi_num_samples,
        noise_sigma=args.mppi_noise_sigma,
        temperature=args.mppi_temperature,
        terminal_weight=args.mppi_terminal_weight,
        terminal_samples=args.mppi_terminal_samples,
    )

    if "hybrid_astar" in methods:
        env = make_comm_inspection_env(args)
        records.extend(_evaluate_controller(
            "hybrid_astar", HybridAStarController(astar_cfg), env, episode_seeds,
            model_run="model", output_dir=output_dir, viz_cfg=viz_cfg, counters=counters,
        ))
    if "mppi_no_terminal" in methods:
        env = make_comm_inspection_env(args)
        records.extend(_evaluate_controller(
            "mppi_no_terminal", MPPIController(mppi_cfg, terminal_mode="none"), env, episode_seeds,
            model_run="model", output_dir=output_dir, viz_cfg=viz_cfg, counters=counters,
        ))
    if "model_mppi" in methods:
        env = make_comm_inspection_env(args)
        records.extend(_evaluate_controller(
            "model_mppi", MPPIController(mppi_cfg, terminal_mode="model"), env, episode_seeds,
            model_run="model", output_dir=output_dir, viz_cfg=viz_cfg, counters=counters,
        ))

    qrl_methods = {"qrl_greedy", "qrl_mppi"} & set(methods)
    if qrl_methods and not args.qrl_checkpoints:
        raise ValueError("QRL methods require --qrl-checkpoints")
    for qrl_index, checkpoint in enumerate(args.qrl_checkpoints):
        if not qrl_methods:
            break
        env = make_comm_inspection_env(args)
        qrl_args = argparse.Namespace(**vars(args))
        qrl_args.checkpoint = checkpoint
        qrl_args.env_name = f"{args.env_name}_qrl_{qrl_index}"
        qrl_agent, _ = build_qrl_adapter(qrl_args, device, env)
        model_run = f"qrl_{qrl_index}"
        if "qrl_greedy" in methods:
            records.extend(_evaluate_controller(
                "qrl_greedy", PolicyController(qrl_agent, name="qrl_greedy"), env,
                episode_seeds, model_run=model_run, output_dir=output_dir,
                viz_cfg=viz_cfg, counters=counters,
            ))
        if "qrl_mppi" in methods:
            records.extend(_evaluate_controller(
                "qrl_mppi", MPPIController(mppi_cfg, terminal_mode="qrl", qrl_agent=qrl_agent),
                env, episode_seeds, model_run=model_run, output_dir=output_dir,
                viz_cfg=viz_cfg, counters=counters,
            ))

    sac_paths = [Path(path) for path in args.sac_checkpoints]
    if "goal_set_sac" in methods and args.train_sac:
        sac_cfg = GoalSetSACConfig(
            hidden_dim=args.sac_hidden_dim,
            batch_size=args.sac_batch_size,
            total_env_steps=sac_steps,
            start_random_steps=min(args.sac_start_random_steps, sac_steps),
            checkpoint_interval=max(sac_steps, 1),
        )
        for sac_seed in sac_seeds:
            train_env = make_comm_inspection_env(args)
            run_dir = output_dir / "goal_set_sac" / f"seed_{sac_seed}"
            train_goal_set_sac(train_env, sac_cfg, device, run_dir, seed=sac_seed)
            sac_paths.append(run_dir / "checkpoint_final.pth")
    if "goal_set_sac" in methods and not sac_paths:
        raise ValueError("goal_set_sac requires --train-sac or --sac-checkpoints")
    for sac_index, checkpoint in enumerate(sac_paths):
        if "goal_set_sac" not in methods:
            break
        env = make_comm_inspection_env(args)
        sac_agent, metadata = load_goal_set_sac_checkpoint(checkpoint, env, device)
        records.extend(_evaluate_controller(
            "goal_set_sac", PolicyController(
                sac_agent,
                name="goal_set_sac",
                static_diagnostics={"training_env_steps": int(metadata["env_steps"])},
            ), env,
            episode_seeds, model_run=f"sac_seed_{metadata['seed']}_{sac_index}",
            output_dir=output_dir, viz_cfg=viz_cfg, counters=counters,
        ))

    summary = _summarize(records, seed=args.seed, n_boot=args.bootstrap_samples)
    payload = {
        "stage": args.stage,
        "methods": methods,
        "episode_seeds": episode_seeds,
        "qrl_checkpoints": args.qrl_checkpoints,
        "sac_checkpoints": [str(path) for path in sac_paths],
        "summary": summary,
        "episode_results": records,
        "config": vars(args),
    }
    with open(output_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_csv(output_dir / "baseline_results.csv", records)
    print(f"Saved {output_dir / 'baseline_results.json'}")
    print(f"Saved {output_dir / 'baseline_results.csv'}")


if __name__ == "__main__":
    main()
