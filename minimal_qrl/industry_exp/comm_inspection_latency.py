#!/usr/bin/env python3
"""Benchmark QRL repeated cost-to-go queries and end-to-end first-decision latency."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from minimal_qrl.industry_exp.comm_inspection_experiment_common import (
    add_astar_arguments,
    add_environment_arguments,
    add_mppi_arguments,
    make_env,
    metric_summary,
    parse_int_list,
    write_json,
)
from minimal_qrl.baselines import (
    HybridAStarConfig,
    HybridAStarController,
    MPPIConfig,
    MPPIController,
)
from minimal_qrl.eval.comm_inspection_execution_eval import build_qrl_adapter
from minimal_qrl.eval.utils import auto_device


ALLOWED_CONTROLLER_METHODS = {"qrl_mppi", "mppi_no_terminal", "hybrid_astar"}


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _query_pool(env, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    observations = []
    goals = []
    for index in range(int(size)):
        device_id = env.device_ids[index % len(env.device_ids)]
        obs, _ = env.reset(
            seed=int(seed) + index * 101,
            options={"device_id": device_id},
        )
        observations.append(np.asarray(obs, dtype=np.float32))
        goals.append(env.abstract_goal_observation().astype(np.float32))
    return np.asarray(observations), np.asarray(goals)


def _benchmark_queries(
    agent,
    env,
    device: torch.device,
    *,
    batch_sizes: list[int],
    repeats: int,
    warmup: int,
    seed: int,
) -> list[dict]:
    pool_obs, pool_goals = _query_pool(env, max(batch_sizes), seed)
    warm_size = min(max(batch_sizes), 32)
    for _ in range(max(0, int(warmup))):
        agent.batch_value(pool_obs[:warm_size], pool_goals[:warm_size])
    _sync(device)

    rows = []
    for batch_size in batch_sizes:
        observations = pool_obs[:batch_size]
        goals = pool_goals[:batch_size]
        for repeat in range(int(repeats)):
            _sync(device)
            start = perf_counter()
            values = agent.batch_value(observations, goals)
            _sync(device)
            elapsed = perf_counter() - start
            if len(values) != batch_size:
                raise RuntimeError("QRL batch_value returned an unexpected batch dimension")
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "repeat": int(repeat),
                    "latency_sec": float(elapsed),
                    "latency_per_pair_sec": float(elapsed / batch_size),
                }
            )
    return rows


def _build_controllers(args, qrl_agent) -> dict[str, object]:
    mppi_cfg = MPPIConfig(
        horizon=int(args.mppi_horizon),
        num_samples=int(args.mppi_num_samples),
        noise_sigma=float(args.mppi_noise_sigma),
        temperature=float(args.mppi_temperature),
        terminal_weight=float(args.mppi_terminal_weight),
        terminal_samples=int(args.mppi_terminal_samples),
    )
    astar_cfg = HybridAStarConfig(
        position_resolution=float(args.astar_position_resolution),
        heading_bins=int(args.astar_heading_bins),
        primitive_steps=int(args.astar_primitive_steps),
        heuristic_weight=float(args.astar_heuristic_weight),
        max_expansions=int(args.astar_max_expansions),
        timeout_sec=float(args.astar_timeout_sec),
        terminal_samples=int(args.astar_terminal_samples),
    )
    return {
        "qrl_mppi": MPPIController(mppi_cfg, terminal_mode="qrl", qrl_agent=qrl_agent),
        "mppi_no_terminal": MPPIController(mppi_cfg, terminal_mode="none"),
        "hybrid_astar": HybridAStarController(astar_cfg),
    }


def _benchmark_controllers(
    args,
    env,
    device: torch.device,
    controllers: dict[str, object],
    methods: list[str],
) -> list[dict]:
    rows = []
    for method in methods:
        controller = controllers[method]
        for trial in range(int(args.controller_trials)):
            device_id = env.device_ids[trial % len(env.device_ids)]
            episode_seed = int(args.seed) + 1_000_003 * (trial % len(env.device_ids)) + trial
            obs, _ = env.reset(seed=episode_seed, options={"device_id": device_id})
            goal = env.abstract_goal_observation().astype(np.float32)
            _sync(device)
            start = perf_counter()
            begin_diag = controller.begin_episode(env, goal, episode_seed) or {}
            action, action_diag = controller.act(obs, env)
            _sync(device)
            elapsed = perf_counter() - start
            rows.append(
                {
                    "method": method,
                    "trial": int(trial),
                    "device_id": device_id,
                    "episode_seed": int(episode_seed),
                    "first_decision_latency_sec": float(elapsed),
                    "internal_initial_planning_sec": float(
                        begin_diag.get("initial_planning_time_sec", 0.0)
                    ),
                    "internal_action_planning_sec": float(
                        action_diag.get("planning_time_sec", 0.0)
                    ),
                    "planner_success": bool(begin_diag.get("planner_success", True)),
                    "expanded_nodes": int(begin_diag.get("expanded_nodes", 0)),
                    "action": float(np.asarray(action).reshape(-1)[0]),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(query_rows: list[dict], controller_rows: list[dict], output_path: Path) -> None:
    batch_sizes = sorted({int(row["batch_size"]) for row in query_rows})
    total_ms = [
        1000.0
        * float(np.median([row["latency_sec"] for row in query_rows if row["batch_size"] == size]))
        for size in batch_sizes
    ]
    per_pair_us = [
        1e6
        * float(
            np.median(
                [row["latency_per_pair_sec"] for row in query_rows if row["batch_size"] == size]
            )
        )
        for size in batch_sizes
    ]
    methods = list(dict.fromkeys(str(row["method"]) for row in controller_rows))
    controller_ms = [
        [
            1000.0 * float(row["first_decision_latency_sec"])
            for row in controller_rows
            if row["method"] == method
        ]
        for method in methods
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].plot(batch_sizes, total_ms, marker="o")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("QRL query batch size")
    axes[0].set_ylabel("Median batch latency (ms)")
    axes[1].plot(batch_sizes, per_pair_us, marker="o", color="#54a24b")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("QRL query batch size")
    axes[1].set_ylabel("Median latency per pair (microseconds)")
    axes[2].boxplot(controller_ms, tick_labels=methods, showfliers=False)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("End-to-end first-decision latency (ms)")
    axes[2].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/experiments/comm_inspection_latency",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--env-name", default="comm_inspection_latency")
    parser.add_argument("--batch-sizes", default="1,8,24,128,600")
    parser.add_argument("--query-repeats", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--controller-methods", default="qrl_mppi,mppi_no_terminal,hybrid_astar")
    parser.add_argument("--controller-trials", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260712)
    add_environment_arguments(parser)
    add_mppi_arguments(parser)
    add_astar_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.query_repeats <= 0 or args.controller_trials <= 0:
        raise ValueError("query repeats and controller trials must be positive")
    methods = [item.strip() for item in args.controller_methods.split(",") if item.strip()]
    unknown = set(methods) - ALLOWED_CONTROLLER_METHODS
    if unknown:
        raise ValueError(f"unknown controller method(s): {sorted(unknown)}")
    batch_sizes = parse_int_list(args.batch_sizes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = auto_device(args.device)
    env = make_env(args)
    qrl_agent, checkpoint_step = build_qrl_adapter(args, device, env)
    query_rows = _benchmark_queries(
        qrl_agent,
        env,
        device,
        batch_sizes=batch_sizes,
        repeats=int(args.query_repeats),
        warmup=int(args.warmup),
        seed=int(args.seed),
    )
    controllers = _build_controllers(args, qrl_agent)
    controller_rows = _benchmark_controllers(args, env, device, controllers, methods)

    _write_csv(output_dir / "qrl_query_latency.csv", query_rows)
    _write_csv(output_dir / "first_decision_latency.csv", controller_rows)
    query_summary = {
        str(size): {
            "batch_latency_sec": metric_summary(
                [row["latency_sec"] for row in query_rows if row["batch_size"] == size],
                seed=int(args.seed) + size,
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "per_pair_latency_sec": metric_summary(
                [row["latency_per_pair_sec"] for row in query_rows if row["batch_size"] == size],
                seed=int(args.seed) + size + 17,
                bootstrap_samples=int(args.bootstrap_samples),
            ),
        }
        for size in batch_sizes
    }
    controller_summary = {
        method: metric_summary(
            [
                row["first_decision_latency_sec"]
                for row in controller_rows
                if row["method"] == method
            ],
            seed=int(args.seed) + sum(map(ord, method)),
            bootstrap_samples=int(args.bootstrap_samples),
        )
        for method in methods
    }
    write_json(
        output_dir / "latency_results.json",
        {
            "experiment": "online_latency_and_repeated_queries",
            "checkpoint_step": checkpoint_step,
            "device": str(device),
            "query_summary": query_summary,
            "controller_summary": controller_summary,
            "config": vars(args),
            "notes": {
                "timing": "wall-clock with accelerator synchronization before and after each measurement",
                "warmup": "warm-up is excluded from reported QRL query latency",
                "first_decision": "includes controller begin_episode plus first act",
            },
        },
    )
    _plot(query_rows, controller_rows, output_dir / "latency_summary.png")
    print(f"Saved latency experiment to {output_dir}")


if __name__ == "__main__":
    main()
