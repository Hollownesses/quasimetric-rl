#!/usr/bin/env python3
"""Visualize and quantify observation/communication/safety feasible-set intersections."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from minimal_qrl.industry_exp.comm_inspection_experiment_common import (
    add_environment_arguments,
    make_env,
    parse_float_list,
    write_json,
)
from minimal_qrl.envs import CircleObstacle
from minimal_qrl.visualize_comm_inspection_dubins_uav import compute_feasibility_masks


def _choose_devices(env, raw_device_ids: str | None) -> list[str]:
    if raw_device_ids:
        requested = [item.strip() for item in raw_device_ids.split(",") if item.strip()]
        unknown = set(requested) - set(env.device_ids)
        if unknown:
            raise ValueError(f"unknown device id(s): {sorted(unknown)}")
        return requested

    station = np.asarray(env.ground_station, dtype=np.float32)
    ranked = sorted(
        env.device_catalog.devices,
        key=lambda task: float(np.linalg.norm(np.asarray(task.position) - station)),
    )
    indices = sorted(set([0, len(ranked) // 2, len(ranked) - 1]))
    return [ranked[index].device_id for index in indices]


def _sample_components(env, *, num_samples: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    states = np.column_stack(
        [
            rng.uniform(env.x_min, env.x_max, size=num_samples),
            rng.uniform(env.y_min, env.y_max, size=num_samples),
            rng.uniform(-np.pi, np.pi, size=num_samples),
        ]
    ).astype(np.float32)
    safe = np.fromiter((env.is_valid_state(state) for state in states), dtype=bool, count=num_samples)
    observation = np.zeros((num_samples,), dtype=bool)
    quality = np.full((num_samples,), -np.inf, dtype=np.float32)
    station_los = np.zeros((num_samples,), dtype=bool)
    for index in np.flatnonzero(safe):
        state = states[index]
        observation[index] = env.is_observation_feasible(state)
        comm = env.compute_comm_quality(state)
        quality[index] = float(comm["quality"])
        station_los[index] = bool(comm["has_los"])
    return {
        "states": states,
        "safe": safe,
        "observation": observation,
        "comm_quality": quality,
        "station_los": station_los,
    }


def _summarize_thresholds(
    env,
    components: dict[str, np.ndarray],
    *,
    device_id: str,
    thresholds: list[float],
) -> list[dict[str, float | int | str | bool]]:
    safe = components["safe"]
    observation = components["observation"]
    safe_count = max(int(np.sum(safe)), 1)
    rows = []
    for threshold in thresholds:
        communication = safe & (components["comm_quality"] >= float(threshold))
        if env.require_ground_station_los:
            communication &= components["station_los"]
        task = safe & observation & communication
        rows.append(
            {
                "device_id": device_id,
                "comm_threshold": float(threshold),
                "require_ground_station_los": bool(env.require_ground_station_los),
                "num_uniform_samples": int(len(safe)),
                "num_safe_samples": int(np.sum(safe)),
                "safe_fraction_of_bounds": float(np.mean(safe)),
                "observation_fraction_given_safe": float(np.sum(safe & observation) / safe_count),
                "communication_fraction_given_safe": float(np.sum(communication) / safe_count),
                "joint_task_fraction_given_safe": float(np.sum(task) / safe_count),
                "joint_task_sample_count": int(np.sum(task)),
            }
        )
    return rows


def _draw_obstacles(ax, env) -> None:
    for obstacle in env.obstacles:
        if isinstance(obstacle, CircleObstacle):
            patch = patches.Circle(
                (obstacle.x, obstacle.y), obstacle.radius, color="0.25", alpha=0.8
            )
        else:
            patch = patches.Rectangle(
                (obstacle.x_min, obstacle.y_min),
                obstacle.x_max - obstacle.x_min,
                obstacle.y_max - obstacle.y_min,
                color="0.25",
                alpha=0.8,
            )
        ax.add_patch(patch)


def _plot_device_masks(env, *, device_id: str, resolution: int, output_path: Path, seed: int) -> None:
    env.set_task_by_device_id(device_id)
    terminal = env.sample_task_terminal_state(seed=int(seed))
    heading = float(terminal[2])
    xs, ys, obs_mask, comm_mask, task_mask = compute_feasibility_masks(
        env, theta=heading, resolution=int(resolution)
    )
    masks = [obs_mask, comm_mask, task_mask]
    titles = ["Observation feasible", "Communication feasible", "Joint task feasible"]
    colors = ["Greens", "Blues", "Oranges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, mask, title, cmap in zip(axes, masks, titles, colors):
        ax.imshow(
            mask,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            alpha=0.65,
        )
        _draw_obstacles(ax, env)
        ax.scatter(*env.inspection_target, marker="X", s=80, c="gold", edgecolors="black")
        ax.scatter(*env.ground_station, marker="s", s=55, c="navy")
        ax.scatter(terminal[0], terminal[1], marker="o", s=35, c="red")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("y")
    fig.suptitle(
        f"{device_id} | heading slice={heading:.2f} rad | comm threshold={env.comm_threshold:.2f}"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_sensitivity(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for device_id in sorted({str(row["device_id"]) for row in rows}):
        selected = sorted(
            (row for row in rows if row["device_id"] == device_id),
            key=lambda row: float(row["comm_threshold"]),
        )
        ax.plot(
            [float(row["comm_threshold"]) for row in selected],
            [float(row["joint_task_fraction_given_safe"]) for row in selected],
            marker="o",
            label=device_id,
        )
    ax.set_xlabel("Communication threshold")
    ax.set_ylabel("Joint feasible fraction among safe states")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/experiments/comm_inspection_feasible_set",
    )
    parser.add_argument("--device-ids", default=None, help="Comma-separated IDs; default: near/mid/far")
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--grid-resolution", type=int, default=140)
    parser.add_argument("--thresholds", default="-0.5,0.0,0.5,1.0,1.5")
    parser.add_argument("--seed", type=int, default=20260712)
    add_environment_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples <= 0 or args.grid_resolution < 10:
        raise ValueError("--num-samples must be positive and --grid-resolution must be >= 10")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    device_ids = _choose_devices(env, args.device_ids)
    thresholds = sorted(set(parse_float_list(args.thresholds) + [float(args.comm_threshold)]))

    rows: list[dict] = []
    for device_index, device_id in enumerate(device_ids):
        env.set_task_by_device_id(device_id)
        components = _sample_components(
            env,
            num_samples=int(args.num_samples),
            seed=int(args.seed) + device_index * 1009,
        )
        rows.extend(
            _summarize_thresholds(
                env,
                components,
                device_id=device_id,
                thresholds=thresholds,
            )
        )
        _plot_device_masks(
            env,
            device_id=device_id,
            resolution=int(args.grid_resolution),
            output_path=output_dir / "feasible_set_maps" / f"{device_id}.png",
            seed=int(args.seed) + device_index,
        )

    csv_path = output_dir / "feasible_set_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot_threshold_sensitivity(rows, output_dir / "threshold_sensitivity.png")
    write_json(
        output_dir / "feasible_set_results.json",
        {
            "experiment": "task_feasible_set_quantification",
            "device_ids": device_ids,
            "thresholds": thresholds,
            "rows": rows,
            "config": vars(args),
            "notes": {
                "monte_carlo_space": "uniform x, y, heading over configured bounds",
                "reported_conditioning": "observation/communication/joint fractions are conditioned on safe states",
                "map_heading": "each 2D map uses the heading of one sampled task-feasible terminal state",
            },
        },
    )
    print(f"Saved feasible-set experiment to {output_dir}")


if __name__ == "__main__":
    main()
