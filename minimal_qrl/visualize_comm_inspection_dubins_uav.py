#!/usr/bin/env python3
"""
Task-conditioned terminal-state 通信感知巡检 Dubins UAV 环境可视化脚本。
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from minimal_qrl.envs import CircleObstacle, CommInspectionDubinsUAV2D


OUTPUT_DIR = Path(__file__).parent.parent / "results" / "minimal_qrl" / "comm_inspection_dubins_uav_vis"


def _normalize_angle(theta: float) -> float:
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta


def _obstacles_from_args(obstacles_raw: Optional[List[float]]) -> List:
    if not obstacles_raw:
        return []
    if len(obstacles_raw) % 3 != 0:
        raise ValueError("--circle-obstacles 必须是 3 的倍数个数字 (x y r ...)")
    return [
        CircleObstacle(
            x=float(obstacles_raw[i]),
            y=float(obstacles_raw[i + 1]),
            radius=float(obstacles_raw[i + 2]),
        )
        for i in range(0, len(obstacles_raw), 3)
    ]


def make_env(obstacles: Optional[List] = None, **kwargs) -> CommInspectionDubinsUAV2D:
    default = dict(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_steps=220,
        inspection_target=(7.5, 6.5),
        ground_station=(1.5, 2.0),
        observation_radius=1.8,
        fov_angle=np.pi / 2.0,
        require_target_los=True,
        require_ground_station_los=False,
        comm_alpha=2.0,
        comm_bias=5.0,
        comm_occlusion_penalty=6.0,
        comm_threshold=0.5,
    )
    default.update(kwargs)
    return CommInspectionDubinsUAV2D(obstacles=obstacles, **default)


def rollout(
    env: CommInspectionDubinsUAV2D,
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    max_steps: int = 220,
) -> Tuple[List[np.ndarray], List[bool], bool]:
    env.reset(options={"start": start, "goal": goal})
    states = [env.state.copy()]
    task_flags = [env.is_task_feasible(env.state)]
    success = False
    for _ in range(max_steps):
        x, y, theta = env.state
        gx, gy, gtheta = env.goal
        dx, dy = gx - x, gy - y
        target_theta = np.arctan2(dy, dx)
        err = _normalize_angle(target_theta - theta)
        goal_heading_err = _normalize_angle(gtheta - theta)
        omega = np.clip(1.5 * err + 0.3 * goal_heading_err, -env.omega_max, env.omega_max)
        _, _, terminated, truncated, info = env.step(np.array([omega], dtype=np.float32))
        states.append(env.state.copy())
        task_flags.append(bool(info["task_feasible"]))
        if terminated:
            success = True
            break
        if truncated:
            break
    return states, task_flags, success


def compute_feasibility_masks(
    env: CommInspectionDubinsUAV2D,
    theta: float,
    resolution: int = 140,
):
    xs = np.linspace(env.x_min, env.x_max, resolution)
    ys = np.linspace(env.y_min, env.y_max, resolution)
    obs_mask = np.zeros((resolution, resolution), dtype=np.float32)
    comm_mask = np.zeros((resolution, resolution), dtype=np.float32)
    task_mask = np.zeros((resolution, resolution), dtype=np.float32)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state = np.array([x, y, theta], dtype=np.float32)
            if not env.is_valid_state(state):
                continue
            obs_ok = env.is_observation_feasible(state)
            comm_ok = env.is_communication_feasible(state)
            obs_mask[iy, ix] = 1.0 if obs_ok else 0.0
            comm_mask[iy, ix] = 1.0 if comm_ok else 0.0
            task_mask[iy, ix] = 1.0 if (obs_ok and comm_ok) else 0.0

    return xs, ys, obs_mask, comm_mask, task_mask


def plot_environment(
    env: CommInspectionDubinsUAV2D,
    states: List[np.ndarray],
    task_flags: List[bool],
    out_path: Path,
) -> Path:
    os.makedirs(out_path.parent, exist_ok=True)
    start = states[0]
    goal = np.asarray(env.goal, dtype=np.float32)

    xs, ys, obs_mask, comm_mask, task_mask = compute_feasibility_masks(env, theta=float(goal[2]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    titles = [
        "Observation Feasible Region",
        "Communication Feasible Region",
        "Joint Task Feasible Region",
    ]
    masks = [obs_mask, comm_mask, task_mask]
    cmaps = ["Greens", "Blues", "Oranges"]

    for ax, title, mask, cmap in zip(axes, titles, masks, cmaps):
        ax.imshow(
            mask,
            origin="lower",
            extent=[env.x_min, env.x_max, env.y_min, env.y_max],
            cmap=cmap,
            alpha=0.35,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(env.x_min - 0.2, env.x_max + 0.2)
        ax.set_ylim(env.y_min - 0.2, env.y_max + 0.2)
        ax.grid(True, alpha=0.2)

        for obs in env.obstacles:
            if isinstance(obs, CircleObstacle):
                patch = patches.Circle((obs.x, obs.y), obs.radius, color="gray", alpha=0.7)
            else:
                patch = patches.Rectangle(
                    (obs.x_min, obs.y_min),
                    obs.x_max - obs.x_min,
                    obs.y_max - obs.y_min,
                    color="gray",
                    alpha=0.7,
                )
            ax.add_patch(patch)

        traj_x = [s[0] for s in states]
        traj_y = [s[1] for s in states]
        for idx in range(1, len(states)):
            seg_color = "darkorange" if task_flags[idx] else "black"
            seg_label = None
            if idx == 1:
                seg_label = "trajectory"
            ax.plot(
                traj_x[idx - 1: idx + 1],
                traj_y[idx - 1: idx + 1],
                color=seg_color,
                linewidth=2.0,
                label=seg_label,
                zorder=4,
            )
        ax.scatter(start[0], start[1], c="green", s=80, label="start", zorder=5)
        ax.scatter(goal[0], goal[1], c="red", s=120, marker="*", label="task terminal goal", zorder=5)
        ax.scatter(env.inspection_target[0], env.inspection_target[1], c="gold", s=90, marker="X", label="inspection target", zorder=5)
        ax.scatter(env.ground_station[0], env.ground_station[1], c="navy", s=90, marker="s", label="ground station", zorder=5)
        first_feasible_idx = next((i for i, flag in enumerate(task_flags) if flag), None)
        if first_feasible_idx is not None:
            feasible_state = states[first_feasible_idx]
            ax.scatter(
                feasible_state[0],
                feasible_state[1],
                c="darkorange",
                s=70,
                marker="o",
                label="first task-feasible state",
                zorder=6,
            )

        circle = patches.Circle(
            (env.inspection_target[0], env.inspection_target[1]),
            env.observation_radius,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            edgecolor="goldenrod",
            alpha=0.9,
        )
        ax.add_patch(circle)

        arrow_len = 0.45
        ax.arrow(
            goal[0],
            goal[1],
            arrow_len * np.cos(goal[2]),
            arrow_len * np.sin(goal[2]),
            head_width=0.15,
            head_length=0.12,
            fc="red",
            ec="red",
            zorder=6,
        )
        ax.arrow(
            start[0],
            start[1],
            arrow_len * np.cos(start[2]),
            arrow_len * np.sin(start[2]),
            head_width=0.15,
            head_length=0.12,
            fc="green",
            ec="green",
            zorder=6,
        )

        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[-1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize task-conditioned comm-aware inspection Dubins UAV environment")
    parser.add_argument("--start", type=float, nargs=3, default=[1.5, 1.5, 0.0])
    parser.add_argument("--goal", type=float, nargs=3, default=None)
    parser.add_argument("--inspection-target", type=float, nargs=2, default=[7.5, 6.5])
    parser.add_argument("--ground-station", type=float, nargs=2, default=[1.5, 2.0])
    parser.add_argument("--circle-obstacles", type=float, nargs="*", default=None)
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR / "comm_inspection_overview.png"))
    args = parser.parse_args()

    obstacles = _obstacles_from_args(args.circle_obstacles)
    env = make_env(
        obstacles=obstacles,
        inspection_target=tuple(args.inspection_target),
        ground_station=tuple(args.ground_station),
        observation_mode="task_context",
    )

    env.reset(seed=42)
    goal = tuple(args.goal) if args.goal is not None else tuple(env.sample_task_feasible_goal(seed=7))
    states, task_flags, success = rollout(env, tuple(args.start), goal)
    out_path = plot_environment(env, states, task_flags, Path(args.out))
    print(f"Saved visualization to: {out_path}")
    print(f"Task terminal goal: {goal}")
    print(f"Success: {success}")


if __name__ == "__main__":
    main()
