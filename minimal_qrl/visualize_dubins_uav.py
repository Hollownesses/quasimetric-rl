#!/usr/bin/env python3
"""
Dubins UAV 2D 环境可视化脚本

生成三种图：
1. 基础轨迹图：起点、目标、UAV 轨迹、朝向箭头、障碍物（sanity check）
2. 非对称性可视化：s1→s2 与 s2→s1 两条路径对比（random shooting）
3. 距离场可视化：固定目标 g 与朝向 theta，V(x,y, theta_fixed) 热力图（为 QRL 准备）

图片保存至 results/minimal_qrl/dubins_uav_vis/

带障碍时请加参数，例如：
  python -m minimal_qrl.visualize_dubins_uav --obstacle-config simple
  python -m minimal_qrl.visualize_dubins_uav --obstacles 5 5 1 2 8 0.5
"""
import argparse
import sys
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from minimal_qrl.envs import DubinsUAV2D, Obstacle, CircleObstacle


# Output directory: project_root/results/minimal_qrl/dubins_uav_vis
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "minimal_qrl" / "dubins_uav_vis"


def _obstacles_from_args(bounds: Tuple[float, float, float, float], obstacle_config: str, obstacles_raw: Optional[List[float]]) -> List:
    """根据预设或自定义列表生成圆形障碍。obstacles_raw 为 [x1,y1,r1, x2,y2,r2, ...]。"""
    if obstacles_raw and len(obstacles_raw) > 0:
        if len(obstacles_raw) % 3 != 0:
            raise ValueError("--obstacles 必须是 3 的倍数个数字 (x y r x y r ...)")
        return [
            CircleObstacle(x=float(obstacles_raw[i]), y=float(obstacles_raw[i + 1]), radius=float(obstacles_raw[i + 2]))
            for i in range(0, len(obstacles_raw), 3)
        ]
    x_min, y_min, x_max, y_max = bounds
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w, h = x_max - x_min, y_max - y_min
    if obstacle_config == "none":
        return []
    if obstacle_config == "simple":
        return [CircleObstacle(x=cx, y=cy, radius=0.12 * min(w, h))]
    if obstacle_config == "medium":
        r = 0.10 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.35 * w, y=cy, radius=r),
            CircleObstacle(x=x_min + 0.65 * w, y=cy, radius=r),
            CircleObstacle(x=cx, y=y_min + 0.3 * h, radius=r * 0.8),
        ]
    if obstacle_config == "hard":
        r = 0.08 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=cx, y=cy, radius=r * 1.2),
        ]
    return []


def _normalize_angle(theta: float) -> float:
    """归一化到 [-pi, pi]"""
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta


def _make_env(obstacles: Optional[List] = None, **kwargs) -> DubinsUAV2D:
    """创建带可选障碍物的环境"""
    default = dict(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_episode_steps=300,
        epsilon_pos=0.2,
        epsilon_theta=0.4,
    )
    default.update(kwargs)
    return DubinsUAV2D(obstacles=obstacles, **default)


def rollout_to_goal(
    env: DubinsUAV2D,
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    policy: str = "greedy",
    max_steps: int = 300,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], int, bool]:
    """
    从 start 滚落到 goal。
    policy: "greedy" 简单朝向目标再前进；"random" 随机动作。
    返回 (states 列表, 步数, 是否成功)
    """
    env.reset(seed=seed, options={"start": start, "goal": goal})
    states = [env.state.copy()]
    for _ in range(max_steps - 1):
        x, y, theta = env.state[0], env.state[1], env.state[2]
        gx, gy, gtheta = env.goal[0], env.goal[1], env.goal[2]
        if policy == "greedy":
            dx, dy = gx - x, gy - y
            target_theta = np.arctan2(dy, dx)
            err = _normalize_angle(target_theta - theta)
            omega = np.clip(err * 2.0, -env.omega_max, env.omega_max)
            action = np.array([omega], dtype=np.float32)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        if terminated:
            return states, len(states) - 1, True
        if truncated:
            return states, len(states) - 1, False
    return states, len(states) - 1, False


def random_shooting_plan(
    env: DubinsUAV2D,
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    num_sequences: int = 80,
    horizon: int = 25,
    max_steps: int = 300,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], int, bool]:
    """
    简单 random shooting：采样 num_sequences 条长度为 horizon 的动作序列，
    选到达最近（位置+朝向）的一条动作序列执行，重复直到到达或超时。
    返回 (states, 总步数, 是否成功)
    """
    rng = np.random.default_rng(seed)
    env.reset(seed=seed, options={"start": start, "goal": goal})
    states = [env.state.copy()]
    total_steps = 0
    for _ in range(max_steps // horizon + 1):
        if total_steps >= max_steps:
            break
        gx, gy, gtheta = env.goal[0], env.goal[1], env.goal[2]
        best_actions: Optional[List[np.ndarray]] = None
        best_dist = float("inf")
        for _ in range(num_sequences):
            actions = [np.array([rng.uniform(-env.omega_max, env.omega_max)], dtype=np.float32) for _ in range(horizon)]
            env.reset(seed=None, options={"start": tuple(env.state), "goal": goal})
            for a in actions:
                env.step(a)
            x, y, theta = env.state[0], env.state[1], env.state[2]
            pos_d = np.sqrt((gx - x) ** 2 + (gy - y) ** 2)
            th_d = abs(_normalize_angle(theta - gtheta))
            d = pos_d + 0.5 * th_d
            if d < best_dist:
                best_dist = d
                best_actions = list(actions)
        if best_actions is None:
            break
        env.reset(seed=None, options={"start": tuple(states[-1]), "goal": goal})
        for a in best_actions:
            obs, reward, terminated, truncated, info = env.step(a)
            states.append(env.state.copy())
            total_steps += 1
            if terminated:
                return states, total_steps, True
            if total_steps >= max_steps:
                return states, total_steps, False
    return states, total_steps, False


def plot_basic_trajectory(
    env: DubinsUAV2D,
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    arrow_every: int = 5,
    out_path: Optional[Path] = None,
) -> Path:
    """图1：基础轨迹图（起点、目标、轨迹、朝向箭头、障碍物）"""
    states, steps, success = rollout_to_goal(env, start, goal, policy="greedy", seed=42)
    out_path = out_path or OUTPUT_DIR / "01_basic_trajectory.png"
    os.makedirs(out_path.parent, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    x_min, y_min, x_max, y_max = env.bounds
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    # 障碍物（圆或矩形）
    for obs in env.obstacles:
        if isinstance(obs, CircleObstacle):
            patch = patches.Circle(
                (obs.x, obs.y), obs.radius,
                linewidth=1.5, edgecolor="k", facecolor="gray", alpha=0.6,
            )
        else:
            patch = patches.Rectangle(
                (obs.x_min, obs.y_min),
                obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                linewidth=1.5, edgecolor="k", facecolor="gray", alpha=0.6,
            )
        ax.add_patch(patch)

    # Trajectory
    xs = [s[0] for s in states]
    ys = [s[1] for s in states]
    ax.plot(xs, ys, "b-", linewidth=2, label="Trajectory", zorder=3)

    # Heading arrows (every arrow_every steps)
    for i in range(0, len(states), arrow_every):
        s = states[i]
        x, y, theta = s[0], s[1], s[2]
        L = 0.4
        ax.arrow(
            x, y,
            L * np.cos(theta), L * np.sin(theta),
            head_width=0.15, head_length=0.08, fc="green", ec="green", zorder=4,
        )

    # Start and goal
    ax.plot(start[0], start[1], "go", markersize=14, label="Start", zorder=5)
    ax.plot(goal[0], goal[1], "r*", markersize=20, label="Goal", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"UAV Basic Trajectory (steps={steps}, success={success})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_asymmetry(
    env: DubinsUAV2D,
    s1: Tuple[float, float, float],
    s2: Tuple[float, float, float],
    num_sequences: int = 80,
    horizon: int = 25,
    out_path: Optional[Path] = None,
) -> Path:
    """图2：非对称性可视化 —— s1→s2 与 s2→s1 两条路径"""
    out_path = out_path or OUTPUT_DIR / "02_asymmetry.png"
    os.makedirs(out_path.parent, exist_ok=True)

    # s1 -> s2
    path_12, steps_12, ok_12 = random_shooting_plan(
        env, s1, s2, num_sequences=num_sequences, horizon=horizon, seed=42
    )
    # s2 -> s1
    path_21, steps_21, ok_21 = random_shooting_plan(
        env, s2, s1, num_sequences=num_sequences, horizon=horizon, seed=43
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")
    x_min, y_min, x_max, y_max = env.bounds
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)

    for obs in env.obstacles:
        if isinstance(obs, CircleObstacle):
            patch = patches.Circle(
                (obs.x, obs.y), obs.radius,
                linewidth=1.5, edgecolor="k", facecolor="gray", alpha=0.6,
            )
        else:
            patch = patches.Rectangle(
                (obs.x_min, obs.y_min),
                obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                linewidth=1.5, edgecolor="k", facecolor="gray", alpha=0.6,
            )
        ax.add_patch(patch)

    ax.plot([s[0] for s in path_12], [s[1] for s in path_12], "b-", linewidth=2, label=f"s1->s2 (steps={steps_12})", zorder=3)
    ax.plot([s[0] for s in path_21], [s[1] for s in path_21], "orange", linewidth=2, linestyle="--", label=f"s2->s1 (steps={steps_21})", zorder=3)
    ax.plot(s1[0], s1[1], "go", markersize=14, label="s1", zorder=5)
    ax.plot(s2[0], s2[1], "rs", markersize=14, label="s2", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Asymmetry: s1->s2 vs s2->s1 (different paths)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def compute_distance_field(
    env: DubinsUAV2D,
    goal: Tuple[float, float, float],
    theta_fixed: float,
    resolution: Tuple[int, int] = (40, 40),
) -> np.ndarray:
    """
    计算 V(x, y, theta_fixed)：从 (x,y, theta_fixed) 到 goal 的近似“成本”（时间）。
    使用解析近似：位置距离/v + 朝向误差/omega_max，体现前方成本低、后方成本高。
    """
    x_min, y_min, x_max, y_max = env.bounds
    nx, ny = resolution
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    gx, gy, gtheta = goal[0], goal[1], goal[2]
    v = env.v
    om = env.omega_max
    V = np.zeros((nx, ny))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if not env._is_valid_position(x, y):
                V[i, j] = np.nan
                continue
            pos_dist = np.sqrt((gx - x) ** 2 + (gy - y) ** 2)
            angle_to_goal = np.arctan2(gy - y, gx - x)
            heading_err = abs(_normalize_angle(theta_fixed - angle_to_goal))
            # 粗略时间：移动时间 + 转向时间
            V[i, j] = pos_dist / v + heading_err / om
    return V, xs, ys


def plot_distance_field(
    env: DubinsUAV2D,
    goal: Tuple[float, float, float],
    theta_fixed: float = 0.0,
    resolution: Tuple[int, int] = (40, 40),
    out_path: Optional[Path] = None,
) -> Path:
    """图3：距离场 V(x, y, theta_fixed) 热力图（固定目标 g，固定朝向）"""
    out_path = out_path or OUTPUT_DIR / "03_distance_field.png"
    os.makedirs(out_path.parent, exist_ok=True)

    V, xs, ys = compute_distance_field(env, goal, theta_fixed, resolution)
    x_min, y_min, x_max, y_max = env.bounds

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.set_aspect("equal")
    vm = np.nanmin(V)
    vM = np.nanmax(V)
    im = ax.imshow(
        V.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
        vmin=vm,
        vmax=vM,
    )
    # Goal
    ax.plot(goal[0], goal[1], "r*", markersize=22, label="Goal g", zorder=5)
    # Fixed heading arrow from goal
    L = 1.2
    ax.arrow(
        goal[0], goal[1],
        L * np.cos(theta_fixed), L * np.sin(theta_fixed),
        head_width=0.2, head_length=0.12, fc="red", ec="red", zorder=6,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Distance field V(x,y, theta={np.degrees(theta_fixed):.0f} deg) — low cost ahead, high behind")
    plt.colorbar(im, ax=ax, label="V (approx. time)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Dubins UAV 2D 可视化（可带障碍）")
    parser.add_argument("--bounds", type=float, nargs=4, default=[0, 0, 10, 10], help="地图边界 x_min y_min x_max y_max")
    parser.add_argument(
        "--obstacle-config",
        type=str,
        default="none",
        choices=["none", "simple", "medium", "hard"],
        help="障碍预设：none=无, simple=单圆, medium=2～3 圆, hard=4～5 圆",
    )
    parser.add_argument(
        "--obstacles",
        type=float,
        nargs="*",
        default=None,
        help="自定义圆形障碍 (x1 y1 r1 x2 y2 r2 ...)，若提供则忽略 --obstacle-config",
    )
    args = parser.parse_args()

    bounds = tuple(args.bounds)
    obstacles = _obstacles_from_args(bounds, args.obstacle_config, args.obstacles)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    if obstacles:
        print(f"Obstacles: {len(obstacles)} circles (config={args.obstacle_config})")

    env = _make_env(obstacles=obstacles, bounds=bounds)

    # 1. Basic trajectory
    start = (1.0, 1.0, 0.0)
    goal = (9.0, 9.0, np.pi / 4)
    p1 = plot_basic_trajectory(env, start, goal)
    print(f"Saved: {p1}")

    # 2. Asymmetry
    s1 = (2.0, 2.0, 0.0)
    s2 = (8.0, 8.0, np.pi)
    p2 = plot_asymmetry(env, s1, s2)
    print(f"Saved: {p2}")

    # 3. Distance field (fixed goal and heading)
    g = (5.0, 5.0, 0.0)
    theta_fixed = 0.0
    p3 = plot_distance_field(env, g, theta_fixed=theta_fixed)
    print(f"Saved: {p3}")

    print("Done.")


if __name__ == "__main__":
    main()
