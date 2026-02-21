#!/usr/bin/env python3
"""
定性评估：可视化 QRL 的 goal-conditioned 多目标导航能力（展示用）

设计目标
- 固定同一个起点
- 在环境中采样多个不同目标
- 对每个目标，使用同一个已训练 QRL 模型进行一次 rollout
- 输出一张叠加可视化图：边界/障碍物/固定起点/多个目标/多条轨迹

说明
- 该脚本是独立评估模块，不耦合/不修改现有 evaluation 脚本
- 会复用 `minimal_qrl.eval.planning_evaluation.greedy_navigation_rollout` 做 greedy rollout
- 加载 checkpoint 时，仅需要 `agent.state_dict()`（与 minimal_qrl/train.py 的保存格式一致）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目根目录到路径（保持与 minimal_qrl/train.py 一致的运行方式：在 repo 根目录执行）
# 注意：本脚本位于 minimal_qrl/eval/ 下，因此需要上溯两级到 repo 根目录
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quasimetric_rl.modules import QRLConf
from quasimetric_rl.data import Dataset

from minimal_qrl.envs import ContinuousObstacle2D
from minimal_qrl.eval.planning_evaluation import navigation_rollout, LookaheadConfig
from minimal_qrl.eval.utils import auto_device, ensure_registered_env


def _project_start_to_valid(env: ContinuousObstacle2D, start: np.ndarray) -> np.ndarray:
    """
    利用 env.reset 内部的投影逻辑把 start 修正到合法位置。
    """
    s = np.array(start, dtype=np.float32).reshape(2)
    # goal 给一个临时值即可（reset 会同时投影 start/goal）
    _, _ = env.reset(options={"start": (float(s[0]), float(s[1])), "goal": (float(s[0]), float(s[1]))})
    return np.array(env.start, dtype=np.float32)


def _sample_diverse_goals(
    env: ContinuousObstacle2D,
    *,
    start: np.ndarray,
    n_goals: int,
    seed: int,
    min_start_goal_dist: float,
    min_goal_separation: float,
    require_reachable: bool,
    max_attempts: int = 20000,
) -> np.ndarray:
    rng = np.random.RandomState(seed)

    goals: List[np.ndarray] = []
    attempts = 0
    while len(goals) < n_goals and attempts < max_attempts:
        attempts += 1

        # 使用 env 的合法采样；为了可复现，用 seed + 随机偏移
        cand = env.sample_valid_state(seed=int(rng.randint(0, 2**31 - 1)))
        cand = np.array(cand, dtype=np.float32).reshape(2)

        if float(np.linalg.norm(cand - start)) < float(min_start_goal_dist):
            continue

        if any(float(np.linalg.norm(cand - g)) < float(min_goal_separation) for g in goals):
            continue

        if require_reachable:
            # 避免采到 A* 判定不可达的点（极少数情况下可能发生）
            try:
                d = env.compute_shortest_path_distance(start=start, goal=cand)
                if not np.isfinite(d):
                    continue
            except Exception:
                # 如果环境不支持/出错，则不强制
                pass

        goals.append(cand)

    if len(goals) < n_goals:
        raise RuntimeError(
            f"目标采样失败：仅采到 {len(goals)}/{n_goals} 个目标。"
            f"你可以尝试减小 min_goal_separation/min_start_goal_dist 或关闭 require_reachable。"
        )

    return np.stack(goals, axis=0)


def _extract_path_xy(rollout_result: dict, *, start: np.ndarray) -> np.ndarray:
    traj = rollout_result.get("trajectory", [])
    if not traj:
        return start[None, :]
    states = [np.array(s, dtype=np.float32).reshape(2) for (s, _a) in traj]
    final_state = np.array(rollout_result.get("final_state", states[-1]), dtype=np.float32).reshape(2)
    states.append(final_state)
    return np.stack(states, axis=0)


def _plot_multigoal_trajectories(
    *,
    env: ContinuousObstacle2D,
    start: np.ndarray,
    goals: np.ndarray,
    paths: List[np.ndarray],
    successes: List[bool],
    output_path: str,
    title: Optional[str] = None,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))

    # 边界
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # 障碍物
    if hasattr(env, "obstacles"):
        for obs in env.obstacles:
            rect = patches.Rectangle(
                (obs.x_min, obs.y_min),
                obs.x_max - obs.x_min,
                obs.y_max - obs.y_min,
                linewidth=1.5,
                edgecolor="black",
                facecolor="gray",
                alpha=0.45,
            )
            ax.add_patch(rect)

    # 起点（固定）
    ax.scatter([start[0]], [start[1]], s=120, c="black", marker="o", zorder=5, label="Start (fixed)")

    # 颜色：每个目标/轨迹一种颜色
    cmap = plt.get_cmap("tab20" if len(goals) <= 20 else "hsv")
    colors = [cmap(i % cmap.N) for i in range(len(goals))]

    # 目标 + 轨迹（同色）
    for i, (goal, path, color, ok) in enumerate(zip(goals, paths, colors, successes)):
        # 轨迹：失败时用虚线，帮助对比但不喧宾夺主
        linestyle = "-" if ok else "--"
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.2, alpha=0.95, linestyle=linestyle, zorder=3)

        # 终点（rollout 最后位置）
        ax.scatter([path[-1, 0]], [path[-1, 1]], s=35, c=[color], marker="x", zorder=4, linewidths=2.0)

        # 目标
        ax.scatter([goal[0]], [goal[1]], s=160, c=[color], marker="*", zorder=6)
        ax.text(goal[0] + 0.012, goal[1] + 0.012, f"{i+1}", color=color, fontsize=10, weight="bold")

    # 说明文字（避免 legend 过长）
    n_success = int(sum(1 for s in successes if s))
    ax.text(
        0.02,
        0.98,
        f"Goals: {len(goals)} | Success: {n_success}/{len(goals)}\n"
        f"Traj color matches goal color; same start for all rollouts.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        zorder=10,
    )

    if title:
        ax.set_title(title)
    else:
        ax.set_title("QRL multi-goal qualitative behavior (fixed start)")

    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="定性评估：固定起点下的多目标 goal-conditioned 行为可视化（QRL）")

    # checkpoint / 输出
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练得到的 checkpoint 路径（例如 results/minimal_qrl/checkpoint_final.pth）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/minimal_qrl/qualitative_multigoal.png",
        help="输出图像路径（png）",
    )

    # 环境
    parser.add_argument("--env-name", type=str, default="obstacle2d", help="环境注册名（用于 env_spec 构建）")
    parser.add_argument("--max-steps", type=int, default=200, help="每条 rollout 的最大步数")
    parser.add_argument("--grid-resolution", type=int, default=80, help="A* 网格分辨率（仅影响可达性判断/距离）")

    # 模型结构（需与训练一致）
    parser.add_argument("--num-critics", type=int, default=2, help="Critic 数量（需与训练一致）")

    # rollout 策略参数（greedy / lookahead）
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="greedy",
        choices=["greedy", "lookahead"],
        help="执行机制：greedy（一步）或 lookahead（短视野仿真规划）",
    )
    parser.add_argument("--num-action-candidates", type=int, default=32, help="每步采样候选动作数量")
    parser.add_argument("--lookahead-horizon", type=int, default=5, help="lookahead 规划步长（仅 lookahead 模式）")
    parser.add_argument("--lookahead-num-sequences", type=int, default=64, help="lookahead 序列数量（仅 lookahead 模式）")

    # 定性评估配置
    parser.add_argument("--seed", type=int, default=0, help="随机种子（影响目标采样与 greedy 采样）")
    parser.add_argument("--num-goals", type=int, default=12, help="目标数量")
    parser.add_argument("--start", type=float, nargs=2, default=[0.12, 0.12], help="固定起点 (x y)")
    parser.add_argument("--min-start-goal-dist", type=float, default=0.25, help="目标与起点的最小距离")
    parser.add_argument("--min-goal-separation", type=float, default=0.18, help="目标之间的最小间距（提升覆盖度）")
    parser.add_argument("--require-reachable", action="store_true", help="仅采样 A* 判定可达的目标（更稳定）")

    # 设备
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备 (auto/cpu/cuda/mps)",
    )

    # 展示信息
    parser.add_argument("--title", type=str, default="", help="图标题（可用于报告/PPT）")

    args = parser.parse_args()

    device = auto_device(args.device)

    # 构建环境（与 minimal_qrl/train.py 的 obstacle 环境一致）
    env_kwargs = dict(max_episode_steps=int(args.max_steps), grid_resolution=int(args.grid_resolution))

    def create_env_fn():
        return ContinuousObstacle2D(**env_kwargs)

    # dummy loader：只为注册占位（本脚本不需要数据集 episode）
    def load_episodes_fn():
        # Dataset(dummy=True) 不会调用此函数；这里留作保险
        return iter(())

    ensure_registered_env("obstacle", args.env_name, create_env_fn=create_env_fn, load_episodes_fn=load_episodes_fn)

    # 仅为了拿 env_spec：dummy=True 避免加载数据集
    dataset_conf = Dataset.Conf(kind="obstacle", name=args.env_name, future_observation_discount=0.99)
    dataset = dataset_conf.make(dummy=True)

    # 构建 agent（结构需与训练一致：actor=None）
    agent_conf = QRLConf(actor=None, num_critics=int(args.num_critics))
    agent, _losses = agent_conf.make(env_spec=dataset.env_spec, total_optim_steps=1)
    agent.to(device)
    agent.eval()

    # 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
        ckpt_step = ckpt.get("optim_steps", None)
    else:
        # 兼容：直接保存了 state_dict
        agent.load_state_dict(ckpt)
        ckpt_step = None

    # 用于 rollout 的环境实例
    env = create_env_fn()

    # 固定起点（确保合法）
    start = _project_start_to_valid(env, np.array(args.start, dtype=np.float32))

    # 采样多个目标
    goals = _sample_diverse_goals(
        env,
        start=start,
        n_goals=int(args.num_goals),
        seed=int(args.seed),
        min_start_goal_dist=float(args.min_start_goal_dist),
        min_goal_separation=float(args.min_goal_separation),
        require_reachable=bool(args.require_reachable),
    )

    # rollout（同一起点，不同目标）
    np.random.seed(int(args.seed))  # greedy_action_selection 内会用 np.random
    paths: List[np.ndarray] = []
    successes: List[bool] = []

    lookahead_cfg = None
    if args.execution_mode == "lookahead":
        lookahead_cfg = LookaheadConfig(
            horizon=int(args.lookahead_horizon),
            num_sequences=int(args.lookahead_num_sequences),
        )

    for goal in goals:
        rr = navigation_rollout(
            agent=agent,
            env=env,
            start=start,
            goal=goal,
            device=str(device),
            max_steps=int(args.max_steps),
            num_action_candidates=int(args.num_action_candidates),
            use_improved_termination=True,
            execution_mode=str(args.execution_mode),
            lookahead_config=lookahead_cfg,
        )
        paths.append(_extract_path_xy(rr, start=start))
        successes.append(bool(rr.get("success", False)))

    # 组装标题
    title = args.title.strip()
    if not title:
        ckpt_tag = f"step={ckpt_step}" if ckpt_step is not None else os.path.basename(args.checkpoint)
        title = f"QRL goal-conditioned multi-goal navigation (fixed start) | {ckpt_tag}"

    _plot_multigoal_trajectories(
        env=env,
        start=start,
        goals=goals,
        paths=paths,
        successes=successes,
        output_path=args.output,
        title=title,
    )

    print(f"[qualitative_multigoal_eval] 已保存可视化图像: {args.output}")


if __name__ == "__main__":
    main()

