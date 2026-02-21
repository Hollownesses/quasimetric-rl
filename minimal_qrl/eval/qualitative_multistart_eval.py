#!/usr/bin/env python3
"""
定性评估：可视化 QRL 在“固定目标”条件下的 multi-start 行为一致性（展示用）

设计目标
- 固定同一个目标
- 在环境中采样多个不同起点（尽量覆盖不同区域）
- 对每个起点，使用同一个已训练 QRL 模型进行一次导航 rollout（goal-conditioned）
- 输出一张叠加可视化图：边界/障碍物/固定目标/多个起点/多条轨迹

说明
- 该脚本是独立评估模块，不耦合/不修改任何已有 evaluation 脚本
- 会复用 `minimal_qrl.eval.planning_evaluation.greedy_navigation_rollout` 做 greedy rollout
- checkpoint 加载格式与 `minimal_qrl/train.py` 保存格式兼容（{agent: state_dict}）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 本脚本位于 minimal_qrl/eval/ 下，因此需要上溯两级到 repo 根目录
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quasimetric_rl.modules import QRLConf
from quasimetric_rl.data import Dataset

from minimal_qrl.envs import ContinuousObstacle2D
from minimal_qrl.eval.planning_evaluation import navigation_rollout, LookaheadConfig
from minimal_qrl.eval.utils import auto_device, ensure_registered_env


def _project_goal_to_valid(env: ContinuousObstacle2D, goal: np.ndarray) -> np.ndarray:
    """
    利用 env.reset 内部的投影逻辑把 goal 修正到合法位置。
    """
    g = np.array(goal, dtype=np.float32).reshape(2)
    _, _ = env.reset(options={"start": (float(g[0]), float(g[1])), "goal": (float(g[0]), float(g[1]))})
    return np.array(env.goal, dtype=np.float32)


def _is_reachable(env: ContinuousObstacle2D, start: np.ndarray, goal: np.ndarray) -> bool:
    try:
        d = env.compute_shortest_path_distance(start=start, goal=goal)
        return bool(np.isfinite(d))
    except Exception:
        # 如果 A* 距离计算不可用/异常，则不强制可达
        return True


def _sample_start_candidates(
    env: ContinuousObstacle2D,
    *,
    goal: np.ndarray,
    n_candidates: int,
    seed: int,
    min_start_goal_dist: float,
    require_reachable: bool,
    max_attempts: int = 200000,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    candidates: List[np.ndarray] = []
    attempts = 0
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        cand = env.sample_valid_state(seed=int(rng.randint(0, 2**31 - 1)))
        cand = np.array(cand, dtype=np.float32).reshape(2)

        if float(np.linalg.norm(cand - goal)) < float(min_start_goal_dist):
            continue

        if require_reachable and not _is_reachable(env, cand, goal):
            continue

        candidates.append(cand)

    if len(candidates) == 0:
        raise RuntimeError("起点候选采样失败：没有采到任何候选起点。请检查 goal 是否可达、或关闭 --require-reachable。")

    return np.stack(candidates, axis=0)


def _greedy_farthest_point_selection(
    candidates: np.ndarray,
    *,
    k: int,
    goal: np.ndarray,
    min_separation: float,
    relax: bool,
    relax_floor: float = 0.06,
    relax_factor: float = 0.90,
) -> np.ndarray:
    """
    从候选集中挑选 k 个尽量分散的起点：
    - 先选距离 goal 最远的
    - 之后每次选“到已选集合的最小距离最大”的点（farthest-point sampling）
    若 strict min_separation 导致无法凑够 k，且 relax=True，则逐步放宽 min_separation 直到满足或触底。
    """
    assert candidates.ndim == 2 and candidates.shape[1] == 2
    cand = candidates
    g = goal.reshape(1, 2)

    # 预计算候选到 goal 距离（用于第一个点）
    d_goal = np.linalg.norm(cand - g, axis=1)
    first_idx = int(np.argmax(d_goal))

    selected_idx: List[int] = [first_idx]
    selected = [cand[first_idx]]

    cur_sep = float(min_separation)
    tried_relax = False

    while len(selected) < k:
        # 对每个候选点，计算其到已选集合的最小距离
        sel = np.stack(selected, axis=0)  # (m,2)
        # (N,m,2) -> (N,m) -> (N,)
        dists = np.linalg.norm(cand[:, None, :] - sel[None, :, :], axis=2)
        min_to_sel = dists.min(axis=1)

        # 排除已选点
        mask_unselected = np.ones(cand.shape[0], dtype=bool)
        mask_unselected[selected_idx] = False

        # 必须满足 min separation
        feasible = mask_unselected & (min_to_sel >= cur_sep)

        if not np.any(feasible):
            if relax and cur_sep > relax_floor:
                cur_sep = max(relax_floor, cur_sep * relax_factor)
                tried_relax = True
                continue
            break

        # 在可行集中选最“远离已选集合”的点
        feasible_idx = np.where(feasible)[0]
        pick = feasible_idx[int(np.argmax(min_to_sel[feasible_idx]))]
        selected_idx.append(int(pick))
        selected.append(cand[pick])

    if len(selected) < k:
        msg = (
            f"起点挑选不足：仅选到 {len(selected)}/{k} 个起点。"
            f"当前 min_start_separation={min_separation:.3f}"
        )
        if relax and tried_relax:
            msg += f"（已自动放宽到 {cur_sep:.3f} 仍不足）"
        msg += "。可尝试减小 --min-start-separation / 增大 --candidate-pool / 关闭 --require-reachable。"
        raise RuntimeError(msg)

    return np.stack(selected, axis=0)


def _extract_path_xy(rollout_result: dict, *, start: np.ndarray) -> np.ndarray:
    traj = rollout_result.get("trajectory", [])
    if not traj:
        return start[None, :]
    states = [np.array(s, dtype=np.float32).reshape(2) for (s, _a) in traj]
    final_state = np.array(rollout_result.get("final_state", states[-1]), dtype=np.float32).reshape(2)
    states.append(final_state)
    return np.stack(states, axis=0)


def _plot_multistart_trajectories(
    *,
    env: ContinuousObstacle2D,
    goal: np.ndarray,
    starts: np.ndarray,
    paths: List[np.ndarray],
    successes: List[bool],
    output_path: str,
    title: Optional[str] = None,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))

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

    # 固定目标
    ax.scatter([goal[0]], [goal[1]], s=220, c="red", marker="*", zorder=7, label="Goal (fixed)")

    # 每个起点/轨迹一种颜色
    cmap = plt.get_cmap("tab20" if len(starts) <= 20 else "hsv")
    colors = [cmap(i % cmap.N) for i in range(len(starts))]

    for i, (start, path, color, ok) in enumerate(zip(starts, paths, colors, successes)):
        linestyle = "-" if ok else "--"
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.2, alpha=0.95, linestyle=linestyle, zorder=3)
        ax.scatter([start[0]], [start[1]], s=110, c=[color], marker="o", zorder=6, edgecolors="black", linewidths=0.6)
        ax.text(start[0] + 0.012, start[1] + 0.012, f"{i+1}", color=color, fontsize=10, weight="bold")

        # 终点（rollout 最后位置）
        ax.scatter([path[-1, 0]], [path[-1, 1]], s=35, c=[color], marker="x", zorder=5, linewidths=2.0)

    n_success = int(sum(1 for s in successes if s))
    ax.text(
        0.02,
        0.98,
        f"Starts: {len(starts)} | Success: {n_success}/{len(starts)}\n"
        f"Traj color matches start color; same goal for all rollouts.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        zorder=10,
    )

    ax.set_title(title or "QRL multi-start qualitative behavior (fixed goal)")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="定性评估：固定目标下的 multi-start 行为可视化（QRL）")

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
        default="./results/minimal_qrl/qualitative_multistart.png",
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
    parser.add_argument("--seed", type=int, default=0, help="随机种子（影响起点采样与 greedy 采样）")
    parser.add_argument("--num-starts", type=int, default=12, help="起点数量")
    parser.add_argument("--goal", type=float, nargs=2, default=[0.88, 0.88], help="固定目标 (x y)")
    parser.add_argument("--min-start-goal-dist", type=float, default=0.25, help="起点与目标的最小距离（避免太简单）")
    parser.add_argument("--min-start-separation", type=float, default=0.18, help="起点之间的最小间距（提升覆盖度）")
    parser.add_argument("--require-reachable", action="store_true", help="仅采样 A* 判定可达的起点（更稳定）")

    # 更稳定的采样：先采候选池再做 farthest-point 选择
    parser.add_argument("--candidate-pool", type=int, default=5000, help="候选起点池大小（越大越容易挑到分散起点）")
    parser.add_argument("--no-relax-separation", action="store_true", help="若挑选不足则直接失败（默认会自动放宽间距）")

    # 设备
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda/mps)")

    # 展示信息
    parser.add_argument("--title", type=str, default="", help="图标题（可用于报告/PPT）")

    args = parser.parse_args()

    device = auto_device(args.device)

    # 构建环境（与 minimal_qrl/train.py 的 obstacle 环境一致）
    env_kwargs = dict(max_episode_steps=int(args.max_steps), grid_resolution=int(args.grid_resolution))

    def create_env_fn():
        return ContinuousObstacle2D(**env_kwargs)

    def load_episodes_fn():
        return iter(())

    ensure_registered_env("obstacle", args.env_name, create_env_fn=create_env_fn, load_episodes_fn=load_episodes_fn)

    dataset_conf = Dataset.Conf(kind="obstacle", name=args.env_name, future_observation_discount=0.99)
    dataset = dataset_conf.make(dummy=True)

    agent_conf = QRLConf(actor=None, num_critics=int(args.num_critics))
    agent, _losses = agent_conf.make(env_spec=dataset.env_spec, total_optim_steps=1)
    agent.to(device)
    agent.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
        ckpt_step = ckpt.get("optim_steps", None)
    else:
        agent.load_state_dict(ckpt)
        ckpt_step = None

    env = create_env_fn()

    # 固定目标（确保合法）
    goal = _project_goal_to_valid(env, np.array(args.goal, dtype=np.float32))

    # 采样并挑选多个起点
    candidates = _sample_start_candidates(
        env,
        goal=goal,
        n_candidates=int(args.candidate_pool),
        seed=int(args.seed),
        min_start_goal_dist=float(args.min_start_goal_dist),
        require_reachable=bool(args.require_reachable),
    )
    starts = _greedy_farthest_point_selection(
        candidates,
        k=int(args.num_starts),
        goal=goal,
        min_separation=float(args.min_start_separation),
        relax=not bool(args.no_relax_separation),
    )

    # rollout（不同起点，同一目标）
    np.random.seed(int(args.seed))  # greedy_action_selection 内会用 np.random
    paths: List[np.ndarray] = []
    successes: List[bool] = []

    lookahead_cfg = None
    if args.execution_mode == "lookahead":
        lookahead_cfg = LookaheadConfig(
            horizon=int(args.lookahead_horizon),
            num_sequences=int(args.lookahead_num_sequences),
        )

    for start in starts:
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

    title = args.title.strip()
    if not title:
        ckpt_tag = f"step={ckpt_step}" if ckpt_step is not None else os.path.basename(args.checkpoint)
        title = f"QRL multi-start (fixed goal) | {ckpt_tag}"

    _plot_multistart_trajectories(
        env=env,
        goal=goal,
        starts=starts,
        paths=paths,
        successes=successes,
        output_path=args.output,
        title=title,
    )

    print(f"[qualitative_multistart_eval] 已保存可视化图像: {args.output}")


if __name__ == "__main__":
    main()

