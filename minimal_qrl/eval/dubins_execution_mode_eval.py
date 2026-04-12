#!/usr/bin/env python3
"""
Dubins UAV 环境下的 QRL 执行机制评估脚本（greedy vs lookahead）。

目标：
- 复用已有的 Dubins QRL checkpoint（minimal_qrl_dubins_initial 下训练得到）
- 在同一个 DubinsUAV2D 环境上，对比两种执行策略的成功率：
  1) greedy：使用当前的 QRLGoalValueAdapter 贪心一阶启发式（现有做法）
  2) lookahead：使用 QRL value 作为终端代价，进行多步随机 shooting 的 lookahead planner

本脚本专注于「执行策略」评估，不改变 QRL 训练逻辑。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

# 允许在 repo 根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minimal_qrl.envs import DubinsUAV2D
from minimal_qrl.gc_agents import GoalConditionedAgentBase
from minimal_qrl.eval.gc_benchmark import build_qrl_adapter, _make_dubins_env  # type: ignore


def _normalize_angle(theta: float) -> float:
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta


def _heading_to_goal(env: DubinsUAV2D) -> float:
    gx, gy, _gtheta = env.goal
    x, y, theta = float(env.state[0]), float(env.state[1]), float(env.state[2])
    return float(np.arctan2(gy - y, gx - x) - theta)


@dataclass
class DubinsLookaheadConfig:
    """
    Dubins UAV 上的 lookahead planner 超参数。

    - horizon: 每条候选动作序列的长度（步数）
    - num_sequences: 每一步规划时采样的动作序列条数
    - step_cost_weight: 对动作幅度的惩罚权重（抑制抖动/绕圈）
    - collision_penalty: 每次碰撞累计的代价惩罚（info['collision']=True 时计数）
    """

    horizon: int = 5
    num_sequences: int = 64
    step_cost_weight: float = 0.0
    collision_penalty: float = 0.0
    # 额外增强：Dubins 上纯随机序列很弱，加入 goal-directed 的偏置序列
    biased_sequences: int = 24
    bias_kp: float = 2.0  # turn-to-goal 的比例控制系数（仅用于生成偏置序列）
    # 可选：CEM 优化（更稳，但更慢）
    use_cem: bool = False
    cem_iters: int = 3
    cem_elite_frac: float = 0.1
    cem_std_init_frac: float = 0.5  # 初始 std 占动作范围的比例
    alpha_subgoal: float = 1.0
    alpha_final: float = 0.3
    alpha_task_terminal: float = 0.5
    use_env_stage_cost: bool = False
    subgoal_reached_pos_tolerance: float = 0.35
    subgoal_reached_theta_tolerance: float = 0.35


def _build_biased_sequences(env: DubinsUAV2D, horizon: int, n: int, *, kp: float) -> np.ndarray:
    """
    构造一些 goal-directed 的候选动作序列，显著提升 Dubins 的 lookahead 成功率上限：
    - 常值转向 primitive（一直左/直/一直右等）
    - turn-to-goal 的“近似闭环”序列（在规划仿真中会因状态变化而失配，但仍能提供强先验）
    """
    if n <= 0:
        return np.zeros((0, horizon), dtype=np.float32)

    om_max = float(env.omega_max)
    primitives = np.array(
        [-om_max, -0.5 * om_max, -0.25 * om_max, 0.0, 0.25 * om_max, 0.5 * om_max, om_max],
        dtype=np.float32,
    )
    seqs: list[np.ndarray] = []

    # 1) 常值 primitive
    for w in primitives:
        seqs.append(np.full((horizon,), float(w), dtype=np.float32))
        if len(seqs) >= n:
            return np.stack(seqs, axis=0)

    # 2) “转向目标方向”的近似闭环：用当前 heading error 生成一个常值 omega
    err = _normalize_angle(_heading_to_goal(env))
    w0 = float(np.clip(kp * err, -om_max, om_max))
    seqs.append(np.full((horizon,), w0, dtype=np.float32))
    if len(seqs) >= n:
        return np.stack(seqs, axis=0)

    # 3) 轻微扰动的 turn-to-goal 变体
    for s in [0.75, 0.5, 0.35, 0.25, 0.15]:
        w = float(np.clip(w0 * s, -om_max, om_max))
        seqs.append(np.full((horizon,), w, dtype=np.float32))
        if len(seqs) >= n:
            return np.stack(seqs, axis=0)

    # 4) 若还不够，用随机常值补齐（比完全随机序列更适合 Dubins）
    while len(seqs) < n:
        # 使用 NumPy 全局 RNG，便于通过 np.random.seed 控制复现性
        w = float(np.random.uniform(-om_max, om_max))
        seqs.append(np.full((horizon,), w, dtype=np.float32))

    return np.stack(seqs, axis=0)


def _evaluate_action_sequences(
    agent: GoalConditionedAgentBase,
    env: DubinsUAV2D,
    goal_obs: np.ndarray,
    cfg: DubinsLookaheadConfig,
    omegas: np.ndarray,
    base_state: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    评估一批候选角速度序列，返回：
    - costs: (N,)
    - first_actions: (N,) 第一个 omega（便于直接取最优）
    """
    horizon = int(omegas.shape[1])
    n = int(omegas.shape[0])

    costs = np.zeros((n,), dtype=np.float32)
    first_actions = omegas[:, 0].astype(np.float32).copy()

    for i in range(n):
        env.set_state(base_state)

        total_step_cost = 0.0
        collision_count = 0
        success = False

        for t in range(horizon):
            w = float(omegas[i, t])
            action = np.array([w], dtype=np.float32)
            _obs, reward, terminated, truncated, info = env.step(action)

            if cfg.step_cost_weight > 0.0:
                total_step_cost += float(cfg.step_cost_weight) * abs(w)

            if cfg.collision_penalty > 0.0 and isinstance(info, dict) and info.get("collision", False):
                collision_count += 1

            if terminated:
                success = True
                break
            if truncated:
                break

        terminal_obs = env.state_to_observation(env.state)

        if success:
            terminal_cost = 0.0
        else:
            terminal_cost = float(agent.value(terminal_obs, goal_obs))

        costs[i] = float(terminal_cost + total_step_cost + float(cfg.collision_penalty) * float(collision_count))

    env.set_state(base_state)
    return costs, first_actions


def dubins_lookahead_action(
    agent: GoalConditionedAgentBase,
    env: DubinsUAV2D,
    goal_obs: np.ndarray,
    cfg: DubinsLookaheadConfig,
) -> np.ndarray:
    """
    使用 QRL value 作为终端代价，在 DubinsUAV2D 上做多步 shooting lookahead：
    - 固定当前状态 s_t
    - 采样 num_sequences 条长度为 horizon 的角速度序列
    - 用环境 step 仿真每条序列，得到终点状态 s_T
    - 终端代价 = agent.value(obs_T, goal_obs)；成功到达则终端代价视为 0
    - 总代价 = 终端代价 + step_cost_weight * 累计动作幅度 + collision_penalty * 碰撞次数
    - 选择总代价最小序列的第一个动作作为当前控制输入
    """
    horizon = max(1, int(cfg.horizon))
    num_sequences = max(1, int(cfg.num_sequences))

    # 当前内部状态快照（包含 start/goal/_t）
    base_state = env.get_state()

    # 动作范围
    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])

    # 1) 生成候选序列：随机 + biased（Dubins 上 biased 非常关键）
    n_bias = int(min(max(0, cfg.biased_sequences), num_sequences))
    n_rand = int(max(0, num_sequences - n_bias))
    rand = np.random.uniform(low, high, size=(n_rand, horizon)).astype(np.float32) if n_rand > 0 else np.zeros((0, horizon), dtype=np.float32)
    bias = _build_biased_sequences(env, horizon, n_bias, kp=float(cfg.bias_kp)) if n_bias > 0 else np.zeros((0, horizon), dtype=np.float32)
    omegas0 = np.concatenate([bias, rand], axis=0) if (n_bias + n_rand) > 0 else np.zeros((1, horizon), dtype=np.float32)

    # 2) 若启用 CEM：在 omegas0 基础上做若干轮分布更新（更稳，但更耗时）
    if cfg.use_cem:
        om_range = float(high - low)
        std = np.full((horizon,), float(cfg.cem_std_init_frac) * 0.5 * om_range, dtype=np.float32)
        mean = np.zeros((horizon,), dtype=np.float32)
        # 用初始候选的均值作为 warm start（更稳定）
        if omegas0.shape[0] > 0:
            mean = np.mean(omegas0, axis=0).astype(np.float32)

        n_elite = max(1, int(float(cfg.cem_elite_frac) * float(num_sequences)))
        best_first = np.array([0.0], dtype=np.float32)
        best_cost = float("inf")

        for _ in range(max(1, int(cfg.cem_iters))):
            samples = np.random.normal(loc=mean[None, :], scale=std[None, :], size=(num_sequences, horizon)).astype(np.float32)
            samples = np.clip(samples, low, high)

            # 强制保留一小部分 biased 序列，避免 CEM 初期陷入坏局部
            if n_bias > 0:
                k = min(n_bias, samples.shape[0])
                samples[:k] = bias[:k]

            costs, firsts = _evaluate_action_sequences(agent, env, goal_obs, cfg, samples, base_state)
            idx = int(np.argmin(costs))
            if float(costs[idx]) < best_cost:
                best_cost = float(costs[idx])
                best_first = np.array([float(firsts[idx])], dtype=np.float32)

            elite_idx = np.argsort(costs)[:n_elite]
            elite = samples[elite_idx]
            mean = np.mean(elite, axis=0).astype(np.float32)
            std = (np.std(elite, axis=0) + 1e-4).astype(np.float32)

        env.set_state(base_state)
        return best_first

    # 3) 不用 CEM：直接评估一次
    costs, firsts = _evaluate_action_sequences(agent, env, goal_obs, cfg, omegas0, base_state)
    best_idx = int(np.argmin(costs))
    best_first_action = np.array([float(firsts[best_idx])], dtype=np.float32)
    env.set_state(base_state)
    return best_first_action


def evaluate_success_rate_lookahead(
    agent: GoalConditionedAgentBase,
    env: DubinsUAV2D,
    cfg: DubinsLookaheadConfig,
    n_trials: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """
    使用 lookahead planner（而不是 agent.act 自带的贪心策略）在 DubinsUAV2D 上评估成功率。
    """
    np.random.seed(seed)

    success = 0
    steps_success = []

    for i in tqdm(range(n_trials), desc="success_rate_lookahead", leave=False):
        # reset 会随机 start/goal
        obs, _ = env.reset(seed=int(seed + i))
        goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))

        done = False
        truncated = False
        t = 0

        while not (done or truncated):
            action = dubins_lookahead_action(agent, env, goal_obs, cfg)
            obs, reward, done, truncated, info = env.step(action)
            t += 1

            if done:
                success += 1
                steps_success.append(t)
                break
            if truncated:
                break

    rate = success / float(n_trials) if n_trials > 0 else 0.0
    avg_steps = float(np.mean(steps_success)) if steps_success else 0.0
    return {
        "success_rate": rate,
        "avg_steps_success": avg_steps,
        "num_success": float(success),
        "num_trials": float(n_trials),
    }


def main():
    parser = argparse.ArgumentParser(description="Dubins UAV 上 QRL greedy vs lookahead 成功率评估")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Dubins QRL checkpoint 路径（如 results/minimal_qrl_dubins_initial/checkpoint_final.pth）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/minimal_qrl_dubins_initial",
        help="输出目录（保存 json）",
    )

    # Dubins 环境参数（需与训练保持一致）
    parser.add_argument("--bounds", type=float, nargs=4, default=[0, 0, 5, 5])
    parser.add_argument("--omega-max", type=float, default=3.0)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--epsilon-pos", type=float, default=0.15)
    parser.add_argument("--epsilon-theta", type=float, default=0.2)

    # 评估配置
    parser.add_argument("--n-trials", type=int, default=200, help="评估 episode 数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps")

    # lookahead 配置
    parser.add_argument("--lookahead-horizon", type=int, default=5)
    parser.add_argument("--lookahead-num-sequences", type=int, default=64)
    parser.add_argument("--lookahead-step-cost-weight", type=float, default=0.0)
    parser.add_argument("--lookahead-collision-penalty", type=float, default=0.0)
    parser.add_argument("--lookahead-biased-sequences", type=int, default=24)
    parser.add_argument("--lookahead-bias-kp", type=float, default=2.0)
    parser.add_argument("--lookahead-use-cem", action="store_true")
    parser.add_argument("--lookahead-cem-iters", type=int, default=3)
    parser.add_argument("--lookahead-cem-elite-frac", type=float, default=0.1)
    parser.add_argument("--lookahead-cem-std-init-frac", type=float, default=0.5)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 复用 gc_benchmark 里的环境构造逻辑，保证与训练/其它评估一致
    # 注意：_make_dubins_env 会忽略 checkpoint 路径，只使用几何与动力学参数
    env = _make_dubins_env(args)

    # 为 build_qrl_adapter 提供 qrl_ckpt 字段
    args.qrl_ckpt = args.checkpoint
    qrl_agent: GoalConditionedAgentBase = build_qrl_adapter(args, device, env)

    # 先用当前 greedy policy（QRLGoalValueAdapter.act）评估一次成功率
    from minimal_qrl.eval.gc_benchmark import evaluate_success_rate as eval_success_greedy  # type: ignore

    greedy_metrics = eval_success_greedy(qrl_agent, env, n_trials=args.n_trials, seed=args.seed)

    # 构造 lookahead 配置并评估
    la_cfg = DubinsLookaheadConfig(
        horizon=args.lookahead_horizon,
        num_sequences=args.lookahead_num_sequences,
        step_cost_weight=args.lookahead_step_cost_weight,
        collision_penalty=args.lookahead_collision_penalty,
        biased_sequences=int(args.lookahead_biased_sequences),
        bias_kp=float(args.lookahead_bias_kp),
        use_cem=bool(args.lookahead_use_cem),
        cem_iters=int(args.lookahead_cem_iters),
        cem_elite_frac=float(args.lookahead_cem_elite_frac),
        cem_std_init_frac=float(args.lookahead_cem_std_init_frac),
    )
    lookahead_metrics = evaluate_success_rate_lookahead(
        qrl_agent, env, cfg=la_cfg, n_trials=args.n_trials, seed=args.seed
    )

    results: Dict[str, Dict[str, float]] = {
        "greedy": greedy_metrics,
        "lookahead": lookahead_metrics,
    }

    out_json = os.path.join(args.output_dir, "dubins_execution_mode_eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[dubins_execution_mode_eval] 已保存评估结果到 {out_json}")
    print(
        f"  greedy:   success_rate={greedy_metrics['success_rate']:.3f}, "
        f"avg_steps_success={greedy_metrics['avg_steps_success']:.1f}"
    )
    print(
        f"  lookahead: success_rate={lookahead_metrics['success_rate']:.3f}, "
        f"avg_steps_success={lookahead_metrics['avg_steps_success']:.1f}"
    )


if __name__ == "__main__":
    main()
