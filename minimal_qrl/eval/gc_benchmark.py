#!/usr/bin/env python3
"""
Dubins UAV 上 QRL vs TD-based goal-conditioned RL 的统一对比评估脚本。

支持的算法：
- QRL（通过 QRLGoalValueAdapter 适配）
- HER + DDPG
- Goal-conditioned SAC
- UVFA-style value learning

统一输出以下核心指标：
1) 成功率（随机起点、随机目标）
2) Triangle Inequality Violation:
   违反比例 & 平均违反幅度
3) Value-to-True-Time 相关性：Pearson / Spearman / MSE

用法示例（假设已经分别训练好 QRL 和各 TD agent 的 checkpoint）：

  python -m minimal_qrl.eval.gc_benchmark \\
      --qrl-ckpt results/minimal_qrl_dubins_initial/checkpoint_final.pth \\
      --output-dir results/minimal_qrl_dubins_initial/gc_benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quasimetric_rl.data import EnvSpec, Dataset  # type: ignore
from quasimetric_rl.modules import QRLConf  # type: ignore
from quasimetric_rl.data.base import register_offline_env  # type: ignore

from minimal_qrl.envs import DubinsUAV2D, CircleObstacle
from minimal_qrl.dataset import create_dataset
from minimal_qrl.eval.planning_evaluation import (
    sample_state_goal_pairs,
    compute_ground_truth_distance,
)
from minimal_qrl.gc_agents import (
    GoalConditionedAgentBase,
    QRLGoalValueAdapter,
)


def _obstacles_from_args(args) -> List:
    """从 args 构建圆形障碍列表，与 run_dubins_gc_experiments 的 _get_obstacles_from_args 一致。"""
    if getattr(args, "obstacles", None) and len(args.obstacles) > 0:
        vals = list(args.obstacles)
        if len(vals) % 3 != 0:
            raise ValueError("--obstacles 必须是 3 的倍数个数字 (x, y, radius) 每组")
        return [
            CircleObstacle(x=float(vals[i]), y=float(vals[i + 1]), radius=float(vals[i + 2]))
            for i in range(0, len(vals), 3)
        ]
    config = getattr(args, "obstacle_config", "none") or "none"
    x_min, y_min, x_max, y_max = args.bounds
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    w, h = x_max - x_min, y_max - y_min
    if config == "none":
        return []
    if config == "simple":
        return [CircleObstacle(x=cx, y=cy, radius=0.12 * min(w, h))]
    if config == "medium":
        r = 0.10 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.35 * w, y=cy, radius=r),
            CircleObstacle(x=x_min + 0.65 * w, y=cy, radius=r),
            CircleObstacle(x=cx, y=y_min + 0.3 * h, radius=r * 0.8),
        ]
    if config == "hard":
        r = 0.08 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=cx, y=cy, radius=r * 1.2),
        ]
    return []


def _make_dubins_env(args) -> DubinsUAV2D:
    obstacles = _obstacles_from_args(args)
    collision_penalty = getattr(args, "collision_penalty", -10.0)
    env_kwargs = {
        "bounds": tuple(args.bounds),
        "omega_max": args.omega_max,
        "v": args.v,
        "dt": args.dt,
        "max_episode_steps": args.max_episode_steps,
        "epsilon_pos": args.epsilon_pos,
        "epsilon_theta": args.epsilon_theta,
        "obstacles": obstacles,
        "collision_penalty": collision_penalty,
        "use_cos_sin_obs": True,
    }
    return DubinsUAV2D(**env_kwargs)


def evaluate_success_rate(
    agent: GoalConditionedAgentBase,
    env: DubinsUAV2D,
    n_trials: int = 200,
    seed: int = 0,
    execution_mode: str = "act",
    lookahead_config: Optional[object] = None,
) -> Dict[str, float]:
    """
    成功率评估：随机起点、随机目标。

    - execution_mode=="act"：每步用 agent.act(obs, goal_obs, eval_mode=True) 选动作（默认）。
    - execution_mode=="lookahead"：每步用 Dubins lookahead planner（以 agent.value 为终端代价）选动作，
      用于与 act 策略及 QRL/TD 算法统一对比。需传入 lookahead_config（DubinsLookaheadConfig），
      若为 None 则使用默认配置。
    """
    if execution_mode == "lookahead":
        from minimal_qrl.eval.dubins_execution_mode_eval import (
            DubinsLookaheadConfig,
            dubins_lookahead_action,
        )
        if lookahead_config is None:
            lookahead_config = DubinsLookaheadConfig()

    success = 0
    steps_success: List[int] = []
    desc = "success_rate_lookahead" if execution_mode == "lookahead" else "success_rate"

    for i in tqdm(range(n_trials), desc=desc, leave=False):
        obs, _ = env.reset(seed=int(seed + i))
        goal_obs = env.state_to_observation(np.asarray(env.goal, dtype=np.float32))

        done = False
        truncated = False
        t = 0
        while not (done or truncated):
            if execution_mode == "lookahead":
                action = dubins_lookahead_action(agent, env, goal_obs, lookahead_config)
            else:
                action = agent.act(obs, goal_obs, eval_mode=True)
            next_obs, reward, done, truncated, info = env.step(action)
            obs = next_obs
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


def evaluate_triangle_inequality(
    agent: GoalConditionedAgentBase,
    env: DubinsUAV2D,
    n_triples: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    对随机三元组 (s, m, g) 检测 triangle inequality:
      V(s,g) <= V(s,m) + V(m,g)
    输出违反比例和平均违反幅度（仅统计违反样本的平均 max(0, lhs-rhs)）。
    """
    rng = np.random.RandomState(seed)
    states, goals = sample_state_goal_pairs(env, n_pairs=3 * n_triples, seed=seed)
    # 为简便：前 n_triples 个作为 s， 中间 n_triples 个作为 m，后 n_triples 个作为 g
    s_raw = states[:n_triples]
    m_raw = states[n_triples : 2 * n_triples]
    g_raw = goals[2 * n_triples : 3 * n_triples]

    u = env
    s_obs = np.array([u.state_to_observation(s) for s in s_raw], dtype=np.float32)
    m_obs = np.array([u.state_to_observation(m) for m in m_raw], dtype=np.float32)
    g_obs = np.array([u.state_to_observation(g) for g in g_raw], dtype=np.float32)

    lhs = []
    rhs = []
    violations = []
    for so, mo, go in tqdm(zip(s_obs, m_obs, g_obs), total=n_triples, desc="triangle", leave=False):
        v_sg = agent.value(so, go)
        v_sm = agent.value(so, mo)
        v_mg = agent.value(mo, go)
        lhs.append(v_sg)
        rhs_val = v_sm + v_mg
        rhs.append(rhs_val)
        violations.append(max(0.0, v_sg - rhs_val))

    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    violations = np.asarray(violations, dtype=np.float64)

    violation_mask = violations > 1e-8
    ratio = float(np.mean(violation_mask.astype(np.float32)))
    mean_mag = float(np.mean(violations[violation_mask])) if violation_mask.any() else 0.0
    return {
        "triangle_violation_ratio": ratio,
        "triangle_violation_mean_magnitude": mean_mag,
    }


def evaluate_value_true_time(
    agent: GoalConditionedAgentBase,
    env: DubinsUAV2D,
    n_pairs: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Value-to-True-Time 相关性：
    - 真值：Dubins 环境的 compute_min_time_to_go
    - 预测：agent.value(s,g)
    """
    states_raw, goals_raw = sample_state_goal_pairs(env, n_pairs=n_pairs, seed=seed)
    u = env
    states_obs = np.array([u.state_to_observation(s) for s in states_raw], dtype=np.float32)
    goals_obs = np.array([u.state_to_observation(g) for g in goals_raw], dtype=np.float32)

    pred_vals: List[float] = []
    true_times: List[float] = []
    for s_raw, g_raw, s_obs, g_obs in tqdm(
        zip(states_raw, goals_raw, states_obs, goals_obs), total=n_pairs, desc="value_true_time", leave=False
    ):
        v = agent.value(s_obs, g_obs)
        t_true = compute_ground_truth_distance(env, s_raw, g_raw)
        pred_vals.append(v)
        true_times.append(t_true)

    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    true_times = np.asarray(true_times, dtype=np.float64)

    # Pearson
    pred_c = pred_vals - pred_vals.mean()
    true_c = true_times - true_times.mean()
    num = np.sum(pred_c * true_c)
    den = np.sqrt(np.sum(pred_c ** 2) * np.sum(true_c ** 2)) + 1e-12
    pearson = float(num / den)

    # Spearman
    spearman_corr, _ = spearmanr(pred_vals, true_times)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    mse = float(np.mean((pred_vals - true_times) ** 2))

    return {
        "pearson_corr": pearson,
        "spearman_corr": float(spearman_corr),
        "mse": mse,
        "pred_mean": float(pred_vals.mean()),
        "true_mean": float(true_times.mean()),
    }


def build_qrl_adapter(args, device: torch.device, env: DubinsUAV2D) -> GoalConditionedAgentBase:
    """
    从 checkpoint 加载 QRL, 并适配为 GoalConditionedAgentBase。
    """
    # 注册一个 dummy offline env 以获取 env_spec
    env_key = ("dubins_uav", "dubins_uav")

    from quasimetric_rl.data.base import CREATE_ENV_REGISTRY  # type: ignore

    if env_key not in CREATE_ENV_REGISTRY:
        def create_env_fn():
            return _make_dubins_env(args)

        def load_episodes():
            e = create_env_fn()
            return create_dataset(e, num_episodes=1, max_steps_per_episode=10, seed=args.seed)

        register_offline_env("dubins_uav", "dubins_uav", create_env_fn=create_env_fn, load_episodes_fn=load_episodes)

    dataset_conf = Dataset.Conf(kind="dubins_uav", name="dubins_uav", future_observation_discount=0.99)
    dataset = dataset_conf.make(dummy=True)
    env_spec = dataset.env_spec

    agent_conf = QRLConf(actor=None, num_critics=2)
    qrl_agent, _ = agent_conf.make(env_spec=env_spec, total_optim_steps=1)
    ckpt = torch.load(args.qrl_ckpt, map_location=device)
    if isinstance(ckpt, dict) and "agent" in ckpt:
        qrl_agent.load_state_dict(ckpt["agent"])
    else:
        qrl_agent.load_state_dict(ckpt)
    qrl_agent.to(device)
    qrl_agent.eval()

    adapter = QRLGoalValueAdapter(qrl_agent, env=env, device=device)
    return adapter


def main():
    parser = argparse.ArgumentParser(description="Dubins UAV 上 QRL vs TD-based GC-RL 统一对比评估")
    parser.add_argument("--qrl-ckpt", type=str, required=True, help="QRL checkpoint 路径")
    parser.add_argument("--output-dir", type=str, default="./results/minimal_qrl_dubins_gc_benchmark")

    # Dubins 参数（需与训练保持一致）
    parser.add_argument("--bounds", type=float, nargs=4, default=[0, 0, 5, 5])
    parser.add_argument(
        "--obstacle-config",
        type=str,
        default="none",
        choices=["none", "simple", "medium", "hard"],
        help="障碍预设",
    )
    parser.add_argument("--obstacles", type=float, nargs="*", default=None, help="自定义圆障 (x y r x y r ...)")
    parser.add_argument("--collision-penalty", type=float, default=-10.0)
    parser.add_argument("--omega-max", type=float, default=0.5)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--epsilon-pos", type=float, default=0.15)
    parser.add_argument("--epsilon-theta", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    env = _make_dubins_env(args)

    # 当前脚本专注于评估，TD-based agent 的 checkpoint 加载接口可以按需扩展；
    # 这里先只对 QRL 进行完整几何指标评估，TD 部分由训练脚本负责构造相同接口后复用本模块。
    results: Dict[str, Dict[str, float]] = {}

    # 1) QRL
    qrl_agent = build_qrl_adapter(args, device, env)

    results["QRL_success"] = evaluate_success_rate(qrl_agent, env, n_trials=200, seed=args.seed)
    results["QRL_triangle"] = evaluate_triangle_inequality(qrl_agent, env, n_triples=1000, seed=args.seed + 1)
    results["QRL_value_true_time"] = evaluate_value_true_time(qrl_agent, env, n_pairs=2000, seed=args.seed + 2)

    out_json = os.path.join(args.output_dir, "gc_benchmark_qrl_only.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[gc_benchmark] 已保存 QRL 评估结果到 {out_json}")


if __name__ == "__main__":
    main()

