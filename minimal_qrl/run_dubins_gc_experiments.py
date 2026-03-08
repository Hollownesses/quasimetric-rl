#!/usr/bin/env python3
"""
Dubins UAV 无障碍环境上 QRL vs TD-based goal-conditioned RL 的统一实验脚本。

功能：
- 复用现有 QRL checkpoint（不改动 QRL 训练逻辑）
- 在同一 Dubins 环境上训练：
    1) HER + DDPG
    2) Goal-conditioned SAC
    3) UVFA-style value learning
- 使用统一训练预算（total_env_steps）
- 使用统一评估模块 minimal_qrl.eval.gc_benchmark 中的指标：
    - 成功率
    - Asymmetry Gap
    - Triangle Inequality Violation
    - Value-to-True-Time 相关性
    - OOD Goal Generalization
- 将所有方法的指标整理为对比表（json + 简单 csv），便于论文整理。

示例：

  python -m minimal_qrl.run_dubins_gc_experiments \\
      --qrl-ckpt results/minimal_qrl_dubins_initial/checkpoint_final.pth \\
      --output-dir results/minimal_qrl_dubins_benchmark \\
      --total-env-steps 200000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimal_qrl.envs import DubinsUAV2D, CircleObstacle
from minimal_qrl.gc_agents import (
    AlgoConfig,
    train_td_agent,
    HERDDPGAgent,
    GCSACAgent,
    UVFAValueAgent,
)
from minimal_qrl.eval.gc_benchmark import (
    evaluate_success_rate,
    evaluate_asymmetry_gap,
    evaluate_triangle_inequality,
    evaluate_value_true_time,
    evaluate_ood_generalization,
    build_qrl_adapter,
)
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig


def _get_obstacles_from_args(args) -> List[CircleObstacle]:
    """根据 args 生成圆形障碍列表。优先使用 --obstacles，否则用 --obstacle-config 预设。"""
    if getattr(args, "obstacles", None) and len(args.obstacles) > 0:
        # --obstacles x1 y1 r1 x2 y2 r2 ...
        vals = list(args.obstacles)
        if len(vals) % 3 != 0:
            raise ValueError("--obstacles 必须是 3 的倍数个数字 (x, y, radius) 每组")
        out = []
        for i in range(0, len(vals), 3):
            out.append(CircleObstacle(x=float(vals[i]), y=float(vals[i + 1]), radius=float(vals[i + 2])))
        return out
    config = getattr(args, "obstacle_config", "none") or "none"
    x_min, y_min, x_max, y_max = args.bounds
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w = x_max - x_min
    h = y_max - y_min
    if config == "none":
        return []
    if config == "simple":
        # 单个中心圆
        return [CircleObstacle(x=cx, y=cy, radius=0.12 * min(w, h))]
    if config == "medium":
        # 2～3 个圆，形成通道
        r = 0.10 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.35 * w, y=cy, radius=r),
            CircleObstacle(x=x_min + 0.65 * w, y=cy, radius=r),
            CircleObstacle(x=cx, y=y_min + 0.3 * h, radius=r * 0.8),
        ]
    if config == "hard":
        # 4～5 个圆，窄通道
        r = 0.08 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=cx, y=cy, radius=r * 1.2),
        ]
    return []


def make_env(args) -> DubinsUAV2D:
    obstacles = _get_obstacles_from_args(args)
    collision_penalty = getattr(args, "collision_penalty", -10.0)
    return DubinsUAV2D(
        bounds=tuple(args.bounds),
        omega_max=args.omega_max,
        v=args.v,
        dt=args.dt,
        max_episode_steps=args.max_episode_steps,
        epsilon_pos=args.epsilon_pos,
        epsilon_theta=args.epsilon_theta,
        obstacles=obstacles,
        collision_penalty=collision_penalty,
        use_cos_sin_obs=True,
    )


def _make_algo_config(args) -> AlgoConfig:
    return AlgoConfig(
        total_env_steps=args.total_env_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        action_noise_std=args.action_noise_std,
        start_random_steps=args.start_random_steps,
        her_k=args.her_k,
        sac_alpha=args.sac_alpha,
        target_entropy=None,
        log_interval=args.log_interval,
    )


def load_td_agent(
    algo_name: str,
    env: DubinsUAV2D,
    device: torch.device,
    args,
    ckpt_path: str,
):
    """从已有 checkpoint 加载 TD 算法 agent，不训练。"""
    cfg = _make_algo_config(args)
    if algo_name == "her_ddpg":
        agent = HERDDPGAgent(env, cfg, device)
    elif algo_name == "gc_sac":
        agent = GCSACAgent(env, cfg, device)
    elif algo_name == "uvfa":
        agent = UVFAValueAgent(env, cfg, device)
    else:
        raise ValueError(f"未知算法: {algo_name}")
    state = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


def run_single_algo(
    algo_name: str,
    env: DubinsUAV2D,
    device: torch.device,
    args,
) -> Dict[str, Dict]:
    """
    在给定环境上训练一个 TD-based 算法（或从已有 checkpoint 加载），然后运行统一评估。
    若提供了该算法对应的 --her-ddpg-ckpt / --gc-sac-ckpt / --uvfa-ckpt，则跳过训练，直接加载并评估。
    """
    # 是否使用已有 checkpoint（参数名 her_ddpg_ckpt, gc_sac_ckpt, uvfa_ckpt）
    ckpt_arg = getattr(args, f"{algo_name}_ckpt", None)
    out_algo_dir = os.path.join(args.output_dir, algo_name)
    os.makedirs(out_algo_dir, exist_ok=True)

    if ckpt_arg is not None and ckpt_arg.strip() != "":
        # 从已有 checkpoint 加载，只做评估
        print(f"  使用已有 checkpoint: {ckpt_arg}")
        agent = load_td_agent(algo_name, env, device, args, ckpt_arg.strip())
    else:
        # 训练并保存
        cfg = _make_algo_config(args)
        def _log(step: int, stats: Dict[str, float]) -> None:
            with open(os.path.join(out_algo_dir, "train.log"), "a", encoding="utf-8") as f:
                f.write(f"step={step} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()) + "\n")
        agent = train_td_agent(
            algo=algo_name,
            env=env,
            cfg=cfg,
            device=device,
            train_goal_radius=args.r_train,
            log_fn=_log,
        )
        ckpt_path = os.path.join(out_algo_dir, "checkpoint_final.pth")
        torch.save(agent.state_dict(), ckpt_path)

    # 评估（使用同一个 env 实例，各指标内已有 tqdm）
    results: Dict[str, Dict] = {}
    results["success"] = evaluate_success_rate(agent, env, n_trials=args.eval_n_trials, seed=args.seed)
    if getattr(args, "eval_lookahead", False):
        la_cfg = DubinsLookaheadConfig(
            horizon=args.lookahead_horizon,
            num_sequences=args.lookahead_num_sequences,
            biased_sequences=args.lookahead_biased_sequences,
            bias_kp=args.lookahead_bias_kp,
            step_cost_weight=args.lookahead_step_cost_weight,
            collision_penalty=args.lookahead_collision_penalty,
        )
        results["success_lookahead"] = evaluate_success_rate(
            agent, env, n_trials=args.eval_n_trials, seed=args.seed,
            execution_mode="lookahead", lookahead_config=la_cfg,
        )
    results["asymmetry"] = evaluate_asymmetry_gap(agent, env, n_pairs=args.eval_n_pairs, seed=args.seed + 1)
    results["triangle"] = evaluate_triangle_inequality(agent, env, n_triples=args.eval_n_pairs, seed=args.seed + 2)
    results["value_true_time"] = evaluate_value_true_time(agent, env, n_pairs=args.eval_n_pairs, seed=args.seed + 3)
    results["ood"] = evaluate_ood_generalization(
        agent, env, r_train=args.r_train, r_test=args.r_test, n_pairs=args.eval_n_pairs * 2, seed=args.seed + 4
    )

    out_json = os.path.join(out_algo_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Dubins UAV QRL vs TD-based GC-RL 统一实验脚本")
    parser.add_argument("--qrl-ckpt", type=str, required=True, help="已有 QRL checkpoint 路径")
    parser.add_argument("--output-dir", type=str, default="./results/minimal_qrl_dubins_benchmark")
    parser.add_argument("--her-ddpg-ckpt", type=str, default=None, help="已有 HER-DDPG checkpoint；若提供则跳过训练、直接加载评估")
    parser.add_argument("--gc-sac-ckpt", type=str, default=None, help="已有 GC-SAC checkpoint；若提供则跳过训练、直接加载评估")
    parser.add_argument("--uvfa-ckpt", type=str, default=None, help="已有 UVFA checkpoint；若提供则跳过训练、直接加载评估")

    # Dubins 环境配置（需与 QRL 训练阶段保持一致）
    parser.add_argument("--bounds", type=float, nargs=4, default=[0, 0, 5, 5])
    parser.add_argument(
        "--obstacle-config",
        type=str,
        default="none",
        choices=["none", "simple", "medium", "hard"],
        help="障碍预设：none=无障碍, simple=单圆, medium=2～3 圆, hard=4～5 圆",
    )
    parser.add_argument(
        "--obstacles",
        type=float,
        nargs="*",
        default=None,
        help="自定义圆形障碍 (x1 y1 r1 x2 y2 r2 ...)，若提供则忽略 --obstacle-config",
    )
    parser.add_argument("--collision-penalty", type=float, default=-10.0, help="碰撞时奖励惩罚")
    parser.add_argument("--omega-max", type=float, default=0.5)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--epsilon-pos", type=float, default=0.15)
    parser.add_argument("--epsilon-theta", type=float, default=0.2)

    # 训练预算 & TD 超参
    parser.add_argument("--total-env-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--action-noise-std", type=float, default=0.2)
    parser.add_argument("--start-random-steps", type=int, default=1_000)
    parser.add_argument("--her-k", type=int, default=4)
    parser.add_argument("--sac-alpha", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=10_000)

    # 评估
    parser.add_argument("--eval-n-trials", type=int, default=200)
    parser.add_argument("--eval-n-pairs", type=int, default=2000)
    parser.add_argument("--eval-lookahead", action="store_true", help="同时用 lookahead planner 评估成功率，与 act 策略对比")
    parser.add_argument("--lookahead-horizon", type=int, default=10)
    parser.add_argument("--lookahead-num-sequences", type=int, default=128)
    parser.add_argument("--lookahead-biased-sequences", type=int, default=48)
    parser.add_argument("--lookahead-bias-kp", type=float, default=2.0)
    parser.add_argument("--lookahead-step-cost-weight", type=float, default=0.0)
    parser.add_argument("--lookahead-collision-penalty", type=float, default=0.0)

    # OOD 设置
    parser.add_argument("--r-train", type=float, default=1.5)
    parser.add_argument("--r-test", type=float, default=2.0)

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

    # 固定随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1) 构建环境
    env = make_env(args)

    # 2) 训练 TD-based 三个算法并评估
    all_results: Dict[str, Dict] = {}
    for algo in tqdm(["her_ddpg", "gc_sac", "uvfa"], desc="Algorithms (train+eval)", unit="algo"):
        print(f"\n=== 训练并评估 {algo} ===")
        res = run_single_algo(algo, env, device, args)
        all_results[algo] = res

    # 3) 加载 QRL 并进行同一评估
    from minimal_qrl.eval.gc_benchmark import _make_dubins_env  # type: ignore

    # 使用与 TD 相同参数的新 env 实例，保证初始状态一致性
    qrl_env = make_env(args)
    qrl_adapter = build_qrl_adapter(args, device, qrl_env)
    qrl_results = {
        "success": evaluate_success_rate(qrl_adapter, qrl_env, n_trials=args.eval_n_trials, seed=args.seed),
        "asymmetry": evaluate_asymmetry_gap(qrl_adapter, qrl_env, n_pairs=args.eval_n_pairs, seed=args.seed + 1),
        "triangle": evaluate_triangle_inequality(qrl_adapter, qrl_env, n_triples=args.eval_n_pairs, seed=args.seed + 2),
        "value_true_time": evaluate_value_true_time(qrl_adapter, qrl_env, n_pairs=args.eval_n_pairs, seed=args.seed + 3),
        "ood": evaluate_ood_generalization(
            qrl_adapter,
            qrl_env,
            r_train=args.r_train,
            r_test=args.r_test,
            n_pairs=args.eval_n_pairs * 2,
            seed=args.seed + 4,
        ),
    }
    if getattr(args, "eval_lookahead", False):
        la_cfg = DubinsLookaheadConfig(
            horizon=args.lookahead_horizon,
            num_sequences=args.lookahead_num_sequences,
            biased_sequences=args.lookahead_biased_sequences,
            bias_kp=args.lookahead_bias_kp,
            step_cost_weight=args.lookahead_step_cost_weight,
            collision_penalty=args.lookahead_collision_penalty,
        )
        qrl_results["success_lookahead"] = evaluate_success_rate(
            qrl_adapter, qrl_env, n_trials=args.eval_n_trials, seed=args.seed,
            execution_mode="lookahead", lookahead_config=la_cfg,
        )
    all_results["qrl"] = qrl_results

    # 4) 保存综合 json
    out_json = os.path.join(args.output_dir, "all_algorithms_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[run_dubins_gc_experiments] 所有指标已保存到 {out_json}")

    # 5) 生成一个简单 CSV 对比表（主要几个核心指标）
    csv_path = os.path.join(args.output_dir, "summary_table.csv")
    has_lookahead = getattr(args, "eval_lookahead", False)
    header = [
        "algo",
        "success_rate",
        "asym_gap",
        "asym_gap_normalized",
        "triangle_violation_ratio",
        "triangle_violation_mean",
        "pearson_corr",
        "spearman_corr",
        "mse",
        "ood_id_pearson",
        "ood_ood_pearson",
        "ood_id_success",
        "ood_ood_success",
    ]
    if has_lookahead:
        header.extend(["success_rate_lookahead", "avg_steps_success_lookahead"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for algo, res in all_results.items():
            succ = res["success"]
            asym = res["asymmetry"]
            tri = res["triangle"]
            vtt = res["value_true_time"]
            ood = res["ood"]
            row = [
                algo,
                succ.get("success_rate", 0.0),
                asym.get("asym_gap", 0.0),
                asym.get("asym_gap_normalized", 0.0),
                tri.get("triangle_violation_ratio", 0.0),
                tri.get("triangle_violation_mean_magnitude", 0.0),
                vtt.get("pearson_corr", 0.0),
                vtt.get("spearman_corr", 0.0),
                vtt.get("mse", 0.0),
                ood["id"].get("pearson_corr", 0.0),
                ood["ood"].get("pearson_corr", 0.0),
                ood["id"].get("success_success_rate", 0.0),
                ood["ood"].get("success_success_rate", 0.0),
            ]
            if has_lookahead:
                sla = res.get("success_lookahead") or {}
                row.append(sla.get("success_rate", 0.0))
                row.append(sla.get("avg_steps_success", 0.0))
            writer.writerow(row)
    print(f"[run_dubins_gc_experiments] 核心对比表已保存到 {csv_path}")


if __name__ == "__main__":
    main()

