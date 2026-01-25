#!/usr/bin/env python3
"""
离线评估脚本：在不重新训练 QRL 的前提下，对比不同 execution mechanism 的 rollout 表现。

对比项（同一环境、同一 checkpoint、同一 start-goal 采样）：
- greedy
- lookahead（QRL-guided local planning / shooting）

输出：
- 控制台打印指标摘要
- 生成 failure case 可视化（可选）
- 生成 failure start 空间分布对比图（当同时评估多种 mode 且启用可视化时）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# 允许在 repo 根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quasimetric_rl.modules import QRLConf
from quasimetric_rl.data import Dataset, register_offline_env
from quasimetric_rl.data.base import CREATE_ENV_REGISTRY, LOAD_EPISODES_REGISTRY

from minimal_qrl.envs import ContinuousObstacle2D
from minimal_qrl.eval import evaluate_planning, LookaheadConfig


def _auto_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_registered_env(kind: str, name: str, *, create_env_fn, load_episodes_fn):
    key = (kind, name)
    if key in CREATE_ENV_REGISTRY and key in LOAD_EPISODES_REGISTRY:
        return
    register_offline_env(kind, name, create_env_fn=create_env_fn, load_episodes_fn=load_episodes_fn)


def main():
    parser = argparse.ArgumentParser(description="QRL execution mechanism 对比评估（greedy vs lookahead）")

    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径（*.pth）")
    parser.add_argument("--output-dir", type=str, default="./results/minimal_qrl", help="输出目录（保存 json/可视化）")

    # 环境
    parser.add_argument("--env-name", type=str, default="obstacle2d", help="环境注册名")
    parser.add_argument("--max-steps", type=int, default=200, help="max_episode_steps")
    parser.add_argument("--grid-resolution", type=int, default=80, help="A* 网格分辨率（仅影响 GT 距离计算）")

    # 评估
    parser.add_argument("--n-trials", type=int, default=200, help="评估对数量（start-goal pairs）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（采样与动作随机性）")
    parser.add_argument("--num-action-candidates", type=int, default=32, help="greedy 每步候选动作数")
    parser.add_argument(
        "--execution-modes",
        type=str,
        default="greedy,lookahead",
        help='逗号分隔，例如 "greedy" 或 "greedy,lookahead"',
    )
    parser.add_argument("--visualize-failures", action="store_true", help="是否生成 failure case 可视化")

    # lookahead 参数
    parser.add_argument("--lookahead-horizon", type=int, default=5)
    parser.add_argument("--lookahead-num-sequences", type=int, default=64)
    parser.add_argument("--lookahead-step-cost-weight", type=float, default=0.0)
    parser.add_argument("--lookahead-collision-penalty", type=float, default=0.0)

    # 模型结构（需与训练一致）
    parser.add_argument("--num-critics", type=int, default=2, help="Critic 数量（需与训练一致）")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = _auto_device(args.device)

    # 构建环境（与训练一致）
    env_kwargs = dict(max_episode_steps=int(args.max_steps), grid_resolution=int(args.grid_resolution))

    def create_env_fn():
        return ContinuousObstacle2D(**env_kwargs)

    def load_episodes_fn():
        return iter(())

    _ensure_registered_env("obstacle", args.env_name, create_env_fn=create_env_fn, load_episodes_fn=load_episodes_fn)

    # dummy Dataset：只为拿 env_spec
    dataset_conf = Dataset.Conf(kind="obstacle", name=args.env_name, future_observation_discount=0.99)
    dataset = dataset_conf.make(dummy=True)

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
        agent.load_state_dict(ckpt)
        ckpt_step = None

    env = create_env_fn()

    execution_modes = [m.strip() for m in str(args.execution_modes).split(",") if m.strip()]

    lookahead_cfg = None
    if "lookahead" in execution_modes:
        lookahead_cfg = LookaheadConfig(
            horizon=int(args.lookahead_horizon),
            num_sequences=int(args.lookahead_num_sequences),
            step_cost_weight=float(args.lookahead_step_cost_weight),
            collision_penalty=float(args.lookahead_collision_penalty),
        )

    # 运行评估
    np.random.seed(int(args.seed))
    results = evaluate_planning(
        agent=agent,
        env=env,
        n_trials=int(args.n_trials),
        device=str(device),
        seed=int(args.seed),
        num_action_candidates=int(args.num_action_candidates),
        visualize_failures=bool(args.visualize_failures),
        output_dir=str(args.output_dir),
        step=int(ckpt_step) if ckpt_step is not None else 0,
        execution_modes=execution_modes,
        lookahead_config=lookahead_cfg,
    )

    # 保存 json
    tag = f"step{int(ckpt_step):05d}" if ckpt_step is not None else "checkpoint"
    out_json = os.path.join(args.output_dir, f"execution_mode_eval_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print(f"[execution_mode_eval] saved: {out_json}")
    for mode in execution_modes:
        key = "greedy_navigation" if mode == "greedy" else f"{mode}_navigation"
        if key in results:
            m = results[key]
            print(
                f"- {mode}: success_rate={m.get('success_rate', 0.0):.3f}, "
                f"avg_steps={m.get('avg_steps', 0.0):.1f}, "
                f"avg_path_length={m.get('avg_path_length', 0.0):.3f}"
            )
    pe_by_mode = results.get("path_efficiency_by_mode", {})
    if isinstance(pe_by_mode, dict):
        for mode, pe in pe_by_mode.items():
            if not isinstance(pe, dict):
                continue
            print(
                f"- {mode}: avg_eff={float(pe.get('avg_efficiency_ratio', 0.0)):.3f}, "
                f"median_eff={float(pe.get('median_efficiency_ratio', 0.0)):.3f}"
            )


if __name__ == "__main__":
    main()

