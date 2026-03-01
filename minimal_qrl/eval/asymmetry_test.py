#!/usr/bin/env python3
"""
QRL 非对称性统计检验

对学到的 d(s,g) 检验是否具有方向性：
- 若 Δ = d(s,g) - d(g,s) ≈ 0，说明仍偏向对称 metric
- 若 Δ 有系统偏移（尤其在 heading 不同时），说明学到 Dubins 方向性

用法:
  # 随机采样检验
  python -m minimal_qrl.eval.asymmetry_test --checkpoint ./results/minimal_qrl_dubins_initial/checkpoint_final.pth --output-dir ./results/minimal_qrl_dubins_initial
  # 针对性检验：固定 s=(2.5,2.5,0)，比较 d(s,g_back) vs d(s,g_front)，验证局部方向性
  python -m minimal_qrl.eval.asymmetry_test --checkpoint ./results/minimal_qrl_dubins_initial/checkpoint_final.pth --output-dir ./results/minimal_qrl_dubins_initial --targeted
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 项目根
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quasimetric_rl.data import EnvSpec
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.data.base import register_offline_env
from minimal_qrl.envs import DubinsUAV2D
from minimal_qrl.dataset import create_dataset
from minimal_qrl.eval.planning_evaluation import sample_state_goal_pairs


def _normalize_angle(theta: float) -> float:
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta


def _run_targeted_test(agent, env, device, output_dir: str):
    """针对性非对称检验：固定 s=(2.5,2.5,0)，g_front 在前方、g_back 在后方，检验 d(s,g_back) >> d(s,g_front)。"""
    u = env
    # 固定状态（地图中心，朝向 0 = 东/右）
    s = np.array([2.5, 2.5, 0.0], dtype=np.float32)
    g_front = np.array([3.5, 2.5, 0.0], dtype=np.float32)  # 前方同向
    g_back = np.array([1.5, 2.5, 0.0], dtype=np.float32)   # 后方，需调头

    s_obs = u.state_to_observation(s)
    g_front_obs = u.state_to_observation(g_front)
    g_back_obs = u.state_to_observation(g_back)

    critic = agent.critics[0]
    with torch.no_grad():
        s_t = torch.tensor(s_obs[None], device=device, dtype=torch.float32)
        gf_t = torch.tensor(g_front_obs[None], device=device, dtype=torch.float32)
        gb_t = torch.tensor(g_back_obs[None], device=device, dtype=torch.float32)
        zs = critic.encoder(s_t)
        zgf = critic.encoder(gf_t)
        zgb = critic.encoder(gb_t)
        d_front = critic.quasimetric_model(zs, zgf).cpu().item()
        d_back = critic.quasimetric_model(zs, zgb).cpu().item()

    scale = u.get_distance_scale() if hasattr(u, "get_distance_scale") else 1.0
    gap = d_back - d_front
    ratio = d_back / (d_front + 1e-8)

    # 图：两根柱子 d_front vs d_back
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar([0], [d_front], width=0.35, label=r"$d(s,\,g_{\mathrm{front}})$", color="steelblue")
    ax.bar([1], [d_back], width=0.35, label=r"$d(s,\,g_{\mathrm{back}})$", color="coral")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$g_{\mathrm{front}}$ (ahead)", r"$g_{\mathrm{back}}$ (behind)"])
    ax.set_ylabel("Distance")
    ax.set_title(r"Targeted test: $s=(2.5,2.5,0)$, same heading")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_plot = os.path.join(output_dir, "targeted_asymmetry_test.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_plot}")

    print("\n===== 针对性非对称检验 =====")
    print(f"  固定 s = (2.5, 2.5, 0)  朝向 0（东）")
    print(f"  g_front = (3.5, 2.5, 0)  前方")
    print(f"  g_back  = (1.5, 2.5, 0)  后方")
    print(f"  d(s, g_front) = {d_front:.4f}" + (f"  (time: {d_front * scale:.4f})" if scale != 1 else ""))
    print(f"  d(s, g_back)  = {d_back:.4f}" + (f"  (time: {d_back * scale:.4f})" if scale != 1 else ""))
    print(f"  差距 gap = d_back - d_front = {gap:.4f}")
    print(f"  比值 d_back/d_front = {ratio:.4f}")
    if gap > 0.5:
        print("  解读: 差距显著，d(s,g_back) > d(s,g_front)，局部方向性正确。")
    elif gap > 0:
        print("  解读: 差距为正但较小，有一定方向性。")
    else:
        print("  解读: 差距非正，未体现「后方更远」的方向性。")


def main():
    parser = argparse.ArgumentParser(description="QRL 非对称性统计检验：Δ = d(s,g) - d(g,s)")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径 (*.pth)")
    parser.add_argument("--output-dir", type=str, default="./results/asymmetry_test", help="输出目录")
    parser.add_argument("--n-pairs", type=int, default=500, help="采样 (s,g) 对数")
    parser.add_argument("--seed", type=int, default=42)
    # 与训练一致的 Dubins 参数
    parser.add_argument("--bounds", type=float, nargs=4, default=[0, 0, 5, 5], metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--omega-max", type=float, default=0.5)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--targeted", action="store_true", help="Fixed s; compare d(s,g_back) vs d(s,g_front) for local directionality")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 环境（与训练一致：use_cos_sin_obs=True）
    env_kwargs = {
        "bounds": tuple(args.bounds),
        "omega_max": args.omega_max,
        "v": args.v,
        "dt": args.dt,
        "max_episode_steps": 200,
        "epsilon_pos": 0.15,
        "epsilon_theta": 0.2,
        "obstacles": [],
        "use_cos_sin_obs": True,
    }
    env = DubinsUAV2D(**env_kwargs)
    env_spec = EnvSpec.from_env(env)

    # 注册并构造 dataset 以兼容 make(dummy=True) 的 env_spec（若需要与 train 完全一致可用 register）
    from quasimetric_rl.data import Dataset
    from quasimetric_rl.data.base import CREATE_ENV_REGISTRY
    env_key = ("dubins_uav", "dubins_uav")
    if env_key not in CREATE_ENV_REGISTRY:
        def create_env_fn():
            return DubinsUAV2D(**env_kwargs)
        def load_episodes():
            e = create_env_fn()
            return create_dataset(e, num_episodes=1, max_steps_per_episode=10, seed=args.seed)
        register_offline_env("dubins_uav", "dubins_uav", create_env_fn=create_env_fn, load_episodes_fn=load_episodes)
    dataset_conf = Dataset.Conf(kind="dubins_uav", name="dubins_uav", future_observation_discount=0.99)
    dataset = dataset_conf.make(dummy=True)
    env_spec = dataset.env_spec

    # Agent
    agent_conf = QRLConf(actor=None, num_critics=2)
    agent, _ = agent_conf.make(env_spec=env_spec, total_optim_steps=1)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
    else:
        agent.load_state_dict(ckpt)
    agent.to(device)
    agent.eval()

    if args.targeted:
        _run_targeted_test(agent, env, device, args.output_dir)
        return

    # 采样 N 对 (s, g)，s/g 为内部状态 (x,y,theta)
    states_raw, goals_raw = sample_state_goal_pairs(env, n_pairs=args.n_pairs, seed=args.seed)
    u = env
    states_obs = np.array([u.state_to_observation(s) for s in states_raw], dtype=np.float32)
    goals_obs = np.array([u.state_to_observation(g) for g in goals_raw], dtype=np.float32)

    # A = d(s,g), B = d(g,s)，使用 raw 距离（未乘 dt），Delta 与尺度无关，仅看方向性
    critic = agent.critics[0]
    with torch.no_grad():
        s_t = torch.tensor(states_obs, device=device, dtype=torch.float32)
        g_t = torch.tensor(goals_obs, device=device, dtype=torch.float32)
        zs = critic.encoder(s_t)
        zg = critic.encoder(g_t)
        A = critic.quasimetric_model(zs, zg).cpu().numpy().flatten()
        B = critic.quasimetric_model(zg, zs).cpu().numpy().flatten()

    delta = A - B
    scale = u.get_distance_scale() if hasattr(u, "get_distance_scale") else 1.0
    delta_time = delta * scale  # 若需要“时间差”解释可选用

    mean_delta = float(np.mean(delta))
    var_delta = float(np.var(delta))
    std_delta = float(np.std(delta))

    # 按 heading 差分层：|goal[2]-start[2]| 归一化到 [0, pi]
    heading_diff = np.array([
        abs(_normalize_angle(goals_raw[i][2] - states_raw[i][2])) for i in range(len(states_raw))
    ])
    bins_deg = [0, 45, 90, 135, 180]
    bins_rad = [b * np.pi / 180 for b in bins_deg]
    bin_indices = np.digitize(heading_diff, bins_rad)  # 1..5，5 个边界成 4 档
    bin_indices = np.clip(bin_indices, 1, 4)
    bin_labels = [f"{bins_deg[i]}°~{bins_deg[i+1]}°" for i in range(len(bins_deg) - 1)]

    # 直方图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    ax0.hist(delta, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax0.axvline(0, color="black", linestyle="--", linewidth=1)
    ax0.axvline(mean_delta, color="red", linestyle="-", linewidth=1.5, label=f"mean = {mean_delta:.4f}")
    ax0.set_xlabel(r"$\Delta = d(s,g) - d(g,s)$")
    ax0.set_ylabel("Count")
    ax0.set_title(r"Asymmetry: distribution of $\Delta$")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # 按 heading 差的均值
    ax1 = axes[1]
    means_per_bin = []
    stds_per_bin = []
    for b in range(len(bin_labels)):
        mask = bin_indices == b + 1
        if mask.sum() == 0:
            means_per_bin.append(0.0)
            stds_per_bin.append(0.0)
        else:
            means_per_bin.append(np.mean(delta[mask]))
            stds_per_bin.append(np.std(delta[mask]))
    x_pos = np.arange(len(bin_labels))
    ax1.bar(x_pos - 0.2, means_per_bin, width=0.4, label=r"mean $\Delta$", color="steelblue")
    ax1.errorbar(x_pos + 0.2, means_per_bin, yerr=stds_per_bin, fmt="none", color="black", capsize=3)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_labels)
    ax1.set_xlabel(r"Heading diff $|\theta_g - \theta_s|$ (deg)")
    ax1.set_ylabel(r"mean $\Delta$ $\pm$ std")
    ax1.set_title(r"Mean $\Delta$ by heading difference")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out_plot = os.path.join(args.output_dir, "asymmetry_test_histogram.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_plot}")

    # 控制台统计
    print("\n===== 非对称性统计 =====")
    print(f"  N = {args.n_pairs} 对 (s,g)")
    print(f"  Δ = d(s,g) - d(g,s)")
    print(f"  均值(Δ) = {mean_delta:.4f}")
    print(f"  方差(Δ) = {var_delta:.4f}")
    print(f"  标准差(Δ) = {std_delta:.4f}")
    if abs(mean_delta) < 0.05 and std_delta < 0.1:
        print("  解读: Δ ≈ 0，模型仍偏向对称 metric。")
    else:
        print("  解读: Δ 存在系统偏移或散布，说明学到方向性/非对称性。")
    print("\n按朝向差分层均值(Δ):")
    for i, label in enumerate(bin_labels):
        m = means_per_bin[i] if i < len(means_per_bin) else 0
        s = stds_per_bin[i] if i < len(stds_per_bin) else 0
        print(f"  {label}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
