#!/usr/bin/env python3
"""Train and evaluate TD-based baselines on the communication-inspection Dubins task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimal_qrl.eval.comm_inspection_execution_eval import (
    VisualizationConfig,
    build_qrl_adapter,
    evaluate_execution_mode,
    make_comm_inspection_env,
)
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.gc_agents import (
    AlgoConfig,
    GCSACAgent,
    HERDDPGAgent,
    UVFAValueAgent,
    GoalConditionedAgentBase,
    train_td_agent,
)


TD_AGENT_TYPES = {
    "gc_sac": GCSACAgent,
    "her_ddpg": HERDDPGAgent,
    "uvfa": UVFAValueAgent,
}


def _bool_from_env_flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _make_algo_config(args: argparse.Namespace) -> AlgoConfig:
    return AlgoConfig(
        total_env_steps=int(args.total_env_steps),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        tau=float(args.tau),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        value_lr=float(args.value_lr),
        action_noise_std=float(args.action_noise_std),
        start_random_steps=int(args.start_random_steps),
        her_k=int(args.her_k),
        sac_alpha=float(args.sac_alpha),
        target_entropy=None,
        log_interval=int(args.log_interval),
    )


def _checkpoint_arg_name(algo: str) -> str:
    return algo.replace("-", "_") + "_ckpt"


def _new_td_agent(algo: str, env, cfg: AlgoConfig, device: torch.device) -> GoalConditionedAgentBase:
    try:
        cls = TD_AGENT_TYPES[algo]
    except KeyError as exc:
        raise ValueError(f"Unknown TD baseline: {algo}") from exc
    return cls(env, cfg, device)


def _train_or_load_td_agent(
    algo: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[GoalConditionedAgentBase, Optional[str]]:
    cfg = _make_algo_config(args)
    out_dir = Path(args.output_dir) / algo
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_arg = getattr(args, _checkpoint_arg_name(algo), None)
    ckpt_path = str(ckpt_arg).strip() if ckpt_arg else ""

    env = make_comm_inspection_env(args)
    if ckpt_path:
        agent = _new_td_agent(algo, env, cfg, device)
        state = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(state)
        agent.eval()
        return agent, ckpt_path

    if bool(args.skip_td_training):
        expected = out_dir / "checkpoint_final.pth"
        if not expected.exists():
            raise FileNotFoundError(
                f"--skip-td-training was set but no checkpoint was found for {algo}: {expected}"
            )
        agent = _new_td_agent(algo, env, cfg, device)
        agent.load_state_dict(torch.load(expected, map_location=device))
        agent.eval()
        return agent, str(expected)

    log_path = out_dir / "train.log"

    def _log(step: int, stats: Dict[str, float]) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"step={step} " + " ".join(f"{k}={float(v):.6f}" for k, v in stats.items()) + "\n")

    agent = train_td_agent(
        algo=algo,
        env=env,
        cfg=cfg,
        device=device,
        train_goal_radius=None,
        log_fn=_log,
    )
    ckpt_path = str(out_dir / "checkpoint_final.pth")
    torch.save(agent.state_dict(), ckpt_path)
    agent.eval()
    return agent, ckpt_path


def _evaluate_agent(
    agent: GoalConditionedAgentBase,
    args: argparse.Namespace,
    *,
    method_prefix: str,
    execution_modes: List[str],
    lookahead_heuristics: str,
    device: torch.device,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    env = make_comm_inspection_env(args)
    if hasattr(agent, "env"):
        agent.env = env
    lookahead_cfg = DubinsLookaheadConfig(
        horizon=int(args.lookahead_horizon),
        num_sequences=int(args.lookahead_num_sequences),
        step_cost_weight=float(args.lookahead_step_cost_weight),
        collision_penalty=float(args.lookahead_collision_penalty),
        biased_sequences=int(args.lookahead_biased_sequences),
        bias_kp=float(args.lookahead_bias_kp),
        use_cem=bool(args.lookahead_use_cem),
        cem_iters=int(args.lookahead_cem_iters),
        cem_elite_frac=float(args.lookahead_cem_elite_frac),
        cem_std_init_frac=float(args.lookahead_cem_std_init_frac),
        alpha_final=float(args.planner_alpha_final),
        alpha_task_terminal=float(args.planner_alpha_task_terminal),
        use_env_stage_cost=bool(args.planner_use_env_stage_cost),
        heuristic_mode="terminal",
        qrl_progress_alpha=float(args.planner_qrl_progress_alpha),
    )
    viz_cfg = VisualizationConfig(
        save_visualizations=bool(args.save_visualizations),
        max_successes=int(args.viz_max_successes),
        max_failures=int(args.viz_max_failures),
        save_gif=False,
        gif_fps=8,
    )
    out: Dict[str, Dict[str, float]] = {}
    visualizations: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    parsed_lookahead_heuristics = _parse_lookahead_heuristics(lookahead_heuristics)
    for mode in execution_modes:
        if mode == "lookahead":
            multi_heuristic = len(parsed_lookahead_heuristics) > 1
            for heuristic in parsed_lookahead_heuristics:
                mode_key = (
                    "lookahead"
                    if not multi_heuristic and heuristic == "terminal"
                    else f"lookahead_{heuristic}"
                )
                metric_key = f"{method_prefix}_{mode_key}"
                metrics, vis_index = evaluate_execution_mode(
                    agent,
                    env,
                    "lookahead",
                    n_trials=int(args.n_trials),
                    seed=int(args.seed),
                    lookahead_cfg=replace(lookahead_cfg, heuristic_mode=heuristic),
                    output_dir=Path(args.output_dir),
                    viz_cfg=viz_cfg,
                    result_name=metric_key,
                )
                out[metric_key] = metrics
                visualizations[metric_key] = vis_index
            continue

        metric_key = f"{method_prefix}_{mode}"
        metrics, vis_index = evaluate_execution_mode(
            agent,
            env,
            "greedy",
            n_trials=int(args.n_trials),
            seed=int(args.seed),
            lookahead_cfg=None,
            output_dir=Path(args.output_dir),
            viz_cfg=viz_cfg,
            result_name=metric_key,
        )
        out[metric_key] = metrics
        visualizations[metric_key] = vis_index
    return out, visualizations


def _write_csv(path: str, results: Dict[str, Dict[str, float]]) -> None:
    metric_names = [
        "success_rate",
        "avg_steps_success",
        "avg_steps_all",
        "avg_total_cost",
        "avg_cost_per_step",
        "ever_task_feasible_rate",
        "avg_first_task_feasible_step",
        "observation_feasible_ratio",
        "communication_feasible_ratio",
        "task_feasible_ratio",
        "avg_final_obs_margin",
        "avg_final_comm_margin",
        "avg_final_task_score",
        "collision_rate",
        "out_of_bounds_rate",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *metric_names])
        writer.writeheader()
        for method, metrics in results.items():
            row = {"method": method}
            row.update({name: metrics.get(name, "") for name in metric_names})
            writer.writerow(row)


def _parse_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_lookahead_heuristics(raw: str) -> List[str]:
    heuristics = _parse_list(raw)
    if not heuristics:
        raise ValueError("--lookahead-heuristics 至少需要一个值")
    valid = {"terminal", "dense"}
    bad = [item for item in heuristics if item not in valid]
    if bad:
        raise ValueError(f"未知 lookahead heuristic: {bad}; 可选值: terminal,dense")
    return heuristics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Communication-inspection Dubins TD baseline benchmark.")
    parser.add_argument("--output-dir", type=str, default="results/comm_inspection_td_baselines")
    parser.add_argument("--qrl-ckpt", type=str, default="results/minimal_qrl_inspection_dubins/checkpoint_final.pth")
    parser.add_argument("--include-qrl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qrl-execution-modes", type=str, default="greedy,lookahead")
    parser.add_argument("--qrl-cost-source", type=str, default="fixed")
    parser.add_argument("--qrl-num-episodes", type=int, default=180)
    parser.add_argument("--qrl-total-steps", type=int, default=20_000)
    parser.add_argument("--qrl-batch-size", type=int, default=256)
    parser.add_argument("--td-algos", type=str, default="gc_sac,her_ddpg,uvfa")
    parser.add_argument("--td-execution-modes", type=str, default="greedy,lookahead")
    parser.add_argument("--skip-td-training", action="store_true")
    parser.add_argument("--gc-sac-ckpt", dest="gc_sac_ckpt", type=str, default=None)
    parser.add_argument("--her-ddpg-ckpt", dest="her_ddpg_ckpt", type=str, default=None)
    parser.add_argument("--uvfa-ckpt", type=str, default=None)

    parser.add_argument("--total-env-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--action-noise-std", type=float, default=0.2)
    parser.add_argument("--start-random-steps", type=int, default=1000)
    parser.add_argument("--her-k", type=int, default=4)
    parser.add_argument("--sac-alpha", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=10000)

    parser.add_argument("--bounds", type=float, nargs=4, default=[0.0, 0.0, 10.0, 10.0])
    parser.add_argument("--omega-max", type=float, default=3.0)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--obstacle-config", type=str, default="medium", choices=["none", "simple", "medium", "hard"])
    parser.add_argument("--obstacles", type=float, nargs="*", default=None)
    parser.add_argument("--observation-mode", type=str, default="task_context", choices=["task_context", "cos_sin", "state"])
    parser.add_argument("--inspection-target", type=float, nargs=2, default=[3.0, 7.5])
    parser.add_argument("--ground-station", type=float, nargs=2, default=[1.5, 2.0])
    parser.add_argument("--randomize-inspection-target", action="store_true")
    parser.add_argument("--randomize-ground-station", action="store_true")
    parser.add_argument("--observation-radius", type=float, default=1.8)
    parser.add_argument("--fov-angle", type=float, default=float(np.pi / 2.0))
    parser.add_argument("--require-target-los", dest="require_target_los", action="store_true", default=True)
    parser.add_argument("--no-require-target-los", dest="require_target_los", action="store_false")
    parser.add_argument("--comm-alpha", type=float, default=2.0)
    parser.add_argument("--comm-bias", type=float, default=5.0)
    parser.add_argument("--comm-occlusion-penalty", type=float, default=6.0)
    parser.add_argument("--comm-threshold", type=float, default=0.5)
    parser.add_argument("--require-ground-station-los", action="store_true")
    parser.add_argument("--goal-sampling-mode", type=str, default="task_feasible", choices=["task_feasible", "valid"])
    parser.add_argument("--goal-position-tolerance", type=float, default=0.25)
    parser.add_argument("--goal-heading-tolerance", type=float, default=0.3)
    parser.add_argument("--collision-cost", type=float, default=10.0)
    parser.add_argument("--out-of-bounds-cost", type=float, default=10.0)
    parser.add_argument("--communication-break-cost", type=float, default=1.0)
    parser.add_argument("--observation-violation-cost-weight", type=float, default=1.0)
    parser.add_argument("--communication-violation-cost-weight", type=float, default=0.5)
    parser.add_argument("--observation-failure-cost", type=float, default=0.25)
    parser.add_argument("--taskscore-beta-obs", type=float, default=1.0)
    parser.add_argument("--taskscore-beta-comm", type=float, default=1.0)
    parser.add_argument("--taskscore-beta-feas", type=float, default=0.5)
    parser.add_argument("--taskscore-margin-clip", type=float, default=2.0)

    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--env-name", type=str, default="comm_inspection_td_baseline_eval")
    parser.add_argument("--lookahead-horizon", type=int, default=10)
    parser.add_argument("--lookahead-num-sequences", type=int, default=128)
    parser.add_argument("--lookahead-heuristics", type=str, default="terminal")
    parser.add_argument("--qrl-lookahead-heuristics", type=str, default=None)
    parser.add_argument("--lookahead-step-cost-weight", type=float, default=0.0)
    parser.add_argument("--lookahead-collision-penalty", type=float, default=0.0)
    parser.add_argument("--lookahead-biased-sequences", type=int, default=24)
    parser.add_argument("--lookahead-bias-kp", type=float, default=2.0)
    parser.add_argument("--lookahead-use-cem", action="store_true")
    parser.add_argument("--lookahead-cem-iters", type=int, default=3)
    parser.add_argument("--lookahead-cem-elite-frac", type=float, default=0.1)
    parser.add_argument("--lookahead-cem-std-init-frac", type=float, default=0.5)
    parser.add_argument("--planner-alpha-final", type=float, default=0.3)
    parser.add_argument("--planner-alpha-task-terminal", type=float, default=0.5)
    parser.add_argument("--planner-qrl-progress-alpha", type=float, default=0.0)
    parser.add_argument("--planner-use-env-stage-cost", dest="planner_use_env_stage_cost", action="store_true", default=True)
    parser.add_argument("--no-planner-use-env-stage-cost", dest="planner_use_env_stage_cost", action="store_false")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--viz-max-successes", type=int, default=3)
    parser.add_argument("--viz-max-failures", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = auto_device(args.device)
    results: Dict[str, Dict[str, float]] = {}
    visualizations: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    checkpoints: Dict[str, Optional[str]] = {}

    if args.include_qrl:
        if not os.path.exists(args.qrl_ckpt):
            raise FileNotFoundError(f"QRL checkpoint not found: {args.qrl_ckpt}")
        qrl_lookahead_heuristics = args.qrl_lookahead_heuristics or args.lookahead_heuristics
        qrl_env = make_comm_inspection_env(args)
        qrl_agent, _ckpt_step = build_qrl_adapter(
            argparse.Namespace(**{**vars(args), "checkpoint": args.qrl_ckpt}),
            device,
            qrl_env,
        )
        qrl_results, qrl_visualizations = _evaluate_agent(
            qrl_agent,
            args,
            method_prefix="qrl",
            execution_modes=_parse_list(args.qrl_execution_modes),
            lookahead_heuristics=qrl_lookahead_heuristics,
            device=device,
        )
        results.update(qrl_results)
        visualizations.update(qrl_visualizations)
        checkpoints["qrl"] = args.qrl_ckpt

    td_algos = _parse_list(args.td_algos)
    td_modes = _parse_list(args.td_execution_modes)
    for algo in td_algos:
        agent, ckpt_path = _train_or_load_td_agent(algo, args, device)
        checkpoints[algo] = ckpt_path
        algo_results, algo_visualizations = _evaluate_agent(
            agent,
            args,
            method_prefix=algo,
            execution_modes=td_modes,
            lookahead_heuristics="terminal",
            device=device,
        )
        results.update(algo_results)
        visualizations.update(algo_visualizations)

    payload = {
        "env_config": {
            "bounds": [float(v) for v in args.bounds],
            "obstacle_config": args.obstacle_config,
            "inspection_target": [float(v) for v in args.inspection_target],
            "ground_station": [float(v) for v in args.ground_station],
            "observation_mode": args.observation_mode,
            "goal_sampling_mode": args.goal_sampling_mode,
        },
        "training_config": {
            "qrl": {
                "cost_source": str(args.qrl_cost_source),
                "num_episodes": int(args.qrl_num_episodes),
                "total_steps": int(args.qrl_total_steps),
                "batch_size": int(args.qrl_batch_size),
            },
            "td_baselines": {
                "total_env_steps": int(args.total_env_steps),
                "batch_size": int(args.batch_size),
                "gamma": float(args.gamma),
                "her_k": int(args.her_k),
                "skip_td_training": bool(args.skip_td_training),
            },
        },
        "evaluation_config": {
            "lookahead_heuristics": _parse_lookahead_heuristics(args.lookahead_heuristics),
            "qrl_lookahead_heuristics": _parse_lookahead_heuristics(
                args.qrl_lookahead_heuristics or args.lookahead_heuristics
            ),
            "td_lookahead_heuristics": ["terminal"],
            "planner_qrl_progress_alpha": float(args.planner_qrl_progress_alpha),
            "planner_alpha_final": float(args.planner_alpha_final),
            "planner_alpha_task_terminal": float(args.planner_alpha_task_terminal),
            "planner_use_env_stage_cost": bool(args.planner_use_env_stage_cost),
        },
        "checkpoints": checkpoints,
        "results": results,
        "visualizations": visualizations,
    }
    out_json = os.path.join(args.output_dir, "comm_inspection_td_baselines.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    out_csv = os.path.join(args.output_dir, "comm_inspection_td_baselines.csv")
    _write_csv(out_csv, results)
    print(f"[comm_td_baselines] saved: {out_json}")
    print(f"[comm_td_baselines] saved: {out_csv}")


if __name__ == "__main__":
    main()
