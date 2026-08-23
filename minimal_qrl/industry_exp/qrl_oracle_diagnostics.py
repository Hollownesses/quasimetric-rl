#!/usr/bin/env python3
"""Post-training Oracle diagnostics for transition-only QRL experiments."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qrl_oracle_diag_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qrl_oracle_diag_xdg"))

import numpy as np
import torch

from minimal_qrl.baselines import HybridAStarConfig, HybridAStarValueOracle
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.gc_agents import QRLGoalValueAdapter
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)
from minimal_qrl.industry_exp.supervised_iqe_oracle import (
    _counts,
    _encode_states,
    _make_agent,
    _ranking_accuracy,
    _regression_metrics,
    _successor_ranking_dataset,
    _u_trap_local_dataset,
)


def _global_oracle_eval_dataset(
    env: CommInspectionDubinsUAV2D,
    oracle: HybridAStarValueOracle,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    counts = _counts(sample_count, len(env.device_ids))
    rng = np.random.default_rng(int(seed))
    observations = []
    goals = []
    targets = []
    states_output = []
    device_indices = []
    for device_index, (device_id, count) in enumerate(zip(env.device_ids, counts)):
        env.reset(seed=int(seed) + device_index, options={"device_id": device_id})
        oracle.begin_episode(env, seed=int(seed) + device_index)
        states, values = oracle.lattice_dataset(env, reachable_only=True)
        indices = rng.choice(len(states), size=count, replace=count > len(states))
        selected = states[indices]
        goal = env.abstract_goal_observation().astype(np.float32)
        states_output.append(selected)
        observations.append(
            _encode_states(env, selected, description=f"encode oracle eval/{device_id}")
        )
        goals.append(np.repeat(goal[None, :], len(selected), axis=0))
        targets.append(values[indices].astype(np.float32))
        device_indices.append(np.full(len(selected), device_index, dtype=np.int16))
    return {
        "states": np.concatenate(states_output),
        "observations": np.concatenate(observations),
        "goals": np.concatenate(goals),
        "targets": np.concatenate(targets),
        "device_indices": np.concatenate(device_indices),
    }


def _adapter_predict(
    value: QRLGoalValueAdapter,
    observations: np.ndarray,
    goals: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    predictions = [
        value.batch_value(
            observations[begin : begin + batch_size],
            goals[begin : begin + batch_size],
        )
        for begin in range(0, len(observations), int(batch_size))
    ]
    return np.concatenate(predictions) if predictions else np.zeros(0, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--global-eval-samples", type=int, default=20_000)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--false-zero-prediction-max", type=float, default=0.1)
    parser.add_argument("--false-zero-target-min", type=float, default=1.0)
    parser.add_argument("--astar-position-resolution", type=float, default=0.25)
    parser.add_argument("--astar-heading-bins", type=int, default=24)
    parser.add_argument("--astar-primitive-steps", type=int, default=5)
    parser.add_argument("--oracle-value-cache-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_config(args.scenario_config)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    device = auto_device(str(args.device))
    config = HybridAStarConfig(
        position_resolution=float(args.astar_position_resolution),
        heading_bins=int(args.astar_heading_bins),
        primitive_steps=int(args.astar_primitive_steps),
    )
    oracle = HybridAStarValueOracle(
        config,
        cache_dir=(
            Path(args.oracle_value_cache_dir)
            if args.oracle_value_cache_dir
            else output_dir / "oracle_value_cache"
        ),
    )
    agent = _make_agent(
        env,
        scenario,
        num_critics=int(args.num_critics),
        total_steps=1,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(checkpoint["agent"] if isinstance(checkpoint, dict) else checkpoint)
    agent.to(device).eval()
    value = QRLGoalValueAdapter(agent, env, device, distance_scale=1.0)

    global_data = _global_oracle_eval_dataset(
        env,
        oracle,
        sample_count=int(args.global_eval_samples),
        seed=int(args.seed),
    )
    global_predictions = _adapter_predict(
        value,
        global_data["observations"],
        global_data["goals"],
        batch_size=int(args.eval_batch_size),
    )
    global_targets = global_data["targets"]
    eligible = global_targets > float(args.false_zero_target_min)
    false_zero = eligible & (
        global_predictions <= float(args.false_zero_prediction_max)
    )

    local = _u_trap_local_dataset(env, oracle, scenario, seed=int(args.seed))
    local_predictions = _adapter_predict(
        value,
        local["observations"],
        local["goals"],
        batch_size=int(args.eval_batch_size),
    )
    successor = _successor_ranking_dataset(
        env,
        oracle,
        local,
        config,
        seed=int(args.seed),
    )
    successor_values = _adapter_predict(
        value,
        successor["observations"],
        successor["goals"],
        batch_size=int(args.eval_batch_size),
    )
    immediate = np.asarray(
        [record["immediate_cost"] for record in successor["details"]],
        dtype=np.float32,
    )
    predicted_scores = immediate + successor_values
    ranking_accuracy, ranking_pairs = _ranking_accuracy(
        predicted_scores,
        successor["oracle_scores"],
        successor["groups"],
    )
    metrics = {
        "global": _regression_metrics(global_predictions, global_targets),
        "u_trap_local": _regression_metrics(local_predictions, local["values"]),
        "successor_ranking_accuracy": ranking_accuracy,
        "successor_ranking_pairs": int(ranking_pairs),
        "false_zero": {
            "rate": float(np.sum(false_zero) / max(int(np.sum(eligible)), 1)),
            "count": int(np.sum(false_zero)),
            "eligible_count": int(np.sum(eligible)),
            "prediction_max": float(args.false_zero_prediction_max),
            "target_min": float(args.false_zero_target_min),
        },
    }
    payload = {
        "experiment": "dense_transition_original_qrl_oracle_evaluation",
        "training_labels": "real transitions only; no Oracle values",
        "oracle_role": "post-training evaluation only",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "config": vars(args),
        "metrics": metrics,
    }
    path = output_dir / "qrl_oracle_metrics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved Oracle-only diagnostics: {path}")


if __name__ == "__main__":
    main()
