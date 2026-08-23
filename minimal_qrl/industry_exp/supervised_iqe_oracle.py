#!/usr/bin/env python3
"""Supervised IQE representability experiment using the reverse-Dijkstra oracle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "supervised_iqe_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "supervised_iqe_xdg"))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLConf

from minimal_qrl.baselines import HybridAStarConfig, HybridAStarValueOracle
from minimal_qrl.dataset import create_dataset
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.u_trap_local_navigability import build_probe_records
from minimal_qrl.eval.utils import auto_device, ensure_registered_env
from minimal_qrl.gc_agents import QRLGoalValueAdapter
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)


def _counts(total: int, size: int) -> list[int]:
    base, remainder = divmod(int(total), int(size))
    return [base + int(index < remainder) for index in range(size)]


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1)
        cursor = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    errors = predictions - targets
    return {
        "count": int(len(targets)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "pearson": _correlation(predictions, targets),
        "spearman": _correlation(_rankdata(predictions), _rankdata(targets)),
        "prediction_mean": float(np.mean(predictions)),
        "target_mean": float(np.mean(targets)),
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "target_min": float(np.min(targets)),
        "target_max": float(np.max(targets)),
    }


def _make_agent(
    env: CommInspectionDubinsUAV2D,
    scenario: Mapping[str, Any],
    *,
    num_critics: int,
    total_steps: int,
):
    env_name = f"supervised_iqe_oracle_{os.getpid()}"

    def create_env_fn():
        return CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))

    def load_episodes_fn():
        dataset_env = create_env_fn()
        return create_dataset(
            dataset_env,
            num_episodes=1,
            max_steps_per_episode=10,
            seed=0,
        )

    ensure_registered_env(
        "comm_inspection_dubins_uav",
        env_name,
        create_env_fn=create_env_fn,
        load_episodes_fn=load_episodes_fn,
    )
    dataset = Dataset.Conf(
        kind="comm_inspection_dubins_uav",
        name=env_name,
        future_observation_discount=0.99,
    ).make(dummy=True)
    agent, _unused_qrl_losses = QRLConf(
        actor=None,
        num_critics=int(num_critics),
    ).make(
        env_spec=dataset.env_spec,
        total_optim_steps=max(1, int(total_steps)),
    )
    if tuple(env.observation_space.shape) != tuple(dataset.env_spec.observation_shape):
        raise RuntimeError("supervised agent EnvSpec does not match the experiment environment")
    return agent


def _encode_states(
    env: CommInspectionDubinsUAV2D,
    states: np.ndarray,
    *,
    description: str,
) -> np.ndarray:
    return np.stack(
        [
            env.state_to_observation(state).astype(np.float32)
            for state in tqdm(states, desc=description, leave=False)
        ],
        axis=0,
    )


def _sample_training_indices(
    values: np.ndarray,
    available: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator,
    low_cost_fraction: float,
) -> np.ndarray:
    count = int(count)
    low_count = int(round(count * float(low_cost_fraction)))
    uniform_count = count - low_count
    uniform = rng.choice(available, size=uniform_count, replace=uniform_count > len(available))
    available_values = values[available]
    low_cutoff = float(np.quantile(available_values, 0.15))
    low_pool = available[available_values <= low_cutoff]
    low = rng.choice(low_pool, size=low_count, replace=low_count > len(low_pool))
    result = np.concatenate([uniform, low])
    rng.shuffle(result)
    return result


def build_supervised_dataset(
    env: CommInspectionDubinsUAV2D,
    oracle: HybridAStarValueOracle,
    *,
    train_samples: int,
    eval_samples: int,
    low_cost_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    device_ids = list(env.device_ids)
    train_counts = _counts(train_samples, len(device_ids))
    eval_counts = _counts(eval_samples, len(device_ids))
    rng = np.random.default_rng(int(seed))
    parts: dict[str, list[np.ndarray]] = {
        "train_state": [],
        "train_observation": [],
        "train_goal": [],
        "train_value": [],
        "train_device_index": [],
        "eval_state": [],
        "eval_observation": [],
        "eval_goal": [],
        "eval_value": [],
        "eval_device_index": [],
    }

    for device_index, device_id in enumerate(device_ids):
        env.reset(seed=int(seed) + device_index, options={"device_id": device_id})
        oracle.begin_episode(env, seed=int(seed) + device_index)
        states, values = oracle.lattice_dataset(env, reachable_only=True)
        if len(states) <= eval_counts[device_index]:
            raise RuntimeError(f"not enough reachable oracle states for {device_id}")
        permutation = rng.permutation(len(states))
        eval_indices = permutation[: eval_counts[device_index]]
        available = permutation[eval_counts[device_index] :]
        train_indices = _sample_training_indices(
            values,
            available,
            train_counts[device_index],
            rng=rng,
            low_cost_fraction=low_cost_fraction,
        )
        goal = env.abstract_goal_observation().astype(np.float32)
        train_states = states[train_indices]
        eval_states = states[eval_indices]
        parts["train_state"].append(train_states)
        parts["train_observation"].append(
            _encode_states(env, train_states, description=f"encode train/{device_id}")
        )
        parts["train_goal"].append(np.repeat(goal[None, :], len(train_states), axis=0))
        parts["train_value"].append(values[train_indices].astype(np.float32))
        parts["train_device_index"].append(
            np.full((len(train_states),), device_index, dtype=np.int16)
        )
        parts["eval_state"].append(eval_states)
        parts["eval_observation"].append(
            _encode_states(env, eval_states, description=f"encode eval/{device_id}")
        )
        parts["eval_goal"].append(np.repeat(goal[None, :], len(eval_states), axis=0))
        parts["eval_value"].append(values[eval_indices].astype(np.float32))
        parts["eval_device_index"].append(
            np.full((len(eval_states),), device_index, dtype=np.int16)
        )

    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}


@torch.no_grad()
def _predict(
    critic,
    observations: np.ndarray,
    goals: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    critic.eval()
    predictions = []
    for begin in range(0, len(observations), int(batch_size)):
        end = min(begin + int(batch_size), len(observations))
        obs = torch.as_tensor(observations[begin:end], device=device, dtype=torch.float32)
        goal = torch.as_tensor(goals[begin:end], device=device, dtype=torch.float32)
        predictions.append(critic(obs, goal).reshape(-1).cpu().numpy().astype(np.float32))
    return np.concatenate(predictions) if predictions else np.zeros((0,), dtype=np.float32)


def train_supervised(
    agent,
    dataset: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    loss_name: str,
    huber_delta: float,
    seed: int,
    log_interval: int,
) -> list[dict[str, float]]:
    agent.to(device)
    agent.train()
    trained_parameters = []
    for critic in agent.critics:
        trained_parameters.extend(critic.encoder.parameters())
        trained_parameters.extend(critic.quasimetric_model.parameters())
    optimizer = torch.optim.AdamW(
        trained_parameters,
        lr=float(learning_rate),
        weight_decay=0.0,
    )
    observations = dataset["train_observation"]
    goals = dataset["train_goal"]
    targets = dataset["train_value"]
    rng = np.random.default_rng(int(seed) + 17)
    history = []
    running = []
    progress = tqdm(range(1, int(steps) + 1), desc="supervised IQE")
    for step in progress:
        indices = rng.integers(0, len(targets), size=int(batch_size))
        obs = torch.as_tensor(observations[indices], device=device, dtype=torch.float32)
        goal = torch.as_tensor(goals[indices], device=device, dtype=torch.float32)
        target = torch.as_tensor(targets[indices], device=device, dtype=torch.float32)
        losses = []
        for critic in agent.critics:
            prediction = critic(obs, goal).reshape(-1)
            if loss_name == "mse":
                losses.append(F.mse_loss(prediction, target))
            else:
                losses.append(F.huber_loss(prediction, target, delta=float(huber_delta)))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained_parameters, max_norm=100.0)
        optimizer.step()
        running.append(float(loss.detach().cpu()))
        if step == 1 or step % max(1, int(log_interval)) == 0 or step == int(steps):
            row = {
                "step": float(step),
                "loss": float(np.mean(running)),
            }
            history.append(row)
            progress.set_postfix(loss=f"{row['loss']:.4f}")
            running.clear()
    return history


def _u_trap_local_dataset(
    env: CommInspectionDubinsUAV2D,
    oracle: HybridAStarValueOracle,
    scenario: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    probes = build_probe_records(scenario)
    device_id = str(probes[0]["device_id"])
    env.reset(
        seed=int(seed),
        options={"device_id": device_id, "start": np.asarray(probes[0]["state"], dtype=np.float32)},
    )
    oracle.begin_episode(env, seed=int(seed))
    states = np.asarray([probe["state"] for probe in probes], dtype=np.float32)
    values = oracle.batch_value(env, states)
    observations = _encode_states(env, states, description="encode U-trap probes")
    goal = env.abstract_goal_observation().astype(np.float32)
    return {
        "records": probes,
        "states": states,
        "observations": observations,
        "goals": np.repeat(goal[None, :], len(states), axis=0),
        "values": values,
    }


def _successor_ranking_dataset(
    env: CommInspectionDubinsUAV2D,
    oracle: HybridAStarValueOracle,
    local: Mapping[str, Any],
    config: HybridAStarConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    successor_observations = []
    successor_goals = []
    oracle_scores = []
    groups = []
    details = []
    goal = np.asarray(local["goals"][0], dtype=np.float32)
    for probe_index, record in enumerate(local["records"]):
        state = np.asarray(record["state"], dtype=np.float32)
        env.reset(
            seed=int(seed) + probe_index,
            options={"device_id": str(record["device_id"]), "start": state},
        )
        base_state = env.get_state()
        for action_index, scale in enumerate(config.primitive_scales):
            env.set_state(base_state)
            omega = float(scale) * float(env.omega_max)
            immediate_cost = 0.0
            valid = True
            success = False
            for _ in range(max(1, int(config.primitive_steps))):
                _obs, reward, terminated, truncated, info = env.step(
                    np.asarray([omega], dtype=np.float32)
                )
                immediate_cost += float(info.get("cost_total", -float(reward)))
                if bool(info.get("collision", False) or info.get("out_of_bounds", False)):
                    valid = False
                    break
                success = bool(info.get("success", False))
                if terminated or truncated:
                    break
            if not valid:
                continue
            successor = env.state.copy()
            terminal_value = 0.0 if success else float(oracle.batch_value(env, successor[None, :])[0])
            successor_observations.append(env.state_to_observation(successor).astype(np.float32))
            successor_goals.append(goal)
            oracle_scores.append(immediate_cost + terminal_value)
            groups.append(probe_index)
            details.append(
                {
                    "probe_id": str(record["probe_id"]),
                    "action_index": int(action_index),
                    "omega": omega,
                    "immediate_cost": immediate_cost,
                    "oracle_terminal_value": terminal_value,
                    "oracle_score": immediate_cost + terminal_value,
                }
            )
        env.set_state(base_state)
    return {
        "observations": np.asarray(successor_observations, dtype=np.float32),
        "goals": np.asarray(successor_goals, dtype=np.float32),
        "oracle_scores": np.asarray(oracle_scores, dtype=np.float32),
        "groups": np.asarray(groups, dtype=np.int32),
        "details": details,
    }


def _ranking_accuracy(
    predicted_scores: np.ndarray,
    oracle_scores: np.ndarray,
    groups: np.ndarray,
) -> tuple[float | None, int]:
    agreements = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                target_delta = float(oracle_scores[left] - oracle_scores[right])
                if abs(target_delta) <= 1e-9:
                    continue
                prediction_delta = float(predicted_scores[left] - predicted_scores[right])
                agreements.append(float(prediction_delta * target_delta > 0.0))
    return (float(np.mean(agreements)) if agreements else None, len(agreements))


def _write_history(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--eval-samples", type=int, default=20_000)
    parser.add_argument("--low-cost-fraction", type=float, default=0.25)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--false-zero-prediction-max", type=float, default=0.1)
    parser.add_argument("--false-zero-target-min", type=float, default=1.0)
    parser.add_argument("--astar-position-resolution", type=float, default=0.25)
    parser.add_argument("--astar-heading-bins", type=int, default=24)
    parser.add_argument("--astar-primitive-steps", type=int, default=5)
    parser.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    parser.add_argument("--astar-max-expansions", type=int, default=50_000)
    parser.add_argument("--astar-timeout-sec", type=float, default=30.0)
    parser.add_argument("--astar-terminal-samples", type=int, default=128)
    parser.add_argument("--oracle-value-cache-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= float(args.low_cost_fraction) <= 1.0:
        raise ValueError("--low-cost-fraction must lie in [0, 1]")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_config(args.scenario_config)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    device = auto_device(str(args.device))
    config = HybridAStarConfig(
        position_resolution=float(args.astar_position_resolution),
        heading_bins=int(args.astar_heading_bins),
        primitive_steps=int(args.astar_primitive_steps),
        heuristic_weight=float(args.astar_heuristic_weight),
        max_expansions=int(args.astar_max_expansions),
        timeout_sec=float(args.astar_timeout_sec),
        terminal_samples=int(args.astar_terminal_samples),
    )
    cache_dir = (
        Path(args.oracle_value_cache_dir)
        if args.oracle_value_cache_dir
        else output_dir / "oracle_value_cache"
    )
    oracle = HybridAStarValueOracle(config, cache_dir=cache_dir)

    dataset_start = perf_counter()
    dataset = build_supervised_dataset(
        env,
        oracle,
        train_samples=int(args.train_samples),
        eval_samples=int(args.eval_samples),
        low_cost_fraction=float(args.low_cost_fraction),
        seed=int(args.seed),
    )
    dataset_path = output_dir / "oracle_supervised_dataset.npz"
    with dataset_path.open("wb") as handle:
        np.savez_compressed(handle, **dataset)
    dataset_time = perf_counter() - dataset_start

    agent = _make_agent(
        env,
        scenario,
        num_critics=int(args.num_critics),
        total_steps=int(args.train_steps),
    )
    model_signature = {
        "num_critics": int(len(agent.critics)),
        "encoder": repr(agent.critics[0].encoder),
        "quasimetric_model": repr(agent.critics[0].quasimetric_model),
        "quasimetric_head": repr(agent.critics[0].quasimetric_model.quasimetric_head),
        "total_parameters": int(sum(parameter.numel() for parameter in agent.parameters())),
        "trained_parameters": int(
            sum(
                parameter.numel()
                for critic in agent.critics
                for module in (critic.encoder, critic.quasimetric_model)
                for parameter in module.parameters()
            )
        ),
    }
    train_start = perf_counter()
    history = train_supervised(
        agent,
        dataset,
        device=device,
        steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        loss_name=str(args.loss),
        huber_delta=float(args.huber_delta),
        seed=int(args.seed),
        log_interval=int(args.log_interval),
    )
    training_time = perf_counter() - train_start
    checkpoint_path = output_dir / "checkpoint_final.pth"
    torch.save(
        {
            "optim_steps": int(args.train_steps),
            "agent": agent.state_dict(),
            "training_mode": "supervised_reverse_dijkstra_oracle",
            "objective": f"{args.loss}(d_theta(s,g), V_oracle(s,g))",
            "model_signature": model_signature,
            "config": vars(args),
        },
        checkpoint_path,
    )
    _write_history(output_dir / "train_history.csv", history)

    critic = agent.critics[0]
    global_predictions = _predict(
        critic,
        dataset["eval_observation"],
        dataset["eval_goal"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    global_targets = dataset["eval_value"]
    global_metrics = _regression_metrics(global_predictions, global_targets)
    nonzero = global_targets > float(args.false_zero_target_min)
    false_zero = nonzero & (global_predictions <= float(args.false_zero_prediction_max))
    false_zero_metrics = {
        "rate": float(np.sum(false_zero) / max(int(np.sum(nonzero)), 1)),
        "count": int(np.sum(false_zero)),
        "eligible_count": int(np.sum(nonzero)),
        "prediction_max": float(args.false_zero_prediction_max),
        "target_min": float(args.false_zero_target_min),
    }

    local = _u_trap_local_dataset(env, oracle, scenario, seed=int(args.seed))
    local_predictions = _predict(
        critic,
        local["observations"],
        local["goals"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    local_metrics = _regression_metrics(local_predictions, local["values"])
    successor = _successor_ranking_dataset(
        env,
        oracle,
        local,
        config,
        seed=int(args.seed),
    )
    successor_values = _predict(
        critic,
        successor["observations"],
        successor["goals"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    immediate_costs = np.asarray(
        [record["immediate_cost"] for record in successor["details"]],
        dtype=np.float32,
    )
    predicted_scores = immediate_costs + successor_values
    ranking_accuracy, ranking_pairs = _ranking_accuracy(
        predicted_scores,
        successor["oracle_scores"],
        successor["groups"],
    )
    for index, record in enumerate(successor["details"]):
        record["predicted_terminal_value"] = float(successor_values[index])
        record["predicted_score"] = float(predicted_scores[index])

    adapter = QRLGoalValueAdapter(agent, env, device, distance_scale=1.0)
    del adapter  # Construction verifies checkpoint-compatible value inference.
    payload = {
        "experiment": "supervised_iqe_oracle_representability",
        "objective": "pure supervised regression only; no global push, local constraint, Lagrange multiplier, trajectory loss, or bootstrap",
        "scenario_config": str(Path(args.scenario_config).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "device": str(device),
        "model_signature": model_signature,
        "config": vars(args),
        "timing": {
            "dataset_generation_sec": float(dataset_time),
            "supervised_training_sec": float(training_time),
        },
        "dataset_summary": {
            "train_samples": int(len(dataset["train_value"])),
            "eval_samples": int(len(dataset["eval_value"])),
            "train_target_min": float(np.min(dataset["train_value"])),
            "train_target_max": float(np.max(dataset["train_value"])),
            "train_target_mean": float(np.mean(dataset["train_value"])),
        },
        "metrics": {
            "global": global_metrics,
            "u_trap_local": local_metrics,
            "successor_ranking_accuracy": ranking_accuracy,
            "successor_ranking_pairs": int(ranking_pairs),
            "false_zero": false_zero_metrics,
        },
        "u_trap_local_records": [
            {
                **dict(record),
                "oracle_value": float(local["values"][index]),
                "predicted_value": float(local_predictions[index]),
                "absolute_error": float(abs(local_predictions[index] - local["values"][index])),
            }
            for index, record in enumerate(local["records"])
        ],
        "successor_records": successor["details"],
        "train_history": history,
    }
    metrics_path = output_dir / "supervised_iqe_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved supervised samples: {dataset_path}")


if __name__ == "__main__":
    main()
