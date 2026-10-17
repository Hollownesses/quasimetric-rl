from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from minimal_qrl.baselines.context_gcrl import (
    ContextContrastiveRLAgent,
    ContextGCRLConfig,
    GoalConditionedAgentBase,
    context_agent_metadata,
    make_context_agent,
)
from minimal_qrl.baselines.context_replay import (
    ContextHERReplayBuffer,
    RawGoalSetEpisode,
    episode_from_observations,
)
from minimal_qrl.dataset import collect_task_aware_comm_teacher_episode_pair
from minimal_qrl.envs import CommInspectionDubinsUAV2D


def catalog_hash(env: CommInspectionDubinsUAV2D) -> str:
    payload = asdict(env.device_catalog)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment_signature(env: CommInspectionDubinsUAV2D) -> Dict[str, object]:
    return {
        "catalog_hash": catalog_hash(env),
        "observation_dim": int(env.observation_space.shape[0]),
        "action_low": np.asarray(env.action_space.low, dtype=np.float32).tolist(),
        "action_high": np.asarray(env.action_space.high, dtype=np.float32).tolist(),
        "bounds": [float(value) for value in env.bounds],
        "dt": float(env.dt),
        "v": float(env.v),
        "omega_max": float(env.omega_max),
        "comm_threshold": float(env.comm_threshold),
    }


def build_value_calibration(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    *,
    seed: int,
    samples_per_device: int = 16,
) -> Dict[str, float]:
    observations = []
    goals = []
    for device_index, device_id in enumerate(env.device_ids):
        for sample_index in range(max(1, int(samples_per_device))):
            sample_seed = int(seed) + device_index * 1_000_003 + sample_index * 97
            state = env.sample_valid_state(seed=sample_seed)
            observations.append(env.observation_for_task(state, device_id))
            goals.append(env.abstract_goal_for_task(device_id))
    values = agent.batch_value(
        np.asarray(observations, dtype=np.float32),
        np.asarray(goals, dtype=np.float32),
    )
    finite = np.asarray(values, dtype=np.float32)[np.isfinite(values)]
    if finite.size == 0:
        return {"p05": 0.0, "p95": 1.0, "num_samples": 0.0}
    low = float(np.percentile(finite, 5.0))
    high = float(np.percentile(finite, 95.0))
    if high - low < 1e-6:
        high = low + 1.0
    return {"p05": low, "p95": high, "num_samples": float(finite.size)}


class CalibratedValueAgent(GoalConditionedAgentBase):
    """Monotone robust value normalization used only by the shared MPPI layer."""

    def __init__(self, agent: GoalConditionedAgentBase, calibration: Dict[str, float]) -> None:
        super().__init__()
        self.agent = agent
        self.calibration = dict(calibration)
        self.device = getattr(agent, "device", torch.device("cpu"))
        self.env = getattr(agent, "env", None)

    def act(self, obs: np.ndarray, goal_obs: np.ndarray, eval_mode: bool = True) -> np.ndarray:
        return self.agent.act(obs, goal_obs, eval_mode=eval_mode)

    def batch_value(self, obs_batch: np.ndarray, goal_obs_batch: np.ndarray) -> np.ndarray:
        raw = self.agent.batch_value(obs_batch, goal_obs_batch)
        low = float(self.calibration.get("p05", 0.0))
        high = float(self.calibration.get("p95", low + 1.0))
        return np.clip((raw - low) / max(high - low, 1e-6), 0.0, 1.0).astype(np.float32)

    def value(self, obs: np.ndarray, goal_obs: np.ndarray) -> float:
        return float(self.batch_value(obs[None], goal_obs[None])[0])


def save_context_checkpoint(
    path: Path,
    algorithm: str,
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    *,
    seed: int,
    env_steps: int,
    teacher_steps: int,
    updates: int,
    replay_diagnostics: Dict[str, float],
    training_diagnostics: Optional[Dict[str, float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    calibration = build_value_calibration(agent, env, seed=int(seed) + 70_000_019)
    torch.save(
        {
            "algorithm": str(algorithm),
            "agent": agent.state_dict(),
            "optimizers": agent.optimizer_state_dict(),  # type: ignore[attr-defined]
            "config": asdict(getattr(agent, "cfg")),
            "seed": int(seed),
            "env_steps": int(env_steps),
            "teacher_steps": int(teacher_steps),
            "updates": int(updates),
            "environment_signature": environment_signature(env),
            "model_metadata": context_agent_metadata(agent),
            "replay_diagnostics": dict(replay_diagnostics),
            "training_diagnostics": dict(training_diagnostics or {}),
            "value_calibration": calibration,
        },
        path,
    )


def load_context_checkpoint(
    path: Path,
    env: CommInspectionDubinsUAV2D,
    device: torch.device,
    *,
    load_optimizers: bool = False,
) -> tuple[GoalConditionedAgentBase, Dict[str, object]]:
    payload = torch.load(path, map_location=device)
    expected = environment_signature(env)
    actual = payload.get("environment_signature", {})
    for key in expected:
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"checkpoint environment mismatch for {key}: "
                f"checkpoint={actual.get(key)!r}, current={expected.get(key)!r}"
            )
    cfg = ContextGCRLConfig(**payload.get("config", {}))
    algorithm = str(payload["algorithm"])
    agent = make_context_agent(algorithm, env, cfg, device)
    agent.load_state_dict(payload["agent"])
    if load_optimizers and payload.get("optimizers"):
        agent.load_optimizer_state_dict(payload["optimizers"])  # type: ignore[attr-defined]
    agent.eval()
    metadata = {
        "algorithm": algorithm,
        "seed": int(payload.get("seed", 0)),
        "env_steps": int(payload.get("env_steps", 0)),
        "teacher_steps": int(payload.get("teacher_steps", 0)),
        "updates": int(payload.get("updates", 0)),
        "model_metadata": dict(payload.get("model_metadata", {})),
        "replay_diagnostics": dict(payload.get("replay_diagnostics", {})),
        "training_diagnostics": dict(payload.get("training_diagnostics", {})),
        "value_calibration": dict(payload.get("value_calibration", {})),
    }
    return agent, metadata


def _teacher_episode(
    env: CommInspectionDubinsUAV2D,
    *,
    device_id: str,
    seed: int,
    max_steps: int,
) -> Optional[RawGoalSetEpisode]:
    episode, _abstract_episode = collect_task_aware_comm_teacher_episode_pair(
        env,
        max_steps=int(max_steps),
        seed=int(seed),
        device_id=str(device_id),
    )
    if episode is None:
        return None
    observations = episode.all_observations.detach().cpu().numpy().astype(np.float32)
    actions = episode.actions.detach().cpu().numpy().astype(np.float32)
    return episode_from_observations(
        env,
        observations,
        actions,
        device_id,
        truncated=episode.timeouts.detach().cpu().numpy().astype(bool),
        source="teacher",
    )


def _validate_agent(
    agent: GoalConditionedAgentBase,
    env: CommInspectionDubinsUAV2D,
    *,
    seed: int,
) -> Dict[str, float]:
    successes = []
    total_costs = []
    for device_index, device_id in enumerate(env.device_ids):
        episode_seed = int(seed) + device_index * 1_000_003
        obs, _ = env.reset(seed=episode_seed, options={"device_id": device_id})
        goal = env.abstract_goal_observation().astype(np.float32)
        terminated = truncated = False
        total_cost = 0.0
        final_info: Dict[str, object] = {}
        while not (terminated or truncated):
            action = agent.act(obs, goal, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_cost += -float(reward)
            final_info = dict(info)
        successes.append(float(bool(final_info.get("success", False))))
        total_costs.append(total_cost)
    return {
        "validation_success_rate": float(np.mean(successes)) if successes else 0.0,
        "validation_total_cost": float(np.mean(total_costs)) if total_costs else 0.0,
        "validation_num_tasks": float(len(successes)),
    }


def train_context_agent(
    algorithm: str,
    env: CommInspectionDubinsUAV2D,
    cfg: ContextGCRLConfig,
    device: torch.device,
    output_dir: Path,
    *,
    seed: int,
    progress_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> GoalConditionedAgentBase:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    env.action_space.seed(int(seed))
    rng = np.random.default_rng(int(seed) + 8_001)
    agent = make_context_agent(algorithm, env, cfg, device)
    replay = ContextHERReplayBuffer(
        env, cfg.replay_size, device, her_k=cfg.her_k, seed=int(seed) + 9_001
    )
    metric_fields = [
        "env_step", "teacher_steps", "updates", "critic_loss", "contrastive_loss",
        "actor_loss", "alpha_loss", "alpha", "q_mean", "mrn_metric_mean",
        "mrn_residual_mean", "contrastive_positive_accuracy",
        "contrastive_negative_accuracy", "contrastive_positive_score",
        "contrastive_negative_score", "replay_size", "policy_transitions",
        "teacher_transitions", "sampled_transitions", "relabel_count",
        "eligible_future_ratio", "positive_relabel_ratio",
    ]
    metrics_path = output_dir / "train_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=metric_fields).writeheader()
    validation_path = output_dir / "validation_metrics.csv"
    validation_fields = [
        "checkpoint_step", "actual_env_step", "teacher_steps", "updates",
        "validation_success_rate", "validation_total_cost", "validation_num_tasks",
    ]
    with validation_path.open("w", encoding="utf-8", newline="") as file:
        csv.DictWriter(file, fieldnames=validation_fields).writeheader()
    writer_tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer_tb = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    except ImportError:
        pass

    env_steps = 0
    teacher_steps = 0
    updates = 0
    episode_index = 0
    next_checkpoint = max(1, int(cfg.checkpoint_interval))
    last_stats: Dict[str, float] = {}
    while env_steps < int(cfg.total_env_steps):
        episode_seed = int(seed) + episode_index * 10_007
        obs, _reset_info = env.reset(seed=episode_seed)
        device_id = str(env.active_device_id)
        goal = env.abstract_goal_observation().astype(np.float32)
        states = [env.state.copy()]
        actions = []
        collisions = []
        out_of_bounds = []
        truncated_flags = []
        terminated = False
        truncated = False
        while not (terminated or truncated) and env_steps < int(cfg.total_env_steps):
            if env_steps < int(cfg.start_random_steps):
                action = env.action_space.sample().astype(np.float32)
            else:
                action = agent.act(obs, goal, eval_mode=False)
            next_obs, _reward, terminated, truncated, info = env.step(action)
            states.append(env.state.copy())
            actions.append(np.asarray(action, dtype=np.float32))
            collisions.append(bool(info.get("collision", False)))
            out_of_bounds.append(bool(info.get("out_of_bounds", False)))
            truncated_flags.append(bool(truncated))
            obs = next_obs
            env_steps += 1
        replay.add_episode(RawGoalSetEpisode(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            device_id=device_id,
            collisions=np.asarray(collisions, dtype=bool),
            out_of_bounds=np.asarray(out_of_bounds, dtype=bool),
            truncated=np.asarray(truncated_flags, dtype=bool),
            source="policy",
        ))

        teacher_floor = int(np.floor(max(0.0, cfg.teacher_ratio)))
        teacher_extra = int(float(rng.uniform()) < max(0.0, cfg.teacher_ratio - teacher_floor))
        for teacher_index in range(teacher_floor + teacher_extra):
            teacher = _teacher_episode(
                env,
                device_id=device_id,
                seed=int(seed) + 40_000_009 + episode_index * 1009 + teacher_index,
                max_steps=int(env.max_episode_steps),
            )
            if teacher is not None:
                replay.add_episode(teacher)
                teacher_steps += teacher.length

        update_count = len(actions) * max(1, int(cfg.updates_per_step))
        if replay.size >= int(cfg.batch_size):
            for _ in range(update_count):
                positive_only = isinstance(agent, ContextContrastiveRLAgent)
                try:
                    batch = replay.sample(cfg.batch_size, positive_only=positive_only)
                except RuntimeError:
                    break
                last_stats = agent.update(batch)  # type: ignore[attr-defined]
                updates += 1

        if last_stats and env_steps % max(1, int(cfg.log_interval)) < max(1, len(actions)):
            row = {
                "env_step": int(env_steps),
                "teacher_steps": int(teacher_steps),
                "updates": int(updates),
                **last_stats,
                **replay.diagnostics(),
            }
            with metrics_path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=metric_fields, extrasaction="ignore")
                writer.writerow(row)
            if writer_tb is not None:
                for key, value in row.items():
                    if key != "env_step":
                        writer_tb.add_scalar(f"train/{key}", float(value), env_steps)
            if progress_fn is not None:
                progress_fn(env_steps, {key: float(value) for key, value in row.items()})

        while env_steps >= next_checkpoint:
            validation = _validate_agent(
                agent,
                env,
                seed=int(seed) + 60_000_011,
            )
            validation_row = {
                "checkpoint_step": int(next_checkpoint),
                "actual_env_step": int(env_steps),
                "teacher_steps": int(teacher_steps),
                "updates": int(updates),
                **validation,
            }
            with validation_path.open("a", encoding="utf-8", newline="") as file:
                csv.DictWriter(file, fieldnames=validation_fields).writerow(validation_row)
            if writer_tb is not None:
                for key, value in validation.items():
                    writer_tb.add_scalar(f"validation/{key}", float(value), next_checkpoint)
            save_context_checkpoint(
                output_dir / f"checkpoint_{next_checkpoint}.pth",
                algorithm,
                agent,
                env,
                seed=seed,
                env_steps=env_steps,
                teacher_steps=teacher_steps,
                updates=updates,
                replay_diagnostics=replay.diagnostics(),
                training_diagnostics=last_stats,
            )
            next_checkpoint += max(1, int(cfg.checkpoint_interval))
        episode_index += 1

    save_context_checkpoint(
        output_dir / "checkpoint_final.pth",
        algorithm,
        agent,
        env,
        seed=seed,
        env_steps=env_steps,
        teacher_steps=teacher_steps,
        updates=updates,
        replay_diagnostics=replay.diagnostics(),
        training_diagnostics=last_stats,
    )
    if writer_tb is not None:
        writer_tb.flush()
        writer_tb.close()
    agent.eval()
    return agent
