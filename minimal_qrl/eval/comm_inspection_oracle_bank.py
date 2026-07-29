"""通信巡检 QRL 的固定 Hybrid A* 参考代价库。

Bank 生成是可恢复的：每完成一个样本就原子写回 JSON。规划失败或超时的
样本仍保留在 bank 中，用于报告 oracle coverage，但不会进入 MAE/MSE。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from minimal_qrl.baselines.hybrid_astar import (
    HybridAStarConfig,
    HybridAStarController,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D


ORACLE_BANK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CommInspectionOracleBankConfig:
    sample_count: int = 192
    generation_seed: int = 20_260_729
    candidate_multiplier: int = 16
    position_resolution: float = 0.25
    heading_bins: int = 24
    primitive_steps: int = 5
    heuristic_weight: float = 1.0
    max_expansions: int = 50_000
    timeout_sec: float = 60.0
    terminal_samples: int = 128

    def __post_init__(self) -> None:
        if int(self.sample_count) <= 0:
            raise ValueError("oracle sample_count must be positive")
        if int(self.candidate_multiplier) <= 0:
            raise ValueError("oracle candidate_multiplier must be positive")
        if float(self.position_resolution) <= 0.0:
            raise ValueError("oracle position_resolution must be positive")
        if int(self.heading_bins) <= 0:
            raise ValueError("oracle heading_bins must be positive")
        if int(self.primitive_steps) <= 0:
            raise ValueError("oracle primitive_steps must be positive")
        if int(self.max_expansions) <= 0:
            raise ValueError("oracle max_expansions must be positive")
        if float(self.timeout_sec) <= 0.0:
            raise ValueError("oracle timeout_sec must be positive")
        if int(self.terminal_samples) < 0:
            raise ValueError("oracle terminal_samples must be nonnegative")

    def planner_config(self) -> HybridAStarConfig:
        return HybridAStarConfig(
            position_resolution=float(self.position_resolution),
            heading_bins=int(self.heading_bins),
            primitive_steps=int(self.primitive_steps),
            heuristic_weight=float(self.heuristic_weight),
            max_expansions=int(self.max_expansions),
            timeout_sec=float(self.timeout_sec),
            terminal_samples=int(self.terminal_samples),
        )


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _planner_payload(config: HybridAStarConfig) -> Dict[str, Any]:
    """Return the planner config after the same tuple-to-list conversion as JSON."""
    return json.loads(json.dumps(asdict(config)))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _obstacle_payload(obstacle: Any) -> Dict[str, Any]:
    fields = {
        key: float(value)
        for key, value in vars(obstacle).items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    return {"type": type(obstacle).__name__, **fields}


def environment_signature(env: CommInspectionDubinsUAV2D) -> Dict[str, Any]:
    """返回足以拒绝陈旧 oracle cache 的环境签名。"""
    payload = {
        "catalog": asdict(env.device_catalog),
        "bounds": [float(value) for value in env.bounds],
        "omega_max": float(env.omega_max),
        "v": float(env.v),
        "dt": float(env.dt),
        "max_episode_steps": int(env.max_episode_steps),
        "obstacles": [_obstacle_payload(obstacle) for obstacle in env.obstacles],
        "comm_alpha": float(env.comm_alpha),
        "comm_bias": float(env.comm_bias),
        "comm_occlusion_penalty": float(env.comm_occlusion_penalty),
        "comm_threshold": float(env.comm_threshold),
        "require_ground_station_los": bool(env.require_ground_station_los),
        "collision_cost": float(env.collision_cost),
        "out_of_bounds_cost": float(env.out_of_bounds_cost),
        "communication_break_cost": float(env.communication_break_cost),
        "observation_violation_cost_weight": float(
            env.observation_violation_cost_weight
        ),
        "communication_violation_cost_weight": float(
            env.communication_violation_cost_weight
        ),
        "observation_failure_cost": float(env.observation_failure_cost),
        "min_start_target_distance": float(env.min_start_target_distance),
        "sample_max_attempts": int(env.sample_max_attempts),
    }
    return {"digest": _json_digest(payload), "payload": payload}


def _split_seed_offset(split: str) -> int:
    if split == "validation":
        return 0
    if split == "final_test":
        return 100_000_007
    raise ValueError(f"unsupported oracle split: {split!r}")


def _candidate_stratum(
    env: CommInspectionDubinsUAV2D,
    state: np.ndarray,
) -> Dict[str, Any]:
    direct_path_blocked = not env._segment_has_los(  # noqa: SLF001
        tuple(float(value) for value in state[:2]),
        env.observation_anchor,
        allow_endpoint_contact=True,
    )
    comm = env.compute_comm_quality(state)
    initial_comm_shadow = not bool(comm["has_los"])
    return {
        "direct_path_blocked": bool(direct_path_blocked),
        "initial_comm_shadow": bool(initial_comm_shadow),
        "initial_comm_feasible": bool(env.is_communication_feasible(state)),
        "start_target_distance": float(
            np.linalg.norm(
                np.asarray(state[:2], dtype=np.float32)
                - np.asarray(env.inspection_target, dtype=np.float32)
            )
        ),
        "stratum": (
            f"{'blocked' if direct_path_blocked else 'unblocked'}__"
            f"{'comm_shadow' if initial_comm_shadow else 'comm_los'}"
        ),
    }


def _evenly_spaced(items: Sequence[Dict[str, Any]], count: int) -> list[Dict[str, Any]]:
    if count <= 0 or not items:
        return []
    ordered = sorted(items, key=lambda row: float(row["start_target_distance"]))
    if len(ordered) <= count:
        return list(ordered)
    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[int(index)] for index in indices]


def _sample_counts_per_device(
    device_ids: Sequence[str],
    sample_count: int,
) -> Dict[str, int]:
    if sample_count <= 0:
        raise ValueError("oracle sample_count must be positive")
    if not device_ids:
        raise ValueError("oracle bank requires at least one device")
    base, remainder = divmod(int(sample_count), len(device_ids))
    return {
        str(device_id): int(base + (index < remainder))
        for index, device_id in enumerate(device_ids)
    }


def _build_task_records(
    env: CommInspectionDubinsUAV2D,
    *,
    split: str,
    config: CommInspectionOracleBankConfig,
) -> list[Dict[str, Any]]:
    counts = _sample_counts_per_device(env.device_ids, int(config.sample_count))
    split_offset = _split_seed_offset(split)
    records: list[Dict[str, Any]] = []
    stratum_names = (
        "unblocked__comm_los",
        "unblocked__comm_shadow",
        "blocked__comm_los",
        "blocked__comm_shadow",
    )

    for device_index, device_id in enumerate(env.device_ids):
        required = counts[str(device_id)]
        if required <= 0:
            continue
        candidate_count = max(64, required * max(4, int(config.candidate_multiplier)))
        candidates: list[Dict[str, Any]] = []
        for candidate_index in range(candidate_count):
            episode_seed = int(
                config.generation_seed
                + split_offset
                + device_index * 1_000_003
                + candidate_index * 101
            )
            observation, _ = env.reset(
                seed=episode_seed,
                options={"device_id": str(device_id)},
            )
            state = env.state.copy().astype(np.float32)
            candidate = {
                "device_id": str(device_id),
                "device_index": int(device_index),
                "episode_seed": int(episode_seed),
                "start_state": [float(value) for value in state],
                "observation": [
                    float(value) for value in np.asarray(observation, dtype=np.float32)
                ],
                "goal_observation": [
                    float(value)
                    for value in env.abstract_goal_observation().astype(np.float32)
                ],
                **_candidate_stratum(env, state),
            }
            candidates.append(candidate)

        selected: list[Dict[str, Any]] = []
        selected_seeds: set[int] = set()
        quota_base, quota_remainder = divmod(required, len(stratum_names))
        for stratum_index, stratum_name in enumerate(stratum_names):
            quota = quota_base + int(stratum_index < quota_remainder)
            group = [row for row in candidates if row["stratum"] == stratum_name]
            for row in _evenly_spaced(group, quota):
                selected.append(row)
                selected_seeds.add(int(row["episode_seed"]))

        if len(selected) < required:
            remaining = [
                row
                for row in candidates
                if int(row["episode_seed"]) not in selected_seeds
            ]
            for row in _evenly_spaced(remaining, required - len(selected)):
                selected.append(row)
                selected_seeds.add(int(row["episode_seed"]))

        if len(selected) != required:
            raise RuntimeError(
                f"failed to sample {required} oracle starts for device {device_id!r}"
            )

        selected.sort(key=lambda row: int(row["episode_seed"]))
        for sample_index, row in enumerate(selected):
            records.append(
                {
                    "task_id": f"{split}:{device_id}:{sample_index:03d}",
                    "split": split,
                    "sample_index": int(sample_index),
                    **row,
                    "status": "pending",
                    "oracle_cost": None,
                    "planner_success": False,
                    "rollout_verified": False,
                    "planner_failure_reason": "",
                    "planning_time_sec": None,
                    "expanded_nodes": None,
                    "generated_nodes": None,
                    "planned_action_count": None,
                }
            )

    if len(records) != int(config.sample_count):
        raise RuntimeError(
            f"oracle bank expected {config.sample_count} records, got {len(records)}"
        )
    return records


def _bank_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    solved = sum(row.get("status") == "solved" for row in records)
    failed = sum(row.get("status") == "failed" for row in records)
    pending = len(records) - solved - failed
    reasons: Dict[str, int] = {}
    for row in records:
        if row.get("status") != "failed":
            continue
        reason = str(row.get("planner_failure_reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    planning_times = [
        float(row["planning_time_sec"])
        for row in records
        if row.get("planning_time_sec") is not None
        and math.isfinite(float(row["planning_time_sec"]))
    ]
    strata: Dict[str, Dict[str, int]] = {}
    for row in records:
        stratum = str(row.get("stratum", "unknown"))
        stats = strata.setdefault(stratum, {"requested": 0, "solved": 0})
        stats["requested"] += 1
        stats["solved"] += int(row.get("status") == "solved")
    return {
        "requested_samples": int(len(records)),
        "solved_samples": int(solved),
        "failed_samples": int(failed),
        "pending_samples": int(pending),
        "oracle_coverage": float(solved / max(len(records), 1)),
        "failure_reasons": reasons,
        "mean_planning_time_sec": (
            float(np.mean(planning_times)) if planning_times else None
        ),
        "planning_time_p95_sec": (
            float(np.percentile(planning_times, 95)) if planning_times else None
        ),
        "strata": strata,
    }


def _validate_bank(
    bank: Mapping[str, Any],
    *,
    env_signature: Mapping[str, Any],
    split: str,
    config: CommInspectionOracleBankConfig,
    path: Path,
) -> None:
    errors = []
    if int(bank.get("schema_version", -1)) != ORACLE_BANK_SCHEMA_VERSION:
        errors.append("schema_version")
    if str(bank.get("split")) != split:
        errors.append("split")
    if int(bank.get("sample_count", -1)) != int(config.sample_count):
        errors.append("sample_count")
    if bank.get("environment_signature", {}).get("digest") != env_signature["digest"]:
        errors.append("environment_signature")
    if bank.get("planner_config") != _planner_payload(config.planner_config()):
        errors.append("planner_config")
    if int(bank.get("generation_seed", -1)) != int(config.generation_seed):
        errors.append("generation_seed")
    if errors:
        raise ValueError(
            f"oracle bank {path} is incompatible ({', '.join(errors)}); "
            "use a different path or remove the stale bank"
        )


def _verify_planned_path(
    controller: HybridAStarController,
    env: CommInspectionDubinsUAV2D,
) -> tuple[bool, Optional[float], str]:
    total_cost = 0.0
    final_info: Dict[str, Any] = {}
    for _ in range(int(env.max_episode_steps) + 1):
        action, action_info = controller.act(env.state_to_observation(env.state), env)
        if bool(action_info.get("plan_exhausted", False)):
            return False, None, "plan_exhausted_before_success"
        _obs, reward, terminated, truncated, info = env.step(action)
        final_info = dict(info)
        total_cost += float(info.get("cost_total", -float(reward)))
        if terminated or truncated:
            break
    if not bool(final_info.get("success", False)):
        return False, None, "rollout_verification_failed"
    return True, float(total_cost), ""


def _plan_record(
    env: CommInspectionDubinsUAV2D,
    record: Dict[str, Any],
    planner_config: HybridAStarConfig,
) -> None:
    env.reset(
        seed=int(record["episode_seed"]),
        options={
            "device_id": str(record["device_id"]),
            "start": np.asarray(record["start_state"], dtype=np.float32),
        },
    )
    controller = HybridAStarController(planner_config)
    diagnostics = controller.begin_episode(
        env,
        env.abstract_goal_observation(),
        seed=int(record["episode_seed"]),
    )
    record.update(
        {
            "planner_success": bool(diagnostics["planner_success"]),
            "planner_failure_reason": str(
                diagnostics.get("planner_failure_reason", "")
            ),
            "planning_time_sec": float(diagnostics["initial_planning_time_sec"]),
            "expanded_nodes": int(diagnostics["expanded_nodes"]),
            "generated_nodes": int(diagnostics["generated_nodes"]),
            "planned_action_count": int(diagnostics["planned_action_count"]),
        }
    )
    if not bool(diagnostics["planner_success"]):
        record["status"] = "failed"
        return

    verified, rollout_cost, verification_reason = _verify_planned_path(
        controller,
        env,
    )
    record["rollout_verified"] = bool(verified)
    if not verified or rollout_cost is None or not math.isfinite(rollout_cost):
        record["status"] = "failed"
        record["planner_failure_reason"] = verification_reason
        record["oracle_cost"] = None
        return

    planned_cost = float(diagnostics["planned_cost"])
    record["planned_cost"] = planned_cost
    record["oracle_cost"] = float(rollout_cost)
    record["planned_rollout_cost_abs_error"] = float(
        abs(planned_cost - rollout_cost)
    )
    record["status"] = "solved"
    record["planner_failure_reason"] = ""


def ensure_comm_inspection_oracle_bank(
    env: CommInspectionDubinsUAV2D,
    path: str | Path,
    *,
    split: str,
    config: CommInspectionOracleBankConfig,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """加载或可恢复地生成一个固定 Hybrid A* oracle bank。"""
    bank_path = Path(path)
    signature = environment_signature(env)
    if bank_path.exists():
        with bank_path.open("r", encoding="utf-8") as handle:
            bank: Dict[str, Any] = json.load(handle)
        _validate_bank(
            bank,
            env_signature=signature,
            split=split,
            config=config,
            path=bank_path,
        )
    else:
        records = _build_task_records(env, split=split, config=config)
        bank = {
            "schema_version": ORACLE_BANK_SCHEMA_VERSION,
            "oracle_kind": "high_budget_hybrid_astar_reference_cost",
            "cost_semantics": "undiscounted_environment_cost_total_to_goal_set",
            "split": split,
            "sample_count": int(config.sample_count),
            "generation_seed": int(config.generation_seed),
            "stratification": (
                "device_id x direct_path_blocked x initial_comm_shadow; "
                "distance-spread fallback for unavailable strata"
            ),
            "environment_signature": signature,
            "planner_config": _planner_payload(config.planner_config()),
            "records": records,
            "summary": _bank_summary(records),
        }
        _atomic_write_json(bank_path, bank)

    pending_indices = [
        index
        for index, record in enumerate(bank["records"])
        if record.get("status") == "pending"
    ]
    if progress_callback is not None:
        progress_callback(
            f"{split} oracle bank: {len(bank['records']) - len(pending_indices)}/"
            f"{len(bank['records'])} already complete"
        )

    planner_config = config.planner_config()
    with tqdm(
        pending_indices,
        desc=f"Hybrid A* oracle/{split}",
        unit="sample",
        dynamic_ncols=True,
    ) as progress:
        for record_index in progress:
            record = bank["records"][record_index]
            _plan_record(env, record, planner_config)
            bank["summary"] = _bank_summary(bank["records"])
            _atomic_write_json(bank_path, bank)
            progress.set_postfix(
                coverage=f"{bank['summary']['oracle_coverage']:.3f}",
                status=record["status"],
                refresh=False,
            )
            if progress_callback is not None:
                progress_callback(
                    f"{record['task_id']}: {record['status']}, "
                    f"time={float(record['planning_time_sec']):.2f}s"
                )
    return bank


def load_comm_inspection_oracle_bank(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        bank: Dict[str, Any] = json.load(handle)
    return bank


def _core_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    errors = prediction - target
    absolute = np.abs(errors)
    mse = float(np.mean(errors ** 2))
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    pearson_den = float(
        np.sqrt(np.sum(pred_centered ** 2) * np.sum(target_centered ** 2))
    )
    pearson = (
        float(np.sum(pred_centered * target_centered) / pearson_den)
        if pearson_den > 1e-12
        else float("nan")
    )
    spearman = float(spearmanr(prediction, target)[0])
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(absolute)),
        "median_absolute_error": float(np.median(absolute)),
        "normalized_mae": float(np.mean(absolute) / max(float(np.mean(target)), 1e-8)),
        "relative_error": float(
            np.mean(absolute / np.maximum(np.abs(target), 1e-6))
        ),
        "spearman_corr": spearman,
        "pearson_corr": pearson,
        "pred_mean": float(np.mean(prediction)),
        "pred_std": float(np.std(prediction)),
        "gt_mean": float(np.mean(target)),
        "gt_std": float(np.std(target)),
    }


def _cluster_bootstrap_intervals(
    prediction: np.ndarray,
    target: np.ndarray,
    device_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    unique_devices = np.unique(device_ids)
    if samples <= 0 or len(unique_devices) < 2:
        return {}
    indices_by_device = {
        device_id: np.flatnonzero(device_ids == device_id)
        for device_id in unique_devices
    }
    rng = np.random.default_rng(seed)
    draws: Dict[str, list[float]] = {
        "mse": [],
        "mae": [],
        "rmse": [],
        "spearman_corr": [],
        "pearson_corr": [],
    }
    for _ in range(int(samples)):
        sampled_devices = rng.choice(
            unique_devices,
            size=len(unique_devices),
            replace=True,
        )
        indices = np.concatenate(
            [indices_by_device[device_id] for device_id in sampled_devices]
        )
        metrics = _core_metrics(prediction[indices], target[indices])
        for key in draws:
            if math.isfinite(metrics[key]):
                draws[key].append(metrics[key])

    result: Dict[str, float] = {}
    for key, values in draws.items():
        if not values:
            continue
        result[f"{key}_ci95_low"] = float(np.percentile(values, 2.5))
        result[f"{key}_ci95_high"] = float(np.percentile(values, 97.5))
    return result


def evaluate_qrl_on_oracle_bank(
    agent: torch.nn.Module,
    bank: Mapping[str, Any],
    *,
    device: str | torch.device,
    distance_scale: float = 1.0,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> Dict[str, float]:
    records = list(bank.get("records", []))
    solved = [
        record
        for record in records
        if record.get("status") == "solved"
        and record.get("oracle_cost") is not None
        and math.isfinite(float(record["oracle_cost"]))
    ]
    if not solved:
        raise RuntimeError("oracle bank contains no solved samples")

    observations = np.asarray(
        [record["observation"] for record in solved],
        dtype=np.float32,
    )
    goals = np.asarray(
        [record["goal_observation"] for record in solved],
        dtype=np.float32,
    )
    target = np.asarray(
        [record["oracle_cost"] for record in solved],
        dtype=np.float64,
    )
    device_ids = np.asarray(
        [str(record["device_id"]) for record in solved],
        dtype=object,
    )
    critic = agent.critics[0]
    with torch.no_grad():
        states_t = torch.as_tensor(observations, device=device, dtype=torch.float32)
        goals_t = torch.as_tensor(goals, device=device, dtype=torch.float32)
        prediction = (
            critic.quasimetric_model(
                critic.encoder(states_t),
                critic.encoder(goals_t),
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )
    prediction *= float(distance_scale)

    result = _core_metrics(prediction, target)
    requested = len(records)
    result.update(
        {
            "requested_samples": float(requested),
            "solved_samples": float(len(solved)),
            "failed_samples": float(requested - len(solved)),
            "solved_devices": float(len(np.unique(device_ids))),
            "oracle_coverage": float(len(solved) / max(requested, 1)),
        }
    )
    result.update(
        _cluster_bootstrap_intervals(
            prediction,
            target,
            device_ids,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed),
        )
    )

    for blocked, label in ((False, "unblocked"), (True, "blocked")):
        mask = np.asarray(
            [bool(record["direct_path_blocked"]) == blocked for record in solved]
        )
        if not np.any(mask):
            continue
        group = _core_metrics(prediction[mask], target[mask])
        result[f"{label}_samples"] = float(np.sum(mask))
        for key in ("mae", "rmse", "spearman_corr"):
            result[f"{label}_{key}"] = float(group[key])
    return result
