#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minimal_qrl.baselines import (  # noqa: E402
    CalibratedValueAgent,
    ContextGCRLConfig,
    GoalSetSACConfig,
    HybridAStarConfig,
    HybridAStarController,
    HybridAStarValueOracle,
    MPPIConfig,
    MPPIController,
    PolicyController,
    load_goal_set_sac_checkpoint,
    load_context_checkpoint,
    rollout_controller_episode,
    train_goal_set_sac,
    train_context_agent,
)
from minimal_qrl.eval.comm_inspection_execution_eval import (  # noqa: E402
    VisualizationConfig,
    _save_rollout_visualization,
    build_qrl_adapter,
    load_task_records,
    make_comm_inspection_env,
)
from minimal_qrl.eval.utils import auto_device  # noqa: E402
from minimal_qrl.industry_exp.scalability_scenarios import (  # noqa: E402
    load_scenario_config,
)


CONTEXT_ALGORITHMS = {
    "context_her_ddpg",
    "context_contrastive_rl",
    "mrn_context_her_ddpg",
}
CONTEXT_MPPI_METHODS = {f"{name}_mppi" for name in CONTEXT_ALGORITHMS}
METHODS = {
    "hybrid_astar", "mppi_no_terminal", "model_mppi", "oracle_mppi", "goal_set_sac",
    "qrl_greedy", "qrl_mppi", "supervised_iqe_mppi", "targeted_supervised_iqe_mppi",
    "dense_transition_qrl_mppi", "full_graph_goal_set_qrl_mppi",
    *CONTEXT_ALGORITHMS, *CONTEXT_MPPI_METHODS,
}
SCALAR_METRICS = (
    "success",
    "num_steps",
    "total_cost",
    "cost_per_step",
    "observation_feasible_ratio",
    "communication_feasible_ratio",
    "task_feasible_ratio",
    "collision",
    "out_of_bounds",
    "first_task_feasible_step",
    "final_obs_margin",
    "final_comm_margin",
    "final_task_score",
    "total_planning_time_sec",
    "first_decision_time_sec",
    "planning_time_p95_sec",
    "planning_time_p99_sec",
    "expanded_nodes",
    "model_rollouts",
    "oracle_value_build_time_sec",
    "oracle_value_grid_states",
    "oracle_value_reachable_fraction",
    "oracle_value_queries",
    "oracle_value_unreachable_queries",
    "oracle_value_unreachable_ratio",
    "training_env_steps",
    "teacher_env_steps",
    "training_updates",
    "relabel_count",
    "cross_context_relabel_count",
    "eligible_future_ratio",
    "positive_relabel_ratio",
    "actor_parameters",
    "critic_parameters",
    "contrastive_positive_accuracy",
    "contrastive_negative_accuracy",
    "mrn_metric_mean",
    "mrn_residual_mean",
)
CSV_FIELDS = [
    "method",
    "model_run",
    "task_id",
    "stratum",
    "difficulty",
    "device_id",
    "episode_seed",
    *SCALAR_METRICS,
    "planner_failure_reason",
    "oracle_value_source",
    "oracle_value_table_digest",
    "oracle_value_cache_path",
]


def _record_key(record: Dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record["method"]),
        str(record["model_run"]),
        str(record["device_id"]),
        int(record["episode_seed"]),
    )


class IncrementalResultWriter:
    """Flush one completed episode immediately so interrupted runs retain data."""

    def __init__(self, output_dir: Path, *, resume: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "baseline_results.partial.csv"
        self.jsonl_path = self.output_dir / "baseline_results.partial.jsonl"
        self.progress_path = self.output_dir / "baseline_progress.json"
        self.existing_records: list[Dict[str, Any]] = []
        if resume and self.jsonl_path.exists():
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        self.existing_records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid partial JSONL record at line {line_number}: "
                            f"{self.jsonl_path}"
                        ) from exc

        self._csv_file = open(self.csv_path, "w", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        self._csv_writer.writeheader()
        self._csv_writer.writerows(self.existing_records)
        self._csv_file.flush()
        self._jsonl_file = open(
            self.jsonl_path,
            "a" if resume else "w",
            encoding="utf-8",
        )
        self.completed_records = len(self.existing_records)
        self.completed_keys = {_record_key(record) for record in self.existing_records}
        self._closed = False
        latest = self.existing_records[-1] if self.existing_records else None
        self._write_progress(status="running", latest=latest)

    def _write_progress(self, *, status: str, latest: Optional[Dict[str, Any]]) -> None:
        payload: Dict[str, Any] = {
            "status": str(status),
            "completed_records": int(self.completed_records),
            "partial_csv": str(self.csv_path),
            "partial_jsonl": str(self.jsonl_path),
        }
        if latest is not None:
            payload["latest"] = {
                key: latest[key]
                for key in ("method", "model_run", "device_id", "episode_seed", "success")
                if key in latest
            }
        tmp_path = self.progress_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.progress_path)

    def write(self, record: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("incremental result writer is already closed")
        self._csv_writer.writerow(record)
        self._csv_file.flush()
        self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()
        self.completed_records += 1
        self.completed_keys.add(_record_key(record))
        self._write_progress(status="running", latest=record)

    def mark_complete(self) -> None:
        self._write_progress(status="complete", latest=None)

    def close(self) -> None:
        if self._closed:
            return
        self._csv_file.close()
        self._jsonl_file.close()
        self._closed = True


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = set(methods) - METHODS
    if unknown:
        raise ValueError(f"Unknown baseline methods: {sorted(unknown)}")
    return methods


def _rollout_record(
    method: str,
    rollout: Dict[str, Any],
    model_run: str,
    task_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    infos = list(rollout["step_infos"])
    final_info = rollout["final_info"]
    steps = int(rollout["num_steps"])
    total_cost = float(-np.sum(np.asarray(rollout["rewards"], dtype=np.float64)))
    first_step = final_info.get("first_task_feasible_step")
    diagnostics = dict(rollout.get("controller_diagnostics", {}))
    record = {
        "method": method,
        "model_run": str(model_run),
        "episode_seed": int(rollout["seed"]),
        "success": float(bool(rollout["success"])),
        "num_steps": float(steps),
        "total_cost": total_cost,
        "cost_per_step": total_cost / max(steps, 1),
        "observation_feasible_ratio": float(np.mean([
            bool(info.get("observation_feasible", False)) for info in infos
        ])) if infos else 0.0,
        "communication_feasible_ratio": float(np.mean([
            bool(info.get("communication_feasible", False)) for info in infos
        ])) if infos else 0.0,
        "task_feasible_ratio": float(np.mean([
            bool(info.get("task_feasible", False)) for info in infos
        ])) if infos else 0.0,
        "collision": float(bool(rollout["collision"])),
        "out_of_bounds": float(bool(rollout["out_of_bounds"])),
        "first_task_feasible_step": float(first_step) if first_step is not None else float(steps),
        "final_obs_margin": float(final_info.get("obs_margin", 0.0)),
        "final_comm_margin": float(final_info.get("comm_margin", 0.0)),
        "final_task_score": float(final_info.get("task_score", 0.0)),
        "total_planning_time_sec": float(diagnostics.get("total_planning_time_sec", 0.0)),
        "first_decision_time_sec": float(diagnostics.get("first_decision_time_sec", 0.0)),
        "planning_time_p95_sec": float(diagnostics.get("planning_time_p95_sec", 0.0)),
        "planning_time_p99_sec": float(diagnostics.get("planning_time_p99_sec", 0.0)),
        "expanded_nodes": float(diagnostics.get("expanded_nodes", 0.0)),
        "model_rollouts": float(diagnostics.get("model_rollouts", 0.0)),
        "oracle_value_build_time_sec": float(
            diagnostics.get("oracle_value_build_time_sec", 0.0)
        ),
        "oracle_value_grid_states": float(
            diagnostics.get("oracle_value_grid_states", 0.0)
        ),
        "oracle_value_reachable_fraction": float(
            diagnostics.get("oracle_value_reachable_fraction", 0.0)
        ),
        "oracle_value_queries": float(diagnostics.get("oracle_value_queries", 0.0)),
        "oracle_value_unreachable_queries": float(
            diagnostics.get("oracle_value_unreachable_queries", 0.0)
        ),
        "oracle_value_unreachable_ratio": float(
            diagnostics.get("oracle_value_unreachable_ratio", 0.0)
        ),
        "training_env_steps": float(diagnostics.get("training_env_steps", 0.0)),
        "teacher_env_steps": float(diagnostics.get("teacher_env_steps", 0.0)),
        "training_updates": float(diagnostics.get("training_updates", 0.0)),
        "relabel_count": float(diagnostics.get("relabel_count", 0.0)),
        "cross_context_relabel_count": float(diagnostics.get("cross_context_relabel_count", 0.0)),
        "eligible_future_ratio": float(diagnostics.get("eligible_future_ratio", 0.0)),
        "positive_relabel_ratio": float(diagnostics.get("positive_relabel_ratio", 0.0)),
        "actor_parameters": float(diagnostics.get("actor_parameters", 0.0)),
        "critic_parameters": float(diagnostics.get("critic_parameters", 0.0)),
        "contrastive_positive_accuracy": float(diagnostics.get("contrastive_positive_accuracy", 0.0)),
        "contrastive_negative_accuracy": float(diagnostics.get("contrastive_negative_accuracy", 0.0)),
        "mrn_metric_mean": float(diagnostics.get("mrn_metric_mean", 0.0)),
        "mrn_residual_mean": float(diagnostics.get("mrn_residual_mean", 0.0)),
        "planner_failure_reason": str(diagnostics.get("planner_failure_reason", "")),
        "oracle_value_source": str(diagnostics.get("oracle_value_source", "")),
        "oracle_value_table_digest": str(
            diagnostics.get("oracle_value_table_digest", "")
        ),
        "oracle_value_cache_path": str(diagnostics.get("oracle_value_cache_path", "")),
        "start": [float(v) for v in rollout["start"]],
        "inspection_target": [float(v) for v in rollout["inspection_target"]],
        "ground_station": [float(v) for v in rollout["ground_station"]],
        "device_id": str(rollout["device_id"]),
    }
    if task_spec is not None:
        record.update(
            {
                "task_id": str(task_spec.get("task_id", "")),
                "stratum": str(task_spec.get("stratum", "")),
                "difficulty": str(task_spec.get("difficulty", "")),
            }
        )
    return record


def _run_task_matrix(
    records: list[Dict[str, Any]],
    key: str,
    episode_seeds: Optional[list[int]] = None,
) -> tuple[np.ndarray, list[int]]:
    runs = sorted({str(row["model_run"]) for row in records})
    seeds = episode_seeds or sorted({int(row["episode_seed"]) for row in records})
    matrix = np.full((len(runs), len(seeds)), np.nan, dtype=np.float64)
    for run_index, run in enumerate(runs):
        rows_by_seed: Dict[int, list[float]] = defaultdict(list)
        for row in records:
            if str(row["model_run"]) == run:
                rows_by_seed[int(row["episode_seed"])].append(float(row[key]))
        for seed_index, episode_seed in enumerate(seeds):
            values = rows_by_seed.get(episode_seed, [])
            if values:
                matrix[run_index, seed_index] = float(np.mean(values))
    return matrix, seeds


def _hierarchical_bootstrap_ci(
    records: list[Dict[str, Any]],
    key: str,
    rng: np.random.Generator,
    n_boot: int,
) -> list[float]:
    matrix, _seeds = _run_task_matrix(records, key)
    if matrix.size == 0:
        return [0.0, 0.0]
    samples = []
    for _ in range(max(1, int(n_boot))):
        run_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        task_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        samples.append(float(np.nanmean(matrix[run_indices][:, task_indices])))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _hierarchical_paired_ci(
    method_records: list[Dict[str, Any]],
    reference_records: list[Dict[str, Any]],
    key: str,
    common_seeds: list[int],
    rng: np.random.Generator,
    n_boot: int,
) -> list[float]:
    method_matrix, _ = _run_task_matrix(method_records, key, common_seeds)
    reference_matrix, _ = _run_task_matrix(reference_records, key, common_seeds)
    samples = []
    for _ in range(max(1, int(n_boot))):
        method_runs = rng.integers(0, method_matrix.shape[0], size=method_matrix.shape[0])
        reference_runs = rng.integers(0, reference_matrix.shape[0], size=reference_matrix.shape[0])
        tasks = rng.integers(0, len(common_seeds), size=len(common_seeds))
        method_mean = float(np.nanmean(method_matrix[method_runs][:, tasks]))
        reference_mean = float(np.nanmean(reference_matrix[reference_runs][:, tasks]))
        samples.append(method_mean - reference_mean)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _summarize(records: list[Dict[str, Any]], *, seed: int, n_boot: int) -> Dict[str, Any]:
    by_method: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[str(record["method"])].append(record)
    summary: Dict[str, Any] = {}
    for method, method_records in by_method.items():
        rng = np.random.default_rng(int(seed) + sum(ord(ch) for ch in method))
        metrics: Dict[str, Any] = {"num_records": len(method_records)}
        for key in SCALAR_METRICS:
            values = np.asarray([float(row[key]) for row in method_records], dtype=np.float64)
            metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "bootstrap_95_ci": _hierarchical_bootstrap_ci(
                    method_records, key, rng, n_boot
                ),
            }
        failure_reasons: Dict[str, int] = defaultdict(int)
        for row in method_records:
            reason = str(row.get("planner_failure_reason", ""))
            if reason:
                failure_reasons[reason] += 1
        metrics["planner_failure_reasons"] = dict(failure_reasons)
        per_device: Dict[str, Any] = {}
        for device_id in sorted({str(row["device_id"]) for row in method_records}):
            rows = [row for row in method_records if str(row["device_id"]) == device_id]
            per_device[device_id] = {
                "num_records": len(rows),
                "success_rate": float(np.mean([float(row["success"]) for row in rows])),
                "avg_total_cost": float(np.mean([float(row["total_cost"]) for row in rows])),
                "avg_steps": float(np.mean([float(row["num_steps"]) for row in rows])),
                "communication_feasible_ratio": float(
                    np.mean([float(row["communication_feasible_ratio"]) for row in rows])
                ),
            }
        metrics["per_device"] = per_device
        metrics["macro_device_success_rate"] = float(
            np.mean([row["success_rate"] for row in per_device.values()])
        ) if per_device else 0.0
        metrics["macro_device_avg_total_cost"] = float(
            np.mean([row["avg_total_cost"] for row in per_device.values()])
        ) if per_device else 0.0
        metrics["macro_device_avg_steps"] = float(
            np.mean([row["avg_steps"] for row in per_device.values()])
        ) if per_device else 0.0
        metrics["macro_device_communication_feasible_ratio"] = float(
            np.mean([row["communication_feasible_ratio"] for row in per_device.values()])
        ) if per_device else 0.0
        per_stratum: Dict[str, Any] = {}
        strata = sorted(
            {str(row.get("stratum", "")) for row in method_records if row.get("stratum")}
        )
        for stratum in strata:
            rows = [row for row in method_records if str(row.get("stratum", "")) == stratum]
            per_stratum[stratum] = {
                "num_records": len(rows),
                "success_rate": float(np.mean([float(row["success"]) for row in rows])),
                "avg_total_cost": float(np.mean([float(row["total_cost"]) for row in rows])),
                "avg_steps": float(np.mean([float(row["num_steps"]) for row in rows])),
                "communication_feasible_ratio": float(
                    np.mean([float(row["communication_feasible_ratio"]) for row in rows])
                ),
            }
        metrics["per_stratum"] = per_stratum
        metrics["macro_stratum_success_rate"] = (
            float(np.mean([row["success_rate"] for row in per_stratum.values()]))
            if per_stratum
            else 0.0
        )
        summary[method] = metrics

    def add_paired_comparison(
        paired: Dict[str, Any],
        method: str,
        reference_name: str,
        comparison_name: Optional[str] = None,
    ) -> None:
        reference = by_method.get(reference_name, [])
        method_records = by_method.get(method, [])
        if not reference or not method_records:
            return
        reference_by_seed: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
        for row in reference:
            reference_by_seed[int(row["episode_seed"])].append(row)
        method_by_seed: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
        for row in method_records:
            method_by_seed[int(row["episode_seed"])].append(row)
        common = sorted(set(reference_by_seed) & set(method_by_seed))
        if not common:
            return
        name = comparison_name or method
        rng = np.random.default_rng(int(seed) + 17 + sum(ord(ch) for ch in name))
        paired[name] = {"reference": reference_name, "method": method}
        for key in ("success", "total_cost", "num_steps", "collision"):
            differences = np.asarray([
                np.mean([float(row[key]) for row in method_by_seed[s]])
                - np.mean([float(row[key]) for row in reference_by_seed[s]])
                for s in common
            ])
            paired[name][f"{key}_difference_vs_{reference_name}"] = {
                "mean": float(np.mean(differences)),
                "bootstrap_95_ci": _hierarchical_paired_ci(
                    method_records,
                    reference,
                    key,
                    common,
                    rng,
                    n_boot,
                ),
                "num_pairs": len(common),
            }

    paired: Dict[str, Any] = {}
    for method in by_method:
        if method in {"qrl_greedy", "qrl_mppi"}:
            continue
        reference_name = "qrl_mppi" if method.endswith("_mppi") else "qrl_greedy"
        add_paired_comparison(paired, method, reference_name)
    add_paired_comparison(
        paired,
        "mrn_context_her_ddpg",
        "context_her_ddpg",
        "mrn_vs_monolithic_ddpg",
    )
    add_paired_comparison(
        paired,
        "mrn_context_her_ddpg_mppi",
        "context_her_ddpg_mppi",
        "mrn_vs_monolithic_ddpg_mppi",
    )
    if paired:
        summary["paired_comparisons"] = paired
    return summary


def _save_visualization(
    env,
    rollout: Dict[str, Any],
    method: str,
    episode_index: int,
    output_dir: Path,
    viz_cfg: VisualizationConfig,
    counters: Dict[str, Dict[str, int]],
) -> None:
    if not viz_cfg.save_visualizations:
        return
    category = "success" if rollout["success"] else "failure"
    limit = viz_cfg.max_successes if category == "success" else viz_cfg.max_failures
    if counters[method][category] >= int(limit):
        return
    category_dir = output_dir / "visualizations" / method / category
    category_dir.mkdir(parents=True, exist_ok=True)
    _save_rollout_visualization(
        env,
        rollout,
        execution_mode=method,
        episode_index=episode_index,
        category=category,
        category_dir=category_dir,
        base_output_dir=output_dir,
        viz_cfg=viz_cfg,
    )
    counters[method][category] += 1


def _evaluate_controller(
    method: str,
    controller,
    env,
    episode_specs: Iterable[Dict[str, Any]],
    *,
    model_run: str,
    output_dir: Path,
    viz_cfg: VisualizationConfig,
    counters: Dict[str, Dict[str, int]],
    on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
    completed_keys: Optional[set[tuple[str, str, str, int]]] = None,
) -> list[Dict[str, Any]]:
    if completed_keys is None:
        completed_keys = set()
    normalized_specs = []
    for spec in episode_specs:
        if isinstance(spec, dict):
            normalized_specs.append(dict(spec))
        else:
            device_id, episode_seed = spec
            normalized_specs.append(
                {
                    "task_id": f"random:{device_id}:{int(episode_seed)}",
                    "device_id": str(device_id),
                    "seed": int(episode_seed),
                    "start": None,
                }
            )
    episode_specs = [
        spec
        for spec in normalized_specs
        if (
            str(method),
            str(model_run),
            str(spec["device_id"]),
            int(spec["seed"]),
        ) not in completed_keys
    ]
    specs_by_device: Dict[str, list[Dict[str, Any]]] = {}
    for spec in episode_specs:
        specs_by_device.setdefault(str(spec["device_id"]), []).append(spec)

    records = []
    success_count = 0
    episode_index = 0
    algorithm_desc = f"{method}/{model_run}"
    with tqdm(
        total=len(episode_specs),
        desc=algorithm_desc,
        unit="episode",
        position=0,
        dynamic_ncols=True,
    ) as algorithm_bar:
        for device_index, (device_id, device_specs) in enumerate(specs_by_device.items(), start=1):
            device_desc = f"{algorithm_desc} [{device_index}/{len(specs_by_device)}] {device_id}"
            with tqdm(
                device_specs,
                desc=device_desc,
                unit="start",
                leave=False,
                position=1,
                dynamic_ncols=True,
            ) as device_bar:
                for task_spec in device_bar:
                    episode_seed = int(task_spec["seed"])
                    rollout = rollout_controller_episode(
                        controller,
                        env,
                        episode_seed=episode_seed,
                        device_id=device_id,
                        start_state=(
                            np.asarray(task_spec["start"], dtype=np.float32)
                            if task_spec.get("start") is not None
                            else None
                        ),
                    )
                    record = _rollout_record(method, rollout, model_run, task_spec)
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
                    success_count += int(bool(record["success"]))
                    _save_visualization(
                        env,
                        rollout,
                        method,
                        episode_index,
                        output_dir,
                        viz_cfg,
                        counters,
                    )
                    episode_index += 1
                    algorithm_bar.update(1)
                    success_rate = success_count / max(episode_index, 1)
                    algorithm_bar.set_postfix(
                        device=device_id,
                        success=f"{success_rate:.3f}",
                        refresh=False,
                    )
                    device_bar.set_postfix(
                        success=f"{float(record['success']):.0f}",
                        steps=f"{int(record['num_steps'])}",
                        refresh=False,
                    )
    return records


def _write_csv(path: Path, records: list[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _stage_defaults(stage: str) -> tuple[int, int, list[int]]:
    if stage == "smoke":
        return 3, 200, [0]
    if stage == "pilot":
        return 50, 300_000, [0]
    return 300, 300_000, [0, 1, 2, 3, 4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified goal-set communication-inspection baselines")
    parser.add_argument("--stage", choices=["smoke", "pilot", "final"], default="smoke")
    parser.add_argument("--methods", default=",".join(sorted(METHODS)))
    parser.add_argument("--output-dir", default="results/comm_inspection_baselines")
    parser.add_argument("--qrl-checkpoints", nargs="*", default=[])
    parser.add_argument("--sac-checkpoints", nargs="*", default=[])
    parser.add_argument("--context-checkpoints", nargs="*", default=[])
    parser.add_argument("--train-sac", action="store_true")
    parser.add_argument("--train-context-agents", action="store_true")
    parser.add_argument("--sac-seeds", default=None)
    parser.add_argument("--sac-total-env-steps", type=int, default=None)
    parser.add_argument("--starts-per-device", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from baseline_results.partial.jsonl and skip completed records.",
    )
    parser.add_argument("--viz-max-successes", type=int, default=3)
    parser.add_argument("--viz-max-failures", type=int, default=3)

    parser.add_argument("--bounds", type=float, nargs=4, default=[0.0, 0.0, 10.0, 10.0])
    parser.add_argument("--omega-max", type=float, default=3.0)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--obstacle-config", choices=["none", "simple", "medium", "hard"], default="medium")
    parser.add_argument("--obstacles", type=float, nargs="*", default=None)
    parser.add_argument("--device-catalog", default=None)
    parser.add_argument("--comm-alpha", type=float, default=2.0)
    parser.add_argument("--comm-bias", type=float, default=5.0)
    parser.add_argument("--comm-occlusion-penalty", type=float, default=6.0)
    parser.add_argument("--comm-threshold", type=float, default=0.5)
    parser.add_argument("--require-ground-station-los", action="store_true")
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
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--env-name", default="comm_inspection_unified_baselines")
    parser.add_argument("--scenario-config", default=None)
    parser.add_argument("--task-bank", default=None)
    parser.add_argument("--task-split", choices=["validation", "test"], default="test")
    parser.add_argument(
        "--task-strata",
        default=None,
        help="Optional comma-separated task-bank strata to evaluate (for example: u_trap).",
    )

    parser.add_argument("--astar-position-resolution", type=float, default=0.25)
    parser.add_argument("--astar-heading-bins", type=int, default=24)
    parser.add_argument("--astar-primitive-steps", type=int, default=5)
    parser.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    parser.add_argument("--astar-max-expansions", type=int, default=50_000)
    parser.add_argument("--astar-timeout-sec", type=float, default=30.0)
    parser.add_argument("--astar-terminal-samples", type=int, default=128)

    parser.add_argument("--mppi-horizon", type=int, default=20)
    parser.add_argument("--mppi-num-samples", type=int, default=256)
    parser.add_argument("--mppi-noise-sigma", type=float, default=0.8)
    parser.add_argument("--mppi-temperature", type=float, default=1.0)
    parser.add_argument("--mppi-terminal-weight", type=float, default=1.0)
    parser.add_argument("--mppi-terminal-samples", type=int, default=128)
    parser.add_argument(
        "--oracle-value-cache-dir",
        default=None,
        help="Directory for reusable exhaustive Hybrid A* cost-to-go tables.",
    )

    parser.add_argument("--sac-batch-size", type=int, default=256)
    parser.add_argument("--sac-start-random-steps", type=int, default=5_000)
    parser.add_argument("--sac-hidden-dim", type=int, default=256)
    parser.add_argument("--context-total-env-steps", type=int, default=None)
    parser.add_argument("--context-seeds", default=None)
    parser.add_argument("--context-batch-size", type=int, default=256)
    parser.add_argument("--context-hidden-dim", type=int, default=256)
    parser.add_argument("--context-representation-dim", type=int, default=64)
    parser.add_argument("--context-replay-size", type=int, default=500_000)
    parser.add_argument("--context-start-random-steps", type=int, default=5_000)
    parser.add_argument("--context-her-k", type=int, default=4)
    parser.add_argument("--context-teacher-ratio", type=float, default=1.0)
    parser.add_argument("--context-checkpoint-interval", type=int, default=50_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenario = None
    args._scenario_data = None
    if args.scenario_config:
        scenario = load_scenario_config(args.scenario_config)
        args._scenario_data = scenario
        args.bounds = [float(value) for value in scenario["bounds"]]
        args.max_episode_steps = int(scenario["max_episode_steps"])
        args.device_catalog = scenario["device_catalog"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = _parse_methods(args.methods)
    stage_trials, stage_steps, stage_seeds = _stage_defaults(args.stage)
    starts_per_device = stage_trials if args.starts_per_device is None else int(args.starts_per_device)
    if starts_per_device <= 0:
        raise ValueError("--starts-per-device must be positive")
    sac_steps = stage_steps if args.sac_total_env_steps is None else int(args.sac_total_env_steps)
    sac_seeds = stage_seeds if args.sac_seeds is None else [
        int(value) for value in str(args.sac_seeds).split(",") if value.strip()
    ]
    context_steps = stage_steps if args.context_total_env_steps is None else int(args.context_total_env_steps)
    context_seeds = stage_seeds if args.context_seeds is None else [
        int(value) for value in str(args.context_seeds).split(",") if value.strip()
    ]
    device = auto_device(args.device)
    spec_env = make_comm_inspection_env(args)
    if args.task_bank:
        _task_bank, task_records = load_task_records(
            args.task_bank,
            split=str(args.task_split),
            bounds=[float(value) for value in args.bounds],
            scenario_id=str(scenario["scenario_id"]) if scenario is not None else None,
        )
        known_devices = set(spec_env.device_ids)
        episode_specs = [
            {
                **row,
                "seed": int(row["seed"]),
                "device_id": str(row["device_id"]),
                "start": [float(value) for value in row["start"]],
            }
            for row in task_records
            if str(row["device_id"]) in known_devices
        ]
        if not episode_specs:
            raise ValueError("task bank contains no records for devices in this environment")
        if args.task_strata:
            requested_strata = {
                item.strip() for item in str(args.task_strata).split(",") if item.strip()
            }
            known_strata = {str(row.get("stratum", "")) for row in episode_specs}
            unknown_strata = requested_strata - known_strata
            if unknown_strata:
                raise ValueError(
                    f"task bank split has no requested strata: {sorted(unknown_strata)}"
                )
            episode_specs = [
                row for row in episode_specs if str(row.get("stratum", "")) in requested_strata
            ]
    else:
        episode_specs = [
            {
                "task_id": f"random:{device_id}:{start_index:03d}",
                "device_id": str(device_id),
                "seed": int(args.seed) + device_index * 1_000_003 + start_index,
                "start": None,
            }
            for device_index, device_id in enumerate(spec_env.device_ids)
            for start_index in range(starts_per_device)
        ]
    viz_cfg = VisualizationConfig(
        save_visualizations=bool(args.save_visualizations),
        max_successes=int(args.viz_max_successes),
        max_failures=int(args.viz_max_failures),
        save_gif=False,
        gif_fps=8,
    )
    counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    incremental_writer = IncrementalResultWriter(output_dir, resume=bool(args.resume))
    atexit.register(incremental_writer.close)
    records: list[Dict[str, Any]] = list(incremental_writer.existing_records)
    completed_keys = incremental_writer.completed_keys
    if args.resume:
        print(
            f"Resuming from {incremental_writer.jsonl_path}: "
            f"{len(records)} completed records loaded"
        )

    astar_cfg = HybridAStarConfig(
        position_resolution=args.astar_position_resolution,
        heading_bins=args.astar_heading_bins,
        primitive_steps=args.astar_primitive_steps,
        heuristic_weight=args.astar_heuristic_weight,
        max_expansions=args.astar_max_expansions,
        timeout_sec=args.astar_timeout_sec,
        terminal_samples=args.astar_terminal_samples,
    )
    mppi_cfg = MPPIConfig(
        horizon=args.mppi_horizon,
        num_samples=args.mppi_num_samples,
        noise_sigma=args.mppi_noise_sigma,
        temperature=args.mppi_temperature,
        terminal_weight=args.mppi_terminal_weight,
        terminal_samples=args.mppi_terminal_samples,
    )

    if "hybrid_astar" in methods:
        env = make_comm_inspection_env(args)
        records.extend(_evaluate_controller(
            "hybrid_astar", HybridAStarController(astar_cfg), env, episode_specs,
            model_run="model", output_dir=output_dir, viz_cfg=viz_cfg, counters=counters,
            on_record=incremental_writer.write,
            completed_keys=completed_keys,
        ))
    if "mppi_no_terminal" in methods:
        env = make_comm_inspection_env(args)
        records.extend(_evaluate_controller(
            "mppi_no_terminal", MPPIController(mppi_cfg, terminal_mode="none"), env, episode_specs,
            model_run="model", output_dir=output_dir, viz_cfg=viz_cfg, counters=counters,
            on_record=incremental_writer.write,
            completed_keys=completed_keys,
        ))
    if "model_mppi" in methods:
        env = make_comm_inspection_env(args)
        records.extend(_evaluate_controller(
            "model_mppi", MPPIController(mppi_cfg, terminal_mode="model"), env, episode_specs,
            model_run="model", output_dir=output_dir, viz_cfg=viz_cfg, counters=counters,
            on_record=incremental_writer.write,
            completed_keys=completed_keys,
        ))
    if "oracle_mppi" in methods:
        env = make_comm_inspection_env(args)
        oracle_cache_dir = (
            Path(args.oracle_value_cache_dir)
            if args.oracle_value_cache_dir
            else output_dir / "oracle_value_cache"
        )
        oracle_value = HybridAStarValueOracle(
            astar_cfg,
            cache_dir=oracle_cache_dir,
            unreachable_cost=float(mppi_cfg.invalid_penalty),
        )
        records.extend(_evaluate_controller(
            "oracle_mppi",
            MPPIController(
                mppi_cfg,
                terminal_mode="oracle",
                terminal_value_provider=oracle_value,
            ),
            env,
            episode_specs,
            model_run="model",
            output_dir=output_dir,
            viz_cfg=viz_cfg,
            counters=counters,
            on_record=incremental_writer.write,
            completed_keys=completed_keys,
        ))

    qrl_methods = {
        "qrl_greedy",
        "qrl_mppi",
        "supervised_iqe_mppi",
        "targeted_supervised_iqe_mppi",
        "dense_transition_qrl_mppi",
        "full_graph_goal_set_qrl_mppi",
    } & set(methods)
    if qrl_methods and not args.qrl_checkpoints:
        raise ValueError("QRL methods require --qrl-checkpoints")
    for qrl_index, checkpoint in enumerate(args.qrl_checkpoints):
        if not qrl_methods:
            break
        env = make_comm_inspection_env(args)
        qrl_args = argparse.Namespace(**vars(args))
        qrl_args.checkpoint = checkpoint
        qrl_args.env_name = f"{args.env_name}_qrl_{qrl_index}"
        qrl_agent, _ = build_qrl_adapter(qrl_args, device, env)
        model_run = f"qrl_{qrl_index}"
        if "qrl_greedy" in methods:
            records.extend(_evaluate_controller(
                "qrl_greedy", PolicyController(qrl_agent, name="qrl_greedy"), env,
                episode_specs, model_run=model_run, output_dir=output_dir,
                viz_cfg=viz_cfg, counters=counters,
                on_record=incremental_writer.write,
                completed_keys=completed_keys,
            ))
        if "qrl_mppi" in methods:
            records.extend(_evaluate_controller(
                "qrl_mppi", MPPIController(mppi_cfg, terminal_mode="qrl", qrl_agent=qrl_agent),
                env, episode_specs, model_run=model_run, output_dir=output_dir,
                viz_cfg=viz_cfg, counters=counters,
                on_record=incremental_writer.write,
                completed_keys=completed_keys,
            ))
        if "supervised_iqe_mppi" in methods:
            supervised_controller = MPPIController(
                mppi_cfg,
                terminal_mode="qrl",
                qrl_agent=qrl_agent,
            )
            supervised_controller.name = "supervised_iqe_mppi"
            records.extend(_evaluate_controller(
                "supervised_iqe_mppi",
                supervised_controller,
                env,
                episode_specs,
                model_run=f"supervised_iqe_{qrl_index}",
                output_dir=output_dir,
                viz_cfg=viz_cfg,
                counters=counters,
                on_record=incremental_writer.write,
                completed_keys=completed_keys,
            ))
        if "targeted_supervised_iqe_mppi" in methods:
            targeted_controller = MPPIController(
                mppi_cfg,
                terminal_mode="qrl",
                qrl_agent=qrl_agent,
            )
            targeted_controller.name = "targeted_supervised_iqe_mppi"
            records.extend(_evaluate_controller(
                "targeted_supervised_iqe_mppi",
                targeted_controller,
                env,
                episode_specs,
                model_run=f"targeted_supervised_iqe_{qrl_index}",
                output_dir=output_dir,
                viz_cfg=viz_cfg,
                counters=counters,
                on_record=incremental_writer.write,
                completed_keys=completed_keys,
            ))
        if "dense_transition_qrl_mppi" in methods:
            dense_controller = MPPIController(
                mppi_cfg,
                terminal_mode="qrl",
                qrl_agent=qrl_agent,
            )
            dense_controller.name = "dense_transition_qrl_mppi"
            records.extend(_evaluate_controller(
                "dense_transition_qrl_mppi",
                dense_controller,
                env,
                episode_specs,
                model_run=f"dense_transition_qrl_{qrl_index}",
                output_dir=output_dir,
                viz_cfg=viz_cfg,
                counters=counters,
                on_record=incremental_writer.write,
                completed_keys=completed_keys,
            ))
        if "full_graph_goal_set_qrl_mppi" in methods:
            full_graph_controller = MPPIController(
                mppi_cfg,
                terminal_mode="qrl",
                qrl_agent=qrl_agent,
            )
            full_graph_controller.name = "full_graph_goal_set_qrl_mppi"
            records.extend(_evaluate_controller(
                "full_graph_goal_set_qrl_mppi",
                full_graph_controller,
                env,
                episode_specs,
                model_run=f"full_graph_goal_set_qrl_{qrl_index}",
                output_dir=output_dir,
                viz_cfg=viz_cfg,
                counters=counters,
                on_record=incremental_writer.write,
                completed_keys=completed_keys,
            ))

    sac_paths = [Path(path) for path in args.sac_checkpoints]
    if "goal_set_sac" in methods and args.train_sac:
        sac_cfg = GoalSetSACConfig(
            hidden_dim=args.sac_hidden_dim,
            batch_size=args.sac_batch_size,
            total_env_steps=sac_steps,
            start_random_steps=min(args.sac_start_random_steps, sac_steps),
            checkpoint_interval=max(sac_steps, 1),
        )
        for sac_seed in sac_seeds:
            train_env = make_comm_inspection_env(args)
            run_dir = output_dir / "goal_set_sac" / f"seed_{sac_seed}"
            train_goal_set_sac(train_env, sac_cfg, device, run_dir, seed=sac_seed)
            sac_paths.append(run_dir / "checkpoint_final.pth")
    if "goal_set_sac" in methods and not sac_paths:
        raise ValueError("goal_set_sac requires --train-sac or --sac-checkpoints")
    for sac_index, checkpoint in enumerate(sac_paths):
        if "goal_set_sac" not in methods:
            break
        env = make_comm_inspection_env(args)
        sac_agent, metadata = load_goal_set_sac_checkpoint(checkpoint, env, device)
        records.extend(_evaluate_controller(
            "goal_set_sac", PolicyController(
                sac_agent,
                name="goal_set_sac",
                static_diagnostics={"training_env_steps": int(metadata["env_steps"])},
            ), env,
            episode_specs, model_run=f"sac_seed_{metadata['seed']}_{sac_index}",
            output_dir=output_dir, viz_cfg=viz_cfg, counters=counters,
            on_record=incremental_writer.write,
            completed_keys=completed_keys,
        ))

    requested_context_algorithms = {
        name for name in CONTEXT_ALGORITHMS
        if name in methods or f"{name}_mppi" in methods
    }
    context_paths = [Path(path) for path in args.context_checkpoints]
    if requested_context_algorithms and args.train_context_agents:
        context_batch_size = int(args.context_batch_size)
        if args.stage == "smoke":
            context_batch_size = min(context_batch_size, 32)
        context_cfg = ContextGCRLConfig(
            hidden_dim=int(args.context_hidden_dim),
            representation_dim=int(args.context_representation_dim),
            residual_dim=int(args.context_representation_dim),
            batch_size=context_batch_size,
            replay_size=int(args.context_replay_size),
            total_env_steps=context_steps,
            start_random_steps=min(int(args.context_start_random_steps), context_steps),
            her_k=int(args.context_her_k),
            teacher_ratio=float(args.context_teacher_ratio),
            checkpoint_interval=min(max(1, int(args.context_checkpoint_interval)), context_steps),
        )
        for algorithm in sorted(requested_context_algorithms):
            for context_seed in context_seeds:
                train_env = make_comm_inspection_env(args)
                run_dir = output_dir / algorithm / f"seed_{context_seed}"
                train_context_agent(
                    algorithm,
                    train_env,
                    context_cfg,
                    device,
                    run_dir,
                    seed=context_seed,
                )
                context_paths.append(run_dir / "checkpoint_final.pth")
    if requested_context_algorithms and not context_paths:
        raise ValueError(
            "context GCRL methods require --train-context-agents or --context-checkpoints"
        )
    loaded_algorithms = set()
    context_checkpoint_metadata = []
    for context_index, checkpoint in enumerate(context_paths):
        env = make_comm_inspection_env(args)
        context_agent, metadata = load_context_checkpoint(checkpoint, env, device)
        algorithm = str(metadata["algorithm"])
        if algorithm not in requested_context_algorithms:
            continue
        loaded_algorithms.add(algorithm)
        context_checkpoint_metadata.append({"checkpoint": str(checkpoint), **metadata})
        model_run = f"{algorithm}_seed_{metadata['seed']}_{context_index}"
        static_diagnostics = {
            "training_env_steps": int(metadata["env_steps"]),
            "teacher_env_steps": int(metadata["teacher_steps"]),
            "training_updates": int(metadata["updates"]),
            **{
                key: float(value)
                for key, value in metadata["replay_diagnostics"].items()
                if key in {
                    "relabel_count",
                    "cross_context_relabel_count",
                    "eligible_future_ratio",
                    "positive_relabel_ratio",
                }
            },
            "actor_parameters": int(metadata["model_metadata"].get("actor_parameters", 0)),
            "critic_parameters": int(metadata["model_metadata"].get("critic_parameters", 0)),
            **{
                key: float(value)
                for key, value in metadata["training_diagnostics"].items()
                if key in {
                    "contrastive_positive_accuracy",
                    "contrastive_negative_accuracy",
                    "mrn_metric_mean",
                    "mrn_residual_mean",
                }
            },
        }
        if algorithm in methods:
            records.extend(_evaluate_controller(
                algorithm,
                PolicyController(
                    context_agent,
                    name=algorithm,
                    static_diagnostics=static_diagnostics,
                ),
                env,
                episode_specs,
                model_run=model_run,
                output_dir=output_dir,
                viz_cfg=viz_cfg,
                counters=counters,
                on_record=incremental_writer.write,
                completed_keys=completed_keys,
            ))
        mppi_method = f"{algorithm}_mppi"
        if mppi_method in methods:
            calibrated_agent = CalibratedValueAgent(
                context_agent, metadata["value_calibration"]
            )
            records.extend(_evaluate_controller(
                mppi_method,
                MPPIController(
                    mppi_cfg,
                    terminal_mode="qrl",
                    qrl_agent=calibrated_agent,
                    static_diagnostics=static_diagnostics,
                ),
                env,
                episode_specs,
                model_run=model_run,
                output_dir=output_dir,
                viz_cfg=viz_cfg,
                counters=counters,
                on_record=incremental_writer.write,
                completed_keys=completed_keys,
            ))
    missing_context_algorithms = requested_context_algorithms - loaded_algorithms
    if missing_context_algorithms:
        raise ValueError(
            f"no matching checkpoints for context algorithms: {sorted(missing_context_algorithms)}"
        )

    summary = _summarize(records, seed=args.seed, n_boot=args.bootstrap_samples)
    payload = {
        "stage": args.stage,
        "methods": methods,
        "episode_specs": [
            {
                key: value
                for key, value in spec.items()
                if key in {"task_id", "stratum", "difficulty", "device_id", "seed", "start"}
            }
            for spec in episode_specs
        ],
        "qrl_checkpoints": args.qrl_checkpoints,
        "sac_checkpoints": [str(path) for path in sac_paths],
        "context_checkpoints": [str(path) for path in context_paths],
        "context_checkpoint_metadata": context_checkpoint_metadata,
        "summary": summary,
        "episode_results": records,
        "scenario_config": (
            str(Path(args.scenario_config).resolve()) if args.scenario_config else None
        ),
        "task_bank": str(Path(args.task_bank).resolve()) if args.task_bank else None,
        "task_split": str(args.task_split),
        "config": vars(args),
    }
    with open(output_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_csv(output_dir / "baseline_results.csv", records)
    incremental_writer.mark_complete()
    incremental_writer.close()
    print(f"Saved {output_dir / 'baseline_results.json'}")
    print(f"Saved {output_dir / 'baseline_results.csv'}")


if __name__ == "__main__":
    main()
