"""Generate, run, evaluate, and report the metric QRL scalability study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.scalability_scenarios import (
    DEFAULT_AREA_SIDE_METRES,
    DEVICE_COUNTS,
    build_metric_scenario,
    build_scalability_scenarios,
    scenario_to_env_kwargs,
    write_scalability_scenarios,
)


DEFAULT_OUTPUT_ROOT = Path("results/qrl_scalability_metric")
DEFAULT_SEEDS = (0, 1, 2)
USABILITY_THRESHOLD = 0.75
TASK_BANK_SEED = 20250714


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_task_bank(
    path: Path,
    *,
    validation_per_device: int = 10,
    test_per_device: int = 25,
    seed: int = TASK_BANK_SEED,
) -> dict[str, Any]:
    """Generate paired normalized starts for all 24 devices."""

    base = build_metric_scenario(100, 24)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(base))
    width = float(base["bounds"][2] - base["bounds"][0])
    height = float(base["bounds"][3] - base["bounds"][1])
    records: list[dict[str, Any]] = []
    total = int(validation_per_device) + int(test_per_device)
    for device_index, device_id in enumerate(env.device_ids):
        for sample_index in range(total):
            episode_seed = int(seed + device_index * 1_000_003 + sample_index * 101)
            _obs, _info = env.reset(seed=episode_seed, options={"device_id": device_id})
            start = [float(v) for v in env.state]
            split = "validation" if sample_index < validation_per_device else "test"
            split_index = sample_index if split == "validation" else sample_index - validation_per_device
            records.append(
                {
                    "task_id": f"{split}:{device_id}:{split_index:03d}",
                    "split": split,
                    "device_id": str(device_id),
                    "device_index": int(device_index),
                    "sample_index": int(split_index),
                    "seed": episode_seed,
                    "start_normalized": [start[0] / width, start[1] / height, start[2]],
                }
            )
    payload = {
        "schema_version": 1,
        "generation_seed": int(seed),
        "validation_per_device": int(validation_per_device),
        "test_per_device": int(test_per_device),
        "device_ids": list(env.device_ids),
        "records": records,
    }
    payload["content_digest"] = _digest(payload)
    _write_json(path, payload)
    return payload


def build_manifest(
    output_root: Path,
    *,
    seeds: Sequence[int],
    target_env_transitions: int,
    total_steps: int,
    checkpoints: Sequence[int],
    validation_per_device: int,
    test_per_device: int,
    area_sides: Sequence[int] = DEFAULT_AREA_SIDE_METRES,
    device_counts: Sequence[int] = DEVICE_COUNTS,
    save_interval: int = 2_000,
) -> dict[str, Any]:
    if int(save_interval) <= 0:
        raise ValueError("save_interval must be positive")
    scenario_dir = output_root / "scenario_configs"
    scenario_paths = write_scalability_scenarios(
        scenario_dir,
        area_sides=area_sides,
        device_counts=device_counts,
    )
    scenarios = {
        item["scenario_id"]: item
        for item in build_scalability_scenarios(
            area_sides=area_sides,
            device_counts=device_counts,
        )
    }
    paths_by_id = {path.stem: path for path in scenario_paths}
    task_bank_path = output_root / "task_bank.json"
    task_bank = generate_task_bank(
        task_bank_path,
        validation_per_device=validation_per_device,
        test_per_device=test_per_device,
    )

    jobs: list[dict[str, Any]] = []
    for scenario_id in sorted(scenarios):
        scenario = scenarios[scenario_id]
        for train_seed in seeds:
            job_id = f"{scenario_id}-qrl-s{int(train_seed)}"
            jobs.append(
                {
                    "job_id": job_id,
                    "scenario_id": scenario_id,
                    "train_seed": int(train_seed),
                    "physical_side_m": float(scenario["physical_side_m"]),
                    "physical_area_m2": float(scenario["physical_area_m2"]),
                    "device_count": int(scenario["device_count"]),
                    "experiment_axes": list(scenario["experiment_axes"]),
                    "scenario_config": str(paths_by_id[scenario_id]),
                    "task_bank": str(task_bank_path),
                    "target_env_transitions": int(target_env_transitions),
                    "total_steps": int(total_steps),
                    "save_interval": int(save_interval),
                    "checkpoints": [int(v) for v in checkpoints],
                    "validation_per_device": int(validation_per_device),
                    "test_per_device": int(test_per_device),
                    "output_dir": str(output_root / "jobs" / job_id),
                }
            )
    if len(jobs) != len(scenarios) * len(seeds):
        raise AssertionError("manifest must contain every selected scenario for every seed")
    if len({row["job_id"] for row in jobs}) != len(jobs):
        raise AssertionError("manifest contains duplicate jobs")
    manifest = {
        "schema_version": 1,
        "usability_threshold": USABILITY_THRESHOLD,
        "task_bank_digest": task_bank["content_digest"],
        "selection": {
            "area_sides_m": [int(value) for value in area_sides],
            "device_counts": [int(value) for value in device_counts],
        },
        "scenarios": list(scenarios.values()),
        "jobs": jobs,
    }
    _write_json(output_root / "job_manifest.json", manifest)
    csv_rows = [
        {
            **{key: value for key, value in row.items() if key not in {"experiment_axes", "checkpoints"}},
            "experiment_axes": json.dumps(row["experiment_axes"]),
            "checkpoints": json.dumps(row["checkpoints"]),
        }
        for row in jobs
    ]
    _write_csv(output_root / "job_manifest.csv", csv_rows)
    return manifest


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _checkpoint_path(training_dir: Path, step: int, total_steps: int) -> Path:
    if int(step) == int(total_steps):
        periodic = training_dir / f"checkpoint_{int(step):05d}.pth"
        return periodic if periodic.exists() else training_dir / "checkpoint_final.pth"
    return training_dir / f"checkpoint_{int(step):05d}.pth"


def run_job(
    job: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
    num_critics: int,
    teacher_ratio: float,
) -> None:
    job_dir = Path(str(job["output_dir"]))
    if (job_dir / "COMPLETE").exists():
        return
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(job_dir / "job_input.json", dict(job))
    training_dir = job_dir / "training"
    training_complete = training_dir / "COMPLETE"
    log_path = job_dir / "runner.log"

    if not training_complete.exists():
        # Model persistence is independent from the more expensive validation
        # schedule.  Old manifests fall back to the historical behavior.
        save_interval = int(
            job.get(
                "save_interval",
                min(int(v) for v in job["checkpoints"] if int(v) > 0),
            )
        )
        command = [
            sys.executable,
            "minimal_qrl/train.py",
            "--scenario-config",
            str(job["scenario_config"]),
            "--output-dir",
            str(training_dir),
            "--seed",
            str(job["train_seed"]),
            "--device",
            str(device),
            "--target-env-transitions",
            str(job["target_env_transitions"]),
            "--total-steps",
            str(job["total_steps"]),
            "--batch-size",
            str(batch_size),
            "--num-critics",
            str(num_critics),
            "--task-aware-teacher-ratio",
            str(teacher_ratio),
            "--save-interval",
            str(save_interval),
            "--log-interval",
            str(max(1, int(job["total_steps"]) // 20)),
            "--eval-interval",
            "0",
            "--visualization-interval",
            "0",
            "--planning-eval-interval",
            "0",
        ]
        checkpoints = sorted(training_dir.glob("checkpoint_[0-9]*.pth"))
        if checkpoints and not training_complete.exists():
            command.extend(["--init-checkpoint", str(checkpoints[-1])])
        _run_logged(command, log_path)

    for step in job["checkpoints"]:
        eval_dir = job_dir / "evaluation" / f"validation_{int(step):06d}"
        result_path = eval_dir / "comm_inspection_execution_eval.json"
        if result_path.exists():
            continue
        checkpoint = _checkpoint_path(training_dir, int(step), int(job["total_steps"]))
        command = [
            sys.executable,
            "minimal_qrl/eval/comm_inspection_execution_eval.py",
            "--scenario-config",
            str(job["scenario_config"]),
            "--task-bank",
            str(job["task_bank"]),
            "--task-split",
            "validation",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(eval_dir),
            "--env-name",
            f"{job['job_id']}_validation_{int(step)}",
            "--seed",
            str(job["train_seed"]),
            "--device",
            str(device),
            "--num-critics",
            str(num_critics),
            "--starts-per-device",
            str(job["validation_per_device"]),
            "--execution-modes",
            "greedy",
        ]
        _run_logged(command, log_path)

    final_eval_dir = job_dir / "evaluation" / "final_test"
    final_result = final_eval_dir / "comm_inspection_execution_eval.json"
    if not final_result.exists():
        command = [
            sys.executable,
            "minimal_qrl/eval/comm_inspection_execution_eval.py",
            "--scenario-config",
            str(job["scenario_config"]),
            "--task-bank",
            str(job["task_bank"]),
            "--task-split",
            "test",
            "--checkpoint",
            str(training_dir / "checkpoint_final.pth"),
            "--output-dir",
            str(final_eval_dir),
            "--env-name",
            f"{job['job_id']}_test",
            "--seed",
            str(job["train_seed"]),
            "--device",
            str(device),
            "--num-critics",
            str(num_critics),
            "--starts-per-device",
            str(job["test_per_device"]),
            "--execution-modes",
            "greedy,lookahead",
            "--lookahead-horizon",
            "5",
            "--lookahead-num-sequences",
            "64",
            "--lookahead-biased-sequences",
            "24",
        ]
        _run_logged(command, log_path)
    (job_dir / "COMPLETE").write_text("", encoding="utf-8")


def run_manifest(
    manifest: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
    num_critics: int,
    teacher_ratio: float,
) -> None:
    for index, job in enumerate(manifest["jobs"], start=1):
        print(f"[scalability] job {index}/{len(manifest['jobs'])}: {job['job_id']}", flush=True)
        run_job(
            job,
            device=device,
            batch_size=batch_size,
            num_critics=num_critics,
            teacher_ratio=teacher_ratio,
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_std(values: Iterable[float]) -> tuple[float | None, float | None, float | None, float | None]:
    data = [float(v) for v in values]
    if not data:
        return None, None, None, None
    return (
        statistics.mean(data),
        statistics.stdev(data) if len(data) > 1 else 0.0,
        min(data),
        max(data),
    )


def _job_result_row(job: Mapping[str, Any]) -> dict[str, Any]:
    job_dir = Path(str(job["output_dir"]))
    timing_path = job_dir / "training" / "timing.json"
    eval_path = job_dir / "evaluation" / "final_test" / "comm_inspection_execution_eval.json"
    row: dict[str, Any] = {
        "job_id": job["job_id"],
        "scenario_id": job["scenario_id"],
        "train_seed": int(job["train_seed"]),
        "physical_side_m": float(job["physical_side_m"]),
        "physical_area_m2": float(job["physical_area_m2"]),
        "device_count": int(job["device_count"]),
        "complete": bool((job_dir / "COMPLETE").exists()),
    }
    if timing_path.exists():
        timing = _read_json(timing_path)
        row.update(
            {
                "data_time_sec": timing.get("data_time_sec"),
                "optimization_core_time_sec": timing.get("optimization_core_time_sec"),
                "end_to_end_time_sec": timing.get("end_to_end_time_sec"),
                "time_per_1000_updates_sec": (
                    float(timing["optimization_core_time_sec"])
                    / max(1, int(timing["global_gradient_updates"]))
                    * 1000.0
                ),
                "resolved_device": timing.get("hardware", {}).get("resolved_device"),
                "hardware_platform": timing.get("hardware", {}).get("platform"),
                "torch_version": timing.get("hardware", {}).get("torch_version"),
            }
        )
    if eval_path.exists():
        payload = _read_json(eval_path)
        greedy = payload["results"].get("greedy", {})
        mppi = payload["results"].get("lookahead", {})
        for prefix, metrics in (("qrl_greedy", greedy), ("qrl_mppi", mppi)):
            for metric in (
                "success_rate",
                "macro_device_success_rate",
                "avg_steps_success",
                "avg_total_cost",
                "collision_rate",
                "out_of_bounds_rate",
                "communication_feasible_ratio",
            ):
                row[f"{prefix}_{metric}"] = metrics.get(metric)
    return row


def _aggregate_scenarios(manifest: Mapping[str, Any], job_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scenario_by_id = {item["scenario_id"]: item for item in manifest["scenarios"]}
    aggregates: list[dict[str, Any]] = []
    for scenario_id, scenario in sorted(scenario_by_id.items()):
        rows = [row for row in job_rows if row["scenario_id"] == scenario_id]
        mppi_values = [
            float(row["qrl_mppi_macro_device_success_rate"])
            for row in rows
            if row.get("qrl_mppi_macro_device_success_rate") is not None
        ]
        greedy_values = [
            float(row["qrl_greedy_macro_device_success_rate"])
            for row in rows
            if row.get("qrl_greedy_macro_device_success_rate") is not None
        ]
        times = [float(row["end_to_end_time_sec"]) for row in rows if row.get("end_to_end_time_sec") is not None]
        data_times = [float(row["data_time_sec"]) for row in rows if row.get("data_time_sec") is not None]
        optimization_times = [
            float(row["optimization_core_time_sec"])
            for row in rows
            if row.get("optimization_core_time_sec") is not None
        ]
        per_1000_times = [
            float(row["time_per_1000_updates_sec"])
            for row in rows
            if row.get("time_per_1000_updates_sec") is not None
        ]
        m_mean, m_std, m_min, m_max = _mean_std(mppi_values)
        g_mean, g_std, g_min, g_max = _mean_std(greedy_values)
        t_mean, t_std, t_min, t_max = _mean_std(times)
        descriptive_means = {}
        for metric in (
            "avg_steps_success",
            "avg_total_cost",
            "collision_rate",
            "out_of_bounds_rate",
            "communication_feasible_ratio",
        ):
            values = [
                float(row[f"qrl_mppi_{metric}"])
                for row in rows
                if row.get(f"qrl_mppi_{metric}") is not None
            ]
            descriptive_means[f"qrl_mppi_{metric}_mean"] = _mean_std(values)[0]
        aggregates.append(
            {
                "scenario_id": scenario_id,
                "physical_side_m": float(scenario["physical_side_m"]),
                "physical_area_m2": float(scenario["physical_area_m2"]),
                "device_count": int(scenario["device_count"]),
                "completed_seeds": len(mppi_values),
                "qrl_mppi_success_mean": m_mean,
                "qrl_mppi_success_std": m_std,
                "qrl_mppi_success_min": m_min,
                "qrl_mppi_success_max": m_max,
                "qrl_greedy_success_mean": g_mean,
                "qrl_greedy_success_std": g_std,
                "qrl_greedy_success_min": g_min,
                "qrl_greedy_success_max": g_max,
                "training_time_mean_sec": t_mean,
                "training_time_std_sec": t_std,
                "training_time_min_sec": t_min,
                "training_time_max_sec": t_max,
                "data_time_mean_sec": _mean_std(data_times)[0],
                "optimization_core_time_mean_sec": _mean_std(optimization_times)[0],
                "time_per_1000_updates_mean_sec": _mean_std(per_1000_times)[0],
                **descriptive_means,
                "usable": bool(len(mppi_values) == 3 and m_mean is not None and m_mean >= USABILITY_THRESHOLD),
            }
        )
    return aggregates


def _plot_axis(rows: Sequence[Mapping[str, Any]], *, x_key: str, x_label: str, path: Path) -> None:
    valid = [row for row in rows if row.get("qrl_mppi_success_mean") is not None]
    if not valid:
        return
    valid = sorted(valid, key=lambda row: float(row[x_key]))
    x = [float(row[x_key]) for row in valid]
    success = [float(row["qrl_mppi_success_mean"]) for row in valid]
    success_std = [float(row["qrl_mppi_success_std"] or 0.0) for row in valid]
    hours = [float(row["training_time_mean_sec"]) / 3600.0 for row in valid]
    hours_std = [float(row["training_time_std_sec"] or 0.0) / 3600.0 for row in valid]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(x, success, yerr=success_std, marker="o", capsize=3)
    axes[0].axhline(USABILITY_THRESHOLD, color="tab:red", linestyle="--", label="75% usable")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("QRL-MPPI macro success rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend()
    axes[1].errorbar(x, hours, yerr=hours_std, marker="o", capsize=3)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("End-to-end training time (hours)")
    if x_key == "physical_area_m2":
        for axis in axes:
            axis.set_xscale("log")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_learning_curves(manifest: Mapping[str, Any], path: Path) -> None:
    curves: dict[str, dict[int, list[float]]] = {}
    for job in manifest["jobs"]:
        for step in job["checkpoints"]:
            result_path = (
                Path(str(job["output_dir"]))
                / "evaluation"
                / f"validation_{int(step):06d}"
                / "comm_inspection_execution_eval.json"
            )
            if not result_path.exists():
                continue
            metrics = _read_json(result_path)["results"]["greedy"]
            curves.setdefault(str(job["scenario_id"]), {}).setdefault(int(step), []).append(
                float(metrics["macro_device_success_rate"])
            )
    if not curves:
        return
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for scenario_id, by_step in sorted(curves.items()):
        steps = sorted(by_step)
        means = [statistics.mean(by_step[step]) for step in steps]
        axis.plot(steps, means, marker="o", label=scenario_id)
    axis.axhline(USABILITY_THRESHOLD, color="tab:red", linestyle="--")
    axis.set_xlabel("Gradient updates")
    axis.set_ylabel("Validation QRL-greedy macro success rate")
    axis.set_ylim(0.0, 1.02)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_report(output_root: Path) -> dict[str, Any]:
    manifest = _read_json(output_root / "job_manifest.json")
    job_rows = [_job_result_row(job) for job in manifest["jobs"]]
    hardware_signatures = {
        (row.get("resolved_device"), row.get("hardware_platform"), row.get("torch_version"))
        for row in job_rows
        if row.get("resolved_device") is not None
    }
    if len(hardware_signatures) > 1:
        raise ValueError(
            "completed jobs use mixed hardware/software environments; refusing to aggregate timings: "
            f"{sorted(hardware_signatures)}"
        )
    aggregates = _aggregate_scenarios(manifest, job_rows)
    area_rows = sorted(
        [row for row in aggregates if int(row["device_count"]) == 24],
        key=lambda row: float(row["physical_area_m2"]),
    )
    device_rows = sorted(
        [row for row in aggregates if float(row["physical_side_m"]) == 100.0],
        key=lambda row: int(row["device_count"]),
    )
    report_dir = output_root / "report"
    _write_csv(report_dir / "job_results.csv", job_rows)
    _write_csv(report_dir / "area_summary.csv", area_rows)
    _write_csv(report_dir / "device_summary.csv", device_rows)
    summary = {
        "schema_version": 1,
        "usability_threshold": USABILITY_THRESHOLD,
        "hardware_signature": list(next(iter(hardware_signatures))) if hardware_signatures else None,
        "jobs": job_rows,
        "scenarios": aggregates,
    }
    _write_json(report_dir / "scalability_summary.json", summary)
    _plot_axis(
        area_rows,
        x_key="physical_area_m2",
        x_label="Physical park area (m²)",
        path=report_dir / "area_scalability",
    )
    _plot_axis(
        device_rows,
        x_key="device_count",
        x_label="Inspection device count",
        path=report_dir / "device_scalability",
    )
    _plot_learning_curves(manifest, report_dir / "checkpoint_learning_curves")

    lines = [
        "# QRL Metric Scalability Experiment Report",
        "",
        f"Usability rule: final QRL-MPPI three-seed mean macro-device success rate >= {USABILITY_THRESHOLD:.0%}.",
        "Collision and out-of-bounds rates are descriptive and do not gate usability.",
        "",
        "| Scenario | Area (m²) | Devices | Seeds | QRL-MPPI success | Training hours | Usable |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregates:
        success = row["qrl_mppi_success_mean"]
        hours = (
            float(row["training_time_mean_sec"]) / 3600.0
            if row["training_time_mean_sec"] is not None
            else None
        )
        lines.append(
            f"| {row['scenario_id']} | {row['physical_area_m2']:.0f} | {row['device_count']} | "
            f"{row['completed_seeds']} | {success:.3f} | {hours:.2f} | "
            f"{'yes' if row['usable'] else 'no'} |"
            if success is not None and hours is not None
            else f"| {row['scenario_id']} | {row['physical_area_m2']:.0f} | {row['device_count']} | "
            f"{row['completed_seeds']} | incomplete | incomplete | no |"
        )
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        sub.add_argument("--seeds", type=_parse_ints, default=list(DEFAULT_SEEDS))
        sub.add_argument(
            "--area-sides",
            type=_parse_ints,
            default=list(DEFAULT_AREA_SIDE_METRES),
            help="comma-separated physical side lengths in metres",
        )
        sub.add_argument(
            "--device-counts",
            type=_parse_ints,
            default=list(DEVICE_COUNTS),
            help="comma-separated device counts for the 100 m map",
        )
        sub.add_argument("--target-env-transitions", type=int, default=60_000)
        sub.add_argument("--total-steps", type=int, default=60_000)
        sub.add_argument(
            "--save-interval",
            type=int,
            default=2_000,
            help="save a training checkpoint every N gradient updates",
        )
        sub.add_argument("--checkpoints", type=_parse_ints, default=[20_000, 40_000, 60_000])
        sub.add_argument("--validation-per-device", type=int, default=10)
        sub.add_argument("--test-per-device", type=int, default=25)
        if name == "run":
            sub.add_argument("--device", type=str, required=True)
            sub.add_argument("--batch-size", type=int, default=256)
            sub.add_argument("--num-critics", type=int, default=2)
            sub.add_argument("--teacher-ratio", type=float, default=1.0)
    report = subparsers.add_parser("report")
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "report":
        write_report(args.output_root)
        return
    if max(args.checkpoints) > int(args.total_steps):
        raise ValueError("checkpoints cannot exceed total_steps")
    if int(args.save_interval) <= 0:
        raise ValueError("save-interval must be positive")
    manifest = build_manifest(
        args.output_root,
        seeds=args.seeds,
        target_env_transitions=args.target_env_transitions,
        total_steps=args.total_steps,
        checkpoints=args.checkpoints,
        validation_per_device=args.validation_per_device,
        test_per_device=args.test_per_device,
        area_sides=args.area_sides,
        device_counts=args.device_counts,
        save_interval=args.save_interval,
    )
    if args.command == "run":
        run_manifest(
            manifest,
            device=args.device,
            batch_size=args.batch_size,
            num_critics=args.num_critics,
            teacher_ratio=args.teacher_ratio,
        )
        write_report(args.output_root)


if __name__ == "__main__":
    main()
