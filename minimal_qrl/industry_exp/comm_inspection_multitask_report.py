#!/usr/bin/env python3
"""Create thesis-ready tables and plots from the unified multi-start/multi-target benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_METRICS = (
    "success",
    "num_steps",
    "total_cost",
    "collision",
    "out_of_bounds",
    "communication_feasible_ratio",
    "first_task_feasible_step",
    "first_decision_time_sec",
    "planning_time_p95_sec",
)


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_summary_csv(payload: dict, path: Path) -> None:
    rows = []
    for method, summary in payload["summary"].items():
        if method == "paired_comparisons":
            continue
        row = {"method": method, "num_records": summary.get("num_records", 0)}
        for metric in SUMMARY_METRICS:
            stats = summary.get(metric, {})
            row[f"{metric}_mean"] = stats.get("mean", "")
            ci = stats.get("bootstrap_95_ci", ["", ""])
            row[f"{metric}_ci_low"] = ci[0] if len(ci) > 0 else ""
            row[f"{metric}_ci_high"] = ci[1] if len(ci) > 1 else ""
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_overall(payload: dict, path: Path) -> None:
    summaries = {
        key: value for key, value in payload["summary"].items() if key != "paired_comparisons"
    }
    methods = list(summaries)
    success = [100.0 * float(summaries[m]["success"]["mean"]) for m in methods]
    latency_ms = [1000.0 * float(summaries[m]["first_decision_time_sec"]["mean"]) for m in methods]
    collision = [100.0 * float(summaries[m]["collision"]["mean"]) for m in methods]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].bar(x, success, color="#4c78a8")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(x, latency_ms, color="#f58518")
    axes[1].set_ylabel("First-decision latency (ms, log scale)")
    axes[1].set_yscale("log")
    axes[2].bar(x, collision, color="#e45756")
    axes[2].set_ylabel("Collision rate (%)")
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xticks(x, methods, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_per_device(payload: dict, path: Path) -> None:
    records = payload.get("episode_results", [])
    methods = list(dict.fromkeys(str(row["method"]) for row in records))
    devices = sorted({str(row["device_id"]) for row in records})
    matrix = np.full((len(methods), len(devices)), np.nan, dtype=np.float32)
    for row_index, method in enumerate(methods):
        for column_index, device_id in enumerate(devices):
            values = [
                float(row["success"])
                for row in records
                if str(row["method"]) == method and str(row["device_id"]) == device_id
            ]
            if values:
                matrix[row_index, column_index] = float(np.mean(values))

    fig_width = max(10.0, 0.43 * len(devices))
    fig, ax = plt.subplots(figsize=(fig_width, 1.2 + 0.65 * len(methods)))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(devices)), devices, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(methods)), methods)
    ax.set_title("Per-device success rate under paired random starts")
    fig.colorbar(image, ax=ax, label="Success rate", fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load(input_path)
    _write_summary_csv(payload, output_dir / "multitask_summary.csv")
    _plot_overall(payload, output_dir / "overall_performance_and_latency.png")
    _plot_per_device(payload, output_dir / "per_device_success_heatmap.png")
    print(f"Saved multi-task report to {output_dir}")


if __name__ == "__main__":
    main()

