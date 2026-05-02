#!/usr/bin/env python3
"""Summarize communication-inspection evaluation JSON files into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any


METRICS = [
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


def _load_rows(label: str, path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    results: Dict[str, Dict[str, Any]] = payload.get("results", {})
    rows = []
    for method, metrics in results.items():
        row = {"group": label, "method": method}
        row.update({name: metrics.get(name, "") for name in METRICS})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize comm-inspection eval JSON files.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input spec formatted as label=/path/to/comm_inspection_execution_eval.json",
    )
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for spec in args.input:
        if "=" not in spec:
            raise ValueError(f"--input must be label=path, got: {spec}")
        label, raw_path = spec.split("=", 1)
        rows.extend(_load_rows(label, Path(raw_path)))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "method", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summarize_comm] saved: {out_path}")


if __name__ == "__main__":
    main()
