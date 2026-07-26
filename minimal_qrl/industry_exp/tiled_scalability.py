"""Run density-preserving tiled QRL scalability experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from minimal_qrl.industry_exp.scalability import (
    USABILITY_THRESHOLD,
    _parse_ints,
    _write_csv,
    _write_json,
    generate_scenario_task_bank,
    run_manifest,
    write_report,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    BASE_DEVICE_COUNT,
    build_tiled_metric_scenario,
)


DEFAULT_OUTPUT_ROOT = Path("results/qrl_tiled_scalability_2x2_seed0")


def _write_scenarios(
    directory: Path,
    scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        path = directory / f"{scenario_id}.json"
        _write_json(path, dict(scenario))
        paths[scenario_id] = path
    return paths


def build_tiled_manifest(
    output_root: Path,
    *,
    tile_grids: Sequence[int],
    seeds: Sequence[int],
    base_target_env_transitions: int,
    base_total_steps: int,
    base_checkpoints: Sequence[int],
    save_interval: int,
    validation_per_device: int,
    test_per_device: int,
) -> dict[str, Any]:
    """Build jobs whose data/update density matches the 24-device base tile."""

    grids = tuple(dict.fromkeys(int(value) for value in tile_grids))
    if not grids or any(value <= 0 for value in grids):
        raise ValueError("tile_grids must contain positive integers")
    if int(base_target_env_transitions) <= 0 or int(base_total_steps) <= 0:
        raise ValueError("base transition and update budgets must be positive")
    if int(save_interval) <= 0:
        raise ValueError("save_interval must be positive")
    if not base_checkpoints or max(int(value) for value in base_checkpoints) > int(
        base_total_steps
    ):
        raise ValueError("base checkpoints must not exceed base_total_steps")

    scenarios = [build_tiled_metric_scenario(grid) for grid in grids]
    scenario_paths = _write_scenarios(output_root / "scenario_configs", scenarios)
    task_bank_path = output_root / "task_bank.json"
    task_bank = generate_scenario_task_bank(
        task_bank_path,
        scenarios,
        validation_per_device=validation_per_device,
        test_per_device=test_per_device,
    )

    jobs: list[dict[str, Any]] = []
    for scenario in sorted(scenarios, key=lambda item: str(item["scenario_id"])):
        scenario_id = str(scenario["scenario_id"])
        density_multiplier = int(scenario["device_count"]) // BASE_DEVICE_COUNT
        target_env_transitions = int(base_target_env_transitions) * density_multiplier
        total_steps = int(base_total_steps) * density_multiplier
        checkpoints = [
            int(value) * density_multiplier for value in base_checkpoints
        ]
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
                    "scenario_config": str(scenario_paths[scenario_id]),
                    "task_bank": str(task_bank_path),
                    "density_multiplier": int(density_multiplier),
                    "target_env_transitions": int(target_env_transitions),
                    "total_steps": int(total_steps),
                    "save_interval": int(save_interval),
                    "checkpoints": checkpoints,
                    "validation_per_device": int(validation_per_device),
                    "test_per_device": int(test_per_device),
                    "output_dir": str(output_root / "jobs" / job_id),
                }
            )

    if len({row["job_id"] for row in jobs}) != len(jobs):
        raise AssertionError("manifest contains duplicate jobs")
    manifest = {
        "schema_version": 1,
        "experiment_kind": "density_preserving_tiled_scalability",
        "usability_threshold": USABILITY_THRESHOLD,
        "task_bank_digest": task_bank["content_digest"],
        "fixed_density": {
            "base_tile_side_m": 100,
            "base_device_count": BASE_DEVICE_COUNT,
            "base_target_env_transitions": int(base_target_env_transitions),
            "base_total_steps": int(base_total_steps),
            "base_checkpoints": [int(value) for value in base_checkpoints],
        },
        "selection": {"tile_grids": list(grids), "seeds": [int(v) for v in seeds]},
        "scenarios": scenarios,
        "jobs": jobs,
    }
    _write_json(output_root / "job_manifest.json", manifest)
    _write_csv(
        output_root / "job_manifest.csv",
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"experiment_axes", "checkpoints"}
                },
                "experiment_axes": json.dumps(row["experiment_axes"]),
                "checkpoints": json.dumps(row["checkpoints"]),
            }
            for row in jobs
        ],
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        sub.add_argument("--tile-grids", type=_parse_ints, default=[2])
        sub.add_argument("--seeds", type=_parse_ints, default=[0])
        sub.add_argument("--base-target-env-transitions", type=int, default=50_000)
        sub.add_argument("--base-total-steps", type=int, default=50_000)
        sub.add_argument(
            "--base-checkpoints",
            type=_parse_ints,
            default=[10_000, 20_000, 30_000, 40_000, 50_000],
            help="checkpoints for one 24-device tile; scaled by tile count",
        )
        sub.add_argument("--save-interval", type=int, default=2_000)
        sub.add_argument("--validation-per-device", type=int, default=3)
        sub.add_argument("--test-per-device", type=int, default=5)
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
    manifest = build_tiled_manifest(
        args.output_root,
        tile_grids=args.tile_grids,
        seeds=args.seeds,
        base_target_env_transitions=args.base_target_env_transitions,
        base_total_steps=args.base_total_steps,
        base_checkpoints=args.base_checkpoints,
        save_interval=args.save_interval,
        validation_per_device=args.validation_per_device,
        test_per_device=args.test_per_device,
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
