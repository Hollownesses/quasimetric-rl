#!/usr/bin/env python3
"""Visualize the three fixed strata in the long-horizon diagnostic map."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

_CACHE_ROOT = Path(tempfile.gettempdir()) / "qrl_diagnostic_visualization_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from minimal_qrl.envs import CircleObstacle, CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.diagnostic_scenario import (
    CHALLENGE_STRATA,
    build_diagnostic_scenario,
    build_diagnostic_task_bank,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)


STRATUM_TITLES = {
    "u_trap": "u_trap: move away before approaching",
    "comm_shadow_corridor": "comm_shadow_corridor: long shadow vs connected route",
    "easy_open": "easy_open: unobstructed direct tasks",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _map_start(row: Mapping[str, Any], bounds: Sequence[float]) -> np.ndarray:
    normalized = row["start_normalized"]
    width = float(bounds[2]) - float(bounds[0])
    height = float(bounds[3]) - float(bounds[1])
    return np.asarray(
        [
            float(bounds[0]) + float(normalized[0]) * width,
            float(bounds[1]) + float(normalized[1]) * height,
            float(normalized[2]),
        ],
        dtype=np.float32,
    )


def _records_for_stratum(
    task_bank: Mapping[str, Any],
    *,
    split: str,
    stratum: str,
) -> list[dict[str, Any]]:
    records = [
        dict(row)
        for row in task_bank["records"]
        if str(row["split"]) == str(split) and str(row["stratum"]) == str(stratum)
    ]
    if not records:
        raise ValueError(f"no {split!r} task records for stratum={stratum!r}")
    return records


def _draw_obstacles(ax, env: CommInspectionDubinsUAV2D) -> None:
    for obstacle in env.obstacles:
        if isinstance(obstacle, CircleObstacle):
            patch = patches.Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                facecolor="#596273",
                edgecolor="#273142",
                linewidth=1.2,
                alpha=0.9,
                zorder=3,
            )
        else:
            patch = patches.Rectangle(
                (obstacle.x_min, obstacle.y_min),
                obstacle.x_max - obstacle.x_min,
                obstacle.y_max - obstacle.y_min,
                facecolor="#596273",
                edgecolor="#273142",
                linewidth=1.2,
                alpha=0.9,
                zorder=3,
            )
        ax.add_patch(patch)


def _draw_station(ax, env: CommInspectionDubinsUAV2D) -> None:
    x, y = env.ground_station
    ax.scatter(
        [x],
        [y],
        marker="s",
        s=90,
        c="#2457a6",
        edgecolors="white",
        linewidths=0.9,
        label="ground station",
        zorder=8,
    )
    ax.annotate("GS", (x, y), xytext=(6, -14), textcoords="offset points", fontsize=8)


def _draw_device_and_terminal_sector(
    ax,
    env: CommInspectionDubinsUAV2D,
    device_id: str,
    *,
    label: bool = True,
) -> None:
    env.set_task_by_device_id(device_id)
    target = env.inspection_target
    ax.scatter(
        [target[0]],
        [target[1]],
        marker="X",
        s=115,
        c="#f2b134",
        edgecolors="#6b4c00",
        linewidths=0.9,
        label="inspection target" if label else None,
        zorder=9,
    )
    sector = patches.Wedge(
        target,
        env.observation_max_distance,
        np.degrees(env.preferred_bearing - env.bearing_tolerance),
        np.degrees(env.preferred_bearing + env.bearing_tolerance),
        width=env.observation_max_distance - env.observation_min_distance,
        facecolor="#f2b134",
        edgecolor="#b27b00",
        linewidth=1.0,
        linestyle="--",
        alpha=0.25,
        label="terminal observation set" if label else None,
        zorder=4,
    )
    ax.add_patch(sector)
    ax.annotate(
        device_id,
        target,
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
        zorder=10,
    )


def _draw_start(
    ax,
    start: np.ndarray,
    *,
    selected: bool,
    label: str | None = None,
) -> None:
    color = "#0b8f87" if selected else "#61bdb6"
    size = 72 if selected else 25
    alpha = 1.0 if selected else 0.45
    ax.scatter(
        [start[0]],
        [start[1]],
        s=size,
        c=color,
        edgecolors="white" if selected else "none",
        linewidths=0.9,
        alpha=alpha,
        label=label,
        zorder=10 if selected else 5,
    )
    length = 0.55 if selected else 0.3
    ax.arrow(
        float(start[0]),
        float(start[1]),
        length * np.cos(float(start[2])),
        length * np.sin(float(start[2])),
        width=0.025 if selected else 0.012,
        head_width=0.18 if selected else 0.1,
        head_length=0.18 if selected else 0.1,
        color=color,
        alpha=alpha,
        length_includes_head=True,
        zorder=11 if selected else 6,
    )


def _communication_mask(
    env: CommInspectionDubinsUAV2D,
    extent: tuple[float, float, float, float],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = extent
    nx = max(40, int(resolution))
    ny = max(30, int(round(nx * (y_max - y_min) / max(x_max - x_min, 1e-6))))
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    mask = np.zeros((ny, nx), dtype=np.uint8)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            state = np.asarray([x, y, 0.0], dtype=np.float32)
            if env.is_valid_state(state):
                mask[iy, ix] = 2 if env.is_communication_feasible(state) else 1
    return xs, ys, mask


def _style_axis(
    ax,
    *,
    title: str,
    extent: tuple[float, float, float, float],
) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (environment units)")
    ax.set_ylabel("y (environment units)")
    ax.grid(True, color="#b8c0cc", linewidth=0.5, alpha=0.35, zorder=0)
    ax.set_facecolor("#f7f9fc")


def _deduplicated_legend(ax, *, location: str = "best") -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            loc=location,
            fontsize=7.5,
            framealpha=0.92,
        )


def _plot_u_trap(
    ax,
    env: CommInspectionDubinsUAV2D,
    records: Sequence[Mapping[str, Any]],
    bounds: Sequence[float],
    selected_index: int,
) -> None:
    extent = (2.0, 8.1, 0.6, 6.7)
    _style_axis(ax, title=STRATUM_TITLES["u_trap"], extent=extent)
    _draw_obstacles(ax, env)
    _draw_station(ax, env)
    _draw_device_and_terminal_sector(ax, env, "u_trap_target")
    starts = [_map_start(row, bounds) for row in records]
    selected = starts[selected_index % len(starts)]
    for index, start in enumerate(starts):
        _draw_start(
            ax,
            start,
            selected=index == selected_index % len(starts),
            label="selected start" if index == selected_index % len(starts) else None,
        )
    target = np.asarray(env.inspection_target, dtype=np.float32)
    ax.plot(
        [selected[0], target[0]],
        [selected[1], target[1]],
        color="#c83e4d",
        linestyle=":",
        linewidth=2.0,
        label="blocked direct path",
        zorder=6,
    )
    route = np.asarray(
        [
            selected[:2],
            [2.25, float(selected[1])],
            [2.25, 0.85],
            [6.35, 0.85],
            [6.45, 2.55],
            target,
        ],
        dtype=np.float32,
    )
    ax.plot(
        route[:, 0],
        route[:, 1],
        color="#d97706",
        linestyle="--",
        linewidth=2.2,
        label="schematic feasible detour",
        zorder=7,
    )
    ax.annotate(
        "initial motion\naway from target",
        xy=(2.55, float(selected[1])),
        xytext=(3.0, 3.3),
        arrowprops={"arrowstyle": "->", "color": "#d97706"},
        fontsize=8,
        color="#8a4b00",
    )
    _deduplicated_legend(ax, location="upper left")


def _plot_comm_shadow_corridor(
    ax,
    env: CommInspectionDubinsUAV2D,
    records: Sequence[Mapping[str, Any]],
    bounds: Sequence[float],
    selected_index: int,
    resolution: int,
) -> None:
    extent = (6.2, 17.3, 2.3, 9.1)
    _style_axis(ax, title=STRATUM_TITLES["comm_shadow_corridor"], extent=extent)
    env.set_task_by_device_id("corridor_target")
    xs, ys, mask = _communication_mask(env, extent, resolution)
    ax.imshow(
        mask,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap=ListedColormap(["#f7f9fc", "#f4a6a6", "#b8ddf2"]),
        vmin=0,
        vmax=2,
        alpha=0.30,
        interpolation="nearest",
        zorder=1,
    )
    ax.add_patch(
        patches.Rectangle(
            (extent[0], extent[2]),
            0.0,
            0.0,
            facecolor="#f4a6a6",
            alpha=0.45,
            label="communication shadow",
        )
    )
    _draw_obstacles(ax, env)
    _draw_station(ax, env)
    _draw_device_and_terminal_sector(ax, env, "corridor_target")
    starts = [_map_start(row, bounds) for row in records]
    selected = starts[selected_index % len(starts)]
    for index, start in enumerate(starts):
        _draw_start(
            ax,
            start,
            selected=index == selected_index % len(starts),
            label="selected start" if index == selected_index % len(starts) else None,
        )
    target = np.asarray(env.inspection_target, dtype=np.float32)
    upper = np.asarray([selected[:2], [8.6, 8.05], [15.4, 8.05], target])
    lower = np.asarray([selected[:2], [8.6, 5.35], [15.4, 5.35], target])
    ax.plot(
        upper[:, 0],
        upper[:, 1],
        color="#c83e4d",
        linewidth=2.4,
        linestyle="--",
        label="upper route: long shadow",
        zorder=7,
    )
    ax.plot(
        lower[:, 0],
        lower[:, 1],
        color="#218c5b",
        linewidth=2.4,
        linestyle="--",
        label="lower route: connected",
        zorder=7,
    )
    ax.annotate(
        "long LOS occlusion",
        xy=(12.7, 8.0),
        xytext=(11.2, 8.65),
        arrowprops={"arrowstyle": "->", "color": "#a52f3f"},
        fontsize=8,
        color="#8f2432",
    )
    _deduplicated_legend(ax, location="lower right")


def _plot_easy_open(
    ax,
    env: CommInspectionDubinsUAV2D,
    records: Sequence[Mapping[str, Any]],
    bounds: Sequence[float],
    selected_index: int,
) -> None:
    extent = (4.8, 12.3, 0.6, 11.5)
    _style_axis(ax, title=STRATUM_TITLES["easy_open"], extent=extent)
    _draw_obstacles(ax, env)
    _draw_station(ax, env)
    _draw_device_and_terminal_sector(ax, env, "easy_north", label=True)
    _draw_device_and_terminal_sector(ax, env, "easy_south", label=False)
    starts = [_map_start(row, bounds) for row in records]
    selected_position = selected_index % len(starts)
    targets = {
        "easy_north": np.asarray([7.8, 10.4], dtype=np.float32),
        "easy_south": np.asarray([11.0, 1.8], dtype=np.float32),
    }
    direct_label_used = False
    for index, (row, start) in enumerate(zip(records, starts)):
        selected = index == selected_position
        _draw_start(
            ax,
            start,
            selected=selected,
            label="selected start" if selected else None,
        )
        target = targets[str(row["device_id"])]
        ax.plot(
            [start[0], target[0]],
            [start[1], target[1]],
            color="#218c5b",
            linewidth=2.0 if selected else 1.0,
            alpha=0.95 if selected else 0.35,
            label="unobstructed direct path" if not direct_label_used else None,
            zorder=6,
        )
        direct_label_used = True
    _deduplicated_legend(ax, location="center right")


def visualize_diagnostic_scenarios(
    output_dir: str | Path,
    *,
    scenario: Mapping[str, Any] | None = None,
    task_bank: Mapping[str, Any] | None = None,
    split: str = "validation",
    sample_index: int = 0,
    communication_resolution: int = 180,
    dpi: int = 180,
) -> dict[str, Path]:
    """Save three stratum maps plus a combined overview figure."""

    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    scenario = dict(scenario or build_diagnostic_scenario())
    task_bank = dict(task_bank or build_diagnostic_task_bank(scenario))
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    bounds = [float(value) for value in scenario["bounds"]]
    records = {
        stratum: _records_for_stratum(task_bank, split=split, stratum=stratum)
        for stratum in CHALLENGE_STRATA
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plotters = {
        "u_trap": lambda ax: _plot_u_trap(
            ax, env, records["u_trap"], bounds, int(sample_index)
        ),
        "comm_shadow_corridor": lambda ax: _plot_comm_shadow_corridor(
            ax,
            env,
            records["comm_shadow_corridor"],
            bounds,
            int(sample_index),
            int(communication_resolution),
        ),
        "easy_open": lambda ax: _plot_easy_open(
            ax, env, records["easy_open"], bounds, int(sample_index)
        ),
    }
    sizes = {
        "u_trap": (7.2, 6.5),
        "comm_shadow_corridor": (10.5, 6.3),
        "easy_open": (7.2, 8.5),
    }
    paths: dict[str, Path] = {}
    for stratum in CHALLENGE_STRATA:
        fig, ax = plt.subplots(figsize=sizes[stratum], constrained_layout=True)
        plotters[stratum](ax)
        path = output / f"{stratum}.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        paths[stratum] = path

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), constrained_layout=True)
    for ax, stratum in zip(axes, CHALLENGE_STRATA):
        plotters[stratum](ax)
    overview_path = output / "diagnostic_scenarios_overview.png"
    fig.savefig(overview_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    paths["overview"] = overview_path
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize u_trap, comm_shadow_corridor, and easy_open maps"
    )
    parser.add_argument(
        "--output-dir",
        default="results/diagnostic_u_shadow_corridors/visualizations",
    )
    parser.add_argument("--scenario-config", default=None)
    parser.add_argument("--task-bank", default=None)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--communication-resolution", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    scenario = (
        load_scenario_config(args.scenario_config)
        if args.scenario_config
        else build_diagnostic_scenario()
    )
    task_bank = _load_json(args.task_bank) if args.task_bank else None
    paths = visualize_diagnostic_scenarios(
        args.output_dir,
        scenario=scenario,
        task_bank=task_bank,
        split=args.split,
        sample_index=args.sample_index,
        communication_resolution=args.communication_resolution,
        dpi=args.dpi,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
