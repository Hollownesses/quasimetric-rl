from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "quasimetric_rl_cache")
for _cache_dir in (
    _CACHE_ROOT,
    os.path.join(_CACHE_ROOT, "matplotlib"),
    os.path.join(_CACHE_ROOT, "xdg"),
    os.path.join(_CACHE_ROOT, "xdg", "fontconfig"),
):
    os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quasimetric_rl.data import EnvSpec
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.optim import AdamWSpec
from quasimetric_rl.modules.quasimetric_critic import QuasimetricCriticConf
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic
from quasimetric_rl.modules.quasimetric_critic.models.encoder import Encoder
from quasimetric_rl.modules.quasimetric_critic.models.latent_dynamics import LatentDynamics
from quasimetric_rl.modules.quasimetric_critic.models.quasimetric_model import QuasimetricModel
from quasimetric_rl.modules.quasimetric_critic.losses import QuasimetricCriticLosses
from quasimetric_rl.modules.quasimetric_critic.losses.global_push import GlobalPushLoss
from quasimetric_rl.modules.quasimetric_critic.losses.latent_dynamics import LatentDynamicsLoss
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss
from minimal_qrl.envs import Maze2DNavigation, MountainCar2D


Cell = Tuple[int, int]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: Sequence[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    sorted_values = values[order]
    n = values.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    return pearson_corr(rankdata_average(x), rankdata_average(y))


def affine_fit_to_target(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target)
    fitted = np.full_like(pred, np.nan, dtype=np.float64)
    if mask.sum() < 2:
        return fitted, float("nan"), float("nan")
    x = pred[mask]
    y = target[mask]
    if np.std(x) <= 1e-12:
        scale = 0.0
        bias = float(y.mean())
    else:
        design = np.stack([x, np.ones_like(x)], axis=1)
        scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        scale = float(scale)
        bias = float(bias)
    fitted[mask] = scale * x + bias
    return np.maximum(fitted, 0.0), scale, bias


def regression_metrics(pred: np.ndarray, target: np.ndarray, fit_affine: bool = True) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target)
    raw_pred = pred[mask]
    raw_target = target[mask]
    if raw_pred.size == 0:
        return {
            "n": 0,
            "mse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "affine_scale": float("nan"),
            "affine_bias": float("nan"),
        }
    if fit_affine:
        eval_pred, scale, bias = affine_fit_to_target(raw_pred, raw_target)
    else:
        eval_pred = raw_pred
        scale = 1.0
        bias = 0.0
    err = eval_pred - raw_target
    return {
        "n": int(raw_pred.size),
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "pearson": pearson_corr(raw_pred, raw_target),
        "spearman": spearman_corr(raw_pred, raw_target),
        "affine_scale": float(scale),
        "affine_bias": float(bias),
    }


def load_qrl_agent(env, checkpoint_path: str, num_critics: int, device: str, agent_conf=None):
    agent_conf = agent_conf or QRLConf(actor=None, num_critics=num_critics)
    agent, _ = agent_conf.make(
        env_spec=EnvSpec.from_env(env),
        total_optim_steps=1,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("agent", ckpt) if isinstance(ckpt, dict) else ckpt
    agent.load_state_dict(state_dict)
    agent.to(torch.device(device))
    agent.eval()
    return agent


def qrl_distance(agent, states: np.ndarray, goals: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    critic = agent.critics[0]
    out = []
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            s_t = torch.as_tensor(states[i : i + batch_size], dtype=torch.float32, device=device)
            g_t = torch.as_tensor(goals[i : i + batch_size], dtype=torch.float32, device=device)
            zx = critic.encoder(s_t)
            zy = critic.encoder(g_t)
            out.append(critic.quasimetric_model(zx, zy).detach().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(-1)


def parse_goals(text: str) -> np.ndarray:
    goals = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(x) for x in item.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError("Goals must be formatted as pos,vel or pos,vel,indicator.")
        goals.append(parts)
    return np.asarray(goals, dtype=np.float32)


def mountaincar_baselines(states: np.ndarray, goal: np.ndarray, mode: str) -> Dict[str, np.ndarray]:
    env = MountainCar2D()
    pos = (states[:, 0] - env.min_position) / (env.max_position - env.min_position)
    vel = (states[:, 1] - env.min_velocity) / (env.max_velocity - env.min_velocity)
    gpos = (goal[0] - env.min_position) / (env.max_position - env.min_position)
    gvel = (goal[1] - env.min_velocity) / (env.max_velocity - env.min_velocity)
    if mode == "threshold":
        dx = np.maximum(gpos - pos, 0.0)
        dy = np.where(pos >= gpos, 0.0, vel - gvel)
        position_distance = dx
        euclidean_distance = np.sqrt(dx * dx + dy * dy)
    else:
        position_distance = np.abs(pos - gpos)
        euclidean_distance = np.sqrt((pos - gpos) ** 2 + (vel - gvel) ** 2)
    return {
        "Position distance": position_distance.astype(np.float32),
        "Euclidean distance": euclidean_distance.astype(np.float32),
    }


def eval_mountaincar(args) -> None:
    env = MountainCar2D(
        max_episode_steps=args.max_steps_per_episode,
        gt_pos_bins=args.gt_pos_bins,
        gt_vel_bins=args.gt_vel_bins,
        gt_goal_mode=args.gt_goal_mode,
    )
    agent_conf = QRLConf(
        actor=None,
        num_critics=args.num_critics,
        quasimetric_critic=QuasimetricCriticConf(
            model=QuasimetricCritic.Conf(
                encoder=Encoder.Conf(arch=tuple(args.encoder_arch), latent_size=args.latent_size),
                quasimetric_model=QuasimetricModel.Conf(
                    projector_arch=tuple(args.projector_arch),
                    quasimetric_head_spec=f"iqe(dim={args.iqe_dim},components={args.iqe_components})",
                ),
                latent_dynamics=LatentDynamics.Conf(arch=tuple(args.transition_arch), residual=True),
            ),
            losses=QuasimetricCriticLosses.Conf(
                global_push=GlobalPushLoss.Conf(softplus_beta=args.global_beta, softplus_offset=args.global_offset),
                local_constraint=LocalConstraintLoss.Conf(epsilon=0.25, step_cost=1.0, init_lagrange_multiplier=0.01),
                latent_dynamics=LatentDynamicsLoss.Conf(weight=args.transition_loss_weight),
                critic_optim=AdamWSpec.Conf(lr=5e-4),
                lagrange_mult_optim=AdamWSpec.Conf(lr=0.3),
            ),
        ),
    )
    agent = load_qrl_agent(env, args.checkpoint, args.num_critics, args.device, agent_conf=agent_conf)
    goals = parse_goals(args.goals) if args.goals else np.asarray([[0.5, 0.0, 1.0]], dtype=np.float32)
    states_physical = np.stack([env._index_to_state(i) for i in range(env.gt_pos_bins * env.gt_vel_bins)], axis=0)
    states_obs = np.stack([env._as_observation(s) for s in states_physical], axis=0)

    rows = []
    aggregate = defaultdict(lambda: {"pred": [], "target": []})
    for goal_id, goal in enumerate(goals):
        goal_obs = goal if goal.shape[0] == 3 else (
            np.array([goal[0], goal[1], 1.0], dtype=np.float32)
            if args.gt_goal_mode == "threshold"
            else env._as_observation(goal)
        )
        goal_physical = goal_obs[:2]
        true_dist = env._distance_grid(goal_obs, mode=args.gt_goal_mode)
        finite = np.isfinite(true_dist)
        qrl_pred = qrl_distance(agent, states_obs, np.repeat(goal_obs[None, :], len(states_obs), axis=0), args.device)
        predictions = mountaincar_baselines(states_physical, goal_physical, args.gt_goal_mode)
        predictions["QRL distance"] = qrl_pred
        for method, pred in predictions.items():
            result = regression_metrics(pred[finite], true_dist[finite], fit_affine=args.fit_affine)
            rows.append({"goal_id": goal_id, "goal_position": float(goal_physical[0]), "goal_velocity": float(goal_physical[1]), "method": method, **result})
            aggregate[method]["pred"].append(pred[finite])
            aggregate[method]["target"].append(true_dist[finite])
        plot_mountaincar_heatmap(args.output_dir, env, states_physical, goal_physical, qrl_pred, true_dist, goal_id)

    for method, data in aggregate.items():
        result = regression_metrics(np.concatenate(data["pred"]), np.concatenate(data["target"]), fit_affine=args.fit_affine)
        rows.append({"goal_id": "all", "goal_position": "", "goal_velocity": "", "method": method, **result})
    write_csv(os.path.join(args.output_dir, "distance_metrics.csv"), rows)


def plot_mountaincar_heatmap(output_dir: str, env: MountainCar2D, states: np.ndarray, goal: np.ndarray, qrl_pred: np.ndarray, true_dist: np.ndarray, goal_id: int) -> None:
    ensure_dir(output_dir)
    qrl_grid = qrl_pred.reshape(env.gt_pos_bins, env.gt_vel_bins)
    true_grid = np.where(np.isfinite(true_dist), true_dist, np.nan).reshape(env.gt_pos_bins, env.gt_vel_bins)
    extent = [env.min_position, env.max_position, env.min_velocity, env.max_velocity]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150, sharex=True, sharey=True)
    for ax, title, grid, cmap in [
        (axes[0], "QRL learned distance", qrl_grid, "viridis"),
        (axes[1], "Graph shortest steps", true_grid, "magma"),
    ]:
        im = ax.imshow(grid.T, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.scatter([goal[0]], [goal[1]], marker="*", s=120, color="white", edgecolor="black", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("position")
        ax.set_ylabel("velocity")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"mountaincar_heatmap_goal_{goal_id}.png"))
    plt.close(fig)


def maze_cells(env: Maze2DNavigation) -> List[Cell]:
    return list(env.valid_cells)


def maze_states(env: Maze2DNavigation, cells: Sequence[Cell]) -> np.ndarray:
    return np.stack([env._cell_to_obs(cell) for cell in cells], axis=0).astype(np.float32)


def maze_baselines(cells: Sequence[Cell], goal: Cell) -> Dict[str, np.ndarray]:
    arr = np.asarray(cells, dtype=np.float32)
    g = np.asarray(goal, dtype=np.float32)[None, :]
    diff = np.abs(arr - g)
    return {
        "Euclidean distance": np.sqrt((diff * diff).sum(axis=1)).astype(np.float32),
        "Manhattan distance": diff.sum(axis=1).astype(np.float32),
    }


def eval_maze2d(args) -> None:
    env = Maze2DNavigation(grid_size=tuple(args.grid_size), max_episode_steps=args.max_steps_per_episode)
    agent = load_qrl_agent(env, args.checkpoint, args.num_critics, args.device)
    goals = [(1, env.width - 2), (env.height - 2, env.width - 2), (env.height // 2, env.width - 2)]
    goals = [g for g in goals if g in env.valid_set]
    cells = maze_cells(env)
    states = maze_states(env, cells)
    rows = []
    aggregate = defaultdict(lambda: {"pred": [], "target": []})
    for goal_id, goal in enumerate(goals):
        true_dist = np.array([env.compute_shortest_path_distance(env._cell_to_obs(cell), env._cell_to_obs(goal)) for cell in cells], dtype=np.float32)
        finite = np.isfinite(true_dist)
        goal_obs = np.repeat(env._cell_to_obs(goal)[None, :], len(cells), axis=0)
        predictions = maze_baselines(cells, goal)
        predictions["QRL distance"] = qrl_distance(agent, states, goal_obs, args.device)
        for method, pred in predictions.items():
            result = regression_metrics(pred[finite], true_dist[finite], fit_affine=args.fit_affine)
            rows.append({"goal_id": goal_id, "goal_row": goal[0], "goal_col": goal[1], "method": method, **result})
            aggregate[method]["pred"].append(pred[finite])
            aggregate[method]["target"].append(true_dist[finite])
        plot_maze_heatmap(args.output_dir, env, cells, goal, predictions["QRL distance"], true_dist, goal_id)
    for method, data in aggregate.items():
        result = regression_metrics(np.concatenate(data["pred"]), np.concatenate(data["target"]), fit_affine=args.fit_affine)
        rows.append({"goal_id": "all", "goal_row": "", "goal_col": "", "method": method, **result})
    write_csv(os.path.join(args.output_dir, "distance_metrics.csv"), rows)
    eval_maze_navigation(args, env, agent)


def plot_maze_heatmap(output_dir: str, env: Maze2DNavigation, cells: Sequence[Cell], goal: Cell, qrl_pred: np.ndarray, true_dist: np.ndarray, goal_id: int) -> None:
    ensure_dir(output_dir)
    grids = []
    for values in [qrl_pred, true_dist, maze_baselines(cells, goal)["Euclidean distance"]]:
        grid = np.full((env.height, env.width), np.nan, dtype=np.float32)
        for cell, value in zip(cells, values):
            grid[cell] = value
        grids.append(grid)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#202020")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150, sharex=True, sharey=True)
    for ax, title, grid in zip(axes, ["QRL learned distance", "BFS shortest path", "Euclidean distance"], grids):
        im = ax.imshow(np.ma.masked_invalid(grid), origin="upper", cmap=cmap, interpolation="nearest")
        ax.scatter([goal[1]], [goal[0]], marker="*", s=140, color="white", edgecolor="black", linewidth=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"maze2d_heatmap_goal_{goal_id}.png"))
    plt.close(fig)


def select_qrl_action(agent, env: Maze2DNavigation, cell: Cell, goal: Cell, device: str, horizon: int) -> int:
    frontier = [(env._next_cell(cell, action), action, 1) for action in range(4)]
    for _ in range(max(1, horizon) - 1):
        expanded = []
        for cur, first_action, depth in frontier:
            if cur == goal:
                expanded.append((cur, first_action, depth))
                continue
            for action in range(4):
                expanded.append((env._next_cell(cur, action), first_action, depth + 1))
        frontier = expanded
    endpoints = [item[0] for item in frontier]
    scores = qrl_distance(
        agent,
        maze_states(env, endpoints),
        np.repeat(env._cell_to_obs(goal)[None, :], len(endpoints), axis=0),
        device,
    )
    adjusted = np.array([depth if endpoint == goal else depth + score for (endpoint, _, depth), score in zip(frontier, scores)])
    return int(frontier[int(np.argmin(adjusted))][1])


def eval_maze_navigation(args, env: Maze2DNavigation, agent) -> None:
    rng = np.random.default_rng(args.seed)
    pairs = []
    attempts = 0
    while len(pairs) < args.eval_pairs and attempts < args.eval_pairs * 200:
        attempts += 1
        start = env.valid_cells[int(rng.integers(0, len(env.valid_cells)))]
        goal = env.valid_cells[int(rng.integers(0, len(env.valid_cells)))]
        if start == goal:
            continue
        d = env.compute_shortest_path_distance(env._cell_to_obs(start), env._cell_to_obs(goal))
        if np.isfinite(d) and d >= args.min_pair_distance:
            pairs.append((start, goal))

    rows = []
    for method in ["Euclidean Greedy", "QRL Greedy", "QRL Lookahead"]:
        success, steps = [], []
        for start, goal in pairs:
            cell = start
            executed_steps = 0
            for t in range(args.max_steps_per_episode):
                if cell == goal:
                    break
                if method == "Euclidean Greedy":
                    candidates = [(a, env._next_cell(cell, a)) for a in range(4)]
                    action = min(candidates, key=lambda x: np.linalg.norm(np.asarray(x[1]) - np.asarray(goal)))[0]
                elif method == "QRL Greedy":
                    action = select_qrl_action(agent, env, cell, goal, args.device, horizon=1)
                else:
                    action = select_qrl_action(agent, env, cell, goal, args.device, horizon=args.lookahead_horizon)
                cell = env._next_cell(cell, action)
                executed_steps = t + 1
            reached = cell == goal
            success.append(float(reached))
            if reached:
                steps.append(float(executed_steps))
        rows.append({
            "method": method,
            "n_eval": len(pairs),
            "success_rate": float(np.mean(success)) if success else float("nan"),
            "avg_steps_success": float(np.mean(steps)) if steps else float("nan"),
            "avg_path_length_success": float(np.mean(steps)) if steps else float("nan"),
            "max_episode_steps": args.max_steps_per_episode,
        })
    write_csv(os.path.join(args.output_dir, "navigation_metrics.csv"), rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing-QRL checkpoint for thesis value experiments.")
    parser.add_argument("--env-type", choices=["mountaincar", "maze2d"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, nargs=2, default=[15, 15])
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    parser.add_argument("--eval-pairs", type=int, default=200)
    parser.add_argument("--min-pair-distance", type=int, default=6)
    parser.add_argument("--lookahead-horizon", type=int, default=4)
    parser.add_argument("--goals", default="")
    parser.add_argument("--gt-pos-bins", type=int, default=160)
    parser.add_argument("--gt-vel-bins", type=int, default=160)
    parser.add_argument("--gt-goal-mode", choices=["threshold", "point"], default="threshold")
    parser.add_argument("--fit-affine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encoder-arch", type=int, nargs="*", default=[1024, 1024, 1024])
    parser.add_argument("--transition-arch", type=int, nargs="*", default=[1024, 1024, 1024])
    parser.add_argument("--projector-arch", type=int, nargs="*", default=[1024, 1024, 1024])
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--iqe-dim", type=int, default=512)
    parser.add_argument("--iqe-components", type=int, default=16)
    parser.add_argument("--transition-loss-weight", type=float, default=75.0)
    parser.add_argument("--global-offset", type=float, default=500.0)
    parser.add_argument("--global-beta", type=float, default=0.01)
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    if args.env_type == "mountaincar":
        eval_mountaincar(args)
    else:
        eval_maze2d(args)
    print(f"QRL checkpoint evaluation finished. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
