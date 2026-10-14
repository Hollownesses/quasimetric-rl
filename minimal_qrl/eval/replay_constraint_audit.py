"""Audit every recorded one-step constraint without training or sampled futures.

Reconstruct qrl_explore replay from saved starts and literal training arguments,
then require the exact training replay SHA256 before evaluating any checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qrl_replay_audit_mpl")

import numpy as np
import torch

from minimal_qrl.dataset import QRLExploreConfig, create_dataset, EXPLORE_OUTCOME_NAMES
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.global_push_sampling import replay_fingerprint
from minimal_qrl.train import get_env_kwargs
from quasimetric_rl.data import MultiEpisodeData, EnvSpec
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss
from minimal_qrl.eval.utils import auto_device


REPLAY_INFO_KEYS = ("abstract_goal_edge", "source_terminal_goal_state",
                    "task_goal_observations", "context_id", "task_success_episode",
                    "device_index", "exploration_outcome", "planner_path_support")


def load_training_args(path):
    """Parse logged Namespace literals, never evaluate executable log content."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    literal = next((line.split("参数: ", 1)[1] for line in reversed(lines)
                    if "参数: Namespace(" in line), None)
    if literal is None:
        raise ValueError(f"Missing literal training Namespace in {path}")
    node = ast.parse(literal, mode="eval").body
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "Namespace" and not node.args
            and all(keyword.arg is not None for keyword in node.keywords)):
        raise ValueError("Training arguments must be a literal Namespace")
    return argparse.Namespace(**{k.arg: ast.literal_eval(k.value) for k in node.keywords})


def load_run_spec(checkpoint):
    directory = Path(checkpoint).resolve().parent
    args = load_training_args(directory / "train.log")
    if args.comm_dataset_mode not in ("qrl_explore", "path_support_only"):
        raise ValueError("Exact replay reconstruction supports qrl_explore and path_support_only")
    if args.qrl_cost_source != "negative_reward":
        raise ValueError("Communication replay audit requires negative_reward cost semantics")
    stats = json.loads((directory / "global_push_sampling_stats.json").read_text())
    scenario = args._scenario_data
    if not scenario:
        raise ValueError("Replay audit needs the saved diagnostic scenario")
    if getattr(args, 'path_support_bank', None):
        from minimal_qrl.path_support import load_bank
        bank = load_bank(directory / 'path_support_bank.json')
        if bank['content_digest'] != args.path_support_bank_sha256:
            raise ValueError('path-support snapshot disagrees with training log')
        args.path_support_bank = str(directory / 'path_support_bank.json')
    if args.comm_dataset_mode == 'path_support_only':
        from minimal_qrl.path_support import validate_training_protocol
        validate_training_protocol(args)
        # No exploration start bank exists in the bank-only experiment.
        return {"checkpoint": str(Path(checkpoint).resolve()), "args": args,
                "config": None, "expected_sha256": stats["physical_replay_sha256"], "scenario": scenario}
    bank = json.loads((directory / "exploration_start_bank.json").read_text())
    claimed_digest = bank.pop("content_digest")
    actual_digest = hashlib.sha256(json.dumps(
        bank, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if actual_digest != claimed_digest:
        raise ValueError(f"Start bank content digest mismatch: {directory}")
    if int(bank["seed"]) != int(args.seed):
        raise ValueError("Start bank seed disagrees with training seed")
    records = sorted(bank["records"], key=lambda r: r["start_index"])
    if [r["start_index"] for r in records] != list(range(len(records))):
        raise ValueError("Start bank indices must be contiguous")
    config = QRLExploreConfig(
        attempted_env_steps=int(args.explore_attempted_env_steps),
        start_position_resolution=float(args.explore_start_position_resolution),
        start_heading_bins=int(args.explore_start_heading_bins),
        action_hold_min_steps=int(args.explore_action_hold_min_steps),
        action_hold_max_steps=int(args.explore_action_hold_max_steps),
        straight_action_probability=float(args.explore_straight_action_probability),
        exclusion_radius=float(args.explore_exclusion_radius),
        start_states=tuple(tuple(float(x) for x in r["state"]) for r in records),
        diagnostic_regions=scenario.get("metadata", {}).get("exploration_diagnostic_regions"),
        diagnostic_routes=scenario.get("metadata", {}).get("exploration_diagnostic_routes"),
        start_strata=tuple((r["name"], r["weight"], tuple(r["bounds"]))
                           for r in scenario["metadata"]["exploration_start_strata"]),
        start_boundary_margin=float(args.explore_start_boundary_margin),
        local_safety_lookahead_steps=int(args.explore_local_safety_lookahead_steps),
        # Logs predating the fix used the hold-capped policy. Reconstruct that
        # exact replay instead of silently changing the data behind a checkpoint.
        safety_lookahead_mode=getattr(args, "explore_safety_lookahead_mode", "legacy_action_hold"),
    )
    return {"checkpoint": str(Path(checkpoint).resolve()), "args": args,
            "config": config, "expected_sha256": stats["physical_replay_sha256"],
            "scenario": scenario}


def validate_replay(raw, expected_sha256):
    actual = replay_fingerprint(raw)
    if actual != expected_sha256:
        raise ValueError(f"Reconstructed replay SHA256 mismatch: expected {expected_sha256}, got {actual}")
    return actual


def load_replay(spec, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        raw = MultiEpisodeData(**payload["raw"])
        validate_replay(raw, spec["expected_sha256"])
        print(f"Loaded verified replay cache: {cache_path}", flush=True)
        return raw
    print("Reconstructing replay from saved starts or frozen path bank; environment push pairs disabled", flush=True)
    env = CommInspectionDubinsUAV2D(**get_env_kwargs(spec["args"]))
    episodes = []
    for episode in create_dataset(
            env, max_steps_per_episode=spec["args"].max_steps_per_episode,
            seed=spec["args"].seed, qrl_explore_config=spec["config"],
            path_support_bank=getattr(spec['args'], 'path_support_bank', None),
            path_support_only=getattr(spec['args'], 'comm_dataset_mode', None) == 'path_support_only',
            include_global_push_pairs=False):
        # Drop the large push-only tensors; the constraint replay remains exact.
        episode.transition_infos = {k: episode.transition_infos[k] for k in REPLAY_INFO_KEYS}
        episodes.append(episode)
    raw = MultiEpisodeData.cat(episodes)
    digest = validate_replay(raw, spec["expected_sha256"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save({"schema_version": 1, "sha256": digest,
                "raw": {name: getattr(raw, name) for name in (
                    "episode_lengths", "all_observations", "actions", "rewards",
                    "terminals", "timeouts", "observation_infos", "transition_infos")}}, temporary)
    temporary.replace(cache_path)
    print(f"Verified replay: {raw.num_transitions} edges, SHA256={digest}", flush=True)
    return raw


def edge_metadata(raw):
    lengths = raw.episode_lengths.numpy()
    episode_ids = np.repeat(np.arange(len(lengths)), lengths)
    timesteps = np.concatenate([np.arange(length) for length in lengths])
    sources = np.arange(raw.num_transitions) + episode_ids
    return episode_ids, timesteps, sources


@torch.no_grad()
def evaluate_edges(critic, raw, *, chunk_size, device, local=None):
    """Every replay row exactly once, using its actual recorded next observation."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    _, _, sources = edge_metadata(raw)
    distances = np.empty(raw.num_transitions, dtype=np.float32)
    dual = np.empty_like(distances) if local is not None else None
    raw_dual = np.empty_like(distances) if local is not None else None
    was_training = critic.training
    critic.eval()
    try:
        for start in range(0, raw.num_transitions, chunk_size):
            stop = min(raw.num_transitions, start + chunk_size)
            indices = torch.from_numpy(sources[start:stop])
            obs = raw.all_observations[indices].to(device)
            next_obs = raw.all_observations[indices + 1].to(device)
            d = critic(obs, next_obs).reshape(-1)
            distances[start:stop] = d.cpu().numpy()
            if local is not None:
                cost = (-raw.rewards[start:stop].to(device=device, dtype=d.dtype)).clamp_min(0)
                lam, unbounded, _ = local._functional_lagrange_multiplier(obs, next_obs, cost)
                dual[start:stop] = lam.cpu().numpy()
                raw_dual[start:stop] = unbounded.cpu().numpy()
    finally:
        critic.train(was_training)
    return distances, dual, raw_dual


def summarize_edges(distances, costs, *, tolerance, weights=None, dual=None, raw_dual=None,
                    dual_raw_floor=-10.0):
    """Uniform coverage first; weighted rates separately match the train sampler."""
    distances = np.asarray(distances)
    costs = np.asarray(costs, dtype=distances.dtype)
    residual = distances - costs  # Same float32 subtraction as training.
    finite = np.isfinite(residual) & np.isfinite(distances) & np.isfinite(costs)
    r = residual[finite].astype(np.float64)
    positive = np.maximum(r, 0)
    count = len(residual)
    invalid = int(np.sum(~finite))
    result = {"edges": count, "nonfinite_edges": invalid, "audit_valid": invalid == 0,
              "tolerance": float(tolerance), "strict_violation_count": int(np.sum(r > 0)),
              "violation_count": int(np.sum(r > tolerance)),
              "violation_fraction": float(np.mean(r > tolerance)) if len(r) and not invalid else None,
              "strict_violation_fraction": float(np.mean(r > 0)) if len(r) and not invalid else None,
              "distance_mean": float(np.mean(distances[finite])) if len(r) else None,
              "cost_mean": float(np.mean(costs[finite])) if len(r) else None,
              "residual_mean": float(r.mean()) if len(r) else None,
              "positive_residual_mean": float(positive.mean()) if len(r) else None,
              "squared_positive_residual_mean": float(np.mean(positive ** 2)) if len(r) else None,
              "max_violation": float(positive.max()) if len(r) else None}
    for label, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("p999", .999)):
        result[f"positive_residual_{label}"] = float(np.quantile(positive, q)) if len(r) else None
    violated = r[r > tolerance]
    result["violating_residual_p99"] = float(np.quantile(violated, .99)) if len(violated) else None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != count or np.any(~np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("Sampler weights must be finite, positive and match all edges")
        result["sampler_weighted_violation_fraction"] = float(np.average(residual > tolerance, weights=w)) if count and not invalid else None
    if dual is not None:
        lam = np.asarray(dual)
        mask = finite & np.isfinite(lam)
        vm = mask & (residual > tolerance)
        result.update({
            "dual_nonfinite_edges": int(np.sum(~np.isfinite(lam))),
            "dual_mean": float(lam[mask].mean()) if mask.any() else None,
            "violation_dual_mean": float(lam[vm].mean()) if vm.any() else None,
            "violation_dual_below_1e_3_fraction": float(np.mean(lam[vm] < 1e-3)) if vm.any() else None,
            "complementarity_abs_mean": float(np.mean(np.abs(lam[mask] * residual[mask]))) if mask.any() else None,
        })
        if raw_dual is not None:
            raw_lam = np.asarray(raw_dual)
            result["dual_raw_floor"] = float(dual_raw_floor)
            result["violation_raw_dual_below_floor_fraction"] = float(np.mean(raw_lam[vm] < dual_raw_floor)) if vm.any() else None
    return result


def group_masks(raw, env, scenario):
    infos = raw.transition_infos
    abstract = infos["abstract_goal_edge"].numpy().astype(bool)
    success = infos["task_success_episode"].numpy().astype(bool)
    costs = (-raw.rewards).clamp_min(0).numpy()
    real = ~abstract
    masks = {"all": np.ones(len(real), dtype=bool), "real": real,
             "abstract": abstract, "zero_cost": costs == 0,
             "anchor": abstract & (costs == 0) & infos["source_terminal_goal_state"].numpy().astype(bool),
             "real_success": real & success, "real_failure": real & ~success}
    masks["real_terminal_success"] = real & success & raw.terminals.numpy().astype(bool)
    masks["real_terminal_failure"] = real & ~success & raw.terminals.numpy().astype(bool)
    masks["real_timeout"] = real & raw.timeouts.numpy().astype(bool)
    for idx, name in enumerate(env.device_ids):
        device = infos["device_index"].numpy() == idx
        for family in ("real", "anchor"):
            masks[f"device/{name}/{family}"] = device & masks[family]
    _, _, sources = edge_metadata(raw)
    xy = raw.all_observations.numpy()[sources, :2]
    for name, (xmin, ymin, xmax, ymax) in scenario.get("metadata", {}).get("exploration_diagnostic_regions", {}).items():
        masks[f"region/{name}/real"] = real & (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
    outcomes = infos.get("exploration_outcome")
    if outcomes is not None:
        for idx, name in EXPLORE_OUTCOME_NAMES.items():
            masks[f"outcome/{name}/real"] = real & (outcomes.numpy() == idx)
    return masks


def load_models(spec, device):
    ckpt = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    experiment = ckpt.get("experiment", {})
    if (experiment.get("local_constraint_mode", spec["args"].qrl_local_constraint_mode)
            != spec["args"].qrl_local_constraint_mode):
        raise ValueError("Checkpoint constraint mode disagrees with its training log")
    sampling = experiment.get("global_push_sampling", {})
    if sampling.get("seed", spec["args"].seed) != spec["args"].seed:
        raise ValueError("Checkpoint sampling seed disagrees with its training log")
    env = CommInspectionDubinsUAV2D(**get_env_kwargs(spec["args"]))
    conf = QRLConf(actor=None, num_critics=spec["args"].num_critics)
    agent, _ = conf.make(env_spec=EnvSpec.from_env(env), total_optim_steps=1)
    agent.load_state_dict(ckpt["agent"], strict=True)
    agent.to(device).eval()
    duals = []
    args = spec["args"]
    for idx, critic in enumerate(agent.critics):
        local = None
        if args.qrl_local_constraint_mode == "kkt_functional":
            local = LocalConstraintLoss.Conf(
                mode="kkt_functional", cost_source="negative_reward",
                dual_hidden_sizes=tuple(args.qrl_kkt_dual_hidden_sizes),
                dual_feature_scale=args.qrl_kkt_dual_feature_scale,
                dual_raw_min=args.qrl_kkt_dual_raw_min,
                dual_max=args.qrl_kkt_dual_max,
            ).make(critic)
            prefix = f"critic_losses.{idx}.local_constraint."
            loss_module = ckpt["losses"].get("module", ckpt["losses"])
            state = {k[len(prefix):]: v for k, v in loss_module.items() if k.startswith(prefix)}
            local.load_state_dict(state, strict=True)
            local.to(device).eval()
        duals.append(local)
    return agent, duals, int(ckpt["optim_steps"]), env, experiment


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def top_edge_rows(raw, distances, costs, label, critic_idx, masks, top_k,
                  feature_indices=None):
    episode_ids, times, sources = edge_metadata(raw)
    residual = distances - costs
    obs = raw.all_observations.numpy()
    infos = raw.transition_infos
    feature_indices = feature_indices or {}
    rows = []
    for family in ("real", "anchor"):
        indices = np.flatnonzero(masks[family] & (residual > 0))
        indices = indices[np.argsort(residual[indices], kind="stable")[-top_k:][::-1]]
        for i in indices:
            state, next_state = obs[sources[i]], obs[sources[i] + 1]
            rows.append({"checkpoint_id": label, "critic": critic_idx, "family": family,
                         "transition_index": int(i), "episode_index": int(episode_ids[i]),
                         "timestep": int(times[i]), "context_id": int(infos["context_id"][i]),
                         "device_index": int(infos["device_index"][i]),
                         "success_episode": bool(infos["task_success_episode"][i]),
                         "terminal": bool(raw.terminals[i]), "timeout": bool(raw.timeouts[i]),
                         "exploration_outcome": int(infos["exploration_outcome"][i]),
                         "x": float(state[0]), "y": float(state[1]),
                         "heading": float(np.arctan2(state[3], state[2])),
                         "next_x": float(next_state[0]), "next_y": float(next_state[1]),
                         "distance": float(distances[i]), "cost": float(costs[i]),
                         "residual": float(residual[i])})
            if "task_feasible" in feature_indices:
                task_idx = feature_indices["task_feasible"]
                rows[-1]["source_task_feasible"] = float(state[task_idx])
                rows[-1]["next_task_feasible"] = float(next_state[task_idx])
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-cache", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--anchor-tolerance", type=float, default=1e-6)
    parser.add_argument("--top-k", type=int, default=100)
    return parser


def main():
    args = build_parser().parse_args()
    if min(args.chunk_size, args.cpu_threads, args.top_k) <= 0 or min(args.tolerance, args.anchor_tolerance) < 0:
        raise ValueError("Chunk/thread/top-k counts must be positive and tolerances nonnegative")
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = [load_run_spec(path) for path in args.checkpoints]
    if len({spec["expected_sha256"] for spec in specs}) != 1:
        raise ValueError("Checkpoint runs have different replay fingerprints; audit them separately")
    started = perf_counter()
    raw = load_replay(specs[0], args.replay_cache or output / "verified_replay.pt")
    costs = (-raw.rewards).clamp_min(0).numpy()
    summary_rows, top_rows, reports = [], [], []
    for checkpoint_index, spec in enumerate(specs):
        agent, duals, step, env, experiment = load_models(spec, auto_device(args.device))
        label = f"{Path(spec['checkpoint']).parent.name}_step{step:05d}_{checkpoint_index}"
        print(f"Auditing {label}: {raw.num_transitions} edges x {len(agent.critics)} critics", flush=True)
        masks = group_masks(raw, env, spec["scenario"])
        weights = np.where(raw.transition_infos["task_success_episode"].numpy(),
                           spec["args"].qrl_success_transition_weight, 1.0)
        all_distances, all_dual, all_raw_dual = [], [], []
        critic_reports = []
        for idx, (critic, local) in enumerate(zip(agent.critics, duals)):
            distances, dual, unbounded = evaluate_edges(
                critic, raw, chunk_size=args.chunk_size, device=auto_device(args.device), local=local)
            all_distances.append(distances)
            if dual is not None:
                all_dual.append(dual)
                all_raw_dual.append(unbounded)
            groups = {}
            for name, mask in masks.items():
                tolerance = args.anchor_tolerance if name == "anchor" or name.endswith("/anchor") else args.tolerance
                groups[name] = summarize_edges(distances[mask], costs[mask], tolerance=tolerance,
                                               weights=weights[mask],
                                               dual=dual[mask] if dual is not None else None,
                                               raw_dual=unbounded[mask] if unbounded is not None else None,
                                               dual_raw_floor=local.dual_raw_min if local is not None else -10.0)
                summary_rows.append({"checkpoint_id": label, "optim_steps": step,
                                     "critic": idx, "group": name, **groups[name]})
            top_rows.extend(top_edge_rows(raw, distances, costs, label, idx, masks, args.top_k,
                                         feature_indices=getattr(env, "task_context_indices", {})))
            critic_reports.append({"critic": idx, "groups": groups})
            print(f"  critic {idx}: real violation={groups['real']['violation_fraction']}, max={groups['real']['max_violation']}; anchor violation={groups['anchor']['violation_fraction']}", flush=True)
        # Per-edge worst critic avoids hiding a violation through ensemble averaging.
        worst = np.max(all_distances, axis=0)
        worst_groups = {}
        for name, mask in masks.items():
            tolerance = args.anchor_tolerance if name == "anchor" or name.endswith("/anchor") else args.tolerance
            worst_groups[name] = summarize_edges(worst[mask], costs[mask], tolerance=tolerance, weights=weights[mask])
            summary_rows.append({"checkpoint_id": label, "optim_steps": step,
                                 "critic": "worst_per_edge", "group": name, **worst_groups[name]})
        archive = {"distances": np.stack(all_distances), "costs": costs,
                   "residuals": np.stack(all_distances) - costs,
                   "abstract_goal_edge": raw.transition_infos["abstract_goal_edge"].numpy(),
                   "replay_sha256": np.asarray(spec["expected_sha256"])}
        if all_dual:
            archive.update(dual=np.stack(all_dual), raw_dual=np.stack(all_raw_dual))
        np.savez_compressed(output / f"{label}_edges.npz", **archive)
        reports.append({"checkpoint_id": label, "checkpoint": spec["checkpoint"], "optim_steps": step,
                        "experiment": experiment, "critics": critic_reports,
                        "worst_per_edge": worst_groups})
    payload = {"schema_version": 1, "replay_sha256": replay_fingerprint(raw),
               "replay_verified_against_all_runs": True, "edges": raw.num_transitions,
               "cost_semantics": "max(-recorded_reward, 0)",
               "constraint": "d(recorded_source, recorded_next_state) <= cost",
               "counts_are": "uniform_all_replay_edges; sampler weighting reported separately",
               "device": str(auto_device(args.device)), "chunk_size": args.chunk_size,
               "tolerance": args.tolerance, "anchor_tolerance": args.anchor_tolerance,
               "elapsed_seconds": perf_counter() - started, "checkpoints": reports}
    (output / "replay_constraint_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    write_csv(output / "replay_constraint_audit.csv", summary_rows)
    write_csv(output / "top_violations.csv", top_rows)
    print(f"Full replay audit saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
