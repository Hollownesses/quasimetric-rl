"""Restore replay feasibility with push disabled and complete uniform replay passes.

This diagnostic keeps the source KKT optimizer and its saved state. It measures
constraint recovery; a collapsed distance field is not a navigation solution.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qrl_constraint_recovery_mpl")

import numpy as np
import torch

from minimal_qrl.eval.replay_constraint_audit import (
    edge_metadata, evaluate_edges, get_env_kwargs, group_masks, load_replay,
    load_run_spec, summarize_edges, top_edge_rows, write_csv,
)
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.global_push_sampling import replay_fingerprint
from quasimetric_rl.data import BatchData, EnvSpec
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.optim import AdamWSpec
from quasimetric_rl.modules.quasimetric_critic import QuasimetricCriticConf
from quasimetric_rl.modules.quasimetric_critic.losses import QuasimetricCriticLosses
from quasimetric_rl.modules.quasimetric_critic.losses.abstract_goal_edge import AbstractGoalEdgeLoss
from quasimetric_rl.modules.quasimetric_critic.losses.global_push import GlobalPushLoss
from quasimetric_rl.modules.quasimetric_critic.losses.latent_dynamics import LatentDynamicsLoss
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss
from quasimetric_rl.modules.quasimetric_critic.losses.temporal_path import (
    GoalReturnConstraintLoss, MQEInspiredWaypointConsistencyLoss,
    NstepGoalConsistencyLoss, TemporalPathConstraintLoss,
)


def recovery_conf(args):
    if args.qrl_local_constraint_mode != "kkt_functional":
        raise ValueError("Constraint-only recovery currently requires a KKT functional checkpoint")
    local = LocalConstraintLoss.Conf(
        mode="kkt_functional", cost_source="negative_reward",
        init_lagrange_multiplier=args.qrl_kkt_init_lagrange_multiplier,
        augmented_lagrangian_rho=args.qrl_kkt_augmented_lagrangian_rho,
        dual_hidden_sizes=tuple(args.qrl_kkt_dual_hidden_sizes),
        dual_max=args.qrl_kkt_dual_max, dual_steps=args.qrl_kkt_dual_steps,
        dual_start_violation_fraction=args.qrl_kkt_dual_start_violation_fraction,
        dual_projected_step_size=args.qrl_kkt_dual_projected_step_size,
        dual_feature_scale=args.qrl_kkt_dual_feature_scale,
        dual_raw_min=args.qrl_kkt_dual_raw_min,
    )
    return QRLConf(actor=None, num_critics=args.num_critics,
                   quasimetric_critic=QuasimetricCriticConf(losses=QuasimetricCriticLosses.Conf(
                       local_constraint=local,
                       global_push=GlobalPushLoss.Conf(objective=args.global_push_objective,
                                                       abstract_goal_ratio=0.0, state_goal_ratio=0.0),
                       latent_dynamics=LatentDynamicsLoss.Conf(weight=0.0),
                       abstract_goal_edge=AbstractGoalEdgeLoss.Conf(weight=args.abstract_goal_edge_loss_weight),
                       temporal_path=TemporalPathConstraintLoss.Conf(weight=0.0),
                       goal_return=GoalReturnConstraintLoss.Conf(weight=0.0),
                       nstep_goal=NstepGoalConsistencyLoss.Conf(weight=0.0),
                       mqe_waypoint=MQEInspiredWaypointConsistencyLoss.Conf(weight=0.0),
                       critic_optim=AdamWSpec.Conf(lr=5e-5),
                       lagrange_mult_optim=AdamWSpec.Conf(lr=args.qrl_kkt_dual_lr),
                   )))


def load_recovery_models(spec, env, device):
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    if not {"agent", "losses", "optim_steps"}.issubset(checkpoint):
        raise ValueError("Recovery requires complete agent, dual and optimizer checkpoint states")
    experiment = checkpoint.get("experiment", {})
    if experiment.get("local_constraint_mode") != spec["args"].qrl_local_constraint_mode:
        raise ValueError("Checkpoint mode disagrees with training metadata")
    if experiment.get("global_push_sampling", {}).get("seed", spec["args"].seed) != spec["args"].seed:
        raise ValueError("Checkpoint seed disagrees with training metadata")
    conf = recovery_conf(spec["args"])
    agent, losses = conf.make(env_spec=EnvSpec.from_env(env),
                             total_optim_steps=max(1, int(checkpoint["optim_steps"])))
    agent.to(device)
    losses.to(device)
    agent.load_state_dict(checkpoint["agent"], strict=True)
    # Includes network/buffers, Adam moments/step counters, and schedulers.
    # Intentionally no critic-only fallback.
    losses.load_state_dict(checkpoint["losses"])
    agent.train()
    return agent, losses, int(checkpoint["optim_steps"]), experiment


def replay_batches(raw, batch_size, *, seed, pass_index):
    """One deterministic shuffle without replacement, including the final tail."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _, _, sources = edge_metadata(raw)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 87_129, int(pass_index)]))
    order = rng.permutation(raw.num_transitions)
    for start in range(0, raw.num_transitions, batch_size):
        indices = torch.from_numpy(order[start:start + batch_size])
        obs_indices = torch.from_numpy(sources[indices.numpy()])
        next_obs = raw.all_observations[obs_indices + 1]
        yield indices, BatchData(
            observations=raw.all_observations[obs_indices], next_observations=next_obs,
            actions=raw.actions[indices], rewards=raw.rewards[indices],
            terminals=raw.terminals[indices], timeouts=raw.timeouts[indices],
            future_observations=next_obs,  # Unused: all future/path objectives are disabled.
            transition_infos={k: v[indices] for k, v in raw.transition_infos.items()},
        )


@torch.no_grad()
def audit_snapshot(raw, agent, losses, env, scenario, output, stage, step, recovery_updates,
                   *, device, chunk_size, tolerance, anchor_tolerance):
    masks = group_masks(raw, env, scenario)
    costs = (-raw.rewards).clamp_min(0).numpy()
    reports, csv_rows, top_rows, distances, duals, raw_duals = [], [], [], [], [], []
    for idx, (critic, critic_loss) in enumerate(zip(agent.critics, losses.critic_losses)):
        local = critic_loss.local_constraint
        d, lam, raw_lam = evaluate_edges(critic, raw, chunk_size=chunk_size, device=device, local=local)
        distances.append(d)
        duals.append(lam)
        raw_duals.append(raw_lam)
        groups = {}
        for name, mask in masks.items():
            tol = anchor_tolerance if name == "anchor" or name.endswith("/anchor") else tolerance
            groups[name] = summarize_edges(d[mask], costs[mask], tolerance=tol,
                                           weights=np.ones(int(mask.sum())), dual=lam[mask],
                                           raw_dual=raw_lam[mask], dual_raw_floor=local.dual_raw_min)
            csv_rows.append({"stage": stage, "optim_steps": step, "recovery_updates": recovery_updates,
                             "critic": idx, "group": name, **groups[name]})
        if not groups["all"]["audit_valid"] or groups["all"]["dual_nonfinite_edges"]:
            raise RuntimeError("Nonfinite full replay audit; recovery cannot be treated as valid")
        reports.append({"critic": idx, "groups": groups})
        top_rows.extend(top_edge_rows(raw, d, costs, stage, idx, masks, 100,
                                      feature_indices=env.task_context_indices))
    worst = np.max(distances, axis=0)
    worst_groups = {}
    for name, mask in masks.items():
        tol = anchor_tolerance if name == "anchor" or name.endswith("/anchor") else tolerance
        worst_groups[name] = summarize_edges(worst[mask], costs[mask], tolerance=tol,
                                             weights=np.ones(int(mask.sum())))
        csv_rows.append({"stage": stage, "optim_steps": step, "recovery_updates": recovery_updates,
                         "critic": "worst_per_edge", "group": name, **worst_groups[name]})
    np.savez_compressed(output / f"{stage}_edges.npz", distances=np.stack(distances), costs=costs,
                        residuals=np.stack(distances) - costs, dual=np.stack(duals),
                        raw_dual=np.stack(raw_duals), replay_sha256=np.asarray(replay_fingerprint(raw)))
    write_csv(output / f"{stage}_top_violations.csv", top_rows)
    # A fixed probe bank reports distance shrinkage; it is never a training target.
    physical = np.flatnonzero(masks["real"])
    probe_indices = physical[np.linspace(0, len(physical) - 1, min(1024, len(physical)), dtype=int)]
    _, _, sources = edge_metadata(raw)
    probe = raw.all_observations[torch.from_numpy(sources[probe_indices])].to(device)
    goals = raw.transition_infos["task_goal_observations"][probe_indices].to(device)
    probe_stats = []
    for critic in agent.critics:
        value = critic(probe, goals).cpu().numpy()
        probe_stats.append({"mean": float(value.mean()), "std": float(value.std()),
                            "min": float(value.min()), "max": float(value.max())})
    report = {"stage": stage, "optim_steps": step, "recovery_updates": recovery_updates,
              "critics": reports, "worst_per_edge": worst_groups,
              "fixed_task_goal_probe": probe_stats}
    print(f"Audit {stage}: real violation={worst_groups['real']['violation_fraction']:.3%}, "
          f"max={worst_groups['real']['max_violation']:.3f}, "
          f"anchor max={worst_groups['anchor']['max_violation']:.3f}", flush=True)
    return report, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-cache", default=None)
    parser.add_argument("--recovery-passes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--audit-chunk-size", type=int, default=1024)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--anchor-tolerance", type=float, default=1e-6)
    parser.add_argument("--log-interval", type=int, default=100)
    cli = parser.parse_args()
    if min(cli.recovery_passes, cli.cpu_threads, cli.audit_chunk_size, cli.log_interval) <= 0:
        raise ValueError("Pass/thread/chunk/log counts must be positive")
    if min(cli.tolerance, cli.anchor_tolerance) < 0:
        raise ValueError("Audit tolerances must be nonnegative")
    spec = load_run_spec(cli.init_checkpoint)
    output = Path(cli.output_dir)
    if output.resolve() == Path(spec["checkpoint"]).parent:
        raise ValueError("Recovery output must be separate from the source training directory")
    if (output / "recovery_manifest.json").exists():
        raise ValueError("Recovery output already contains a run; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cli.cpu_threads)
    device = auto_device(cli.device)
    raw = load_replay(spec, cli.replay_cache or output / "verified_replay.pt")
    batch_size = cli.batch_size if cli.batch_size is not None else spec["args"].batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches_per_pass = math.ceil(raw.num_transitions / batch_size)
    env = CommInspectionDubinsUAV2D(**get_env_kwargs(spec["args"]))
    agent, losses, start_step, experiment = load_recovery_models(spec, env, device)
    experiment = copy.deepcopy(experiment)
    experiment.update(variant="constraint_only_recovery", global_push_abstract_goal_ratio=0.0,
                      global_push_state_goal_ratio=0.0, latent_dynamics_weight=0.0,
                      replay_sha256=spec["expected_sha256"])
    recovery = {"source_checkpoint": spec["checkpoint"], "source_step": start_step,
                "replay_sha256": spec["expected_sha256"], "optimizer_state_restored_strictly": True,
                "sampler": "uniform_shuffle_without_replacement_including_tail",
                "passes": cli.recovery_passes, "batch_size": batch_size,
                "batches_per_pass": batches_per_pass, "dual_steps_per_batch": spec["args"].qrl_kkt_dual_steps,
                "anchor_loss_weight": spec["args"].abstract_goal_edge_loss_weight,
                "future_path_targets_added": False}
    experiment["constraint_only_recovery"] = recovery
    args = copy.deepcopy(spec["args"])
    args.output_dir, args.init_checkpoint, args.device = str(output), spec["checkpoint"], str(device)
    args.total_steps = start_step + batches_per_pass * cli.recovery_passes
    args.global_push_abstract_goal_ratio = args.global_push_state_goal_ratio = 0.0
    args.qrl_latent_dynamics_weight = 0.0
    args.qrl_temporal_constraint_weight = args.qrl_goal_return_constraint_weight = 0.0
    args.qrl_nstep_goal_constraint_weight = args.qrl_mqe_waypoint_consistency_weight = 0.0
    args.qrl_success_transition_weight, args.batch_size = 1.0, batch_size
    # Preserve provenance files so the general offline auditor can read recovery checkpoints.
    (output / "train.log").write_text(f"INFO - 参数: {args!r}\n", encoding="utf-8")
    (output / "exploration_start_bank.json").write_bytes(
        (Path(spec["checkpoint"]).parent / "exploration_start_bank.json").read_bytes())
    (output / "global_push_sampling_stats.json").write_text(json.dumps(
        {"physical_replay_sha256": spec["expected_sha256"], "pair_bank": "inactive_during_recovery",
         "task_push_weight": 0.0, "state_push_weight": 0.0}, indent=2))
    (output / "recovery_manifest.json").write_text(json.dumps(recovery, indent=2))
    audits, audit_rows, train_rows = [], [], []
    step, updates = start_step, 0
    initial_latent = [{k: v.detach().cpu().clone() for k, v in c.latent_dynamics.state_dict().items()}
                      for c in agent.critics]
    started = perf_counter()
    for pass_index in range(cli.recovery_passes + 1):
        stage = f"pass_{pass_index:02d}"
        if pass_index:
            visits = np.zeros(raw.num_transitions, dtype=np.int32)
            for indices, batch in replay_batches(raw, batch_size, seed=args.seed, pass_index=pass_index):
                visits[indices.numpy()] += 1
                result = losses(agent, batch.to(device), optimize=True)
                step += 1
                updates += 1
                if not torch.isfinite(result.loss):
                    raise RuntimeError("Nonfinite recovery training loss")
                if updates % cli.log_interval == 0 or updates % batches_per_pass == 0:
                    row = {"step": step, "recovery_updates": updates, "pass": pass_index,
                           "loss": float(result.loss.detach().cpu())}
                    train_rows.append(row)
                    print(f"Recovery pass {pass_index}, update {updates}, total step {step}, loss={row['loss']:.5g}", flush=True)
            if not np.all(visits == 1):
                raise RuntimeError("Recovery pass did not visit every replay edge exactly once")
            np.savez_compressed(output / f"{stage}_visits.npz", transition_visits=visits)
        for critic, original in zip(agent.critics, initial_latent):
            if not all(torch.equal(v.detach().cpu(), original[k]) for k, v in critic.latent_dynamics.state_dict().items()):
                raise RuntimeError("Disabled latent dynamics moved during recovery")
        checkpoint_path = output / f"checkpoint_{step:05d}.pth"
        torch.save({"optim_steps": step, "recovery_updates": updates,
                    "recovery_pass": pass_index, "agent": agent.state_dict(),
                    "losses": losses.state_dict(), "experiment": experiment}, checkpoint_path)
        report, rows = audit_snapshot(raw, agent, losses, env, spec["scenario"], output, stage,
                                       step, updates, device=device, chunk_size=cli.audit_chunk_size,
                                       tolerance=cli.tolerance, anchor_tolerance=cli.anchor_tolerance)
        report["checkpoint"] = str(checkpoint_path.resolve())
        audits.append(report)
        audit_rows.extend(rows)
        payload = {"recovery": recovery, "elapsed_seconds": perf_counter() - started,
                   "device": str(device), "tolerance": cli.tolerance,
                   "anchor_tolerance": cli.anchor_tolerance, "audits": audits}
        (output / "recovery_audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
        write_csv(output / "recovery_audit.csv", audit_rows)
        write_csv(output / "recovery_train_metrics.csv", train_rows)
    (output / "COMPLETE").write_text("")
    print(f"Constraint-only recovery complete: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
