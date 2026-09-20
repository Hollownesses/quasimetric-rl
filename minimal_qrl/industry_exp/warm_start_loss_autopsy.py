#!/usr/bin/env python3
"""Locate which QRL loss first destroys a correct warm-started U-trap value.

Every arm starts from the same targeted-supervised IQE checkpoint and sees the
same seeded full-graph batches.  Oracle values are evaluation-only.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "warm_start_loss_autopsy_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "warm_start_loss_autopsy_xdg")
)

from minimal_qrl.baselines import (
    HybridAStarConfig,
    HybridAStarValueOracle,
    MPPIConfig,
)
from minimal_qrl.dataset import (
    FullGraphGoalSetQRLConfig,
    create_full_graph_goal_set_qrl_dataset,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.gc_agents import QRLGoalValueAdapter
from minimal_qrl.industry_exp.joint_feasible_iqe import (
    _predict_edges,
    _predict_values,
    build_joint_feasible_problem,
    lattice_successor_ranking,
)
from minimal_qrl.industry_exp.qrl_pipeline_metrics import (
    build_mppi_ranking_contract,
    evaluate_mppi_ranking_contract,
    u_region_constraint_tails,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)
from minimal_qrl.industry_exp.supervised_iqe_oracle import _regression_metrics
from minimal_qrl.iqe_capacity import IQECapacity, iqe_capacity_from_checkpoint
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.optim import AdamWSpec
from quasimetric_rl.modules.quasimetric_critic import QuasimetricCriticConf
from quasimetric_rl.modules.quasimetric_critic.losses import (
    AbstractGoalEdgeLoss,
    CriticBatchInfo,
    GlobalPushLoss,
    LatentDynamicsLoss,
    LocalConstraintLoss,
    QuasimetricCriticLosses,
)
from quasimetric_rl.modules.quasimetric_critic.losses.temporal_path import (
    GoalReturnConstraintLoss,
    NstepGoalConsistencyLoss,
    TemporalPathConstraintLoss,
)
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic
from quasimetric_rl.modules.quasimetric_critic.models.quasimetric_model import (
    QuasimetricModel,
)
from quasimetric_rl.modules.utils import LossResult


ARM_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "a0_no_update": (),
    "a1_global_push": ("global_push",),
    "a2_local_constraint": ("local_constraint",),
    "a3_latent_dynamics": ("latent_dynamics",),
    "a4_abstract_goal_edge": ("abstract_goal_edge",),
    "a5_push_plus_local": ("global_push", "local_constraint"),
    "a6_full_qrl": (
        "global_push",
        "local_constraint",
        "latent_dynamics",
        "abstract_goal_edge",
    ),
}
ARM_ALIASES = {f"a{index}": name for index, name in enumerate(ARM_COMPONENTS)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def normalize_arms(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for raw in values:
        key = str(raw).strip().lower()
        key = ARM_ALIASES.get(key, key)
        if key not in ARM_COMPONENTS:
            raise ValueError(
                f"unknown arm {raw!r}; expected one of {sorted(ARM_COMPONENTS)}"
            )
        if key not in result:
            result.append(key)
    return result


def _make_qrl(
    dataset,
    *,
    num_critics: int,
    capacity: IQECapacity,
    total_steps: int,
    critic_lr: float,
    dual_lr: float,
    ordinary_epsilon: float,
    direct_goal_epsilon: float,
    terminal_goal_epsilon: float,
    latent_dynamics_weight: float,
    abstract_goal_edge_weight: float,
):
    model = QuasimetricCritic.Conf(
        quasimetric_model=QuasimetricModel.Conf(
            projector_arch=(512,),
            quasimetric_head_spec=capacity.spec,
        )
    )
    losses = QuasimetricCriticLosses.Conf(
        global_push=GlobalPushLoss.Conf(
            abstract_goal_ratio=1.0,
            state_goal_ratio=0.0,
        ),
        local_constraint=LocalConstraintLoss.Conf(
            epsilon=float(ordinary_epsilon),
            step_cost=1.0,
            cost_source="negative_reward",
            constraint_mode="full_graph_stratified",
            direct_goal_epsilon=float(direct_goal_epsilon),
            terminal_goal_epsilon=float(terminal_goal_epsilon),
        ),
        latent_dynamics=LatentDynamicsLoss.Conf(
            weight=float(latent_dynamics_weight)
        ),
        abstract_goal_edge=AbstractGoalEdgeLoss.Conf(
            weight=float(abstract_goal_edge_weight)
        ),
        temporal_path=TemporalPathConstraintLoss.Conf(weight=0.0),
        goal_return=GoalReturnConstraintLoss.Conf(weight=0.0),
        nstep_goal=NstepGoalConsistencyLoss.Conf(weight=0.0),
        critic_optim=AdamWSpec.Conf(lr=float(critic_lr)),
        lagrange_mult_optim=AdamWSpec.Conf(lr=float(dual_lr)),
    )
    return QRLConf(
        actor=None,
        num_critics=int(num_critics),
        quasimetric_critic=QuasimetricCriticConf(model=model, losses=losses),
    ).make(
        env_spec=dataset.env_spec,
        total_optim_steps=max(1, int(total_steps)),
    )


def _critic_batch_info(critic, batch) -> CriticBatchInfo:
    zx, zy = critic.encoder(
        torch.stack([batch.observations, batch.next_observations], dim=0)
    ).unbind(0)
    return CriticBatchInfo(critic=critic, zx=zx, zy=zy)


def optimize_selected_components(
    agent,
    losses,
    batch,
    components: Sequence[str],
) -> LossResult:
    """Perform one ordinary QRL optimizer step using only selected components."""

    results: dict[str, LossResult] = {}
    uses_local = "local_constraint" in components
    for index, (critic, critic_losses) in enumerate(
        zip(agent.critics, losses.critic_losses)
    ):
        dual_context = (
            critic_losses.lagrange_mult_optim.update_context(optimize=True)
            if uses_local
            else contextlib.nullcontext()
        )
        with critic_losses.critic_optim.update_context(optimize=True), dual_context:
            batch_info = _critic_batch_info(critic, batch)
            component_results = {
                name: getattr(critic_losses, name)(batch, batch_info)
                for name in components
            }
            result = LossResult.combine(component_results)
            result.loss.backward()
        critic_losses.critic_sched.step()
        if uses_local:
            critic_losses.lagrange_mult_sched.step()
        results[f"critic_{index:02d}"] = result
    return LossResult.combine(results)


def _gradient_vector(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        pieces.append(
            torch.zeros(parameter.numel(), dtype=torch.float32)
            if gradient is None
            else gradient.detach().reshape(-1).float().cpu()
        )
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def gradient_autopsy(agent, losses, batch, *, critic_index: int) -> dict[str, Any]:
    """Measure component gradients on one common warm-start batch."""

    critic = agent.critics[int(critic_index)]
    critic_losses = losses.critic_losses[int(critic_index)]
    named_parameters = list(critic.named_parameters())
    parameters = [parameter for _name, parameter in named_parameters]
    named_dual_parameters = list(critic_losses.local_constraint.named_parameters())
    dual_parameters = [parameter for _name, parameter in named_dual_parameters]
    vectors: dict[str, torch.Tensor] = {}
    components: dict[str, Any] = {}
    for component in (
        "global_push",
        "local_constraint",
        "latent_dynamics",
        "abstract_goal_edge",
    ):
        batch_info = _critic_batch_info(critic, batch)
        result = getattr(critic_losses, component)(batch, batch_info)
        all_gradients = torch.autograd.grad(
            result.loss,
            parameters + dual_parameters,
            allow_unused=True,
            retain_graph=False,
        )
        gradients = all_gradients[: len(parameters)]
        dual_gradients = all_gradients[len(parameters) :]
        vector = _gradient_vector(gradients, parameters)
        vectors[component] = vector
        group_norms: dict[str, float] = {}
        for group_name, prefix in (
            ("encoder", "encoder."),
            ("projector", "quasimetric_model.projector."),
            ("latent_dynamics", "latent_dynamics."),
        ):
            group_gradients = [
                gradient.detach().reshape(-1).float().cpu()
                for (name, _parameter), gradient in zip(named_parameters, gradients)
                if name.startswith(prefix) and gradient is not None
            ]
            group_norms[group_name] = float(
                torch.linalg.vector_norm(torch.cat(group_gradients)).item()
                if group_gradients
                else 0.0
            )
        components[component] = {
            "loss": float(result.loss.detach().cpu()),
            "total_gradient_norm": float(torch.linalg.vector_norm(vector).item()),
            "gradient_norm_by_parameter_group": group_norms,
            "dual_gradients": {
                name: (
                    float(gradient.detach().cpu())
                    if gradient is not None and gradient.numel() == 1
                    else None
                )
                for (name, _parameter), gradient in zip(
                    named_dual_parameters, dual_gradients
                )
            },
        }
    cosines: dict[str, float | None] = {}
    names = list(vectors)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_norm = float(torch.linalg.vector_norm(vectors[left]))
            right_norm = float(torch.linalg.vector_norm(vectors[right]))
            key = f"{left}__{right}"
            cosines[key] = (
                float(
                    torch.dot(vectors[left], vectors[right])
                    / (left_norm * right_norm)
                )
                if left_norm > 0.0 and right_norm > 0.0
                else None
            )
    return {
        "critic_index": int(critic_index),
        "components": components,
        "gradient_cosines": cosines,
    }


def _parameter_drift(critic, initial: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    groups = {
        "encoder": "encoder.",
        "projector": "quasimetric_model.projector.",
        "latent_dynamics": "latent_dynamics.",
    }
    state = dict(critic.named_parameters())
    result: dict[str, Any] = {}
    for group_name, prefix in groups.items():
        squared_delta = 0.0
        squared_initial = 0.0
        for name, parameter in state.items():
            if not name.startswith(prefix):
                continue
            base = initial[name].to(device=parameter.device, dtype=parameter.dtype)
            squared_delta += float(torch.sum((parameter.detach() - base) ** 2))
            squared_initial += float(torch.sum(base**2))
        norm = math.sqrt(squared_delta)
        result[group_name] = {
            "l2": norm,
            "relative_l2": norm / max(math.sqrt(squared_initial), 1e-12),
        }
    return result


def _lagrange_values(losses, critic_index: int) -> dict[str, float]:
    local = losses.critic_losses[int(critic_index)].local_constraint
    result = {
        "ordinary": float(
            torch.nn.functional.softplus(local.raw_lagrange_multiplier)
            .detach()
            .cpu()
        ),
    }
    if local.raw_direct_goal_lagrange_multiplier is not None:
        result["direct_goal"] = float(
            torch.nn.functional.softplus(local.raw_direct_goal_lagrange_multiplier)
            .detach()
            .cpu()
        )
        result["terminal_goal"] = float(
            torch.nn.functional.softplus(local.raw_terminal_goal_lagrange_multiplier)
            .detach()
            .cpu()
        )
    return result


def evaluate_checkpoint(
    agent,
    losses,
    problem,
    tensors: Mapping[str, torch.Tensor],
    contract_cases,
    mppi_config: MPPIConfig,
    env,
    *,
    critic_index: int,
    eval_batch_size: int,
    initial_parameters: Mapping[str, torch.Tensor],
    epsilons: Mapping[str, float],
) -> dict[str, Any]:
    agent.eval()
    critic = agent.critics[int(critic_index)]
    values = _predict_values(
        critic,
        tensors["observations"],
        tensors["goal"],
        batch_size=int(eval_batch_size),
    )
    edges = _predict_edges(
        critic,
        tensors["observations"],
        tensors["goal"],
        tensors["sources"],
        tensors["destinations"],
        batch_size=int(eval_batch_size),
    )
    topology = lattice_successor_ranking(
        problem,
        values,
        source_mask=problem.u_trap_mask,
    )
    tails = u_region_constraint_tails(
        problem,
        edges,
        values,
        epsilons=epsilons,
    )
    adapter = QRLGoalValueAdapter(agent, env, tensors["observations"].device, distance_scale=1.0)
    contract = evaluate_mppi_ranking_contract(
        adapter,
        contract_cases,
        config=mppi_config,
    )
    metrics = {
        "goal_slice_global": _regression_metrics(
            values, problem.reference_values
        ),
        "goal_slice_u_region": _regression_metrics(
            values[problem.u_trap_mask],
            problem.reference_values[problem.u_trap_mask],
        ),
        "u_region_successor_topology": topology,
        "u_region_constraint_tails": tails,
        "mppi_horizon_ranking_contract": contract,
        "parameter_drift": _parameter_drift(critic, initial_parameters),
        "lagrange_multipliers": _lagrange_values(losses, int(critic_index)),
    }
    agent.train()
    return metrics


def _nested(payload: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _summary_row(arm: str, step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    pairwise_tail = "u_region_constraint_tails.groups.all_outgoing.pairwise_quasimetric_edge.ordinary"
    bellman_tail = "u_region_constraint_tails.groups.all_outgoing.goal_slice_bellman.ordinary"
    optimal_tail = "u_region_constraint_tails.groups.oracle_optimal_outgoing.goal_slice_bellman.ordinary"
    return {
        "arm": arm,
        "step": int(step),
        "u_value_mae": _nested(metrics, "goal_slice_u_region.mae"),
        "u_value_pearson": _nested(metrics, "goal_slice_u_region.pearson"),
        "u_successor_top1": _nested(metrics, "u_region_successor_topology.top1_accuracy"),
        "u_successor_pairwise": _nested(metrics, "u_region_successor_topology.pairwise_accuracy"),
        "u_successor_regret_mean": _nested(metrics, "u_region_successor_topology.oracle_regret.mean"),
        "u_successor_regret_p95": _nested(metrics, "u_region_successor_topology.oracle_regret.p95"),
        "u_successor_regret_max": _nested(metrics, "u_region_successor_topology.oracle_regret.max"),
        "u_pairwise_excess_p99": _nested(metrics, f"{pairwise_tail}.positive_excess_p99"),
        "u_pairwise_excess_p999": _nested(metrics, f"{pairwise_tail}.positive_excess_p999"),
        "u_pairwise_excess_max": _nested(metrics, f"{pairwise_tail}.positive_excess_max"),
        "u_bellman_excess_p99": _nested(metrics, f"{bellman_tail}.positive_excess_p99"),
        "u_bellman_excess_p999": _nested(metrics, f"{bellman_tail}.positive_excess_p999"),
        "u_bellman_excess_max": _nested(metrics, f"{bellman_tail}.positive_excess_max"),
        "u_optimal_edge_bellman_excess_p99": _nested(metrics, f"{optimal_tail}.positive_excess_p99"),
        "u_optimal_edge_bellman_excess_max": _nested(metrics, f"{optimal_tail}.positive_excess_max"),
        "mppi_pairwise": _nested(metrics, "mppi_horizon_ranking_contract.pairwise_concordance_mean"),
        "mppi_spearman": _nested(metrics, "mppi_horizon_ranking_contract.spearman_mean"),
        "mppi_topk_overlap": _nested(metrics, "mppi_horizon_ranking_contract.top_k_overlap_mean"),
        "mppi_oracle_regret_mean": _nested(metrics, "mppi_horizon_ranking_contract.learned_selected_oracle_regret.mean"),
        "mppi_oracle_regret_p95": _nested(metrics, "mppi_horizon_ranking_contract.learned_selected_oracle_regret.p95"),
        "mppi_weighted_action_abs_error": _nested(metrics, "mppi_horizon_ranking_contract.weighted_action_abs_error_mean"),
        "encoder_relative_drift": _nested(metrics, "parameter_drift.encoder.relative_l2"),
        "projector_relative_drift": _nested(metrics, "parameter_drift.projector.relative_l2"),
        "lambda_ordinary": _nested(metrics, "lagrange_multipliers.ordinary"),
        "lambda_direct_goal": _nested(metrics, "lagrange_multipliers.direct_goal"),
        "lambda_terminal_goal": _nested(metrics, "lagrange_multipliers.terminal_goal"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_scalars(value: Any, *, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.update(
                _flatten_scalars(item, prefix=f"{prefix}{key}/")
            )
        return result
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            result[prefix.rstrip("/")] = float(value.detach().cpu())
        return result
    if isinstance(value, (int, float, np.number)):
        result[prefix.rstrip("/")] = float(value)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--task-bank", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-id", default="u_trap_target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--critic-index", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=list(ARM_COMPONENTS))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--scheduler-total-steps",
        type=int,
        default=20_000,
        help="Cosine-scheduler horizon; defaults to the original warm-start run.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=(0, 1, 10, 50, 100, 250, 500, 1000, 2000),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
    parser.add_argument("--dual-lr", type=float, default=5e-3)
    parser.add_argument("--ordinary-epsilon", type=float, default=0.25)
    parser.add_argument("--direct-goal-epsilon", type=float, default=0.25)
    parser.add_argument("--terminal-goal-epsilon", type=float, default=0.0)
    parser.add_argument("--latent-dynamics-weight", type=float, default=0.1)
    parser.add_argument("--abstract-goal-edge-weight", type=float, default=1.0)
    parser.add_argument("--position-resolution", type=float, default=0.25)
    parser.add_argument("--heading-bins", type=int, default=24)
    parser.add_argument("--primitive-steps", type=int, default=5)
    parser.add_argument(
        "--primitive-scales",
        type=float,
        nargs="+",
        default=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    parser.add_argument("--uniform-push-seed", type=int, default=20260824)
    parser.add_argument("--contract-split", default="validation")
    parser.add_argument("--contract-repeats", type=int, default=4)
    parser.add_argument("--mppi-horizon", type=int, default=10)
    parser.add_argument("--mppi-num-samples", type=int, default=128)
    parser.add_argument("--mppi-noise-sigma", type=float, default=0.8)
    parser.add_argument("--mppi-temperature", type=float, default=1.0)
    parser.add_argument("--mppi-terminal-weight", type=float, default=1.0)
    parser.add_argument("--oracle-value-cache-dir", default=None)
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    started = perf_counter()
    arms = normalize_arms(args.arms)
    eval_steps = sorted(
        {int(step) for step in args.eval_steps if 0 <= int(step) <= int(args.steps)}
        | {0, int(args.steps)}
    )
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_config(args.scenario_config)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.reset(seed=int(args.seed), options={"device_id": str(args.device_id)})
    device = auto_device(str(args.device))
    graph_config = HybridAStarConfig(
        position_resolution=float(args.position_resolution),
        heading_bins=int(args.heading_bins),
        primitive_steps=int(args.primitive_steps),
        primitive_scales=tuple(float(value) for value in args.primitive_scales),
    )
    problem = build_joint_feasible_problem(env, scenario, graph_config)
    collection_stats: dict[str, Any] = {}
    dataset = create_full_graph_goal_set_qrl_dataset(
        env,
        FullGraphGoalSetQRLConfig(
            device_id=str(args.device_id),
            position_resolution=float(args.position_resolution),
            heading_bins=int(args.heading_bins),
            primitive_steps=int(args.primitive_steps),
            primitive_scales=tuple(float(value) for value in args.primitive_scales),
            uniform_push_seed=int(args.uniform_push_seed),
            stratified_constraints=True,
        ),
        collection_stats=collection_stats,
    )
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
    capacity = iqe_capacity_from_checkpoint(checkpoint)
    state_dict = checkpoint["agent"] if isinstance(checkpoint, dict) else checkpoint
    mppi_config = MPPIConfig(
        horizon=int(args.mppi_horizon),
        num_samples=int(args.mppi_num_samples),
        noise_sigma=float(args.mppi_noise_sigma),
        temperature=float(args.mppi_temperature),
        terminal_weight=float(args.mppi_terminal_weight),
    )
    oracle = HybridAStarValueOracle(
        graph_config,
        cache_dir=(
            Path(args.oracle_value_cache_dir)
            if args.oracle_value_cache_dir
            else output_dir / "oracle_value_cache"
        ),
        unreachable_cost=float(mppi_config.invalid_penalty),
    )
    contract_cases, contract_provenance = build_mppi_ranking_contract(
        env,
        oracle,
        task_bank_path=str(args.task_bank),
        scenario_id=str(scenario["scenario_id"]),
        split=str(args.contract_split),
        stratum="u_trap",
        config=mppi_config,
        repeats=int(args.contract_repeats),
    )
    tensors = {
        "observations": torch.as_tensor(
            problem.observations, device=device, dtype=torch.float32
        ),
        "goal": torch.as_tensor(
            problem.goal_observation, device=device, dtype=torch.float32
        ).reshape(1, -1),
        "sources": torch.as_tensor(problem.sources, device=device, dtype=torch.long),
        "destinations": torch.as_tensor(
            problem.destinations, device=device, dtype=torch.long
        ),
    }
    epsilons = {
        "ordinary": float(args.ordinary_epsilon),
        "direct_goal": float(args.direct_goal_epsilon),
        "terminal_goal": float(args.terminal_goal_epsilon),
    }

    def make_loaded_qrl():
        agent, losses = _make_qrl(
            dataset,
            num_critics=int(args.num_critics),
            capacity=capacity,
            total_steps=int(args.scheduler_total_steps),
            critic_lr=float(args.critic_lr),
            dual_lr=float(args.dual_lr),
            ordinary_epsilon=float(args.ordinary_epsilon),
            direct_goal_epsilon=float(args.direct_goal_epsilon),
            terminal_goal_epsilon=float(args.terminal_goal_epsilon),
            latent_dynamics_weight=float(args.latent_dynamics_weight),
            abstract_goal_edge_weight=float(args.abstract_goal_edge_weight),
        )
        agent.load_state_dict(state_dict)
        agent.to(device).train()
        losses.to(device).train()
        return agent, losses

    torch.manual_seed(int(args.seed))
    gradient_agent, gradient_losses = make_loaded_qrl()
    gradient_loader = dataset.get_dataloader(
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        full_graph_stratified_constraints=True,
    )
    gradient_batch = next(iter(gradient_loader)).to(device)
    gradient_metrics = gradient_autopsy(
        gradient_agent,
        gradient_losses,
        gradient_batch,
        critic_index=int(args.critic_index),
    )
    (output_dir / "step0_gradient_autopsy.json").write_text(
        json.dumps(_jsonable(gradient_metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del gradient_agent, gradient_losses, gradient_batch

    all_arm_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_started = perf_counter()
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        agent, losses = make_loaded_qrl()
        critic = agent.critics[int(args.critic_index)]
        initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in critic.named_parameters()
        }
        components = ARM_COMPONENTS[arm]
        arm_metrics: list[dict[str, Any]] = []
        train_history: list[dict[str, Any]] = []

        def run_eval(step: int) -> None:
            metrics = evaluate_checkpoint(
                agent,
                losses,
                problem,
                tensors,
                contract_cases,
                mppi_config,
                env,
                critic_index=int(args.critic_index),
                eval_batch_size=int(args.eval_batch_size),
                initial_parameters=initial_parameters,
                epsilons=epsilons,
            )
            record = {"step": int(step), "metrics": metrics}
            arm_metrics.append(record)
            summary_rows.append(_summary_row(arm, int(step), metrics))
            if args.save_eval_checkpoints:
                torch.save(
                    {
                        "optim_steps": int(step),
                        "agent": agent.state_dict(),
                        "losses": losses.state_dict(),
                        "training_mode": "warm_start_loss_autopsy",
                        "arm": arm,
                        "components": list(components),
                        "model_capacity": capacity.to_dict(),
                    },
                    arm_dir / f"checkpoint_{int(step):05d}.pth",
                )

        run_eval(0)
        if components:
            dataloader = dataset.get_dataloader(
                batch_size=int(args.batch_size),
                shuffle=True,
                drop_last=True,
                num_workers=0,
                full_graph_stratified_constraints=True,
            )
            step = 0
            progress = tqdm(total=int(args.steps), desc=arm)
            while step < int(args.steps):
                for batch in dataloader:
                    batch = batch.to(device)
                    loss_result = optimize_selected_components(
                        agent,
                        losses,
                        batch,
                        components,
                    )
                    step += 1
                    train_history.append(
                        {
                            "step": int(step),
                            "total_loss": float(loss_result.loss.detach().cpu()),
                            **_flatten_scalars(loss_result.info),
                        }
                    )
                    progress.update(1)
                    if step in eval_steps:
                        run_eval(step)
                    if step >= int(args.steps):
                        break
            progress.close()
        if train_history:
            _write_csv(arm_dir / "train_loss_history.csv", train_history)
        torch.save(
            {
                "optim_steps": int(args.steps if components else 0),
                "agent": agent.state_dict(),
                "losses": losses.state_dict(),
                "training_mode": "warm_start_loss_autopsy",
                "arm": arm,
                "components": list(components),
                "model_capacity": capacity.to_dict(),
                "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
            },
            arm_dir / "checkpoint_final.pth",
        )
        arm_payload = {
            "arm": arm,
            "components": list(components),
            "checkpoints": arm_metrics,
            "train_loss_history_csv": (
                str((arm_dir / "train_loss_history.csv").resolve())
                if train_history
                else None
            ),
            "elapsed_sec": float(perf_counter() - arm_started),
        }
        (arm_dir / "metrics.json").write_text(
            json.dumps(_jsonable(arm_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_arm_results[arm] = arm_payload
        del agent, losses

    _write_csv(output_dir / "warm_start_loss_autopsy_summary.csv", summary_rows)
    payload = {
        "experiment": "warm_start_qrl_loss_autopsy",
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "checkpoint_training_mode": (
            checkpoint.get("training_mode", "unknown")
            if isinstance(checkpoint, Mapping)
            else "unknown"
        ),
        "arms": all_arm_results,
        "step0_gradient_autopsy": gradient_metrics,
        "graph": problem.graph_stats,
        "training_dataset": collection_stats.get("full_graph_goal_set_qrl"),
        "mppi_contract": contract_provenance,
        "evaluation_uses_oracle_labels": True,
        "training_uses_oracle_labels": False,
        "config": vars(args),
        "elapsed_sec": float(perf_counter() - started),
    }
    (output_dir / "warm_start_loss_autopsy.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "arms": arms,
                "summary_rows": len(summary_rows),
                "elapsed_sec": payload["elapsed_sec"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
