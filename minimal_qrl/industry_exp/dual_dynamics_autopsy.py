#!/usr/bin/env python3
"""Screen dual updates, then run a Dual x Dynamics-gradient 2x2 autopsy.

All variants start from the same targeted-supervised checkpoint and consume the
same seeded full-graph minibatches.  Oracle values are used only by evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dual_dynamics_autopsy_mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "dual_dynamics_autopsy_xdg")
)

import numpy as np
import torch
from tqdm import tqdm

from minimal_qrl.baselines import HybridAStarConfig, HybridAStarValueOracle, MPPIConfig
from minimal_qrl.dataset import (
    FullGraphGoalSetQRLConfig,
    create_full_graph_goal_set_qrl_dataset,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.industry_exp.joint_feasible_iqe import build_joint_feasible_problem
from minimal_qrl.industry_exp.qrl_pipeline_metrics import build_mppi_ranking_contract
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)
from minimal_qrl.industry_exp.warm_start_loss_autopsy import (
    _flatten_scalars,
    _jsonable,
    _make_qrl,
    _summary_row,
    _write_csv,
    _critic_batch_info,
    evaluate_checkpoint,
)
from minimal_qrl.iqe_capacity import iqe_capacity_from_checkpoint
from quasimetric_rl.modules.quasimetric_critic.losses import CriticBatchInfo
from quasimetric_rl.modules.utils import LossResult


DEFAULT_DUAL_CANDIDATES = (
    "baseline",
    "fixed:0.1",
    "fixed:1",
    "fixed:10",
    "projected:0.0001:1",
    "projected:0.0001:5",
    "projected:0.0001:10",
)
DYNAMICS_MODES = ("shared", "head_only")


@dataclass(frozen=True)
class DualScheme:
    name: str
    mode: str
    fixed_lambda: float | None = None
    projected_lr: float | None = None
    updates_per_primal: int = 1
    initial_lambda: float = 0.01
    minimum_lambda: float = 1e-8
    maximum_lambda: float = 100.0

    @property
    def selectable(self) -> bool:
        return self.mode != "softplus_adam"


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    dual: DualScheme
    dynamics_gradient_mode: str


def _float_key(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_dual_scheme(
    specification: str,
    *,
    minimum_lambda: float = 1e-8,
    maximum_lambda: float = 100.0,
) -> DualScheme:
    """Parse baseline, fixed:<lambda>, or projected:<lr>:<updates>."""

    fields = str(specification).strip().lower().split(":")
    if fields == ["baseline"]:
        return DualScheme(
            name="baseline_softplus_adam",
            mode="softplus_adam",
            minimum_lambda=float(minimum_lambda),
            maximum_lambda=float(maximum_lambda),
        )
    if len(fields) == 2 and fields[0] == "fixed":
        value = float(fields[1])
        if value <= 0.0:
            raise ValueError("fixed lambda must be positive")
        return DualScheme(
            name=f"fixed_lambda_{_float_key(value)}",
            mode="fixed",
            fixed_lambda=value,
            initial_lambda=value,
            minimum_lambda=float(minimum_lambda),
            maximum_lambda=float(maximum_lambda),
        )
    if len(fields) == 3 and fields[0] == "projected":
        learning_rate = float(fields[1])
        updates = int(fields[2])
        if learning_rate <= 0.0 or updates <= 0:
            raise ValueError("projected dual lr and update count must be positive")
        return DualScheme(
            name=(
                f"projected_lr_{_float_key(learning_rate)}"
                f"_k_{updates}"
            ),
            mode="projected",
            projected_lr=learning_rate,
            updates_per_primal=updates,
            minimum_lambda=float(minimum_lambda),
            maximum_lambda=float(maximum_lambda),
        )
    raise ValueError(
        f"invalid dual scheme {specification!r}; expected baseline, "
        "fixed:<lambda>, or projected:<lr>:<updates>"
    )


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    return value if value > 20.0 else math.log(math.expm1(value))


def _dual_parameters(local_loss) -> dict[str, torch.nn.Parameter]:
    result = {"ordinary": local_loss.raw_lagrange_multiplier}
    if local_loss.raw_direct_goal_lagrange_multiplier is not None:
        result.update(
            {
                "direct_goal": local_loss.raw_direct_goal_lagrange_multiplier,
                "terminal_goal": local_loss.raw_terminal_goal_lagrange_multiplier,
            }
        )
    return result


def _set_lambda(parameter: torch.nn.Parameter, value: float) -> None:
    with torch.no_grad():
        parameter.copy_(
            torch.as_tensor(
                _inverse_softplus(value),
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )


def configure_dual_scheme(losses, scheme: DualScheme) -> None:
    """Initialize lambda values and disable autograd for manually managed duals."""

    initial = (
        float(scheme.fixed_lambda)
        if scheme.mode == "fixed"
        else float(scheme.initial_lambda)
    )
    for critic_losses in losses.critic_losses:
        for parameter in _dual_parameters(critic_losses.local_constraint).values():
            _set_lambda(parameter, initial)
            parameter.requires_grad_(scheme.mode == "softplus_adam")


@contextlib.contextmanager
def _temporarily_frozen(module: torch.nn.Module):
    parameters = list(module.parameters())
    previous = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous):
            parameter.requires_grad_(requires_grad)


def _latent_dynamics_result(
    critic_losses,
    batch,
    batch_info: CriticBatchInfo,
    *,
    gradient_mode: str,
) -> LossResult:
    if gradient_mode == "shared":
        return critic_losses.latent_dynamics(batch, batch_info)
    if gradient_mode != "head_only":
        raise ValueError(f"unknown latent-dynamics gradient mode {gradient_mode!r}")

    # Keep gradients through q(z_pred, z_next) to z_pred and hence the dynamics
    # head, while treating the encoder and quasimetric geometry as a fixed
    # teacher for this component.
    detached_info = CriticBatchInfo(
        critic=batch_info.critic,
        zx=batch_info.zx.detach(),
        zy=batch_info.zy.detach(),
    )
    with _temporarily_frozen(batch_info.critic.quasimetric_model):
        return critic_losses.latent_dynamics(batch, detached_info)


def _projected_dual_step(local_loss, local_result: LossResult, scheme: DualScheme) -> None:
    if scheme.mode != "projected":
        return
    assert scheme.projected_lr is not None
    parameters = _dual_parameters(local_loss)
    for family, parameter in parameters.items():
        info = local_result.info[family]
        residual = float(info["violation"].detach().cpu())
        current = float(torch.nn.functional.softplus(parameter).detach().cpu())
        updated = current + (
            float(scheme.projected_lr)
            * int(scheme.updates_per_primal)
            * residual
        )
        updated = min(
            float(scheme.maximum_lambda),
            max(float(scheme.minimum_lambda), updated),
        )
        _set_lambda(parameter, updated)


def optimize_variant_step(
    agent,
    losses,
    batch,
    *,
    scheme: DualScheme,
    dynamics_gradient_mode: str,
) -> LossResult:
    """Run one full-QRL primal step and the selected dual update."""

    results: dict[str, LossResult] = {}
    for index, (critic, critic_losses) in enumerate(
        zip(agent.critics, losses.critic_losses)
    ):
        uses_adam_dual = scheme.mode == "softplus_adam"
        dual_context = (
            critic_losses.lagrange_mult_optim.update_context(optimize=True)
            if uses_adam_dual
            else contextlib.nullcontext()
        )
        with critic_losses.critic_optim.update_context(optimize=True), dual_context:
            batch_info = _critic_batch_info(critic, batch)
            component_results = {
                "global_push": critic_losses.global_push(batch, batch_info),
                "local_constraint": critic_losses.local_constraint(batch, batch_info),
                "latent_dynamics": _latent_dynamics_result(
                    critic_losses,
                    batch,
                    batch_info,
                    gradient_mode=dynamics_gradient_mode,
                ),
                "abstract_goal_edge": critic_losses.abstract_goal_edge(
                    batch, batch_info
                ),
            }
            result = LossResult.combine(component_results)
            result.loss.backward()
        if scheme.mode == "projected":
            _projected_dual_step(
                critic_losses.local_constraint,
                component_results["local_constraint"],
                scheme,
            )
        elif scheme.mode == "fixed":
            for parameter in _dual_parameters(
                critic_losses.local_constraint
            ).values():
                _set_lambda(parameter, float(scheme.fixed_lambda))
        critic_losses.critic_sched.step()
        if uses_adam_dual:
            critic_losses.lagrange_mult_sched.step()
        results[f"critic_{index:02d}"] = result
    return LossResult.combine(results)


def _metric(row: Mapping[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    return default if value is None else float(value)


def summarize_candidate_trajectory(
    scheme: DualScheme,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    trained = [row for row in ordered if int(row["step"]) > 0]
    if not trained:
        raise ValueError(f"candidate {scheme.name} has no post-update evaluations")
    final = trained[-1]
    catastrophe_count = sum(
        _metric(row, "u_value_pearson", -math.inf) < 0.0
        or _metric(row, "u_successor_top1", -math.inf) < 0.25
        or _metric(row, "mppi_spearman", -math.inf) < 0.0
        for row in trained
    )
    return {
        "scheme": asdict(scheme),
        "selectable": scheme.selectable,
        "final_step": int(final["step"]),
        "catastrophe_count": int(catastrophe_count),
        "final_u_successor_top1": _metric(final, "u_successor_top1", -math.inf),
        "worst_u_successor_top1": min(
            _metric(row, "u_successor_top1", -math.inf) for row in trained
        ),
        "final_mppi_spearman": _metric(final, "mppi_spearman", -math.inf),
        "worst_mppi_spearman": min(
            _metric(row, "mppi_spearman", -math.inf) for row in trained
        ),
        "final_u_value_mae": _metric(final, "u_value_mae", math.inf),
        "worst_u_value_mae": max(
            _metric(row, "u_value_mae", math.inf) for row in trained
        ),
        "final_u_successor_regret": _metric(
            final, "u_successor_regret_mean", math.inf
        ),
        "final_mppi_oracle_regret": _metric(
            final, "mppi_oracle_regret_mean", math.inf
        ),
        "final_u_bellman_excess_p99": _metric(
            final, "u_bellman_excess_p99", math.inf
        ),
    }


SELECTION_METRICS = (
    ("final_u_successor_top1", True),
    ("worst_u_successor_top1", True),
    ("final_mppi_spearman", True),
    ("worst_mppi_spearman", True),
    ("final_u_value_mae", False),
    ("worst_u_value_mae", False),
    ("final_u_successor_regret", False),
    ("final_mppi_oracle_regret", False),
    ("final_u_bellman_excess_p99", False),
)


def select_best_dual_scheme(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select among non-baseline candidates by stability then rank aggregation."""

    ranked = [dict(candidate) for candidate in candidates if candidate["selectable"]]
    if not ranked:
        raise ValueError("dual screening needs at least one non-baseline candidate")
    for candidate in ranked:
        candidate["metric_ranks"] = {}
    for metric, higher_is_better in SELECTION_METRICS:
        ordered = sorted(
            ranked,
            key=lambda item: (
                -float(item[metric]) if higher_is_better else float(item[metric]),
                item["scheme"]["name"],
            ),
        )
        for rank, candidate in enumerate(ordered, start=1):
            candidate["metric_ranks"][metric] = rank
    for candidate in ranked:
        candidate["rank_sum"] = sum(candidate["metric_ranks"].values())
    ranked.sort(
        key=lambda item: (
            int(item["catastrophe_count"]),
            int(item["rank_sum"]),
            -float(item["final_u_successor_top1"]),
            item["scheme"]["name"],
        )
    )
    return {
        "selection_rule": {
            "primary": "fewest catastrophic evaluated checkpoints",
            "catastrophe_definition": (
                "u_value_pearson < 0 or u_successor_top1 < 0.25 "
                "or mppi_spearman < 0"
            ),
            "secondary": "lowest equal-weight ordinal rank sum",
            "rank_metrics": [
                {"name": metric, "higher_is_better": higher}
                for metric, higher in SELECTION_METRICS
            ],
            "baseline_is_reference_only": True,
        },
        "best_scheme": ranked[0]["scheme"],
        "ranking": ranked,
        "all_candidates": list(candidates),
    }


def compute_2x2_effects(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return cell values, two main effects, and the interaction by step."""

    cell_names = (
        "baseline_dual__shared_dynamics",
        "baseline_dual__head_only_dynamics",
        "selected_dual__shared_dynamics",
        "selected_dual__head_only_dynamics",
    )
    by_step: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), {})[str(row["variant"])] = row
    effects: list[dict[str, Any]] = []
    excluded = {
        "variant",
        "dual_scheme",
        "dual_mode",
        "dynamics_gradient_mode",
        "step",
    }
    for step in sorted(by_step):
        cells = by_step[step]
        if any(name not in cells for name in cell_names):
            raise ValueError(f"incomplete 2x2 cells at step {step}")
        metric_names = [key for key in cells[cell_names[0]] if key not in excluded]
        for metric in metric_names:
            raw_values = [cells[name].get(metric) for name in cell_names]
            if any(value is None for value in raw_values):
                continue
            try:
                baseline_shared, baseline_head, selected_shared, selected_head = (
                    float(value) for value in raw_values
                )
            except (TypeError, ValueError):
                continue
            dual_effect = 0.5 * (
                selected_shared + selected_head - baseline_shared - baseline_head
            )
            dynamics_effect = 0.5 * (
                baseline_head + selected_head - baseline_shared - selected_shared
            )
            interaction = (
                selected_head
                - selected_shared
                - baseline_head
                + baseline_shared
            )
            effects.append(
                {
                    "step": int(step),
                    "metric": metric,
                    "baseline_shared": baseline_shared,
                    "baseline_head_only": baseline_head,
                    "selected_shared": selected_shared,
                    "selected_head_only": selected_head,
                    "selected_dual_main_effect": dual_effect,
                    "head_only_main_effect": dynamics_effect,
                    "dual_x_head_only_interaction": interaction,
                }
            )
    return effects


def _scheme_from_mapping(payload: Mapping[str, Any]) -> DualScheme:
    fields = {
        key: payload[key]
        for key in DualScheme.__dataclass_fields__
        if key in payload
    }
    return DualScheme(**fields)


def _variants(args) -> tuple[list[ExperimentVariant], dict[str, Any] | None]:
    if args.experiment == "dual_screen":
        schemes = [
            parse_dual_scheme(
                spec,
                minimum_lambda=float(args.projected_dual_min),
                maximum_lambda=float(args.projected_dual_max),
            )
            for spec in args.dual_candidates
        ]
        names = [scheme.name for scheme in schemes]
        if len(names) != len(set(names)):
            raise ValueError("dual candidates must be unique")
        return (
            [
                ExperimentVariant(
                    name=scheme.name,
                    dual=scheme,
                    dynamics_gradient_mode="shared",
                )
                for scheme in schemes
            ],
            None,
        )

    if not args.selected_dual_json:
        raise ValueError("orthogonal_2x2 requires --selected-dual-json")
    selection_path = Path(args.selected_dual_json)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = _scheme_from_mapping(selection["best_scheme"])
    if selected.mode == "softplus_adam":
        raise ValueError("selected Dual must differ from the baseline for a 2x2")
    baseline = parse_dual_scheme(
        "baseline",
        minimum_lambda=float(args.projected_dual_min),
        maximum_lambda=float(args.projected_dual_max),
    )
    variants = []
    for dual_factor, scheme in (("baseline", baseline), ("selected", selected)):
        for dynamics_mode in DYNAMICS_MODES:
            variants.append(
                ExperimentVariant(
                    name=f"{dual_factor}_dual__{dynamics_mode}_dynamics",
                    dual=scheme,
                    dynamics_gradient_mode=dynamics_mode,
                )
            )
    return variants, {
        "path": str(selection_path.resolve()),
        "selection": selection,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("dual_screen", "orthogonal_2x2"),
        required=True,
    )
    parser.add_argument("--scenario-config", required=True)
    parser.add_argument("--task-bank", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-dual-json", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-id", default="u_trap_target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-critics", type=int, default=2)
    parser.add_argument("--critic-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--scheduler-total-steps", type=int, default=20_000)
    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=(0, 10, 25, 50, 75, 100, 150, 250),
    )
    parser.add_argument(
        "--dual-candidates", nargs="+", default=DEFAULT_DUAL_CANDIDATES
    )
    parser.add_argument("--dual-lr", type=float, default=5e-3)
    parser.add_argument("--projected-dual-min", type=float, default=1e-8)
    parser.add_argument("--projected-dual-max", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
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
    variants, selected_dual_provenance = _variants(args)
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

    def make_loaded_qrl(scheme: DualScheme):
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
        configure_dual_scheme(losses, scheme)
        return agent, losses

    summary_rows: list[dict[str, Any]] = []
    variant_payloads: dict[str, Any] = {}
    candidate_summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_started = perf_counter()
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        agent, losses = make_loaded_qrl(variant.dual)
        critic = agent.critics[int(args.critic_index)]
        initial_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in critic.named_parameters()
        }
        metrics_history: list[dict[str, Any]] = []
        variant_summary_rows: list[dict[str, Any]] = []
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
            metrics_history.append({"step": int(step), "metrics": metrics})
            base_row = _summary_row(variant.name, int(step), metrics)
            row = {
                "variant": base_row.pop("arm"),
                "dual_scheme": variant.dual.name,
                "dual_mode": variant.dual.mode,
                "dynamics_gradient_mode": variant.dynamics_gradient_mode,
                **base_row,
            }
            summary_rows.append(row)
            variant_summary_rows.append(row)
            if args.save_eval_checkpoints:
                torch.save(
                    {
                        "optim_steps": int(step),
                        "agent": agent.state_dict(),
                        "losses": losses.state_dict(),
                        "training_mode": args.experiment,
                        "variant": variant.name,
                        "dual_scheme": asdict(variant.dual),
                        "dynamics_gradient_mode": variant.dynamics_gradient_mode,
                        "model_capacity": capacity.to_dict(),
                    },
                    variant_dir / f"checkpoint_{int(step):05d}.pth",
                )

        run_eval(0)
        dataloader = dataset.get_dataloader(
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=0,
            full_graph_stratified_constraints=True,
        )
        step = 0
        progress = tqdm(total=int(args.steps), desc=variant.name)
        while step < int(args.steps):
            for batch in dataloader:
                batch = batch.to(device)
                loss_result = optimize_variant_step(
                    agent,
                    losses,
                    batch,
                    scheme=variant.dual,
                    dynamics_gradient_mode=variant.dynamics_gradient_mode,
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

        _write_csv(variant_dir / "train_loss_history.csv", train_history)
        torch.save(
            {
                "optim_steps": int(args.steps),
                "agent": agent.state_dict(),
                "losses": losses.state_dict(),
                "training_mode": args.experiment,
                "variant": variant.name,
                "dual_scheme": asdict(variant.dual),
                "dynamics_gradient_mode": variant.dynamics_gradient_mode,
                "model_capacity": capacity.to_dict(),
                "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
            },
            variant_dir / "checkpoint_final.pth",
        )
        variant_payload = {
            "variant": variant.name,
            "dual_scheme": asdict(variant.dual),
            "dynamics_gradient_mode": variant.dynamics_gradient_mode,
            "checkpoints": metrics_history,
            "elapsed_sec": float(perf_counter() - variant_started),
        }
        (variant_dir / "metrics.json").write_text(
            json.dumps(_jsonable(variant_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        variant_payloads[variant.name] = variant_payload
        if args.experiment == "dual_screen":
            candidate_summaries.append(
                summarize_candidate_trajectory(variant.dual, variant_summary_rows)
            )
        del agent, losses

    stem = (
        "dual_screen" if args.experiment == "dual_screen" else "dual_dynamics_2x2"
    )
    _write_csv(output_dir / f"{stem}_summary.csv", summary_rows)
    selection = None
    if args.experiment == "dual_screen":
        selection = select_best_dual_scheme(candidate_summaries)
        selection.update(
            {
                "screening_steps": int(args.steps),
                "eval_steps": eval_steps,
                "output_dir": str(output_dir.resolve()),
            }
        )
        (output_dir / "best_dual_scheme.json").write_text(
            json.dumps(_jsonable(selection), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        effects = compute_2x2_effects(summary_rows)
        _write_csv(output_dir / "dual_dynamics_2x2_effects.csv", effects)
        (output_dir / "dual_dynamics_2x2_effects.json").write_text(
            json.dumps(_jsonable(effects), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    payload = {
        "experiment": args.experiment,
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "checkpoint_training_mode": (
            checkpoint.get("training_mode", "unknown")
            if isinstance(checkpoint, Mapping)
            else "unknown"
        ),
        "variants": variant_payloads,
        "selection": selection,
        "selected_dual_provenance": selected_dual_provenance,
        "graph": problem.graph_stats,
        "training_dataset": collection_stats.get("full_graph_goal_set_qrl"),
        "mppi_contract": contract_provenance,
        "evaluation_uses_oracle_labels": True,
        "training_uses_oracle_labels": False,
        "config": vars(args),
        "elapsed_sec": float(perf_counter() - started),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "output_dir": str(output_dir.resolve()),
                    "experiment": args.experiment,
                    "variants": [variant.name for variant in variants],
                    "best_dual_scheme": (
                        selection["best_scheme"] if selection is not None else None
                    ),
                    "elapsed_sec": payload["elapsed_sec"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
