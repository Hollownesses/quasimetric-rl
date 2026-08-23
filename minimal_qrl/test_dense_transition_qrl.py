from __future__ import annotations

import numpy as np
import torch

from quasimetric_rl.data.base import DenseTransitionBatchSampler

from minimal_qrl.dataset import (
    DENSE_U_TRAP_STRATA,
    DenseUTrapTransitionConfig,
    build_dense_u_trap_state_bank,
    create_dense_u_trap_transition_dataset,
)
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs


FAILED_STARTS = (
    (4.75, 3.15, 3.0),
    (4.75, 4.15, -3.0),
    (4.9, 3.45, 3.1),
    (4.9, 3.9, -3.1),
)


def _config(*, resolution=0.5, heading_bins=8, primitive_scales=(-1.0, 1.0)):
    scenario = build_diagnostic_scenario()
    return DenseUTrapTransitionConfig(
        position_resolution=resolution,
        heading_bins=heading_bins,
        primitive_steps=2,
        primitive_scales=primitive_scales,
        diagnostic_regions=scenario["metadata"]["exploration_diagnostic_regions"],
        failed_start_states=FAILED_STARTS,
    )


def test_dense_u_trap_lattice_covers_all_five_strata_without_oracle_values():
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    states, strata = build_dense_u_trap_state_bank(env, _config())
    assert states.shape[1] == 3
    assert set(np.unique(strata)) == set(range(len(DENSE_U_TRAP_STRATA)))
    assert all(env.is_valid_state(state) for state in states)
    assert len(np.unique(states, axis=0)) == len(states)


def test_dense_transitions_are_real_dynamics_edges_with_no_oracle_labels():
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    config = _config(resolution=1.0, heading_bins=8, primitive_scales=(0.0,))
    states, _strata = build_dense_u_trap_state_bank(env, config)
    stats = {}
    episodes = list(
        create_dense_u_trap_transition_dataset(
            env,
            config,
            seed=17,
            collection_stats=stats,
        )
    )
    assert len(episodes) == len(states)
    assert stats["dense_u_trap"]["oracle_used_for_training"] is False
    assert len(states) <= stats["dense_u_trap"]["transitions"] <= 2 * len(states)
    first = episodes[0]
    assert 1 <= first.num_transitions <= config.primitive_steps
    assert bool(first.transition_infos["dense_u_trap_transition"][0])
    assert float(first.rewards[0]) <= 0.0
    assert "oracle_value" not in first.transition_infos
    source = env.observation_to_state(first.all_observations[0].numpy())
    target = env.observation_to_state(first.all_observations[1].numpy())
    assert np.linalg.norm(target[:2] - source[:2]) <= env.v * env.dt + 1e-5


def test_dense_batch_sampler_is_exactly_half_local_and_stratum_balanced():
    dense_mask = torch.tensor([False] * 20 + [True] * 25)
    dense_strata = torch.tensor([-1] * 20 + list(range(5)) * 5)
    sampler = DenseTransitionBatchSampler(
        dense_mask=dense_mask,
        dense_strata=dense_strata,
        batch_size=20,
        local_fraction=0.5,
        num_batches=2,
    )
    for batch in sampler:
        batch_mask = dense_mask[batch]
        assert int(batch_mask.sum()) == 10
        counts = torch.bincount(dense_strata[batch][batch_mask], minlength=5)
        assert torch.equal(counts, torch.full((5,), 2))
