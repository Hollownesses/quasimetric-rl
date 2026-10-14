"""Full safety horizon must remain independent of the action hold duration."""
from dataclasses import replace
import numpy as np
import pytest

from minimal_qrl import dataset as ds
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs


def test_short_hold_turns_before_entering_wall_dead_end():
    # A three-step straight prefix is collision-free, but leaves too little
    # turning room before the closed U wall. Ten-step lookahead detects it.
    config = ds.QRLExploreConfig(action_hold_min_steps=3, action_hold_max_steps=3,
                                 straight_action_probability=1., local_safety_lookahead_steps=10)
    result = {}
    for mode in ['legacy_action_hold', 'full_horizon']:
        env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(build_diagnostic_scenario()))
        episode, _, _ = ds.collect_qrl_explore_episode_pair(
            env, start_state=np.array([4.8, 3.65, 0.], np.float32), start_index=0,
            max_steps=3, seed=42, context_id=0, device_id='u_trap_target',
            config=replace(config, safety_lookahead_mode=mode), budget_cutoff=True,
            include_global_push_pairs=False)
        result[mode] = episode.actions.numpy()
    assert np.all(result['legacy_action_hold'] == 0.)
    assert np.all(np.abs(result['full_horizon']) == 3.)


def test_execution_never_exceeds_checked_horizon(monkeypatch):
    calls = []
    def safe(env, state, omega, steps):
        calls.append(steps)
        return steps
    monkeypatch.setattr(ds, '_locally_safe_steps', safe)
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(build_diagnostic_scenario()))
    ds.collect_qrl_explore_episode_pair(
        env, start_state=np.array([3.5, 3.65, np.pi], np.float32), start_index=0,
        max_steps=6, seed=42, context_id=0, device_id='u_trap_target',
        config=ds.QRLExploreConfig(action_hold_min_steps=10, action_hold_max_steps=10,
            straight_action_probability=1., local_safety_lookahead_steps=3),
        budget_cutoff=True, include_global_push_pairs=False)
    assert calls == [3, 3]


def test_unknown_safety_mode_cannot_silently_change_replay():
    with pytest.raises(ValueError, match='safety_lookahead_mode'):
        ds.QRLExploreConfig(safety_lookahead_mode='typo')
