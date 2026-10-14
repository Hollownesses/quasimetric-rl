import numpy as np
import pytest

from minimal_qrl.eval import task_path_support_audit as audit
from minimal_qrl.test_global_push_sampling import dataset, episode


def test_failed_rollout_does_not_connect_to_its_independently_sampled_anchor():
    raw = dataset(episode([0, 1], context=0), episode([2, 999], context=0, anchor=True)).raw_data
    eps, anchors, sources = audit.physical_episodes(raw)
    for scope in ['strict_context', 'same_task_goal']:
        reachable = audit.exact_reachable(raw, eps, anchors, sources, scope=scope)
        assert not reachable[0]


def test_exact_cross_context_join_can_connect_fragments_without_spatial_rounding():
    raw = dataset(episode([0, 1], context=0), episode([2, 999], context=0, anchor=True),
                  episode([1, 2], context=1, success=True),
                  episode([2, 999], context=1, anchor=True)).raw_data
    eps, anchors, sources = audit.physical_episodes(raw)
    strict = audit.exact_reachable(raw, eps, anchors, sources, scope='strict_context')
    shared = audit.exact_reachable(raw, eps, anchors, sources, scope='same_task_goal')
    assert not strict[0] and shared[0]
    assert strict[2] and shared[2]


@pytest.mark.parametrize('change', ['position', 'heading', 'goal', 'device'])
def test_nearby_or_different_task_nodes_cannot_be_exact_joins(change):
    data = dataset(episode([0, 1], context=0), episode([2, 999], context=0, anchor=True),
                   episode([1, 2], context=1, success=True), episode([2, 999], context=1, anchor=True))
    raw = data.raw_data
    if change == 'position':
        raw.all_observations[4, 0] += .001
    elif change == 'heading':
        raw.all_observations[4, 3] += .001
    elif change == 'goal':
        raw.transition_infos['task_goal_observations'][2:, 0] = 998
    else:
        raw.transition_infos['device_index'][2:] = 1
    eps, anchors, sources = audit.physical_episodes(raw)
    shared = audit.exact_reachable(raw, eps, anchors, sources, scope='same_task_goal')
    assert not shared[0]


@pytest.mark.parametrize('change', ['terminal', 'context', 'goal', 'abstract'])
def test_internal_resets_are_rejected_rather_than_inventing_chains(change):
    raw = dataset(episode([0, 1, 2])).raw_data
    if change == 'terminal':
        raw.terminals[0] = True
    elif change == 'context':
        raw.transition_infos['context_id'][1] = 7
    elif change == 'goal':
        raw.transition_infos['task_goal_observations'][1, 0] = 998
    else:
        raw.transition_infos['abstract_goal_edge'][1] = True
    with pytest.raises(ValueError, match='ambiguous'):
        audit.physical_episodes(raw)


def test_funnel_requires_order_and_joint_terminal_conditions():
    result = audit.ordered_stages({'exit': np.array([False, True, False, False]),
                                   'around': np.array([True, False, False, False]),
                                   'goal': np.array([False, False, False, True])}, 0)
    assert result == {'exit': 1, 'around': None, 'goal': None}
    # Range/heading true at separate states cannot stand in for a joint goal.
    range_ok = np.array([True, False])
    heading_ok = np.array([False, True])
    assert audit.ordered_stages({'goal': range_ok & heading_ok}, 0)['goal'] is None


def test_nearest_heading_wrap_and_proximity_is_not_exact_evidence():
    def observation(x, theta):
        return [x, 0, np.cos(theta), np.sin(theta)]
    query = np.array(observation(0, np.pi-.01))
    candidates = np.array([observation(.001, 0), observation(.05, -np.pi+.01)])
    near = audit.nearest_state(query, candidates, heading_tolerance=.03)
    assert near['candidate_index'] == 1
    assert near['position_distance'] == pytest.approx(.05)
    assert not near['exact_full_observation_match']
    assert audit.nearest_state(query, candidates, heading_tolerance=.001) is None
    assert audit.nearest_state(query, np.empty((0, 4))) is None


def test_anchor_validity_requires_geometry_goal_zero_cost_and_terminal():
    class Env:
        device_ids = ['task']
        observation_to_state = staticmethod(lambda o: o[:3])
        is_task_feasible_for_task = staticmethod(lambda state, _: state[0] == 2)
    raw = dataset(episode([2, 999], anchor=True), episode([1, 999], anchor=True),
                  episode([2, 998], anchor=True), episode([2, 999], anchor=True)).raw_data
    raw.rewards[-1] = -1
    _, anchors, sources = audit.physical_episodes(raw)
    valid, records = audit.valid_anchor_rows(raw, anchors, sources, Env())
    assert valid == {0}
    assert not records[1]['legal_source_recomputed']
    assert not records[2]['valid_zero_cost_anchor']
    assert not records[3]['valid_zero_cost_anchor']


def test_full_support_rejects_sampled_anchor_as_failed_U_endpoint():
    from minimal_qrl.dataset import QRLExploreConfig, collect_qrl_explore_episode_pair
    from minimal_qrl.envs import CommInspectionDubinsUAV2D
    from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario
    from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs
    from quasimetric_rl.data import MultiEpisodeData
    scenario = build_diagnostic_scenario()
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    real, anchor, _ = collect_qrl_explore_episode_pair(
        env, start_state=np.array([4.8, 3.65, np.pi]), start_index=0,
        max_steps=5, seed=42, context_id=0, device_id='u_trap_target',
        config=QRLExploreConfig(straight_action_probability=1), budget_cutoff=True,
        include_global_push_pairs=False)
    raw = MultiEpisodeData.cat([real, anchor])
    support = audit.build_support(raw, env, scenario, 'u_trap_target')
    assert support['interior'].sum() == 5
    assert not support['witnessed'].any()
    assert not support['shared'].any()
    record = support['episode_records'][0]
    assert record['has_valid_anchor_in_context']
    assert not record['endpoint_equals_legal_anchor']
    assert record['outcome'] == 'budget_cutoff'
    assert not record['end_task_feasible']
    assert record['last_ordered_stage'] == 'before_west_exit'
    np.testing.assert_allclose(support['tail_costs'][:5], np.cumsum((-raw.rewards[:5].numpy())[::-1])[::-1])
    with pytest.raises(ValueError, match='No recorded physical trajectories'):
        audit.build_support(raw, env, scenario, 'easy_south')
    raw.all_observations[1, 0] += .02
    with pytest.raises(ValueError, match='dynamics mismatch'):
        audit.build_support(raw, env, scenario, 'u_trap_target')


def test_audit_phase_passes_parameters_without_regenerating_scenario(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import json
    fake = tmp_path/'python'
    capture = tmp_path/'calls.jsonl'
    fake.write_text(f'#!{sys.executable}\nimport os,sys,json\n'
                    'with open(os.environ["CAPTURE_ARGS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = {**os.environ, 'PHASE': 'task_path_support_audit', 'PYTHON_BIN': str(fake),
           'OUTPUT_ROOT': str(tmp_path), 'QRL_CHECKPOINTS': 'control.pth\nb1.pth',
           'REPLAY_AUDIT_CACHE': 'verified_replay.pt', 'TASK_PATH_HEADING_TOLERANCE_DEG': '10',
           'CAPTURE_ARGS': str(capture)}
    result = subprocess.run(['bash', str(Path(__file__).with_name('run_comm_inspection_diagnostic.sh'))],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(r) for r in capture.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][:2] == ['-m', 'minimal_qrl.eval.task_path_support_audit']
    assert calls[0][calls[0].index('--checkpoints')+1:calls[0].index('--output-dir')] == ['control.pth', 'b1.pth']
    assert calls[0][calls[0].index('--replay-cache')+1] == 'verified_replay.pt'
    assert calls[0][calls[0].index('--heading-tolerance-deg')+1] == '10'
