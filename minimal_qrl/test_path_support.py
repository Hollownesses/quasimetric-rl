"""Positive control adds physical support, never oracle-return targets."""
import argparse
import json
import uuid

import numpy as np
import pytest

from minimal_qrl import path_support as ps
from minimal_qrl.dataset import QRLExploreConfig, create_dataset
from minimal_qrl.global_push_sampling import configure_global_push_sampling, replay_fingerprint
from minimal_qrl.industry_exp.diagnostic_scenario import build_diagnostic_scenario, build_diagnostic_task_bank
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.test_comm_inspection_oracle_bank import make_env
from quasimetric_rl.data import Dataset, MultiEpisodeData, register_offline_env


@pytest.fixture
def bank_path(tmp_path):
    env = make_env()
    start, device = [2.,4.,0.], 'device_a'
    env.reset(seed=5,options={'device_id':device,'start':start})
    actions=[]
    for _ in range(30):
        _,_,done,timeout,info=env.step(np.array([0.],np.float32))
        actions.append([0.])
        if done or timeout:
            break
    assert info['success']
    trajectory=ps.replay_actions(env,start,actions,device_id=device,seed=5)
    record={'candidate_index':0,'start':start,'device_id':device,'seed':5,**trajectory,
            'diagnostic_path_cost_only':float(-sum(trajectory['rewards']))}
    bank={'schema_version':1,'purpose':'planner_path_support_positive_control',
          'expert_free':False,'path_cost_targets_added':False,'complete':True,
          'environment_signature':ps.environment_signature(env),
          'collection_config':{'paths':1,'exclusion_radius':.25},
          'excluded_starts':[[0.,0.,0.]],'records':[record],
          'summary':{'added_real_transitions':len(actions),'added_anchors':1}}
    path=tmp_path/'bank.json'
    ps.save_bank(path,bank)
    return path


def test_bank_has_exact_physical_chain_legal_anchor_and_no_return_target(bank_path):
    episodes=list(ps.bank_episodes(make_env(),bank_path,context_start=7,include_global_push_pairs=False))
    real, anchor=episodes
    assert real.terminals[-1] and not real.timeouts.any()
    assert np.array_equal(real.all_observations[-1],anchor.all_observations[0])
    assert anchor.rewards[0] == 0
    assert real.transition_infos['teacher_guided'].all()
    assert real.transition_infos['planner_path_support'].all()
    assert not real.transition_infos['goal_return_mask'].any()
    assert not real.transition_infos['goal_return_cost'].any()
    assert all(int(e.transition_infos['context_id'][0])==7 for e in episodes)
    key=uuid.uuid4().hex
    register_offline_env(key,key,load_episodes_fn=lambda:iter(episodes),create_env_fn=make_env)
    dataset=Dataset(key,key,future_observation_discount=.99)
    configure_global_push_sampling(dataset,state_mode='trajectory')
    assert dataset.raw_data.transition_infos['global_push_task_witness_mask'][:-1].all()
    assert not dataset.raw_data.transition_infos['global_push_task_witness_mask'][-1]


def test_append_preserves_every_base_edge_and_uses_new_contexts(bank_path):
    config=QRLExploreConfig(attempted_env_steps=40,start_position_resolution=2.,start_heading_bins=2)
    base=list(create_dataset(make_env(),seed=42,max_steps_per_episode=10,qrl_explore_config=config,include_global_push_pairs=False))
    stats={}
    combined=list(create_dataset(make_env(),seed=42,max_steps_per_episode=10,qrl_explore_config=config,
             path_support_bank=str(bank_path),include_global_push_pairs=False,collection_stats=stats))
    before,after=MultiEpisodeData.cat(base),MultiEpisodeData.cat(combined)
    assert len(combined)==len(base)+2
    assert ps.base_fingerprint(after,len(base))==replay_fingerprint(before)
    assert replay_fingerprint(after)!=replay_fingerprint(before)
    assert int(combined[-2].transition_infos['context_id'].min())>int(before.transition_infos['context_id'].max())
    assert stats['path_support']['expert_free'] is False
    assert stats['stored_real_transitions']==40
    assert stats['path_support']['base_episode_count']==len(base)


@pytest.mark.parametrize('change',['digest','reward','start','terminal_suffix','environment','incomplete'])
def test_bank_rejects_tampering_and_invalid_paths(bank_path,change):
    bank=ps.load_bank(bank_path)
    env=make_env()
    if change=='digest':
        bank['records'][0]['rewards'][0]-=1
        bank_path.write_text(json.dumps(bank))
    else:
        if change=='reward': bank['records'][0]['rewards'][0]-=1
        if change=='start': bank['excluded_starts']=[bank['records'][0]['start']]
        if change=='terminal_suffix': bank['records'][0]['actions'].append([0.])
        if change=='environment': env.comm_bias+=1
        if change=='incomplete': bank['complete']=False
        ps.save_bank(bank_path,bank)
    with pytest.raises(ValueError):
        list(ps.bank_episodes(env,bank_path,context_start=0,include_global_push_pairs=False))


def test_training_starts_exclude_all_eval_splits_and_probe_positions():
    scenario=build_diagnostic_scenario()
    bank=build_diagnostic_task_bank(scenario)
    excluded=ps.excluded_starts(scenario,bank)
    env=CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.set_task_by_device_id('u_trap_target')
    first=ps.training_starts(env,scenario,excluded,seed=31,radius=.25)
    assert first==ps.training_starts(env,scenario,excluded,seed=31,radius=.25)
    assert len(first)>=4
    assert all(np.all(np.linalg.norm(excluded[:,:2]-np.asarray(s)[:2],axis=1)>.25) for s in first)
    assert first[0][1]<3.65<first[1][1]


def protocol_args():
    return argparse.Namespace(path_support_bank='bank.json',env_type='comm_inspection_dubins_uav',
        comm_dataset_mode='qrl_explore',qrl_local_constraint_mode='goal_set_barrier',
        global_push_objective='linear',global_push_task_sampling='all',global_push_state_sampling='trajectory',
        global_push_abstract_goal_ratio=.6,global_push_state_goal_ratio=.4,qrl_temporal_constraint_weight=0.,
        qrl_goal_return_constraint_weight=0.,qrl_nstep_goal_constraint_weight=0.,
        qrl_mqe_waypoint_consistency_weight=0.,init_checkpoint=None)


@pytest.mark.parametrize('key,value',[('init_checkpoint','old.pth'),('qrl_goal_return_constraint_weight',1.),
    ('global_push_task_sampling','witnessed'),('qrl_temporal_constraint_weight',1.),('qrl_local_constraint_mode','kkt_functional')])
def test_training_protocol_rejects_extra_supervision_or_old_optimizer(key,value):
    args=protocol_args()
    ps.validate_training_protocol(args)
    setattr(args,key,value)
    with pytest.raises(ValueError,match='unchanged'):
        ps.validate_training_protocol(args)


def test_bank_is_required_for_data_append_without_explore_base(bank_path):
    with pytest.raises(ValueError,match='requires qrl_explore'):
        list(create_dataset(make_env(),path_support_bank=str(bank_path)))


def test_four_phases_change_only_data_bank_and_state_pair_sampling(tmp_path, bank_path):
    from minimal_qrl.test_barrier_push_support import capture_phase, arg
    normalized = []
    for data in ('baseline', 'positive'):
        for pair, sampling in [('control', 'environment'), ('b1', 'trajectory')]:
            result, args = capture_phase(tmp_path, f'{data}_{pair}',
                PHASE=f'train_qrl_path_support_{data}_{pair}', PATH_SUPPORT_BANK=str(bank_path))
            assert result.returncode == 0, result.stderr
            assert ('--path-support-bank' in args) == (data == 'positive')
            assert arg(args, 'global-push-state-sampling') == sampling
            assert arg(args, 'qrl-local-constraint-mode') == 'goal_set_barrier'
            assert arg(args, 'explore-local-safety-lookahead-steps') == '10'
            assert '--init-checkpoint' not in args
            for name in ('qrl-temporal-constraint-weight', 'qrl-goal-return-constraint-weight',
                         'qrl-nstep-goal-constraint-weight', 'qrl-mqe-waypoint-consistency-weight'):
                assert float(arg(args, name)) == 0
            if data == 'positive':
                i = args.index('--path-support-bank')
                del args[i:i+2]
            for name in ('output-dir', 'oracle-bank-dir', 'global-push-state-sampling'):
                args[args.index('--'+name)+1] = '<paired-difference>'
            normalized.append(args)
    assert all(a == normalized[0] for a in normalized)


def test_audit_reconstructs_and_caches_appended_provenance(bank_path, tmp_path, monkeypatch):
    from minimal_qrl.eval import replay_constraint_audit as audit
    config = QRLExploreConfig(attempted_env_steps=40, start_position_resolution=2., start_heading_bins=2)
    raw = MultiEpisodeData.cat(list(create_dataset(make_env(), seed=42, max_steps_per_episode=10,
        qrl_explore_config=config, path_support_bank=str(bank_path), include_global_push_pairs=False)))
    spec = {'expected_sha256': replay_fingerprint(raw), 'config': config,
            'args': argparse.Namespace(seed=42, max_steps_per_episode=10, path_support_bank=str(bank_path))}
    monkeypatch.setattr(audit, 'get_env_kwargs', lambda args: {})
    monkeypatch.setattr(audit, 'CommInspectionDubinsUAV2D', lambda **kwargs: make_env())
    for _ in range(2):  # First reconstruct; then load the verified cache.
        restored = audit.load_replay(spec, tmp_path/'replay.pt')
        assert np.array_equal(restored.transition_infos['planner_path_support'],
                              raw.transition_infos['planner_path_support'])
        assert replay_fingerprint(restored) == replay_fingerprint(raw)


@pytest.mark.parametrize('success', [True, False])
def test_collection_resume_is_frozen_and_counts_failed_planning(bank_path, tmp_path, monkeypatch, success):
    record = ps.load_bank(bank_path)['records'][0]
    scenario = build_diagnostic_scenario()
    scenario['metadata']['u_trap_local_navigability_probes']['device_id'] = 'device_a'
    monkeypatch.setattr(ps, 'excluded_starts', lambda *args: np.array([[0., 0., 0.]], np.float32))
    monkeypatch.setattr(ps, 'training_starts', lambda *args, **kwargs: [record['start']])
    calls = []
    class Planner:
        def __init__(self, config):
            self._actions = np.asarray(record['actions'], np.float32)
        def begin_episode(self, env, goal, seed):
            calls.append(seed)
            return {'planner_success': success, 'planned_cost': record['diagnostic_path_cost_only'] if success else float('inf'),
                    'simulated_env_steps': 123, 'initial_planning_time_sec': 1.,
                    'planner_search_stop_reason': 'goal_popped' if success else 'max_expansions'}
    monkeypatch.setattr(ps, 'HybridAStarController', Planner)
    path = tmp_path/'collected.json'
    for _ in range(2):
        if success:
            ps.collect_bank(make_env(), scenario, {}, path, paths=1, max_attempts=1)
        else:
            with pytest.raises(RuntimeError, match='incomplete'):
                ps.collect_bank(make_env(), scenario, {}, path, paths=1, max_attempts=1)
    assert len(calls) == 1
    bank = ps.load_bank(path, require_complete=False)
    assert bank['summary']['planner_simulated_env_steps'] == 123
    assert bank['summary']['rollout_env_steps'] == (len(record['actions']) if success else 0)
    assert bank['complete'] == success
    with pytest.raises(ValueError, match='provenance changed'):
        ps.collect_bank(make_env(), scenario, {}, path, paths=1, max_attempts=1, seed=99)
