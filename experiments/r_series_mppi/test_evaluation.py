"""Contract tests before any formal controller outcomes are observed."""
import copy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from .common import *
from minimal_qrl.baselines.hybrid_astar_value import _signature_payload, _digest
from minimal_qrl.baselines.mppi import simulate_action_sequences
from experiments.four_path_execution.statistics import paired,aggregate

@pytest.fixture(scope='module')
def data():return load()

@pytest.mark.parametrize('arm',['R0','R1'])
def test_continuous_adapter_full_graph_unseen_and_terminal(data,arm):
    result=validate_model(data,arm)
    assert result['distance_scale']==1 and result['terminal_max']==0 and result['unseen_count']==97


def test_start_draw_is_fixed_balanced_and_score_free(data):
    a=build_states(data.env);b=build_states(data.env);assert a==b
    assert len(a['records'])==32
    for region,n in [('inside',3),('west',1)]:
        for h in range(8):assert sum(r['stratum']==region and r['heading_bin']==h for r in a['records'])==n
    assert len({tuple(r['pose']) for r in a['records']})==32


def test_existing_oracle_cache_signature_and_zero_invalid_queries(data,tmp_path):
    env=data.env;env.reset(seed=0,options={'device_id':'u_trap_target','start':[4.,3.65,0.]})
    import shutil
    before=file_digest(OLD_CACHE);shutil.copy2(OLD_CACHE,tmp_path/OLD_CACHE.name)
    o=HybridAStarValueOracle(ORACLE_CONFIG,cache_dir=tmp_path);diag=o.begin_episode(env,seed=17)
    assert diag['oracle_value_cache_hit'] and diag['oracle_value_table_digest']==_digest(_signature_payload(env,ORACLE_CONFIG))
    grid,_=o._grid(env);idx=np.flatnonzero(np.isfinite(o._active_values)&(o._active_values>0))[0]
    terminal=env.sample_task_terminal_state(seed=91)
    x=np.stack([grid[idx],terminal,[-5.,-5.,0.]])
    v=o.batch_value(env,x);assert 0<v[0]<1e6 and v[1]==0 and v[2]==1e6
    assert o.end_episode()['oracle_value_unreachable_queries']==1
    assert file_digest(OLD_CACHE)==before
    env.observation_failure_cost+=.1
    assert _digest(_signature_payload(env,ORACLE_CONFIG))!=diag['oracle_value_table_digest']
    env.observation_failure_cost-=.1


def test_instrumentation_reset_sampler_and_positive_cost_exact(data,tmp_path):
    env=data.env;cache=OLD_CACHE.parent
    first=[]
    for arm in ARMS:
        logged=make(data,arm,cache,True);plain=make(data,arm,cache,False)
        obs,_=env.reset(seed=0,options={'device_id':'u_trap_target','start':[4.,3.65,.15]});goal=env.abstract_goal_observation()
        for c in (logged,plain):c.begin_episode(env,goal,314159)
        # Shared first candidates, and every action exactly unchanged by logging.
        first.append(logged.sample_candidates(env));plain.sample_candidates(env)
        for _ in range(4):
            a,da=logged.act(obs,env);b,db=plain.act(obs,env)
            np.testing.assert_array_equal(a,b);assert da['best_rollout_cost']==db['best_rollout_cost']
            obs,_,done,truncated,_=env.step(a)
            if done or truncated:break
        logged.begin_episode(env,goal,314159)
        np.testing.assert_array_equal(logged.sample_candidates(env),first[-1])
        assert not logged._nominal.any()
    for x in first:np.testing.assert_array_equal(x,first[0])
    # Verify terminal distance is ADDED without dt/negative scaling.
    env.reset(seed=0,options={'device_id':'u_trap_target','start':[4.,3.65,0.]})
    c=make(data,'R1',cache);c.begin_episode(env,env.abstract_goal_observation(),314159)
    rollout=simulate_action_sequences(env,env.state,c.sample_candidates(env));costs,_=c.score_candidates(env,rollout)
    mask=~(rollout['invalid']|rollout['success']);base=rollout['costs'].astype(float);base[rollout['invalid']]+=1e6
    np.testing.assert_allclose(costs[mask]-base[mask],np.concatenate(c.query_values),rtol=0,atol=1e-9)
    np.testing.assert_array_equal(costs[~mask],base[~mask])


def test_model_dynamics_real_costs_and_termination(data):
    env=data.env;rng=np.random.default_rng(142)
    for pose in ([4.8,3.65,0.],[2.1,4.5,2.],[.05,.2,3.],[7.3,3.65,3.]):
        env.reset(seed=1,options={'device_id':'u_trap_target','start':pose});bank=rng.uniform(-3,3,(12,10)).astype(np.float32)
        sim=simulate_action_sequences(env,env.state,bank)
        for i,actions in enumerate(bank):
            env.reset(seed=1,options={'device_id':'u_trap_target','start':pose});cost=0
            for j,a in enumerate(actions):
                _,r,done,truncated,info=env.step([a]);cost-=r
                assert abs(sum(info[k] for k in COST_KEYS)+r)<1e-12
                if done or truncated:break
            np.testing.assert_allclose(env.state,sim['final_states'][i],atol=4e-6,rtol=0)
            np.testing.assert_allclose(cost,sim['costs'][i],atol=5e-5,rtol=2e-6)
            assert sim['steps'][i]==j+1 and bool(sim['success'][i])==info['success']
            assert bool(sim['invalid'][i])==bool(info['collision'] or info['out_of_bounds'])
    env.reset(seed=1,options={'device_id':'u_trap_target','start':[4.,3.65,3.]});env._t=299
    sim=simulate_action_sequences(env,env.state,np.zeros((1,10),np.float32));_,_,done,truncated,info=env.step([0.])
    assert truncated and not done and sim['steps'][0]==1


def test_success_pairing_and_stage_denominators():
    a=[dict(state_id='a',planner_seed=1,success=True,cost=100,steps=80),dict(state_id='b',planner_seed=1,success=False,cost=1,steps=1)]
    b=[dict(a[0],cost=110),dict(a[1],success=True,cost=200,steps=140)]
    p=paired(a,b);assert p['improved']==1 and p['worsened']==0 and p['joint_success_cost_delta']==10
    inside=stage_metrics([[4,3,3],[2.7,3,3],[2.6,1.3,0],[5.8,1.3,0]],{})
    west=stage_metrics([[2.4,3,3],[2.6,1.3,0],[5.8,1.3,0]],{'prefix_stages':{'exit':True}})
    assert inside['exit_eligible'] and inside['exit_completed'] and inside['wrap_completed']
    assert not west['exit_eligible'] and west['exit_completed'] is None and west['cumulative_exit']
