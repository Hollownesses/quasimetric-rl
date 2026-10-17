import importlib.util
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.four_path_execution.common import *
from experiments.four_path_execution.prepare import build_states
from minimal_qrl.baselines import mppi
from minimal_qrl.envs.dubins_uav_2d import Obstacle


@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1)
    return load_data()


def test_adapter_all_five_seeds_and_both_stages(data):
    for arm in ARMS[2:]:
        result=validate_adapter(data,arm)
        assert result['distance_scale']==1. and result['recorded_pairs']==1009


def test_bank_is_deterministic_valid_and_has_heading(data):
    a=build_states(data);b=build_states(data)
    assert a==b
    assert {k:sum(r['stratum']==k for r in a['records']) for k in ('support','perturbed','holdout')}==dict(support=12,perturbed=12,holdout=8)
    assert all(len(r['pose'])==3 for r in a['records'])
    assert all(r['exclusion_distance']>.25 for r in a['records'] if r['stratum']=='holdout')


def test_rectangle_vectorization_matches_scalar_with_boundary_cases():
    rng=np.random.default_rng(913);box=Obstacle(2.8,5.7,1.4,1.9)
    values=rng.uniform([0,0,0,0],[10,10,10,10],(10000,4)).astype(np.float32)
    boundary=np.array([[2.8,1.4,2.8,1.4],[0,1.4,8,1.4],[0,1.9,8,1.9],[5.7,0,5.7,8],[2,0,2,8],[0,1,8,1]],np.float64)
    for arr in (values,boundary):
        expected=np.array([box.intersects_segment(*map(float,row)) for row in arr])
        np.testing.assert_array_equal(mppi._segment_hits_obstacle(*arr.T,box),expected)


def test_existing_controller_behavior_and_shared_scoring_unchanged(data,tmp_path):
    # Compare with the checked-in pre-change MPPI, including many closed-loop
    # nominal shifts; vectorized collision acceleration must not alter controls.
    source=subprocess.check_output(['git','show','6deee0e:minimal_qrl/baselines/mppi.py'],text=True)
    legacy=tmp_path/'legacy_mppi.py';legacy.write_text(source)
    import sys
    spec=importlib.util.spec_from_file_location('legacy_mppi_for_test',legacy)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    env=data.env
    for mode in ('none','model','qrl'):
        agent,_=adapter(data,checkpoint('decay_101'))
        kw=dict(horizon=9,num_samples=31,terminal_samples=9)
        original=old.MPPIController(old.MPPIConfig(**kw),terminal_mode=mode,qrl_agent=agent)
        current=mppi.MPPIController(mppi.MPPIConfig(**kw),terminal_mode=mode,qrl_agent=agent)
        obs,_=env.reset(seed=32,options={'device_id':'u_trap_target','start':[4.,3.65,3.14]})
        goal=env.abstract_goal_observation()
        original.begin_episode(env,goal,42);current.begin_episode(env,goal,42)
        for _ in range(5):
            before=current._nominal.copy();rng_state=current._rng.bit_generator.state
            actions=current.sample_candidates(env);rollout=mppi.simulate_action_sequences(env,env.state,actions)
            costs,weights=current.score_candidates(env,rollout)
            current._nominal=before;current._rng.bit_generator.state=rng_state
            a,diag=original.act(obs,env);b,newdiag=current.act(obs,env)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(b,np.array([np.sum(weights[:,None]*actions,axis=0).astype(np.float32)[0]]))
            assert diag['best_rollout_cost']==newdiag['best_rollout_cost']
            obs,_,done,timeout,_=env.step(b)
            if done or timeout:break


def test_vectorized_rollouts_match_real_environment(data):
    env=data.env;rng=np.random.default_rng(71)
    for pose in ([4.8,3.65,0.],[3.05,2.15,-3.],[.1,.2,3.14],[7.3,3.65,3.14]):
        env.reset(seed=41,options={'device_id':'u_trap_target','start':pose})
        actual_start=env.state.copy();actions=rng.uniform(-3,3,(12,20)).astype(np.float32)
        rolled=mppi.simulate_action_sequences(env,actual_start,actions,record_trajectory=True)
        for i,sequence in enumerate(actions):
            env.reset(seed=41,options={'device_id':'u_trap_target','start':pose});cost=0
            for j,action in enumerate(sequence):
                _,reward,done,timeout,info=env.step(np.array([action],np.float32));cost-=reward
                if done or timeout:break
            np.testing.assert_allclose(rolled['final_states'][i],env.state,atol=3e-6,rtol=0)
            np.testing.assert_allclose(rolled['costs'][i],cost,atol=5e-5,rtol=2e-6)
            assert rolled['steps'][i]==j+1
            assert bool(rolled['invalid'][i])==bool(info['collision'] or info['out_of_bounds'])
            assert bool(rolled['success'][i])==bool(info['success'])


def test_stage_accounting_does_not_count_prepassed_states():
    full=[[4,3,3],[2.7,3,3],[2.7,1.3,0],[5.8,1.3,0],[7,3,1]]
    a=stage_metrics(full,{})
    assert a['exit_completed'] and a['wrap_completed']
    b=stage_metrics(full[-2:],{'prefix_stages':dict(exit=True,wrap=True)})
    assert not b['exit_eligible'] and not b['wrap_eligible'] and b['exit_completed'] is None
    assert b['cumulative_exit'] and b['cumulative_wrap']
    collision=stage_metrics([[4,3,0],[5.1,3,0]],{})
    assert not collision['exit_completed'] and not collision['wrap_completed']


def test_success_conditioning_and_pairing_do_not_reward_cheap_failure():
    from experiments.four_path_execution.statistics import aggregate,paired
    base=dict(arm='none',state_id='a',planner_seed=1,success=True,cost=100.,steps=80,
        collision=False,out_of_bounds=False,timeout=False,exit_eligible=True,exit_completed=True,
        wrap_eligible=True,wrap_completed=True,cumulative_exit=True,cumulative_wrap=True,
        planning_seconds=1.,model_rollouts=80*256,model_transitions=1000)
    cheap=dict(base,state_id='b',success=False,cost=.1,steps=1,collision=True)
    summary=aggregate([base,cheap])
    assert summary['successful_cost_mean']==100 and summary['success_rate']==.5
    after=[dict(base,cost=90),dict(cheap,success=True,collision=False,cost=150)]
    comparison=paired([base,cheap],after)
    assert comparison['improved']==1 and comparison['joint_success_cost_delta']==-10
    assert comparison['both_success']==1
