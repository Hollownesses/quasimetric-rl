import json
import numpy as np
import pytest
import torch
from .common import SCENARIO,make_env
from .collect import default_config,sample_start,batch_metrics,collect_one,collect
from .data import load_training_data,audit_training_data


def test_random_actions_independent_of_start_and_failure_has_no_anchor():
    env=make_env(json.loads(SCENARIO.read_text()));cfg=default_config()
    a,ra,_=collect_one(env,np.array([7.9,3.65,3.],np.float32),34,0,20,cfg,0,0,0,0)
    b,rb,_=collect_one(env,np.array([5.09,3.65,0.],np.float32),34,1,20,cfg,6,0,0,0)
    n=min(ra['steps'],rb['steps']);assert torch.equal(a[0].actions[:n],b[0].actions[:n])
    assert rb['outcome']!='success' and len(b)==1
    for ep in a+b:
        assert not ep.transition_infos['goal_return_cost'].any()
        assert not ep.transition_infos['goal_return_mask'].any()
        assert not ep.transition_infos['teacher_guided'].any()


def test_start_filter_and_batch_gate():
    env=make_env(json.loads(SCENARIO.read_text()));cfg=default_config();rng=np.random.default_rng(8)
    excluded=np.array([[7.55,3.65,0]],np.float32);seen=set();records=[]
    for i in range(24):
        s,_=sample_start(env,0,rng,excluded,cfg,seen)
        assert not env.is_task_feasible(s) and np.linalg.norm(s[:2]-excluded[0,:2])>.25
        records.append(dict(start=s.tolist(),outcome='success' if i<4 else 'timeout'))
    assert len(seen)==24 and batch_metrics(records,cfg)['passed']
    assert not batch_metrics(records[:8],cfg)['passed']
    assert not batch_metrics([dict(start=r['start'],outcome='timeout') for r in records],cfg)['passed']


@pytest.fixture
def collected(tmp_path):
    cfg=default_config();cfg.update(curriculum_step_budget=1200,max_rounds=2,
        diagnostic_steps_per_stage=120,diagnostic_stages=[5,6])
    root=tmp_path/'collection';collect(root,cfg);return root


def test_budget_failure_masks_and_loader(collected):
    d=load_training_data(collected);w=load_training_data(collected,task_mode='witnessed')
    s=json.loads((collected/'collection_summary.json').read_text());audit=audit_training_data(d)
    assert len(d.original)==s['total_real_steps']<=1440
    assert s['diagnostic_steps']==240
    assert d.raw.transition_infos['global_push_task_mask'][d.original].all()
    assert torch.equal(w.raw.transition_infos['global_push_task_mask'],d.raw.transition_infos['global_push_task_witness_mask'])
    assert audit['failed_episode_anchors']==0 and audit['failed_episodes']>0
    assert not torch.equal(d.raw.transition_infos['global_push_task_mask'],w.raw.transition_infos['global_push_task_mask'])
    assert torch.equal(d.pair,w.pair)
