import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from experiments.four_path_factorial import run as factorial
from experiments.four_path_refinement.run import restore, optimizer_digest
from experiments.four_path_reproducibility import run as experiment


@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1)
    return factorial.load_data()


def test_five_fresh_networks_without_loading_checkpoint(data):
    support_before=factorial.tensor_digest(data.nodes,data.cost,data.task_src,data.goal,data.pair)
    hashes=[];encoders=[];projectors=[]
    with patch.object(torch,'load',side_effect=AssertionError('initialization must not load weights')):
        for seed in experiment.SEEDS:
            model,solver,opt=experiment.fresh(data,seed)
            hashes.append(factorial.tensor_digest(*model.state_dict().values()))
            encoders.append(factorial.tensor_digest(*model.encoder.state_dict().values()))
            projectors.append(factorial.tensor_digest(*model.projector.state_dict().values()))
            assert not opt.state and not solver.multiplier.any()
            assert model.encoder.comm_goal_set_quotient
        repeated,_,_=experiment.fresh(data,experiment.SEEDS[0])
    assert len(set(hashes))==len(set(encoders))==len(set(projectors))==5
    assert factorial.tensor_digest(*repeated.state_dict().values())==hashes[0]
    assert factorial.tensor_digest(data.nodes,data.cost,data.task_src,data.goal,data.pair)==support_before
    for index in (0,1):
        old=factorial.IndexedIQE(data,'encoder',index)
        assert factorial.tensor_digest(*old.state_dict().values()) not in hashes


def test_stage_boundary_preserves_adam_and_dual_and_has_no_references(data,tmp_path):
    model,solver,opt=experiment.fresh(data,101)
    with patch.object(factorial,'graph_reference',side_effect=AssertionError('no reference during training')):
        phr=experiment.train_stage(model,data,solver,opt,101,'phr',2,0,tmp_path,io.StringIO(),io.StringIO())
        assert phr['adam_sha256']==optimizer_digest(opt)
        before=solver.multiplier.clone()
        loaded,dual,adam,saved=restore(data,tmp_path/'phr.pt')
        assert saved['initialization_seed']==101
        assert torch.equal(dual.multiplier,before)
        assert optimizer_digest(adam)==optimizer_digest(opt)
        experiment.train_stage(model,data,solver,opt,101,'decay',1,2,tmp_path,io.StringIO(),io.StringIO())
    assert all(int(state['step'])==3 for state in opt.state.values())
    assert opt.param_groups[0]['lr']==2e-6
    assert factorial.tensor_digest(*model.state_dict().values())!=phr['model_sha256']


def test_all_reference_evaluation_happens_after_final_model_selection(tmp_path):
    out=tmp_path/'seed'
    real_reference=factorial.graph_reference
    calls=[]
    def reference_after_stop(data):
        saved=torch.load(out/'repair/latest.pt',weights_only=True)
        assert saved['stop_reason'] is not None
        calls.append(saved['stop_reason'])
        return real_reference(data)
    args=argparse.Namespace(seed=9002,output=str(out),phr_steps=1,decay_steps=1,qp_iterations=0)
    with patch.object(factorial,'graph_reference',side_effect=reference_after_stop):
        experiment.run(args)
    assert len(calls)==4  # repair final evaluation + the three fixed stage snapshots
    summary=json.loads((out/'summary.json').read_text())
    assert set(summary['stages'])=={'phr','decay','qp'}
    assert not summary['final_joint_pass']  # retained, no reference-triggered retries
    assert (out/'COMPLETE').exists()
    torch.set_num_threads(1)
