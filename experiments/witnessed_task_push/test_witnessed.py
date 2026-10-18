import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.full_replay_four_path.train import masked_mean,push_objective,advance,EpochSampler
from experiments.full_replay_four_path.data import load_union
from experiments.four_path_factorial.run import IndexedIQE


@pytest.mark.parametrize('mismatch',['task_mode','task_mask_sha256'])
def test_resume_rejects_changed_task_objective_before_loading_model(tmp_path,monkeypatch,mismatch):
    import json
    from experiments.full_replay_four_path import train
    from experiments.four_path_factorial.run import tensor_digest
    mask=torch.tensor([True,False])
    data=SimpleNamespace(fingerprint={'test':'unchanged'},raw=SimpleNamespace(
        transition_infos={'global_push_task_mask':mask}))
    monkeypatch.setattr(train,'load_union',lambda **kwargs:data)
    args=SimpleNamespace(output=str(tmp_path),resume='unused.pt',task_mode='witnessed',
        seed=606,phr_steps=20000,decay_steps=5000,batch=256,threads=1)
    config=dict(seed=606,phr_steps=20000,decay_steps=5000,batch=256,threads=1,
                fingerprints=data.fingerprint,task_mode='witnessed',task_mask_sha256=tensor_digest(mask))
    config[mismatch]='all' if mismatch=='task_mode' else 'incorrect_hash'
    (tmp_path/'config.json').write_text(json.dumps(config))
    with pytest.raises(AssertionError):train.run(args)


def test_masked_task_gradient_and_all_b1_retained():
    # Interleave abstract rows to catch the wrong mask/value alignment.
    d=SimpleNamespace(physical=torch.tensor([True,False,True,True]),raw=SimpleNamespace(transition_infos={
        'global_push_task_mask':torch.tensor([True,False,False,True]),
        'global_push_pair_mask':torch.tensor([True,False,True,True])}))
    task=torch.tensor([2.,1000.,6.],requires_grad=True);pair=torch.tensor([3.,9.,12.],requires_grad=True)
    push,info=push_objective(d,torch.arange(4),task,pair)
    torch.testing.assert_close(push,torch.tensor(.6*4+.4*8))
    push.backward();torch.testing.assert_close(task.grad,torch.tensor([.3,0.,.3]))
    torch.testing.assert_close(pair.grad,torch.full((3,),.4/3))
    assert info['task_rows']==2 and info['pair_rows']==3


def test_empty_task_is_zero_without_b1_reweighting_or_nan():
    task=torch.tensor([float('nan'),100.],requires_grad=True);pair=torch.tensor([3.,7.],requires_grad=True)
    d=SimpleNamespace(physical=torch.ones(2,dtype=torch.bool),raw=SimpleNamespace(transition_infos={
        'global_push_task_mask':torch.zeros(2,dtype=torch.bool),'global_push_pair_mask':torch.ones(2,dtype=torch.bool)}))
    push,info=push_objective(d,torch.arange(2),task,pair)
    assert torch.isfinite(push) and float(push.detach())==2. and info['empty_task_batch']
    push.backward();assert torch.equal(task.grad,torch.zeros(2))
    torch.testing.assert_close(pair.grad,torch.full((2,),.2))


def test_real_union_unchanged_and_updates_match_old_all_then_resume_witnessed():
    torch.set_num_threads(1);all_data=load_union();d=load_union(task_mode='witnessed')
    assert d.fingerprint==all_data.fingerprint
    for name in ('nodes','src','dst','cost','original','paths','goal','pair','physical'):
        assert torch.equal(getattr(d,name),getattr(all_data,name)),name
    info=d.raw.transition_infos;mask=info['global_push_task_mask']
    assert torch.equal(mask,info['global_push_task_witness_mask'])
    assert int(mask[:d.nbase].sum())==2575 and int(mask[d.nbase:].sum())==335
    assert int(info['global_push_pair_mask'].sum())==200335
    assert not mask[info['device_index']==1].any()  # no corridor success witness
    # No change in the all-mode objective or local-dual implementation.
    spec=importlib.util.spec_from_file_location('frozen_training',Path('results/full_replay_four_path_20261017/main/source_snapshot/experiments/full_replay_four_path/train.py'))
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    m=IndexedIQE(d.small,'encoder',0,initialization_seed=9003);twin=IndexedIQE(d.small,'encoder',0,initialization_seed=9003)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=0.);other=torch.optim.AdamW(twin.parameters(),lr=2e-5,weight_decay=0.)
    dual=torch.zeros(len(d.src));other_dual=dual.clone();rows=d.original[:64]
    advance(m,all_data,dual,opt,rows);old.advance(twin,all_data,other_dual,other,rows)
    assert torch.equal(dual,other_dual)
    for a,b in zip(m.state_dict().values(),twin.state_dict().values()):assert torch.equal(a,b)
    # Actual empty-task batch still updates B1/primal and per-edge PHR multipliers.
    failed=d.original[~mask[d.original]][:64]
    before={k:v.clone() for k,v in m.state_dict().items()}
    update=advance(m,d,dual,opt,failed)
    assert update['original_components']['task_rows']==0 and update['original_components']['pair_rows']==64
    assert any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    twin=IndexedIQE(d.small,'encoder',0,initialization_seed=9003);twin.load_state_dict(m.state_dict())
    other=torch.optim.AdamW(twin.parameters(),lr=2e-5,weight_decay=0.);other.load_state_dict(copy.deepcopy(opt.state_dict()));other_dual=dual.clone()
    for _ in range(2):
        advance(m,d,dual,opt,failed);advance(twin,d,other_dual,other,failed)
    assert torch.equal(dual,other_dual)
    for a,b in zip(m.state_dict().values(),twin.state_dict().values()):assert torch.equal(a,b)
