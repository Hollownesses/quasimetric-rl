import copy
import pytest
import torch
from experiments.large_phr_calibration.train import advance_outer,update_full_original,learning_rate,SOURCE
from experiments.full_replay_four_path.train import advance,EpochSampler,distances
from experiments.full_replay_four_path.data import load_union
from experiments.four_path_factorial.run import IndexedIQE


def test_full_update_only_original_rows_with_negative_projection():
    dual=torch.tensor([2.,1.,9.,0.,7.]);h=torch.tensor([.5,-.5,99.,.1,99.]);rows=torch.tensor([0,1,3])
    rec=update_full_original(dual,h,rows,10.)
    torch.testing.assert_close(dual,torch.tensor([7.,0.,9.,1.,7.]))
    assert rec['rows']==3 and rec['decreased']==1 and rec['increased']==2
    for bad in (torch.tensor([float('nan')]*5),torch.zeros(4)):
        before=dual.clone()
        with pytest.raises(ValueError):update_full_original(dual,bad,rows,10.)
        assert torch.equal(before,dual)


def test_same_primal_paths_and_frozen_original_then_latest_sweep_and_resume():
    torch.set_num_threads(1);d=load_union(task_mode='witnessed')
    saved=torch.load(SOURCE,map_location='cpu',weights_only=True)
    models=[IndexedIQE(d.small,'encoder',0,initialization_seed=606) for _ in range(2)]
    opts=[]
    for model in models:
        model.load_state_dict(saved['model']);opt=torch.optim.AdamW(model.parameters(),lr=2e-6)
        opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        for group in opt.param_groups:group['lr']=learning_rate(1)
        opts.append(opt)
    duals=[saved['dual'].clone() for _ in range(2)]
    sampler=EpochSampler(len(d.original),0);sampler.restore(saved['sampler']);rows=d.original[sampler.next(256)]
    a=advance(models[0],d,duals[0],opts[0],rows);b=advance_outer(models[1],d,duals[1],opts[1],rows)
    assert a==b
    for k,v in models[0].state_dict().items():assert torch.equal(v,models[1].state_dict()[k]),k
    assert torch.equal(duals[0][d.paths],duals[1][d.paths])
    assert torch.equal(duals[1][:d.nbase],saved['dual'][:d.nbase])
    assert not torch.equal(duals[0][rows],duals[1][rows])
    # Use freshly recomputed post-primal residuals; verify all original entries,
    # including entries outside the primal batch, with independent vector formula.
    with torch.no_grad():
        h=torch.cat([distances(models[1],d,r,False)-d.cost[r] for r in torch.arange(len(d.src)).split(1024)])
        expected=duals[1].clone();expected[d.original]=(expected[d.original]+100*h[d.original]).relu()
        update_full_original(duals[1],h,d.original,100.)
        assert torch.equal(duals[1],expected)
    models[0].load_state_dict(models[1].state_dict());opts[0].load_state_dict(copy.deepcopy(opts[1].state_dict()));duals[0]=duals[1].clone()
    for _ in range(2):
        rows=d.original[sampler.next(256)]
        for m,o,l in zip(models,opts,duals):advance_outer(m,d,l,o,rows)
    for k,v in models[0].state_dict().items():assert torch.equal(v,models[1].state_dict()[k]),k
    assert torch.equal(duals[0],duals[1])


def test_fixed_low_rate_schedule():
    assert learning_rate(1)==2e-6
    assert learning_rate(5000)==pytest.approx(2e-7)
    assert learning_rate(1)>learning_rate(2500)>learning_rate(5000)
