import copy
import torch
from experiments.four_path_factorial.run import IndexedIQE,load_data
from experiments.full_replay_four_path.train import EpochSampler,advance,distances
from experiments.full_replay_four_path.data import load_union


def test_epoch_sampler_unique_and_exact_resume():
    sampler=EpochSampler(23,9)
    first=[sampler.next(8) for _ in range(3)]
    assert torch.equal(torch.cat(first).sort().values,torch.arange(23))
    other=EpochSampler(23,0);other.restore(sampler.state_dict())
    for _ in range(8):
        a,b=sampler.next(8),other.next(8)
        assert torch.equal(a,b) and len(a.unique())==len(a)


def test_union_semantics_and_subset_inference():
    torch.set_num_threads(1);d=load_union()
    assert len(d.src)==201087 and len(d.original)==200000
    m=IndexedIQE(d.small,'encoder',0,initialization_seed=9002)
    expected=m.distances(d.small);observed=distances(m,d,d.paths)
    for x,y in zip(expected,observed):torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-6)
    assert torch.equal(d.src[~d.physical],d.dst[~d.physical])
    # Exact continuation of model/Adam/duals/sampler must reproduce the next updates.
    opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=0.)
    dual=torch.zeros(len(d.src));sampler=EpochSampler(len(d.original),9002)
    advance(m,d,dual,opt,d.original[sampler.next(16)])
    twin=IndexedIQE(d.small,'encoder',0,initialization_seed=9002);twin.load_state_dict(m.state_dict());twin_opt=torch.optim.AdamW(twin.parameters(),lr=2e-5,weight_decay=0.)
    twin_opt.load_state_dict(copy.deepcopy(opt.state_dict()));twin_dual=dual.clone()
    twin_sampler=EpochSampler(len(d.original),0);twin_sampler.restore(sampler.state_dict())
    for _ in range(2):
        advance(m,d,dual,opt,d.original[sampler.next(16)])
        advance(twin,d,twin_dual,twin_opt,d.original[twin_sampler.next(16)])
    assert torch.equal(dual,twin_dual)
    for a,b in zip(m.state_dict().values(),twin.state_dict().values()):assert torch.equal(a,b)
