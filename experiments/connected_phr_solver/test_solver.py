"""Signed path accounting and full-dataset gradient normalization."""
import numpy as np
import pytest
import torch
from experiments.expert_utrap_network.test_network import dataset
from experiments.connected_expert_utrap.test_connected import connected
from experiments.connected_expert_utrap.data import load_training_data
from experiments.four_path_factorial.run import IndexedIQE
from .diagnostic import decompose
from .objective import full_gradients


@pytest.mark.parametrize('edge,pred,expected',[
    ([.5,.3],[.7,.4,0],[1.2,.1,1.3]),
    ([1.5,1],[2.3,1,0],[-.5,.2,-.3])])
def test_signed_decomposition(edge,pred,expected):
    p=decompose(np.array([2.,1.,0.]),np.array([0,1,-1]),np.array([1,2]),
                np.ones(2),np.array(pred),np.array(edge))
    np.testing.assert_allclose([p[k][0] for k in ('edge_slack','triangle_slack','gap')],expected)


def test_invalid_reference_path_rejected():
    with pytest.raises(AssertionError):
        decompose(np.array([2.,1.,0.]),np.array([0,1,-1]),np.array([1,2]),
                  np.array([.5,1]),np.zeros(3),np.zeros(2))


def test_full_gradient_chunking_and_readonly(connected):
    torch.set_num_threads(1)
    d=load_training_data(connected[0]);m=IndexedIQE(d.small,'encoder',0,initialization_seed=606)
    before={k:v.clone() for k,v in m.state_dict().items()}
    dual=torch.linspace(0,3,len(d.src));dual_before=dual.clone()
    a,va,ha=full_gradients(m,d,dual,batch=len(d.original))
    b,vb,hb=full_gradients(m,d,dual,batch=17)
    for k in va:torch.testing.assert_close(va[k],vb[k],rtol=2e-4,atol=2e-5)
    np.testing.assert_allclose(ha,hb,atol=2e-5)
    assert a['augmented_loss']==pytest.approx(b['augmented_loss'],rel=1e-5,abs=1e-4)
    assert not torch.equal(va['phr'],va['lagrange'])
    assert all(p.grad is None for p in m.parameters())
    assert torch.equal(dual,dual_before)
    for k,v in m.state_dict().items():assert torch.equal(v,before[k])


def test_fixed_primal_matches_visit_and_does_not_update_dual(connected):
    from experiments.full_replay_four_path.train import advance
    from .train import advance_fixed,outer_update
    d=load_training_data(connected[0]);a=IndexedIQE(d.small,'encoder',0,initialization_seed=606)
    b=IndexedIQE(d.small,'encoder',0,initialization_seed=606)
    oa=torch.optim.AdamW(a.parameters(),lr=2e-6,weight_decay=0.)
    ob=torch.optim.AdamW(b.parameters(),lr=2e-6,weight_decay=0.)
    da=torch.full((len(d.src),),2.);db=da.clone();rows=d.original[:8]
    advance(a,d,da,oa,rows);advance_fixed(b,d,db,ob,rows)
    for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)
    assert (db==2).all()
    residual=torch.linspace(-3,2,len(db));counts=torch.arange(len(d.original))%4
    before=db.clone();outer_update(db,d.original,residual,counts)
    assert torch.equal(db[d.original],(before[d.original]+counts*residual[d.original]).relu())
    assert torch.equal(db[~d.physical],before[~d.physical])


def test_paired_budgets_and_outer_boundary_resume(connected,tmp_path):
    from experiments.full_replay_four_path import train as original
    from experiments.connected_expert_utrap.compare import equality
    from . import train
    path=connected[0];parent=tmp_path/'parent'
    original.run(original.parse_args(['--connected-expert-data',str(path),'--output',str(parent),
        '--phr-steps','2','--decay-steps','2','--audit-interval','2','--batch','8','--threads','1']))
    states={}
    for arm in ('A','B'):
        out=tmp_path/arm
        argv=['--data',str(path),'--source',str(parent/'final.pt'),'--arm',arm,'--output',str(out),
              '--steps','6','--period','2','--batch','8']
        train.run(train.parse_args(argv));before=torch.load(out/'final.pt',weights_only=True)
        train.run(train.parse_args(argv+['--resume',str(out/'last_block_start.pt')]))
        after=torch.load(out/'final.pt',weights_only=True)
        assert equality(before,after)
        states[arm]=after
        if arm=='B':
            with pytest.raises(AssertionError,match='completed outer boundary'):
                train.run(train.parse_args(argv+['--resume',str(out/'final_before_outer.pt')]))
        with pytest.raises(AssertionError,match='configuration mismatch'):
            train.run(train.parse_args(argv+['--batch','9','--resume',str(out/'last_block_start.pt')]))
    for key in ('visits','nominal_dual_steps','sampler','rng','push_counters'):
        assert equality(states['A'][key],states['B'][key])


def test_pairing_audit_rejects_unmatched_dual_budget():
    import copy
    from .analyze import verify_pair
    parent=dict(step=50000,model={'x':torch.ones(2)},optimizer={},dual=torch.zeros(2),
        visits=torch.tensor([299,300]),sampler={},rng=torch.tensor([1]),push_counters={})
    initials={a:copy.deepcopy(parent) for a in ('A','B')}
    final=copy.deepcopy(parent)
    final.update(step=55000,continuation_step=5000,visits=parent['visits']+30,
                 nominal_dual_steps=torch.tensor([30,30]))
    finals={a:copy.deepcopy(final) for a in ('A','B')}
    assert verify_pair(parent,initials,finals)['final_sampling_and_nominal_budgets_exact']
    finals['B']['nominal_dual_steps'][0]+=1
    with pytest.raises(AssertionError):verify_pair(parent,initials,finals)
