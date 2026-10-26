import copy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest
import torch
from experiments.task_source_balance import train as old
from experiments.full_replay_four_path.train import distances,EpochSampler
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms
from . import train,prepare

@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1)
    return train.load_training_data('results/connected_expert_utrap_20261021/main')


def test_actual_edge_weights(data):
    v,ids,reps,counts,s=train.edge_weights(data);rows=data.original
    assert (s['N'],s['K'])==(42536,6958)
    assert abs(s['mean']-1)<1e-7 and counts.sum()==len(rows)
    total=torch.zeros(len(reps),dtype=torch.float64).scatter_add_(0,ids[rows],v[rows].double())
    torch.testing.assert_close(total,torch.full_like(total,len(rows)/len(reps)),atol=1e-6,rtol=1e-7)
    # Confirm exact identity matches existing evaluation convention, no labels read.
    from experiments.constraint_exposure_diagnostic.core import edge_index
    rr,ii=edge_index(data.src.numpy(),data.dst.numpy(),data.raw.actions.numpy(),data.cost.numpy(),data.physical.numpy())
    assert torch.equal(reps,torch.tensor(rr)) and torch.equal(ids,torch.tensor(ii))


def test_edge_identity_keeps_target_action_cost_and_anchors():
    d=SimpleNamespace(src=torch.tensor([0,0,0,0,0,0]),dst=torch.tensor([1,1,2,1,1,1]),
        cost=torch.tensor([3.,3.,3.,3.,4.,3.]),raw=SimpleNamespace(actions=torch.tensor([[1.],[1.],[1.],[2.],[1.],[1.]])),
        physical=torch.tensor([True,True,True,True,True,False]),original=torch.arange(5))
    v,ids,reps,counts,s=train.edge_weights(d)
    assert s['K']==4 and ids.tolist()==[0,0,1,2,3,-1]
    assert counts.tolist()==[2,1,1,1] and v[-1]==0


def test_weighted_phr_distinct_copy_multipliers_and_gradients():
    h=torch.tensor([.2,.2,-.4],dtype=torch.float64,requires_grad=True)
    lam=torch.tensor([0.,40.,1.],dtype=torch.float64,requires_grad=True)
    v=torch.tensor([.75,.75,1.5],dtype=torch.float64)
    value=train.weighted_phr(h,lam,v,100.)
    # Two copies of one edge have different multipliers; average their INDIVIDUAL terms.
    terms=torch.stack([phr_terms(h[i:i+1],lam[i:i+1],100.)[0] for i in range(3)])
    expected=(terms[:2].mean()+terms[2])/2
    torch.testing.assert_close(value,expected)
    gradient,dual_gradient=torch.autograd.grad(value,(h,lam),allow_unused=True,retain_graph=True)
    torch.testing.assert_close(gradient,v*(lam.detach()+100*h.detach()).relu()/3)
    assert dual_gradient is None
    representative=(terms[0]+terms[2])/2
    assert abs(float((value-representative).detach()))>1
    # This intentionally non-unit-mean batch must not renormalize by sum(v).
    sub=train.weighted_phr(h[:2],lam[:2],v[:2],100.)
    torch.testing.assert_close(sub,(v[:2]*terms[:2]).mean())
    assert not torch.isclose(sub,(v[:2]*terms[:2]).sum()/v[:2].sum())
    torch.autograd.gradcheck(lambda x:train.weighted_phr(x,lam.detach(),v,100.),(h,))


def test_all_one_edge_weights_exact_r1_updates(data):
    w,_=old.task_weights(data);v=torch.ones(len(data.src))
    m,o,d,_,s=old.initial_state(data,606);t,q,dd,_,_=old.initial_state(data,606)
    for _ in range(3):
        rows=data.original[s.next(256)]
        a=train.advance(m,data,d,o,rows,w,v);b=old.advance(t,data,dd,q,rows,'source',w)
        assert a['loss']==b['loss'] and a['connected_expert_push']==b['connected_expert_push']
        assert a['connected_expert_components']==b['connected_expert_components']
        assert old.equal(m.state_dict(),t.state_dict()) and old.equal(o.state_dict(),q.state_dict()) and torch.equal(d,dd)


def test_r2_explicit_task_b1_phr_and_unweighted_dual(data):
    w,_=old.task_weights(data);v,ids,reps,counts,_=train.edge_weights(data)
    # Include exact duplicate copies with deliberately unequal starting multipliers.
    rows=data.original[:128]
    m,o,d,_,_=old.initial_state(data,606);t,q,dd,_,_=old.initial_state(data,606)
    d[rows]=torch.linspace(0,30,len(rows));dd.copy_(d)
    edge,task,pair=distances(t,data,rows);h=edge-data.cost[rows]
    numerator=(dd[rows]+100*h).relu().square()-dd[rows].square()
    loss=-(.6*(w[rows]*task).mean()+.4*pair.mean())+(v[rows]*numerator).mean()/200
    correct=(dd[rows]+h.detach()).relu();wrong=(dd[rows]+v[rows]*h.detach()).relu()
    assert not torch.equal(correct,wrong)
    loss.backward();q.step();dd[rows]=correct
    a=train.advance(m,data,d,o,rows,w,v)
    assert a['loss']==float(loss.detach()) and old.equal(m.state_dict(),t.state_dict())
    assert old.equal(o.state_dict(),q.state_dict()) and torch.equal(d,dd)


def test_full_sampling_budget(data):
    sampler=EpochSampler(len(data.original),2307);visits=torch.zeros(len(data.original),dtype=torch.long);chain='0'*64
    for _ in range(50000):
        ids=sampler.next(256);visits[ids]+=1;chain=hashlib.sha256(bytes.fromhex(chain)+data.original[ids].numpy().tobytes()).hexdigest()
    for folder in [prepare.OLD/'R0',prepare.OLD/'R1',prepare.HALF/'Rhalf']:
        saved=torch.load(folder/'final.pt',weights_only=True)
        assert old.equal(sampler.state_dict(),saved['sampler']) and torch.equal(visits,saved['visits'])
        assert chain==saved['sampling_chain_sha256']
    assert int(visits.sum())==12735416


def test_actual_run_resume_and_weight_mode_guards(data,tmp_path,monkeypatch):
    monkeypatch.setattr(prepare,'load_training_data',lambda _:data)
    monkeypatch.setattr(prepare,'audit_training_data',lambda _:dict(fixture=True))
    p=prepare.run(tmp_path/'run');p.update(phr_steps=2,decay_steps=2,audit_interval=2,threads=1)
    root=tmp_path/'run';train.write(root/'protocol.json',p)
    monkeypatch.setattr(train,'load_training_data',lambda *a,**kw:data)
    args=SimpleNamespace(protocol=str(root/'protocol.json'),output=str(root/'R2'),resume=None)
    train.run(args);expected=torch.load(root/'R2/final.pt',weights_only=True)
    args.resume=str(root/'R2/phr.pt');train.run(args);actual=torch.load(root/'R2/final.pt',weights_only=True)
    assert old.equal(expected,actual)
    initial=torch.load(root/'R2/initial.pt',weights_only=True);prior=torch.load(prepare.OLD/'R1/initial.pt',weights_only=True)
    for k in prepare.FIELDS:assert old.equal(initial[k],prior[k])
    sig=actual['training_signature'];m,o,_,_,sampler=old.initial_state(data,606)
    for field in ('edge_weights','task_weights'):
        changed=copy.deepcopy(actual);changed[field][data.original[0]]+=.1
        with pytest.raises(ValueError,match='weights'):train.restore(changed,sig,m,o,sampler)
    changed=copy.deepcopy(actual);changed['edge_ids'][data.original[0]]+=1
    with pytest.raises(ValueError,match='partition'):train.restore(changed,sig,m,o,sampler)
    changed=copy.deepcopy(actual);changed['phr_weight_mode']='records'
    with pytest.raises(ValueError,match='mode'):train.restore(changed,sig,m,o,sampler)
    with pytest.raises(ValueError,match='configuration'):train.restore(actual,dict(sig,protocol_sha256='wrong'),m,o,sampler)
