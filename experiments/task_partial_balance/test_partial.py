import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.task_source_balance import train as legacy
from experiments.full_replay_four_path.train import distances,EpochSampler
from . import train,prepare

@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1)
    return train.load_training_data('results/connected_expert_utrap_20261021/main')

@pytest.mark.parametrize('alpha,mode',[(0.,'record'),(1.,'source')])
def test_endpoints_exact_to_frozen_original(data,alpha,mode):
    path=prepare.OLD/'source_snapshot/experiments/task_source_balance/train.py'
    spec=importlib.util.spec_from_file_location('frozen_balance',path);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    weights,_,_=train.mixed_weights(data,alpha);source,_=original.task_weights(data)
    if alpha==1:assert torch.equal(weights,source)
    m,o,d,v,s=legacy.initial_state(data,606);t,q,dd,_,_=legacy.initial_state(data,606)
    for _ in range(3):
        rows=data.original[s.next(256)]
        result=train.advance(m,data,d,o,rows,alpha,weights)
        expected=original.advance(t,data,dd,q,rows,mode,source)
        assert result==expected and legacy.equal(m.state_dict(),t.state_dict())
        assert legacy.equal(o.state_dict(),q.state_dict()) and torch.equal(d,dd)


def test_half_global_value_and_gradient(data):
    w,stats,_=train.mixed_weights(data,.5);source,_=legacy.task_weights(data);rows=data.original
    assert torch.equal(w[rows],.5+.5*source[rows])
    assert (stats['N'],stats['M'])==(42536,6766) and abs(stats['mean']-1)<1e-7
    values=torch.linspace(-3,8,len(data.nodes),requires_grad=True)
    pair=torch.randn(len(rows),requires_grad=True);task=values[data.src[rows]]
    actual,_=legacy.weighted_push(data,rows,task,pair,w)
    expected=.6*(.5*task.mean()+.5*values[data.src[rows].unique()].mean())+.4*pair.mean()
    torch.testing.assert_close(actual,expected)
    ga=torch.autograd.grad(actual,(values,pair),retain_graph=True);ge=torch.autograd.grad(expected,(values,pair))
    for a,b in zip(ga,ge):torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-5)
    r=rows[:17];t=torch.linspace(1,5,17,requires_grad=True);p=torch.ones(17,requires_grad=True)
    loss,_=legacy.weighted_push(data,r,t,p,w);gt,gp=torch.autograd.grad(loss,(t,p))
    torch.testing.assert_close(gt,.6*w[r]/len(r));torch.testing.assert_close(gp,torch.full_like(p,.4/len(r)))
    for a in (-1,.25,.75,2):
        with pytest.raises(ValueError):train.mixed_weights(data,a)


def test_half_b1_phr_preprimal_dual(data):
    m,o,d,_,s=legacy.initial_state(data,606);t,q,dd,_,_=legacy.initial_state(data,606)
    d.fill_(2);dd.fill_(2);weights,_,_=train.mixed_weights(data,.5);rows=data.original[s.next(256)]
    edge,task,pair=distances(t,data,rows);h=edge-data.cost[rows]
    phr=((dd[rows]+100*h).relu().square()-dd[rows].square()).mean()/200
    loss=-(.6*(weights[rows]*task).mean()+.4*pair.mean())+phr
    pre=(dd[rows]+h.detach()).relu();loss.backward();q.step();dd[rows]=pre
    r=train.advance(m,data,d,o,rows,.5,weights)
    assert r['loss']==float(loss.detach())
    assert legacy.equal(m.state_dict(),t.state_dict()) and legacy.equal(o.state_dict(),q.state_dict())
    assert torch.equal(d,dd)


def test_full_sampler_budget_matches_existing(data):
    import hashlib
    sampler=EpochSampler(len(data.original),2307);visits=torch.zeros(len(data.original),dtype=torch.long);chain='0'*64
    for _ in range(50000):
        ids=sampler.next(256);visits[ids]+=1
        chain=hashlib.sha256(bytes.fromhex(chain)+data.original[ids].numpy().tobytes()).hexdigest()
    for arm in ('R0','R1'):
        saved=torch.load(prepare.OLD/arm/'final.pt',weights_only=True)
        assert torch.equal(visits,saved['visits']) and legacy.equal(sampler.state_dict(),saved['sampler'])
        assert chain==saved['sampling_chain_sha256']
    assert int(visits.sum())==12735416


def test_actual_run_resume_alpha_and_weight_guards(data,tmp_path,monkeypatch):
    monkeypatch.setattr(prepare,'load_training_data',lambda _:data)
    monkeypatch.setattr(prepare,'audit_training_data',lambda _:dict(fixture=True))
    root=tmp_path/'test';p=prepare.run(root)
    p.update(phr_steps=2,decay_steps=2,audit_interval=2,threads=1)
    train.write(root/'protocol.json',p)
    monkeypatch.setattr(train,'load_training_data',lambda *a,**kw:data)
    a=SimpleNamespace(protocol=str(root/'protocol.json'),output=str(root/'Rhalf'),alpha=.5,resume=None)
    train.run(a);expected=torch.load(root/'Rhalf/final.pt',weights_only=True)
    a.resume=str(root/'Rhalf/phr.pt');train.run(a)
    actual=torch.load(root/'Rhalf/final.pt',weights_only=True)
    assert legacy.equal(expected,actual)
    old=torch.load(prepare.OLD/'R0/initial.pt',weights_only=True)
    initial=torch.load(root/'Rhalf/initial.pt',weights_only=True)
    for key in prepare.FIELDS:assert legacy.equal(initial[key],old[key])
    signature=actual['training_signature'];m,o,_,_,sampler=legacy.initial_state(data,606)
    for alpha in (0.,1.):
        with pytest.raises(ValueError,match='alpha'):
            train.restore(actual,dict(signature,alpha=alpha),m,o,sampler)
    changed=copy.deepcopy(actual);changed['task_weights'][data.original[0]]+=.1
    with pytest.raises(ValueError,match='weights'):
        train.restore(changed,signature,m,o,sampler)
    with pytest.raises(ValueError,match='configuration'):
        train.restore(actual,dict(signature,protocol_sha256='wrong'),m,o,sampler)
    a.alpha=1.
    with pytest.raises(AssertionError,match='protocol alpha'):train.run(a)
