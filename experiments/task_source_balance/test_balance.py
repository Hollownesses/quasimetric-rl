import copy
import importlib.util
import subprocess
from types import SimpleNamespace
import pytest
import torch
from . import train
from experiments.full_replay_four_path import train as base
from experiments.connected_expert_utrap.data import load_training_data
from experiments.four_path_factorial.run import IndexedIQE

@pytest.fixture(scope='module')
def data():
    torch.set_num_threads(1)
    return load_training_data('results/connected_expert_utrap_20261021/main')


def test_global_values_and_gradients(data):
    w,stats=train.task_weights(data);rows=data.original
    assert (stats['N'],stats['M'])==(42536,6766)
    unique=data.src[rows].unique()
    values=torch.linspace(-3,8,len(data.nodes),requires_grad=True)
    task=values[data.src[rows]];pair=torch.ones(len(rows),requires_grad=True)
    actual,_=train.weighted_push(data,rows,task,pair,w)
    expected=.6*values[unique].mean()+.4*pair.mean()
    torch.testing.assert_close(actual,expected)
    ga=torch.autograd.grad(actual,(values,pair),retain_graph=True)
    ge=torch.autograd.grad(expected,(values,pair))
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,atol=1e-9,rtol=1e-5)
    # Nonrepresentative batch explicitly must NOT normalize by its weight sum.
    r=rows[:8];t=torch.arange(8,dtype=torch.float32,requires_grad=True);p=t.detach().clone().requires_grad_()
    v,_=train.weighted_push(data,r,t,p,w)
    torch.testing.assert_close(v,.6*(w[r]*t).mean()+.4*p.mean())
    gt,gp=torch.autograd.grad(v,(t,p))
    torch.testing.assert_close(gt,.6*w[r]/8);torch.testing.assert_close(gp,torch.full_like(p,.4/8))


def test_r0_historical_e06_update(data,tmp_path):
    path=tmp_path/'historical.py'
    path.write_bytes(subprocess.check_output(['git','show','e06e2ad:experiments/full_replay_four_path/train.py']))
    spec=importlib.util.spec_from_file_location('historical',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    m,o,d,v,s=train.initial_state(data,606);t,q,_,_,_=train.initial_state(data,606)
    e=d.clone();w,_=train.task_weights(data)
    # Several updates compare Adam moments, duals, model and original objective exactly.
    for _ in range(3):
        rows=data.original[s.next(256)]
        a=train.advance(m,data,d,o,rows,'record',w);b=old.advance(t,data,e,q,rows)
        assert a['loss']==b['loss'] and train.equal(m.state_dict(),t.state_dict())
        assert train.equal(o.state_dict(),q.state_dict()) and torch.equal(d,e)


def test_r1_phr_b1_preprimal_dual(data):
    m,o,d,v,s=train.initial_state(data,606);t,q,_,_,_=train.initial_state(data,606)
    rows=data.original[s.next(256)];d.fill_(2);e=d.clone();w,_=train.task_weights(data)
    edge,task,pair=base.distances(t,data,rows);h=edge-data.cost[rows]
    # Independent explicit PHR formula with fixed multipliers.
    phr=(((e[rows]+100*h).relu().square()-e[rows].square()).mean()/200)
    loss=-.6*(w[rows]*task).mean()-.4*pair.mean()+phr
    # Match arithmetic association of the historical loss for exact Adam comparison.
    exact=-(.6*(w[rows]*task).mean()+.4*pair.mean())+phr
    torch.testing.assert_close(exact,loss)
    expected=(e[rows]+h.detach()).relu();exact.backward();q.step();e[rows]=expected
    train.advance(m,data,d,o,rows,'source',w)
    assert train.equal(m.state_dict(),t.state_dict()) and train.equal(o.state_dict(),q.state_dict())
    assert torch.equal(d,e)


def test_pair_budget(data):
    a=base.EpochSampler(len(data.original),2307);b=base.EpochSampler(len(data.original),2307)
    va=torch.zeros(len(data.original),dtype=torch.long);vb=va.clone()
    for _ in range(50000):
        x,y=a.next(256),b.next(256);assert torch.equal(x,y);va[x]+=1;vb[y]+=1
    assert train.equal(a.state_dict(),b.state_dict()) and torch.equal(va,vb)
    assert (int(va.min()),int(va.max()),int(va.sum()))==(299,300,12735416)
    assert train.learning_rate(45000,45000,5000)==2e-5
    assert train.learning_rate(45001,45000,5000)==2e-6
    assert train.learning_rate(50000,45000,5000)==2e-7


def test_resume_complete_state_and_mode_guard(data,tmp_path):
    w,_=train.task_weights(data)
    for mode in ('record','source'):
        m,o,d,v,s=train.initial_state(data,606);signature=dict(task_weight_mode=mode,weights_sha256='tested')
        for _ in range(2):
            idx=s.next(32);v[idx]+=1;train.advance(m,data,d,o,data.original[idx],mode,w)
        saved=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),dual=d,visits=v,
            sampler=s.state_dict(),rng=torch.get_rng_state(),training_signature=signature))
        torch.save(saved,tmp_path/'saved.pt');saved=torch.load(tmp_path/'saved.pt',weights_only=True)
        idx=s.next(32);v[idx]+=1;train.advance(m,data,d,o,data.original[idx],mode,w)
        t,q,_,_,ss=train.initial_state(data,606)
        dd,vv=train.restore(saved,signature,t,q,ss)
        ii=ss.next(32);vv[ii]+=1;train.advance(t,data,dd,q,data.original[ii],mode,w)
        assert train.equal(m.state_dict(),t.state_dict()) and train.equal(o.state_dict(),q.state_dict())
        assert torch.equal(d,dd) and torch.equal(v,vv) and train.equal(s.state_dict(),ss.state_dict())
        assert torch.equal(torch.get_rng_state(),saved['rng'])
        with pytest.raises(ValueError,match='configuration/weight mode'):
            train.restore(saved,dict(signature,task_weight_mode='other'),t,q,ss)
        with pytest.raises(ValueError,match='configuration/weight mode'):
            train.restore(saved,dict(signature,weights_sha256='changed'),t,q,ss)
