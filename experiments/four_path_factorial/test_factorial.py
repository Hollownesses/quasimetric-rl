import importlib.util
from pathlib import Path
import torch
import pytest
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms, TabularPHR

spec=importlib.util.spec_from_file_location('factorial',Path(__file__).with_name('run.py'))
f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


def test_phr_boundary_gradient_and_inactive_slack():
    h=torch.tensor([0.,-2.,.5],dtype=torch.double,requires_grad=True)
    lam=torch.tensor([3.,1.,2.],dtype=torch.double,requires_grad=True)
    loss,effective=phr_terms(h,lam,2.)
    loss.backward()
    torch.testing.assert_close(h.grad,torch.tensor([1.,0.,1.],dtype=torch.double))
    assert lam.grad is None
    assert torch.autograd.gradcheck(lambda x:phr_terms(x,lam,2.)[0],(h,))
    linear,_=phr_terms(h,lam,0.)
    torch.testing.assert_close(linear,(lam.detach()*h).mean())


def test_tabular_dual_updates_from_residual_only():
    solver=TabularPHR(4,rho=10.,dual_step=.5)
    solver.update(torch.tensor([2.,1.,-3.,0.]))
    torch.testing.assert_close(solver.multiplier,torch.tensor([1.,.5,0.,0.]))
    solver.update(torch.tensor([-1.,-2.,4.,0.]))
    torch.testing.assert_close(solver.multiplier,torch.tensor([.5,0.,2.,0.]))
    with pytest.raises(ValueError): solver(torch.zeros(3))


def test_exact_nodes_preserve_heading_and_context_and_merge_terminals():
    x=torch.zeros(5,42); x[:,2]=1
    x[1,3]=.00001  # no position-neighbour merging
    x[2,20]=1      # task context identity
    x[3,40]=1; x[3,0]=7
    x[4,41]=1
    nodes,ids,_=f.canonical_nodes(x)
    assert len(nodes)==4 and ids[3]==ids[4]
    assert len(set(ids[:3].tolist()))==3


def test_initial_function_and_frozen_pairs_and_barrier_adapter():
    torch.set_num_threads(2)
    data=f.load_data()
    a=f.IndexedIQE(data,'encoder',0); b=f.IndexedIQE(data,'latent',0)
    torch.testing.assert_close(a.project(),b.project(),rtol=0,atol=0)
    for x,y in zip(a.distances(data),b.distances(data)):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    ref=f.graph_reference(data)
    assert abs(float(.6*ref[data.task_src].mean()+.4*(ref[data.task_src]-ref[data.pair]).mean())-81.6174)<.001
    # Compare inherited production workset/loss to a literal head evaluation.
    from quasimetric_rl.modules.quasimetric_critic.losses.goal_set_barrier import GoalSetBarrierConf
    solver=f.CachedBarrier(GoalSetBarrierConf()); solver.set_replay(data.replay)
    edge,task,pair=a.distances(data); solver.residual=edge-data.cost; solver.cost=data.cost
    solver.audit(a,None)
    torch.manual_seed(123); indices=solver._workset(a,None)
    real=solver.residual[indices][data.cost[indices]>0]
    expected=-.6*task.mean()-.4*pair.mean()+500*(.01*torch.nn.functional.softplus((real+.01)/.01)).topk(8).values.mean()
    torch.manual_seed(123); actual,_,_=f.training_objective(a,data,solver,'barrier')
    torch.testing.assert_close(actual,expected)


def test_iqe_head_learning_rate_is_matched_across_parameterizations():
    data=f.load_data()
    for param,lr in [('latent',.01),('encoder',2e-5)]:
        model=f.IndexedIQE(data,param,0)
        optimizer=f.make_optimizer(model,lr)
        head_ids={id(p) for p in model.head.parameters()}
        all_ids=[]
        for group in optimizer.param_groups:
            for p in group['params']:
                all_ids.append(id(p))
                assert group['lr']==(2e-5 if id(p) in head_ids else lr)
        assert len(all_ids)==len(set(all_ids))==len(list(model.parameters()))


@pytest.mark.parametrize('parameterization',['latent','encoder'])
@pytest.mark.parametrize('method',['barrier','phr'])
def test_training_is_independent_of_graph_reference(monkeypatch,parameterization,method):
    torch.set_num_threads(2)
    data=f.load_data()
    def forbidden(*args):
        raise AssertionError('graph labels entered training')
    monkeypatch.setattr(f,'graph_reference',forbidden)
    model=f.IndexedIQE(data,parameterization,0)
    solver=f.TabularPHR(len(data.src),100.,1.) if method=='phr' else f.CachedBarrier(f.GoalSetBarrierConf())
    if method=='barrier': solver.set_replay(data.replay)
    loss,residual,push=f.training_objective(model,data,solver,method)
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.square().sum()) for g in gradients)>0
