"""Ablation changes only the B1 gradient; checkpoint mode cannot be confused."""
import pytest
import torch
from experiments.connected_expert_utrap.test_connected import connected, dataset
from experiments.connected_expert_utrap.data import load_training_data
from experiments.full_replay_four_path import train
from experiments.four_path_factorial.run import IndexedIQE
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms


def test_task_only_gradient_and_explicit_update(connected):
    torch.set_num_threads(1)
    data=load_training_data(connected[0]);rows=data.original[[0,8,9,17]]
    task=torch.ones(4,requires_grad=True);pair=torch.ones(4,requires_grad=True)
    push,info=train.push_objective(data,rows,task,pair,'task-only');push.backward()
    torch.testing.assert_close(task.grad,torch.full((4,),.6/4))
    assert not pair.grad.any() and info['optimized_pair_rows']==0
    model=IndexedIQE(data.small,'encoder',0,initialization_seed=606)
    twin=IndexedIQE(data.small,'encoder',0,initialization_seed=606)
    a=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.)
    b=torch.optim.AdamW(twin.parameters(),lr=2e-5,weight_decay=0.)
    dual=torch.ones(len(data.src));expected=dual.clone()
    edge,task,pair=train.distances(twin,data,rows);h=edge-data.cost[rows]
    loss=-.6*task.mean()+phr_terms(h,expected[rows],100.)[0]
    expected[rows]=(expected[rows]+h.detach()).relu();loss.backward();b.step()
    result=train.advance(model,data,dual,a,rows,'task-only')
    assert result['loss']==float(loss.detach()) and torch.equal(dual,expected)
    for x,y in zip(model.parameters(),twin.parameters()):assert torch.equal(x,y)


def test_task_only_resume_and_mode_guard(connected,tmp_path):
    out=tmp_path/'run'
    argv=['--connected-expert-data',str(connected[0]),'--output',str(out),
          '--phr-steps','2','--decay-steps','2','--audit-interval','2',
          '--threads','1','--batch','8','--push-mode','task-only']
    train.run(train.parse_args(argv));expected=torch.load(out/'final.pt',weights_only=True)
    train.run(train.parse_args(argv+['--resume',str(out/'phr.pt')]))
    actual=torch.load(out/'final.pt',weights_only=True)
    for key in ('dual','visits','rng'):assert torch.equal(expected[key],actual[key])
    for key in expected['model']:assert torch.equal(expected['model'][key],actual['model'][key])
    for key in ('order','rng'):assert torch.equal(expected['sampler'][key],actual['sampler'][key])
    for key in expected['optimizer']['state']:
        for name in expected['optimizer']['state'][key]:
            assert torch.equal(expected['optimizer']['state'][key][name],actual['optimizer']['state'][key][name])
    with pytest.raises(AssertionError,match='push mode'):
        train.run(train.parse_args(argv+['--resume',str(out/'phr.pt'),'--push-mode','task-b1']))
