"""Production-IQE gradient, budget and fixed-dual continuation contracts."""
import copy
import json
import pytest
import torch
from experiments.expert_utrap_network.test_network import dataset
from experiments.connected_expert_utrap.test_connected import connected
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.compare import equality
from experiments.four_path_factorial.run import IndexedIQE
from experiments.full_replay_four_path import train as original
from experiments.connected_phr_solver.train import advance_fixed
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms
from . import train


def test_chunk_weighting_frozen_parameters_and_dense_adam_equivalence(connected):
    torch.set_num_threads(1);d=load_training_data(connected[0]);d.nodes=d.nodes.double()
    # Float64 equivalence avoids Adam amplifying float32 cancellation noise on
    # near-zero-gradient IQE coordinates.
    # The continuation/parity tests below exercise production float32 states.
    a=IndexedIQE(d.small,'encoder',0,initialization_seed=606).double()
    oa=torch.optim.AdamW(a.parameters(),lr=2e-6,weight_decay=0.)
    dual=torch.linspace(0,3,len(d.src),dtype=torch.float64);fixed=dual.clone()
    advance_fixed(a,d,dual,oa,d.original[:8])  # Nonempty inherited Adam moments.
    b=IndexedIQE(d.small,'encoder',0,initialization_seed=606).double();b.load_state_dict(a.state_dict())
    ob=torch.optim.AdamW(b.parameters(),lr=2e-6,weight_decay=0.);ob.load_state_dict(copy.deepcopy(oa.state_dict()))
    before={k:v.clone() for k,v in a.state_dict().items()};calls=[]
    def observe(model):
        assert equality(model.state_dict(),before);calls.append(True)
    batch=len(d.original)//3+1
    assert len(d.original)%batch>0
    result=train.accumulate(a,d,dual,d.original,batch,observer=observe)
    assert len(calls)>1 and equality(before,a.state_dict())
    edge,task,pair=original.distances(b,d,d.original)
    loss=-.6*task.mean()-.4*pair.mean()+phr_terms(edge-d.cost[d.original],dual[d.original],100.)[0]
    ob.zero_grad(set_to_none=True);loss.backward()
    assert result['loss']==pytest.approx(float(loss.detach()),rel=2e-5,abs=2e-5)
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x.grad,y.grad,rtol=3e-4,atol=3e-5)
    oa.step();ob.step()
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7)
    assert torch.equal(dual,fixed)


@pytest.fixture
def parent(connected,tmp_path):
    folder=tmp_path/'parent';path=connected[0]
    original.run(original.parse_args(['--connected-expert-data',str(path),'--output',str(folder),
        '--phr-steps','2','--decay-steps','2','--audit-interval','2','--batch','8','--threads','1']))
    return path,folder/'final.pt'


def test_paired_record_budget_fixed_dual_and_boundary_resume(parent,tmp_path):
    path,checkpoint=parent;source=torch.load(checkpoint,weights_only=True);states={};logs={}
    for arm in ('mini','full'):
        out=tmp_path/arm
        argv=['--arm',arm,'--data',str(path),'--source',str(checkpoint),'--output',str(out),
              '--passes','3','--batch','17']
        train.run(train.parse_args(argv));expected=torch.load(out/'final.pt',weights_only=True)
        initial=torch.load(out/'initial.pt',weights_only=True)
        for key in ('model','optimizer','dual','visits','rng','push_counters'):
            assert equality(initial[key],source[key]),key
        train.run(train.parse_args(argv+['--resume',str(out/'pass_02.pt')]))
        actual=torch.load(out/'final.pt',weights_only=True)
        for key in expected:
            if key!='timing':assert equality(expected[key],actual[key]),key
        states[arm]=actual;assert torch.equal(actual['dual'],source['dual'])
        assert torch.equal(actual['visits']-source['visits'],torch.full_like(source['visits'],3))
        logs[arm]=[json.loads(s) for s in (out/'training.jsonl').read_text().splitlines()][:4]
        with pytest.raises(AssertionError,match='configuration mismatch'):
            train.run(train.parse_args(argv+['--lr','1e-6','--resume',str(out/'pass_02.pt')]))
    for key in ('visits','sampler','rng','push_counters','processed_records','dual'):
        assert equality(states['mini'][key],states['full'][key]),key
    assert states['mini']['primal_updates']>states['full']['primal_updates']==3
    assert [x['order_sha256'] for x in logs['mini']]==[x['order_sha256'] for x in logs['full']]


def test_mini_matches_existing_update(parent,tmp_path):
    path,checkpoint=parent;d=load_training_data(path);s=torch.load(checkpoint,weights_only=True)
    # One complete pass should be exactly the existing fixed-dual mini updates.
    model=IndexedIQE(d.small,'encoder',0,initialization_seed=s['initialization_seed']);model.load_state_dict(s['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-6);opt.load_state_dict(s['optimizer'])
    for g in opt.param_groups:g['lr']=2e-6
    sampler=original.EpochSampler(len(d.original),2026102401);dual=s['dual'].clone()
    for _ in range((len(d.original)+16)//17):advance_fixed(model,d,dual,opt,d.original[sampler.next(17)])
    out=tmp_path/'mini_parity'
    train.run(train.parse_args(['--arm','mini','--data',str(path),'--source',str(checkpoint),'--output',str(out),
                                '--passes','2','--batch','17']))
    saved=torch.load(out/'pass_01.pt',weights_only=True)
    assert equality(model.state_dict(),saved['model']) and equality(opt.state_dict(),saved['optimizer'])
    assert torch.equal(dual,saved['dual'])


def test_one_step_probes_leave_input_state_unchanged(parent):
    from .probe import probe_state
    data=load_training_data(parent[0]);state=torch.load(parent[1],weights_only=True)
    before=copy.deepcopy(state);result=probe_state(data,state)
    assert equality(state,before)
    assert len(result['probes'])==6 and not result['affects_main_checkpoints']
    assert result['forward_records']==7*len(data.original)
    assert result['backward_records']==len(data.original)


def test_resume_history_deduplicates_only_identical_measurements(tmp_path):
    from .analyze import read_history
    path=tmp_path/'history.jsonl'
    a=dict(completed_passes=1,augmented_loss=-3.,elapsed_seconds=1.)
    b=dict(a,elapsed_seconds=2.)
    path.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
    assert read_history(path)==[b]
    b['augmented_loss']=-4.
    path.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
    with pytest.raises(AssertionError,match='non-deterministic'):read_history(path)


def test_branch_exposure_reconstructs_path_regret():
    from .analyze import regret_exposure
    d=dict(branches=[dict(source=1,chosen_row=5,regret=3.)],
           episodes=[dict(rows=[5,6],regret=3.),dict(rows=[7],regret=0.)])
    assert regret_exposure(d)==[dict(source=1,visits=1,regret=3.,mean_contribution=1.5)]
    d['episodes'][0]['regret']=4.
    with pytest.raises(AssertionError):regret_exposure(d)
