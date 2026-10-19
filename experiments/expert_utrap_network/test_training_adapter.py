"""Expert-only integration, optimizer/restart checks, and legacy default contract."""
import copy
import json
from pathlib import Path
import pytest
import torch
from experiments.full_replay_four_path import train
from experiments.four_path_factorial.run import IndexedIQE
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms
from .training_adapter import load_training_data,audit_training_data
from .test_network import dataset  # tiny real-dynamics two-tail fixture

ROOT=Path(__file__).resolve().parents[2]


def test_real_network_index_alignment_and_defaults():
    torch.set_num_threads(1)
    data=load_training_data(ROOT/'results/expert_utrap_network_20261019/main')
    assert len(data.original)==11949 and len(data.src)==12013 and not len(data.paths)
    assert data.small.goal.shape==(11949,) and data.goal.shape==(12013,)
    assert torch.equal(data.goal[data.original],data.small.goal)
    assert torch.equal(data.pair[data.original],data.small.pair)
    # Include rows after every interleaved anchor, not just the first episode.
    index=data.raw.transition_infos['global_push_pair_goal_obs_index'][data.original]
    assert torch.equal(data.pair[data.original],data.obs_ids[index])
    report=audit_training_data(data)
    assert report['task_sources_with_strict_success_witness']==11949
    args=train.parse_args(['--output','unused'])
    old=json.loads((ROOT/'results/full_replay_four_path_20261017/main/train/config.json').read_text())
    for key in ('seed','phr_steps','decay_steps','batch','threads','audit_interval'):
        assert getattr(args,key)==old[key]
    assert args.task_mode=='all' and args.expert_data is None


def test_single_source_full_weight_pre_primal_dual_and_all_witnessed(dataset):
    path,_,_=dataset;torch.set_num_threads(1)
    data=load_training_data(path);witness=load_training_data(path,task_mode='witnessed')
    assert data.fingerprint==witness.fingerprint
    model=IndexedIQE(data.small,'encoder',0,initialization_seed=606)
    twin=IndexedIQE(data.small,'encoder',0,initialization_seed=606)
    twin.load_state_dict(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.)
    refopt=torch.optim.AdamW(twin.parameters(),lr=2e-5,weight_decay=0.)
    dual=torch.ones(len(data.src));expected_dual=dual.clone()
    rows=data.original[torch.tensor([0,8,9,17])]
    edge,task,pair=train.distances(twin,witness,rows)
    h=edge-witness.cost[rows]
    # Explicit independent single-source objective: catches a leftover .5
    # factor, accidentally repeated stratum, or empty four-path contribution.
    loss=-(.6*task.mean()+.4*pair.mean())+phr_terms(h,expected_dual[rows],100.)[0]
    expected_dual[rows]=(expected_dual[rows]+h.detach()).relu()
    loss.backward();refopt.step()
    update=train.advance(model,data,dual,opt,rows)
    assert update['loss']==float(loss.detach()) and 'expert_components' in update
    assert torch.equal(dual,expected_dual)
    for a,b in zip(model.parameters(),twin.parameters()):assert torch.equal(a,b)
    assert torch.equal(dual[~data.physical],torch.ones_like(dual[~data.physical]))


@pytest.mark.parametrize('mode',['all','witnessed'])
def test_cli_smoke_resume_and_configuration_guard(dataset,tmp_path,mode):
    source,_,_=dataset;out=tmp_path/'training'
    argv=['--expert-data',str(source),'--output',str(out),'--phr-steps','2',
          '--decay-steps','2','--batch','8','--threads','1','--audit-interval','2','--task-mode',mode]
    args=train.parse_args(argv);train.run(args)
    expected=torch.load(out/'final.pt',weights_only=True)
    config=json.loads((out/'config.json').read_text())
    assert config['rho']==100 and config['dual_step']==1 and config['dataset_kind']=='expert_network'
    assert config['push']==f'1.0 expert; 0.6 task({mode}) + 0.4 fixed B1'
    assert expected['step']==4 and expected['sampler']['epochs']>=1
    assert expected['push_counters']['expert_task_visits']==int(expected['visits'].sum())
    assert torch.count_nonzero(expected['dual'][~load_training_data(source).physical])==0
    assert all('expert' in r and 'paths' not in r for r in
               [json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()])
    resumed=train.parse_args(argv+['--resume',str(out/'phr.pt')]);train.run(resumed)
    actual=torch.load(out/'final.pt',weights_only=True)
    for key in ('dual','visits','rng'):assert torch.equal(expected[key],actual[key])
    for key in expected['model']:assert torch.equal(expected['model'][key],actual['model'][key])
    assert expected['push_counters']==actual['push_counters']
    assert expected['sampler']['cursor']==actual['sampler']['cursor']
    assert torch.equal(expected['sampler']['order'],actual['sampler']['order'])
    # A checkpoint from a different budget must not be silently resumed.
    bad=copy.deepcopy(expected);bad['training_signature']['phr_steps']=999
    torch.save(bad,out/'wrong_budget.pt')
    with pytest.raises(AssertionError,match='training configuration'):
        train.run(train.parse_args(argv+['--resume',str(out/'wrong_budget.pt')]))
    with pytest.raises(AssertionError,match='task mode'):
        opposite='witnessed' if mode=='all' else 'all'
        train.run(train.parse_args(argv+['--resume',str(out/'phr.pt'),'--task-mode',opposite]))
