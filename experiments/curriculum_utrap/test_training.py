import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from quasimetric_rl.data import MultiEpisodeData
from minimal_qrl.global_push_sampling import replay_fingerprint
from minimal_qrl.eval.comm_inspection_oracle_bank import environment_signature
from experiments.full_replay_four_path import train
from .common import SCENARIO,FIELDS,make_env,write,sha
from .collect import default_config,collect_one
from .data import load_training_data,audit_training_data


@pytest.fixture
def mixed(tmp_path):
    scenario=json.loads(SCENARIO.read_text());env=make_env(scenario);cfg=default_config()
    # Real success and real collision, both governed by the same random collector.
    episodes=[];success=None
    for seed in range(30):
        ep,r,_=collect_one(env,np.array([7.56,3.65,3.],np.float32),seed,0,20,cfg,0,0,0,0)
        if r['outcome']=='success':success=ep;break
    assert success is not None;episodes.extend(success)
    failure,r,_=collect_one(env,np.array([5.09,3.65,0.],np.float32),34,1,20,cfg,6,1,1,2)
    assert r['outcome']=='collision';episodes.extend(failure)
    raw=MultiEpisodeData.cat(episodes);write(tmp_path/'scenario.json',scenario)
    torch.save(dict(schema_version=1,sha256=replay_fingerprint(raw),raw={k:getattr(raw,k) for k in FIELDS}),tmp_path/'verified_replay.pt')
    write(tmp_path/'manifest.json',dict(dataset_kind='curriculum_network',replay_sha256=replay_fingerprint(raw),
        environment_signature=environment_signature(env),files={n:sha(tmp_path/n) for n in ('scenario.json','verified_replay.pt')}))
    return tmp_path


@pytest.mark.parametrize('mode',['all','witnessed'])
def test_training_and_resume_with_failed_rows(mixed,tmp_path,mode):
    torch.set_num_threads(1);d=load_training_data(mixed,task_mode=mode);audit_training_data(d)
    out=tmp_path/'train';argv=['--curriculum-data',str(mixed),'--output',str(out),
        '--phr-steps','2','--decay-steps','2','--batch','2','--threads','1','--audit-interval','2','--task-mode',mode]
    train.run(train.parse_args(argv));a=torch.load(out/'final.pt',weights_only=True)
    assert a['dataset_kind']=='curriculum_network' and a['step']==4
    assert a['push_counters']['curriculum_pair_visits']==int(a['visits'].sum())
    if mode=='witnessed':
        assert a['push_counters']['curriculum_task_visits']<a['push_counters']['curriculum_pair_visits']
    else:assert a['push_counters']['curriculum_task_visits']==a['push_counters']['curriculum_pair_visits']
    train.run(train.parse_args(argv+['--resume',str(out/'phr.pt')]))
    b=torch.load(out/'final.pt',weights_only=True)
    for k in ('dual','visits','rng'):assert torch.equal(a[k],b[k])
    for k in a['model']:assert torch.equal(a['model'][k],b['model'][k])
    assert a['push_counters']==b['push_counters']
    assert 'curriculum' in json.loads((out/'training.jsonl').read_text().splitlines()[-1])
    bad=copy.deepcopy(b);bad['dataset_kind']='expert_network';torch.save(bad,out/'wrong.pt')
    with pytest.raises(AssertionError,match='dataset kind'):
        train.run(train.parse_args(argv+['--resume',str(out/'wrong.pt')]))


def test_failed_sources_not_lost_from_constraints_or_b1(mixed):
    d=load_training_data(mixed,task_mode='witnessed');infos=d.raw.transition_infos
    failures=d.original[~infos['global_push_task_witness_mask'][d.original]]
    assert len(failures)>0 and infos['global_push_pair_mask'][failures].all()
    # A failed-only minibatch has zero task push, live B1, and local edge constraints.
    task=torch.ones(len(failures),requires_grad=True);pair=torch.ones(len(failures),requires_grad=True)
    push,info=train.push_objective(d,failures,task,pair)
    push.backward();assert not task.grad.any() and pair.grad.abs().sum()>0
    assert info['empty_task_batch'] and train.training_strata(d,failures)[0][1]==1.
    assert torch.equal(train.training_strata(d,failures)[0][2],failures)
    with pytest.raises(SystemExit):train.parse_args(['--output','unused','--expert-data','x','--curriculum-data','y'])


def test_near_target_cost_uses_environment_authority(tmp_path):
    # Real rollout reaching 0.0022 from the target: vector target rounding
    # produces a 9e-5 error although the recorded environment reward is exact.
    scenario=json.loads(SCENARIO.read_text());env=make_env(scenario)
    start=np.array([6.6883378,3.5494323,1.3234004],np.float32)
    from minimal_qrl.baselines.mppi import _state_terms
    env.reset(seed=1,options={'device_id':'u_trap_target','start':start})
    _,reward,_,_,info=env.step(np.array([1.5],np.float32))
    scalar=env.compute_step_terms(env.state,info['collision'],info['out_of_bounds'])['cost_total']
    vector,_=_state_terms(env,env.state[None],np.array([False]),np.array([False]))
    assert abs(scalar+reward)<1e-12
    assert abs(float(vector[0])+reward)>5e-5
    # The actual audited collection must pass without modifying stored costs.
    root=Path(__file__).resolve().parents[2]/'results/curriculum_utrap_20261020/main'
    if root.exists():
        check=audit_training_data(load_training_data(root))
        assert check['recomputed_cost_max_error']<5e-5
        assert check['mppi_vector_cost_max_error']>5e-5
