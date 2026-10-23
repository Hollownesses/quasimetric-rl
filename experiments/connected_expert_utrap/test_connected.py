"""Real-environment composition, legacy compatibility, and training continuation."""
import copy
import json
import numpy as np
import pytest
import torch

from experiments.expert_utrap_network.test_network import dataset
from experiments.expert_utrap_network.build import execute_segment
from experiments.expert_utrap_network.training_adapter import load_training_data as old_loader
from experiments.full_replay_four_path import train
from experiments.four_path_factorial.run import IndexedIQE
from .build import compose, backbone_paths
from .data import load_training_data, audit_training_data


@pytest.fixture
def connected(dataset):
    path,env,tails=dataset
    manifest=json.loads((path/'manifest.json').read_text())
    manifest['dataset_kind']='connected_expert_network'
    (path/'manifest.json').write_text(json.dumps(manifest))
    return path,env,tails


def test_cli_preserves_old_modes_and_rejects_multiple_sources():
    args=train.parse_args(['--output','unused'])
    assert args.expert_data is None and args.connected_expert_data is None
    assert args.phr_steps==20000 and args.decay_steps==5000
    assert train.parse_args(['--output','unused','--expert-data','old']).expert_data=='old'
    with pytest.raises(SystemExit):
        train.parse_args(['--output','unused','--expert-data','old','--connected-expert-data','new'])


def test_builder_rejects_environment_drift(connected,tmp_path):
    _,env,_=connected
    other=tmp_path/'wrong_backbone';other.mkdir()
    (other/'manifest.json').write_text(json.dumps({'environment_signature':{'digest':'wrong'}}))
    with pytest.raises(ValueError,match='environment signature mismatch'):
        backbone_paths(env,other)


def test_loader_kind_masks_and_index_alignment(connected):
    path,_,_=connected
    d=load_training_data(path);w=load_training_data(path,task_mode='witnessed')
    assert d.fingerprint==w.fingerprint
    assert d.dataset_kind=='connected_expert_network'
    assert torch.equal(d.goal[d.original],d.small.goal)
    assert torch.equal(d.pair[d.original],d.small.pair)
    report=audit_training_data(d)
    assert report['all_equals_witnessed'] and not report['graph_references_loaded']
    assert not d.raw.transition_infos['goal_return_mask'].any()
    manifest=json.loads((path/'manifest.json').read_text());manifest['dataset_kind']='expert_network'
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='expected connected'):
        load_training_data(path)


def test_exact_connector_composition_and_near_join_rejection(connected):
    _,env,tails=connected;tail=tails[0]
    start=np.asarray(tail['states'][0],np.float32)
    connection=execute_segment(env,start,tail['actions'][:4])
    target=dict(path='tail',index=4,state=np.asarray(tail['states'][4],np.float32))
    actual,j=compose(env,start,[],connection,target,{'tail':tail})
    assert j==4
    for name in ('states','observations','actions','rewards'):
        np.testing.assert_array_equal(np.asarray(actual[name],np.float32),np.asarray(tail[name],np.float32))
    wrong=copy.deepcopy(connection)
    wrong['states'][-1][0]=float(np.nextafter(np.float32(wrong['states'][-1][0]),np.float32(np.inf)))
    with pytest.raises(ValueError,match='near endpoint'):
        compose(env,start,[],wrong,target,{'tail':tail})


def test_same_single_source_update_as_old_expert(connected):
    path,_,_=connected;torch.set_num_threads(1)
    a=load_training_data(path);b=old_loader(path)
    m=IndexedIQE(a.small,'encoder',0,initialization_seed=606)
    n=IndexedIQE(b.small,'encoder',0,initialization_seed=606)
    oa=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=0.)
    ob=torch.optim.AdamW(n.parameters(),lr=2e-5,weight_decay=0.)
    da=torch.zeros(len(a.src));db=da.clone();rows=a.original[[0,8,9,17]]
    ra=train.advance(m,a,da,oa,rows);rb=train.advance(n,b,db,ob,rows)
    assert ra['loss']==rb['loss'] and 'connected_expert_components' in ra
    assert torch.equal(da,db)
    for x,y in zip(m.parameters(),n.parameters()):assert torch.equal(x,y)


@pytest.mark.parametrize('mode',['all','witnessed'])
def test_real_training_resume_and_signature_guard(connected,tmp_path,mode):
    path,_,_=connected;out=tmp_path/'training'
    argv=['--connected-expert-data',str(path),'--output',str(out),'--phr-steps','2',
          '--decay-steps','2','--audit-interval','2','--batch','8','--threads','1','--task-mode',mode]
    train.run(train.parse_args(argv))
    expected=torch.load(out/'final.pt',weights_only=True)
    assert expected['dataset_kind']=='connected_expert_network'
    train.run(train.parse_args(argv+['--resume',str(out/'phr.pt')]))
    actual=torch.load(out/'final.pt',weights_only=True)
    for key in ('dual','visits','rng'):assert torch.equal(expected[key],actual[key])
    for key in expected['model']:assert torch.equal(expected['model'][key],actual['model'][key])
    assert expected['push_counters']==actual['push_counters']
    assert actual['push_counters']['connected_expert_task_visits']==int(actual['visits'].sum())
    bad=copy.deepcopy(expected);bad['training_signature']['dataset_kind']='expert_network'
    torch.save(bad,out/'wrong.pt')
    with pytest.raises(AssertionError,match='training configuration'):
        train.run(train.parse_args(argv+['--resume',str(out/'wrong.pt')]))
