"""Contract tests for exact joins, serialized labels, masks, and production loading."""
import copy
import json
import numpy as np
import pytest
import torch

from quasimetric_rl.data import MultiEpisodeData
from minimal_qrl.global_push_sampling import replay_fingerprint
from minimal_qrl.eval.comm_inspection_oracle_bank import environment_signature
from experiments.four_path_factorial.run import IndexedIQE
from .build import (SCENARIO, FIELDS, JOIN, make_env, terminal_tail, episode_pair,
                    require_exact_join, sha256, write_json)
from .data import load_data
from .audit import exact_ids, reverse_distances, require_equal


@pytest.fixture
def dataset(tmp_path):
    torch.set_num_threads(1)
    scenario=json.loads(SCENARIO.read_text());env=make_env(scenario)
    tails=[terminal_tail(env,s) for s in (-1,1)]
    raw=MultiEpisodeData.cat(sum([episode_pair(env,t,i) for i,t in enumerate(tails)],[]))
    write_json(tmp_path/'scenario.json',scenario)
    torch.save(dict(schema_version=1,sha256=replay_fingerprint(raw),
        raw={k:getattr(raw,k) for k in FIELDS}),tmp_path/'verified_replay.pt')
    write_json(tmp_path/'manifest.json',dict(environment_signature=environment_signature(env),
        replay_sha256=replay_fingerprint(raw),
        files={n:sha256(tmp_path/n) for n in ('scenario.json','verified_replay.pt')}))
    return tmp_path,env,tails


def test_all_witnessed_b1_and_terminal_quotient(dataset):
    path,env,tails=dataset
    a=load_data(path);w=load_data(path,task_mode='witnessed')
    assert a.fingerprint==w.fingerprint
    assert a.raw.transition_infos['global_push_task_mask'][a.physical].all()
    assert not a.raw.transition_infos['goal_return_mask'].any()
    assert not a.raw.transition_infos['goal_return_cost'].any()
    assert torch.equal(a.src[~a.physical],a.dst[~a.physical])
    assert len(a.goal.unique())==1
    assert not np.array_equal(tails[0]['states'][-1],tails[1]['states'][-1])
    # Compatibility forward only: no optimizer, backward, parameter update or checkpoint.
    model=IndexedIQE(a,'encoder',0,initialization_seed=606)
    with torch.no_grad():
        edge,task,pair=model.distances(a)
    assert torch.isfinite(edge).all() and torch.isfinite(task).all() and torch.isfinite(pair).all()
    assert (edge[~a.physical]==0).all()


def test_same_position_different_heading_or_task_is_not_join(dataset):
    _,env,tails=dataset
    obs=np.asarray(tails[0]['observations'],np.float32)
    near=obs[0].copy();near[0]=np.nextafter(near[0],np.float32(np.inf))
    rotated=env.state_to_observation(JOIN+np.array([0,0,.01],np.float32))
    task=obs[0].copy();task[4]+=1
    ids,_=exact_ids(np.stack([obs[0],near,rotated,task,obs[0]]))
    assert ids.tolist()==[0,1,2,3,0]
    left=dict(states=[tails[0]['states'][0]],observations=[obs[0].tolist()])
    right=copy.deepcopy(tails[0]);require_exact_join(left,right)
    right['states'][0][2]+=.00001
    with pytest.raises(ValueError,match='non-exact'):require_exact_join(left,right)
    with pytest.raises(ValueError,match='exact replay mismatch'):require_equal(obs[0],near,'near endpoint')


def test_loader_rejects_metadata_tampering(dataset):
    path,_,_=dataset
    p=torch.load(path/'verified_replay.pt',weights_only=True)
    p['raw']['transition_infos']['goal_return_cost'][0]=1
    torch.save(p,path/'verified_replay.pt')
    with pytest.raises(ValueError,match='fingerprint'):load_data(path)


def test_graph_audit_uses_real_branch_cost_and_shared_join():
    # Same start and join, distinct costly and cheap routes, then one legal goal.
    src=np.array([0,1,0,2,3]);dst=np.array([1,3,2,3,4]);cost=np.array([1.,6.,2.,1.,2.])
    dist,edge=reverse_distances(5,src,dst,cost,[4])
    assert dist.tolist()==[5.,8.,3.,2.,0.]
    assert edge[0]==2


def test_graph_reference_accumulates_float32_costs_in_float64():
    src=np.arange(200);dst=src+1;cost=np.full(200,.1234567,dtype=np.float32)
    dist,_=reverse_distances(201,src,dst,cost,[200])
    assert abs(dist[0]-cost.sum(dtype=np.float64))<1e-12
    assert np.all(dist[src]<=cost.astype(np.float64)+dist[dst]+1e-12)
