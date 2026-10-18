"""Immutable replay union. No trajectories are collected and no graph labels are built."""
from types import SimpleNamespace
import numpy as np
import torch
from experiments.four_path_factorial.run import (ROOT, Replay, canonical_nodes, load_data,
    MultiEpisodeData, Encoder, replay_fingerprint, configure_global_push_sampling, file_digest, tensor_digest)

BASE = ROOT/'results/diagnostic_path_support_positive/audit_baseline/verified_replay.pt'
PATHS = ROOT/'results/diagnostic_path_support_only/task_path_support_audit/verified_replay.pt'
EXPECTED = ('7ae2a77d8ff942de38762781996d15a6972ccf5f3074ea3075f2919f3e98914a',
            'af269a835ad45bab817230ed2f3f950d8b0d1781de51c146a2217de95f0d5b73')


def load_union():
    small = load_data(); raws=[]; source_fingerprints=[]
    for path,expected in zip((BASE,PATHS),EXPECTED):
        p=torch.load(path,map_location='cpu',weights_only=True);r=MultiEpisodeData(**p['raw'])
        assert replay_fingerprint(r)==p['sha256']==expected
        source_fingerprints.append(dict(path=str(path.relative_to(ROOT)),file_sha256=file_digest(path),replay_sha256=expected))
        ds=Replay(r,small.env)
        for k in ('global_push_source_observations','global_push_goal_observations'):
            r.transition_infos[k]=torch.zeros((len(ds),42))
        configure_global_push_sampling(ds,state_mode='trajectory',task_mode='all',seed=42)
        raws.append(r)
    a,b=raws; n=a.num_transitions; obs_offset=len(a.all_observations)
    # Sampling each immutable constituent separately preserves the historical B1 bank.
    assert torch.equal(b.transition_infos['global_push_pair_goal_obs_index'][small.rows],
                       small.raw.transition_infos['global_push_pair_goal_obs_index'][small.rows])
    b.transition_infos['context_id']+=int(a.transition_infos['context_id'].max())+1
    for key in ('global_push_pair_source_obs_index','global_push_pair_goal_obs_index'):
        value=b.transition_infos[key];value[value>=0]+=obs_offset
    fields={k:torch.cat([getattr(a,k),getattr(b,k)]) for k in
            ('episode_lengths','all_observations','actions','rewards','terminals','timeouts')}
    raw=MultiEpisodeData(**fields,observation_infos={},transition_infos={k:torch.cat([a.transition_infos[k],b.transition_infos[k]]) for k in a.transition_infos})
    replay=Replay(raw,small.env); src_obs=torch.arange(len(replay))+replay.indices_to_episode_indices
    nodes,ids,mapping=canonical_nodes(raw.all_observations)
    goals=Encoder._canonicalize_comm_goal_set(None,raw.transition_infos['task_goal_observations'])
    goals[goals==0]=0
    goal=torch.tensor([mapping[x.numpy().tobytes()] for x in goals])
    pair_obs=raw.transition_infos['global_push_pair_goal_obs_index']; pair=ids[pair_obs.clamp_min(0)]
    physical=~raw.transition_infos['abstract_goal_edge'];src=ids[src_obs];dst=ids[src_obs+1]
    assert int(physical[:n].sum())==200000 and int(physical[n:].sum())==335
    assert torch.equal(src[~physical],dst[~physical]) and (raw.rewards[~physical]==0).all()
    assert raw.transition_infos['global_push_pair_mask'][physical].all()
    assert raw.transition_infos['global_push_task_mask'][physical].all()
    assert (pair_obs[physical]>src_obs[physical]).all()
    fingerprint=dict(sources=source_fingerprints,merged_replay=replay_fingerprint(raw),
        nodes=tensor_digest(nodes,ids),pairs=tensor_digest(src,goal,pair),costs=tensor_digest(-raw.rewards),
        b1_seed=42,b1_policy='separately reconstruct constituents, then offset indices; four-path bank unchanged')
    return SimpleNamespace(small=small,raw=raw,replay=replay,nodes=nodes,obs_ids=ids,src=src,dst=dst,
        src_obs=src_obs,goal=goal,pair=pair,cost=-raw.rewards,physical=physical,nbase=n,env=small.env,
        original=torch.arange(n)[physical[:n]],paths=torch.arange(n,len(src)),fingerprint=fingerprint)


def audit(data):
    from minimal_qrl.eval.task_path_support_audit import physical_episodes,validate_physical_dynamics,valid_anchor_rows
    from minimal_qrl.baselines.mppi import _state_terms
    episodes,anchors,sources=physical_episodes(data.raw)
    dynamics=validate_physical_dynamics(data.raw,episodes,data.env)
    valid,_=valid_anchor_rows(data.raw,anchors,sources,data.env);assert len(valid)==752
    # Independently recompute all one-step costs, including collision/OOB self-loops.
    obs=data.raw.all_observations.numpy();rows=np.flatnonzero(data.physical.numpy())
    following=obs[sources[rows]+1];states=np.column_stack((following[:,:2],np.arctan2(following[:,3],following[:,2])))
    collision=np.zeros(len(data.src),bool);oob=collision.copy()
    for e in episodes:
        collision[e['rows'][-1]]=e['outcome']==2;oob[e['rows'][-1]]=e['outcome']==3
    errors=[]
    for device in range(len(data.env.device_ids)):
        mask=data.raw.transition_infos['device_index'][rows].numpy()==device
        data.env.set_task_by_device_id(data.env.device_ids[device])
        costs,_=_state_terms(data.env,states[mask],collision[rows][mask],oob[rows][mask])
        errors.extend(np.abs(costs-data.cost[rows[mask]].numpy()).tolist())
    assert max(errors)<5e-5,max(errors)
    # Pair indices must remain in the same physical episode, including its last state.
    for e in episodes:
        r=e['rows'];p=data.raw.transition_infos['global_push_pair_goal_obs_index'][r]
        assert (p>torch.from_numpy(sources[r])).all() and (p<=int(e['obs_ids'][-1])).all()
    return dict(physical_edges=int(data.physical.sum()),original_physical=200000,path_physical=335,
        abstract_anchors=len(valid),nodes=len(data.nodes),episodes=len(episodes),dynamics=dynamics,
        recomputed_cost_max_error=max(errors),b1_episode_boundary_checks=True,
        task_witness_original=int(data.raw.transition_infos['global_push_task_witness_mask'][:data.nbase].sum()),
        task_witness_paths=int(data.raw.transition_infos['global_push_task_witness_mask'][data.nbase:].sum()),
        fingerprints=data.fingerprint)
