"""Training loader: all successes/failures; normal one-step fields; no references."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from quasimetric_rl.data import MultiEpisodeData
from experiments.four_path_factorial.run import Replay,canonical_nodes,tensor_digest
from minimal_qrl.global_push_sampling import replay_fingerprint,configure_global_push_sampling
from minimal_qrl.eval.comm_inspection_oracle_bank import environment_signature
from minimal_qrl.eval.task_path_support_audit import (
    physical_episodes,validate_physical_dynamics,valid_anchor_rows,exact_reachable)
from minimal_qrl.baselines.mppi import _state_terms
from .common import make_env,sha


def load_training_data(root,*,task_mode='all'):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['dataset_kind']!='curriculum_network':raise ValueError('wrong dataset kind')
    for name in ('verified_replay.pt','scenario.json'):
        if sha(root/name)!=manifest['files'][name]:raise ValueError('data file fingerprint mismatch: '+name)
    payload=torch.load(root/'verified_replay.pt',map_location='cpu',weights_only=True)
    raw=MultiEpisodeData(**payload['raw'])
    if replay_fingerprint(raw)!=payload['sha256'] or payload['sha256']!=manifest['replay_sha256']:
        raise ValueError('replay fingerprint mismatch')
    env=make_env(json.loads((root/'scenario.json').read_text()))
    if environment_signature(env)['digest']!=manifest['environment_signature']['digest']:
        raise ValueError('environment mismatch')
    replay=Replay(raw,env);configure_global_push_sampling(replay,state_mode='trajectory',task_mode=task_mode,seed=42)
    infos=raw.transition_infos
    if infos['goal_return_mask'].any() or infos['goal_return_cost'].any():raise ValueError('return targets forbidden')
    if infos['teacher_guided'].any() or infos['planner_path_support'].any():raise ValueError('expert rows forbidden')
    # Include the abstract task even if an alternate small collection has zero successes.
    all_obs=torch.cat((raw.all_observations,infos['task_goal_observations'][:1]))
    nodes,full_ids,mapping=canonical_nodes(all_obs);ids=full_ids[:-1]
    source=torch.arange(len(replay))+replay.indices_to_episode_indices
    physical=~infos['abstract_goal_edge'];rows=physical.nonzero().flatten()
    goal=torch.full((len(replay),),int(full_ids[-1]),dtype=torch.long)
    pair_obs=infos['global_push_pair_goal_obs_index'];pair=ids[pair_obs.clamp_min(0)]
    assert infos['global_push_pair_mask'][rows].all()
    d=SimpleNamespace(raw=raw,replay=replay,env=env,nodes=nodes,obs_ids=ids,src_obs=source,
        src=ids[source],dst=ids[source+1],cost=-raw.rewards,physical=physical,rows=rows,
        task_src=ids[source[rows]],goal=goal,pair=pair,original=rows,paths=torch.empty(0,dtype=torch.long),
        nbase=len(replay),dataset_kind='curriculum_network',source_root=str(root.resolve()))
    d.small=SimpleNamespace(nodes=nodes,env=env)
    d.fingerprint=dict(replay=payload['sha256'],file=manifest['files']['verified_replay.pt'],
        nodes=tensor_digest(nodes,ids),pairs=tensor_digest(d.src,goal,pair),costs=tensor_digest(d.cost),
        task_mask=tensor_digest(infos['global_push_task_mask']),b1_seed=42,
        training_dataset='curriculum_network',training_sampling='single_source_all_physical_records_v1')
    return d


def audit_training_data(d):
    raw,env=d.raw,d.env
    eps,anchors,sources=physical_episodes(raw)
    dynamics=validate_physical_dynamics(raw,eps,env)
    valid,_=valid_anchor_rows(raw,anchors,sources,env)
    assert len(valid)==len(anchors)==sum(e['outcome']==1 for e in eps)
    anchor_contexts={int(raw.transition_infos['context_id'][i]) for i in anchors}
    assert all((e['context'] in anchor_contexts)==(e['outcome']==1) for e in eps)
    reach=exact_reachable(raw,eps,valid,sources,scope='strict_context')
    assert np.array_equal(reach,raw.transition_infos['global_push_task_witness_mask'].numpy())
    collision=np.zeros(len(d.src),bool);oob=collision.copy()
    for e in eps:
        collision[e['rows'][-1]]=e['outcome']==2;oob[e['rows'][-1]]=e['outcome']==3
        r=e['rows'];p=raw.transition_infos['global_push_pair_goal_obs_index'][r].numpy()
        assert (p>sources[r]).all() and (p<=e['obs_ids'][-1]).all()
    rows=d.original.numpy();following=raw.all_observations.numpy()[sources[rows]+1]
    states=np.column_stack((following[:,:2],np.arctan2(following[:,3],following[:,2])))
    # The environment's scalar calculation is authoritative. MPPI's vector
    # approximation rounds the target position to float32; near the target this
    # can amplify bearing error. Report it separately without changing either.
    vector_costs,_=_state_terms(env,states,collision[rows],oob[rows])
    costs=np.array([env.compute_step_terms(s,bool(collision[r]),bool(oob[r]))['cost_total']
                    for s,r in zip(states,rows)])
    error=float(np.abs(costs-d.cost[rows].numpy()).max());assert error<5e-5
    vector_error=float(np.abs(vector_costs-d.cost[rows].numpy()).max())
    if len(anchors):assert torch.equal(d.src[~d.physical],d.dst[~d.physical])
    return dict(physical_edges=len(rows),anchors=len(anchors),episodes=len(eps),dynamics=dynamics,
        recomputed_cost_max_error=error,mppi_vector_cost_max_error=vector_error,
        cost_authority='environment.compute_step_terms; raw env.step bit-exact audit is separate',
        strict_context_witness_rows=int(reach.sum()),
        witnessed_task_mask_sha256=tensor_digest(raw.transition_infos['global_push_task_witness_mask']),
        all_equals_witnessed=bool(reach[rows].all()),failed_episodes=sum(e['outcome']!=1 for e in eps),
        failed_episode_anchors=0,all_failures_retained=True,b1_episode_boundary_checks=True,
        fingerprints=d.fingerprint,graph_references_loaded=False)
