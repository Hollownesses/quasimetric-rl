"""Adapt the frozen expert replay to the existing minibatch PHR training interface.

No graph-reference file is read. The original data loader/artifacts stay frozen.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from .data import load_data
from minimal_qrl.eval.task_path_support_audit import (
    physical_episodes, validate_physical_dynamics, valid_anchor_rows, exact_reachable)
from minimal_qrl.baselines.mppi import _state_terms


def load_training_data(root, *, task_mode='all'):
    base=load_data(root,task_mode=task_mode,seed=42)
    # The old minibatch trainer indexes goals/pairs by ALL replay rows. The
    # construction loader indexes these by physical rows only; anchors interleave.
    goal=torch.zeros(len(base.src),dtype=torch.long)
    pair=torch.zeros_like(goal)
    goal[base.rows]=base.goal;pair[base.rows]=base.pair
    data=SimpleNamespace(**vars(base))
    data.goal=goal;data.pair=pair;data.small=base
    data.original=base.rows;data.paths=torch.empty(0,dtype=torch.long)
    data.nbase=len(base.src);data.dataset_kind='expert_network'
    data.source_root=str(Path(root).resolve())
    data.fingerprint=dict(base.fingerprint,training_dataset='expert_network',
                          training_sampling='single_source_physical_records_v1')
    assert torch.equal(base.src[~base.physical],base.dst[~base.physical])
    assert torch.equal(base.raw.transition_infos['global_push_task_mask'],
                       base.raw.transition_infos['global_push_task_witness_mask'])
    return data


def audit_training_data(data):
    """Startup checks of dynamics, costs and witnesses; no shortest-path values."""
    raw,env=data.raw,data.env
    episodes,anchors,sources=physical_episodes(raw)
    dynamics=validate_physical_dynamics(raw,episodes,env)
    valid,_=valid_anchor_rows(raw,anchors,sources,env)
    assert len(valid)==len(anchors)==len(episodes)
    reachable=exact_reachable(raw,episodes,valid,sources,scope='strict_context')
    assert reachable[data.physical.numpy()].all()
    obs=raw.all_observations.numpy();rows=data.original.numpy()
    following=obs[sources[rows]+1]
    states=np.column_stack((following[:,:2],np.arctan2(following[:,3],following[:,2])))
    zero=np.zeros(len(rows),bool)
    costs,_=_state_terms(env,states,zero,zero)
    error=float(np.abs(costs-data.cost[rows].numpy()).max())
    assert error<5e-5,error
    pair_indices=raw.transition_infos['global_push_pair_goal_obs_index']
    for episode in episodes:
        r=episode['rows'];p=pair_indices[r].numpy()
        assert (p>sources[r]).all() and (p<=episode['obs_ids'][-1]).all()
    return dict(source_root=data.source_root,physical_edges=len(rows),anchors=len(anchors),
        successful_episodes=len(episodes),canonical_nodes=len(data.nodes),dynamics=dynamics,
        recomputed_cost_max_error=error,task_sources_with_strict_success_witness=int(reachable.sum()),
        all_equals_witnessed=True,b1_episode_boundary_checks=True,b1_seed=42,
        graph_references_loaded=False,fingerprints=data.fingerprint)
