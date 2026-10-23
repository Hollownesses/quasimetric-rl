"""Independent replay, exact joins, coverage, and post-build graph comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.expert_utrap_network.audit import reverse_distances, graph_structure
from experiments.expert_utrap_network.data import load_data as load_old_data
from experiments.expert_utrap_network.build import ROOT, DEVICE, sha256, write_json
from minimal_qrl.path_support import replay_actions
from minimal_qrl.eval.task_path_support_audit import physical_episodes
from .build import backbone_paths
from .data import load_training_data, audit_training_data


def audit(root):
    root=Path(root); out=root/'audit'; out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'audit_source.py')
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert sha256(root/name)==digest,name
    for name,digest in manifest['sources'].items():
        assert sha256(root/'source_snapshot'/name)==digest,name
    backbone=Path(manifest['backbone'])
    assert sha256(backbone/'manifest.json')==manifest['backbone_manifest_sha256']
    for name,digest in manifest['backbone_files'].items():
        assert sha256(backbone/name)==digest,('changed old artifact',name)
    data=load_training_data(root); report=audit_training_data(data)
    other=load_training_data(root,task_mode='witnessed')
    assert data.fingerprint==other.fingerprint
    records=json.loads((root/'records.json').read_text())
    trajectories=json.loads((root/'trajectories.json').read_text())
    paths,_,_=backbone_paths(data.env,backbone)
    episodes,anchors,sources=physical_episodes(data.raw)
    assert len(episodes)==len(records)==len(trajectories)
    joins=0; backbone_count=0
    for i,(ep,rec,trajectory) in enumerate(zip(episodes,records,trajectories)):
        actual=replay_actions(data.env,trajectory['states'][0],trajectory['actions'],device_id=DEVICE,seed=20261021+i)
        for field in ('observations','states','actions','rewards'):
            np.testing.assert_array_equal(np.asarray(actual[field],np.float32),np.asarray(trajectory[field],np.float32))
        np.testing.assert_array_equal(np.asarray(actual['observations'],np.float32),data.raw.all_observations[ep['obs_ids']].numpy())
        np.testing.assert_array_equal(np.asarray(actual['rewards'],np.float32),data.raw.rewards[ep['rows']].numpy())
        if rec['kind']=='backbone':
            original=paths[rec['backbone_path']]
            for field in ('observations','states','actions','rewards'):
                np.testing.assert_array_equal(np.asarray(actual[field],np.float32),np.asarray(original[field],np.float32))
            backbone_count+=1
        else:
            suffix=paths[rec['target_path']]
            np.testing.assert_array_equal(np.asarray(actual['observations'][rec['join_index']:],np.float32),
                                          np.asarray(suffix['observations'][rec['target_index']:],np.float32))
            joins+=1
    assert backbone_count==64
    src,dst,cost=data.src.numpy(),data.dst.numpy(),data.cost.numpy()
    rows=data.original.numpy(); goals=data.goal.unique().numpy()
    # Graph references are calculated only after dataset construction; never fed to the loader.
    J,next_edge=reverse_distances(len(data.nodes),src,dst,cost,goals)
    assert np.isfinite(J).all()
    assert np.all(J[src]<=cost.astype(float)+J[dst]+1e-8)
    np.savez_compressed(out/'graph_reference_audit_only.npz',distance=J,next_edge=next_edge,
                        source=src,target=dst,physical=data.physical.numpy())
    inc,adj=graph_structure(src,dst,rows,len(data.nodes))
    unique_edges={(int(src[r]),int(dst[r]),data.raw.actions[r].numpy().tobytes(),float(cost[r])) for r in rows}
    old=load_old_data(backbone)
    # Compare against the original serialized replay, not only a fresh replay of
    # its controls. This catches cost or environment drift with identical poses.
    for field in ('all_observations','actions','rewards','terminals','timeouts','episode_lengths'):
        old_value=getattr(old.raw,field)
        assert torch.equal(getattr(data.raw,field)[:len(old_value)],old_value),('backbone prefix changed',field)
    old_J,_=reverse_distances(len(old.nodes),old.src.numpy(),old.dst.numpy(),old.cost.numpy(),old.goal.unique().numpy())
    lookup={o.numpy().tobytes():i for i,o in enumerate(data.nodes)}
    old_ids=np.array([lookup[o.numpy().tobytes()] for o in old.nodes])
    improvements=old_J-J[old_ids]
    assert improvements.min()>-1e-7
    # Compare the exact old branch successors, without assuming ordering of node IDs.
    _,old_adj=graph_structure(old.src.numpy(),old.dst.numpy(),old.rows.numpy(),len(old.nodes))
    branches=[]
    for node, successors in enumerate(old_adj):
        if len(successors)<2:continue
        successors=sorted(successors)
        branches.append(dict(old_source=node,successors=successors,
            old_distances=old_J[successors].tolist(),new_distances=J[old_ids[successors]].tolist(),
            old_spread=float(np.ptp(old_J[successors])),new_spread=float(np.ptp(J[old_ids[successors]]))))
    attempts=[json.loads(line) for line in (root/'attempts.jsonl').read_text().splitlines()]
    attempt_counts=Counter((a['kind'],bool(a['accepted'])) for a in attempts)
    coverage=Counter(r['heading_bin'] for r in records if r['kind']=='heading_start')
    local={}
    for rec in records:
        if rec['kind']=='local_branch':
            local.setdefault((rec['source_path'],rec['source_index']),set()).add(rec['omega'])
    report.update(physical_records=len(rows),unique_physical_edges=len(unique_edges),
        audit_source_sha256=sha256(__file__),
        duplicate_records=len(rows)-len(unique_edges),canonical_nodes=len(data.nodes),
        branch_nodes=sum(len(x)>1 for x in adj),merge_nodes=sum(len(x)>1 for x in inc),
        preserved_backbone_trajectories=backbone_count,bit_exact_new_success_joins=joins,
        serialized_backbone_prefix_exact=True,
        trajectory_kinds=dict(Counter(r['kind'] for r in records)),
        max_episode_length=max(r['steps'] for r in records),
        all_source_nodes_have_graph_witness=True,
        attempts={f'{kind}_{accepted}':n for (kind,accepted),n in attempt_counts.items()},
        accepted_heading_starts_by_bin=dict(sorted(coverage.items())),
        local_sources_with_multiple_accepted_controls=sum(len(x)>1 for x in local.values()),
        old_node_distance_improvement=dict(count=int((improvements>1e-5).sum()),
            mean=float(improvements.mean()),max=float(improvements.max())),
        old_branch_successor_comparison=branches,
        interpretation='Expert-assisted success-connected graph, not a complete environment or held-out navigation benchmark.')
    write_json(out/'summary.json',report)
    # Static scientific artifact: old backbone and only new pre-join transitions.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(11,7))
    for obstacle in data.env.obstacles:
        if hasattr(obstacle,'x_min'):
            ax.add_patch(Rectangle((obstacle.x_min,obstacle.y_min),obstacle.x_max-obstacle.x_min,
                                   obstacle.y_max-obstacle.y_min,color='0.25'))
    for rec,traj in zip(records,trajectories):
        state=np.asarray(traj['states'])
        if rec['kind']=='backbone':ax.plot(state[:,0],state[:,1],color='0.65',alpha=.25,lw=.7)
        else:
            state=state[:rec['join_index']+1]
            color={'cross_route':'#e45756','local_branch':'#2878b5','heading_start':'#20a486'}[rec['kind']]
            ax.plot(state[:,0],state[:,1],color=color,alpha=.35,lw=.8)
    all_xy=np.concatenate([np.asarray(t['states'])[:,:2] for t in trajectories])
    ax.set(xlim=(0,max(10.,float(all_xy[:,0].max())+.25)),
           ylim=(0,max(8.,float(all_xy[:,1].max())+.25)),xlabel='x',ylabel='y',
           title='Connected expert graph: backbone (grey), local branches (blue)\nHeading starts (green), cross-route joins (red)')
    ax.set_aspect('equal');fig.tight_layout();fig.savefig(out/'network.png',dpi=160);plt.close(fig)
    lines=['# Connected expert U-trap network: construction audit','',
        f"Physical records: **{len(rows)}**; unique physical edges: **{len(unique_edges)}**; canonical nodes: **{len(data.nodes)}**.",
        f"All {len(records)} trajectories replay exactly to legal success; all 64 old trajectories remain unchanged. New exact suffix joins: {joins}.",
        f"Branch nodes: {report['branch_nodes']}; merge nodes: {report['merge_nodes']}; local sources with multiple accepted controls: {report['local_sources_with_multiple_accepted_controls']}.",
        '', '## Accepted construction groups', '', '| Group | Trajectories |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in report['trajectory_kinds'].items()]
    lines += ['',f'New start heading bins (0..7 = -pi .. 3pi/4): `{dict(sorted(coverage.items()))}`.',
        '', '## Old branch successor costs', '', '| Old source | Old spread | New spread |','|---|---:|---:|']
    lines += [f"| {b['old_source']} | {b['old_spread']:.6f} | {b['new_spread']:.6f} |" for b in branches]
    lines += ['',f"Old nodes with a shorter graph path: {report['old_node_distance_improvement']['count']}/{len(old_J)}; maximum reduction {improvements.max():.6f}.",
        '', 'All edges are real env.step transitions; exact full observations define joins. No nearest-neighbour edges, snapping, or Oracle/shortest-path training labels. Graph references are audit-only.',
        '', 'Rejected proposals and search exhaustion are retained in attempts.jsonl, not inserted as successful replay. Search failure does not prove unreachability. Dataset sampling remains uniform over physical records; duplicate/shared suffix weights are explicit, not silently deduplicated.',
        '', 'This expert-assisted construction reuses research states and has no held-out claim. All/witnessed task masks agree because every accepted episode has a real success suffix. No MPPI/learner changes or formal training were performed by this audit.',
        '', '![Network](network.png)','']
    (out/'report.md').write_text('\n'.join(lines))
    write_json(root/'COMPLETE',dict(audit_summary_sha256=sha256(out/'summary.json'),trajectories=len(records),training_started=False))
    print(json.dumps({k:report[k] for k in ('physical_records','unique_physical_edges','branch_nodes','merge_nodes','trajectory_kinds','accepted_heading_starts_by_bin','old_branch_successor_comparison')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);audit(p.parse_args().root)
