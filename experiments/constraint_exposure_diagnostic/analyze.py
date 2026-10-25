"""Post-budget edge localization, path exposure and saved-dual diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.connected_expert_utrap.compare import graph_inputs
from experiments.connected_phr_solver.objective import all_distances
from experiments.four_path_factorial.run import IndexedIQE
from .core import (TOL, edge_index, branch_neighborhood, shortest_exposure,
                   rollout_exposure, residual_stats, dual_window, score_error_identity)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'results/connected_expert_utrap_20261021/main'
REFERENCE = ROOT/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz'
BINS = [('1', 1, 1), ('2-4', 2, 4), ('5-16', 5, 16), ('17-64', 17, 64), ('65+', 65, 100000)]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def inputs():
    runs = {}
    for seed in (606, 707, 808):
        base = (ROOT/'results/task_source_balance_20261025/main' if seed == 606 else
                ROOT/f'results/task_source_balance_multiseed_20261025/seed{seed}/main')
        for arm in ('R0', 'R1'):
            runs[f'{seed}_{arm}'] = dict(folder=base/arm, comparison=base/'comparison', arm=arm, seed=seed)
    base = ROOT/'results/task_partial_balance_20261025/main'
    runs['606_Rhalf'] = dict(folder=base/'Rhalf', comparison=base/'comparison', arm='Rhalf', seed=606)
    return runs


@torch.no_grad()
def checkpoint(folder, file, data, hashes):
    path = folder/file
    hashes[str(path.relative_to(ROOT))] = digest(path)
    ck = torch.load(path, map_location='cpu', weights_only=True)
    assert ck['fingerprints'] == data.fingerprint
    assert ck['step'] == (45000 if file == 'phr.pt' else 50000)
    model = IndexedIQE(data.small, 'encoder', 0, initialization_seed=ck['initialization_seed'])
    model.load_state_dict(ck['model']); model.eval()
    prediction, edges = all_distances(model, data, batch=512)
    h = edges-data.cost.double().numpy()
    if not np.isfinite(h).all() or not np.isfinite(prediction).all():
        raise ValueError('nonfinite frozen prediction')
    return ck, prediction, h


def branch_details(decision, graph, data, reference, prediction, exposure, contribution, row_edge, h):
    src, dst, cost = data.src.numpy(), data.dst.numpy(), data.cost.double().numpy()
    details = []
    for branch in decision['branches']:
        node = branch['source']; choice = branch['chosen_row']
        rows = np.asarray(graph.outgoing[node])
        if choice != rows[np.argmin(cost[rows]+prediction[dst[rows]])]:
            raise ValueError('frozen branch choice mismatch')
        components = score_error_identity(cost, prediction, reference, dst, rows, choice)
        if abs(components['regret']-branch['regret']) > 1e-7:
            raise ValueError('branch regret mismatch')
        index = row_edge[choice]
        details.append(dict(source=node, xy=data.nodes[node, :2].tolist(), chosen_row=choice,
                            exposure=int(exposure[index]), contribution=float(contribution[index]),
                            chosen_h=float(h[choice]), outgoing_max_h=float(h[rows].max()), **components))
    return sorted(details, key=lambda item: -item['contribution'])


def localize(data, ref, reps, row_edge, group_masks, run, h0, h1, ck0, ck1,
             decision, prediction, graph, hashes):
    physical = data.physical.numpy(); rows = data.original.numpy()
    visits = np.zeros(len(data.src), int); visits[rows] = (ck1['visits']-ck0['visits']).numpy()
    total_visits = np.zeros(len(data.src), int); total_visits[rows] = ck1['visits'].numpy()
    edge_visits = np.bincount(row_edge[physical], weights=total_visits[physical], minlength=len(reps))
    lam0, lam1 = ck0['dual'].double().numpy(), ck1['dual'].double().numpy()
    assert ck0['initialization_seed'] == ck1['initialization_seed'] == run['seed']
    assert np.all(visits[rows] > 0) and np.isfinite(lam1).all()
    exposure, contribution, regrets = rollout_exposure(
        decision['episodes'], data.src.numpy(), data.dst.numpy(), data.cost.double().numpy(),
        row_edge, ref, graph.goal, unique=True)
    exposure307, _, regrets307 = rollout_exposure(
        decision['episodes'], data.src.numpy(), data.dst.numpy(), data.cost.double().numpy(),
        row_edge, ref, graph.goal, unique=False)
    expected = decision['summary']['unique_start_rollouts']['mean_successful_regret']
    assert abs(regrets.mean()-expected) < 1e-7
    groups, temporal = {}, {}
    for name, mask in group_masks.items():
        record_mask = physical & mask[np.maximum(row_edge, 0)]
        groups[name] = dict(edges=residual_stats(h1[reps], mask),
                            records=residual_stats(h1, record_mask),
                            visits_per_unique_edge=dict(min=int(edge_visits[mask].min()),
                                max=int(edge_visits[mask].max()), mean=float(edge_visits[mask].mean())) if mask.any() else {})
        temporal[name] = dual_window(h0, h1, lam0, lam1, visits, record_mask)
    persistent_edges = (h0[reps] > TOL) & (h1[reps] > TOL)
    on_path = exposure > 0
    for name, mask in [('on_selected165', on_path), ('both_endpoints_positive', persistent_edges)]:
        temporal[name] = dual_window(h0, h1, lam0, lam1, visits,
                                    physical & mask[np.maximum(row_edge, 0)])
    branch = branch_details(decision, graph, data, ref, prediction, exposure, contribution, row_edge, h1)
    assert abs(sum(b['contribution'] for b in branch)-regrets.mean()) < 1e-7
    return dict(groups=groups, dual45to50=temporal, branches=branch,
                unique_start_regret=float(regrets.mean()), start307_regret=float(regrets307.mean()),
                selected165_path_positive_sum=float(np.dot(exposure, np.maximum(h1[reps], 0))/len(regrets)),
                selected307_path_positive_sum=float(np.dot(exposure307, np.maximum(h1[reps], 0))/len(regrets307)),
                exposed_unique_edges=int(on_path.sum()),
                visits45to50_min=int(visits[rows].min()), visits45to50_max=int(visits[rows].max())), exposure, contribution


def compare(name, data, reps, row_edge, masks, base, arm, summary, exposures, residuals, checkpoints):
    h0, h1 = residuals[base][1], residuals[arm][1]
    p0, p1 = np.maximum(h0[reps], 0), np.maximum(h1[reps], 0)
    delta = p1-p0; new = (h0[reps] <= TOL) & (h1[reps] > TOL)
    vanished = (h0[reps] > TOL) & (h1[reps] <= TOL)
    groups = {}
    for label, mask in masks.items():
        groups[label] = dict(edges=int(mask.sum()), new=int((new & mask).sum()),
                             vanished=int((vanished & mask).sum()),
                             new_fraction=float(new[mask].mean()) if mask.any() else None,
                             R0_mean_positive=float(p0[mask].mean()) if mask.any() else None,
                             arm_mean_positive=float(p1[mask].mean()) if mask.any() else None,
                             sum_positive_delta=float(delta[mask].sum()))
    ex0, ex1 = exposures[base], exposures[arm]
    result = dict(base=base, arm=arm, groups=groups, new_edges=int(new.sum()),
                  vanished_edges=int(vanished.sum()), new_at_1e3=int(((h0[reps]<=1e-3)&(h1[reps]>1e-3)).sum()),
                  sum_positive_delta=float(delta.sum()),
                  R0_paths_R0_h=float(np.dot(ex0, p0)/165), R0_paths_arm_h=float(np.dot(ex0, p1)/165),
                  arm_paths_R0_h=float(np.dot(ex1, p0)/165), arm_paths_arm_h=float(np.dot(ex1, p1)/165),
                  path_regret_delta=summary[arm]['unique_start_regret']-summary[base]['unique_start_regret'])
    # Keep record dual histories; aggregate exact duplicates explicitly for display only.
    physical = data.physical.numpy(); rows_by_edge = [[] for _ in reps]
    for row in np.flatnonzero(physical): rows_by_edge[row_edge[row]].append(row)
    c0, c1 = checkpoints[arm]
    lam0, lam1 = c0['dual'].numpy(), c1['dual'].numpy()
    new_record = physical & new[np.maximum(row_edge, 0)]
    visit_delta = np.zeros(len(physical), int)
    visit_delta[data.original.numpy()] = (c1['visits']-c0['visits']).numpy()
    result['new_edge_dual45to50'] = dual_window(residuals[arm][0], h1, lam0, lam1, visit_delta, new_record)
    ranked = set(np.argsort(-p1)[:10].tolist()) | set(np.argsort(-(p1*ex1))[:10].tolist())
    top = []
    for edge in ranked:
        row = reps[edge]; rr = rows_by_edge[edge]
        top.append(dict(edge=int(edge), row=int(row), source=int(data.src[row]), target=int(data.dst[row]),
                        xy=data.nodes[data.src[row], :2].tolist(), source_count=int((data.src[data.original]==data.src[row]).sum()),
                        copies=len(rr), R0_h=float(h0[row]), h45=float(residuals[arm][0][row]),
                        h50=float(h1[row]), R0_exposure=int(ex0[edge]), arm_exposure=int(ex1[edge]),
                        lambda45_mean=float(lam0[rr].mean()), lambda50_mean=float(lam1[rr].mean()),
                        lambda50_min=float(lam1[rr].min()), lambda50_max=float(lam1[rr].max()),
                        near_branch=bool(masks['near_branch3'][edge]), far_source=bool(masks['far_source'][edge]),
                        far_shortest_used=bool(masks['far_shortest_paths'][edge]), new=bool(new[edge])))
    result['top_edges'] = sorted(top, key=lambda item: -max(item['h50'], 0)*item['arm_exposure'])
    return result, new


def plots(out, data, reps, summaries, comparisons, residuals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    pairs = [('606_R0','606_R1'),('707_R0','707_R1'),('808_R0','808_R1'),('606_R0','606_Rhalf')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    xy = (data.nodes[data.src[reps], :2].numpy()+data.nodes[data.dst[reps], :2].numpy())/2
    for ax, (base, arm) in zip(axes.flat, pairs):
        delta = np.maximum(residuals[arm][1][reps],0)-np.maximum(residuals[base][1][reps],0)
        for obstacle in data.env.obstacles:
            if hasattr(obstacle,'x_min'):
                ax.add_patch(Rectangle((obstacle.x_min,obstacle.y_min),obstacle.x_max-obstacle.x_min,
                                       obstacle.y_max-obstacle.y_min,color='.35'))
        # Fixed display range across all panels, all points retained.
        order = np.argsort(np.abs(delta))
        scatter = ax.scatter(xy[order,0],xy[order,1],c=delta[order],s=6,cmap='coolwarm',vmin=-1,vmax=1)
        ax.set(title=arm+' minus '+base, xlabel='x', ylabel='y', aspect='equal')
    fig.colorbar(scatter,ax=list(axes.flat),label='Change in positive edge residual (color clipped to +/-1)')
    fig.savefig(out/'localization.png',dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    names = list(comparisons); x=np.arange(len(names)); width=.35
    for ax,key,title in [(axes[0],'R0_paths_R0_h','Violation exposure on fixed R0 paths'),
                         (axes[1],'path_regret_delta','Change in mean path regret')]:
        if key=='path_regret_delta':
            ax.bar(x,[comparisons[n][key] for n in names]);ax.axhline(0,color='k',lw=.6)
        else:
            ax.bar(x-width/2,[comparisons[n][key] for n in names],width,label='R0 residual')
            ax.bar(x+width/2,[comparisons[n]['R0_paths_arm_h'] for n in names],width,label='Balanced residual');ax.legend()
        ax.set(xticks=x,xticklabels=names,title=title)
    for arm,color in [('R0','tab:blue'),('R1','tab:orange')]:
        for seed in (606,707,808):
            d=summaries[f'{seed}_{arm}']['dual45to50']['both_endpoints_positive']
            v=d['persistent'];axes[2].scatter(v['lambda50_mean'],v['h50_mean'],c=color,label=arm if seed==606 else None)
            axes[2].annotate(str(seed),(v['lambda50_mean'],v['h50_mean']),fontsize=8)
    axes[2].set(xlabel='Mean multiplier (persistent records)',ylabel='Mean residual at 50k',title='Positive endpoints despite nonzero dual');axes[2].legend()
    fig.savefig(out/'exposure_dual.png',dpi=170);plt.close(fig)


def main(args):
    torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sources=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('PROTOCOL.md')]
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources}
    write(out/'protocol.json',dict(source_hashes=source_hashes,scope='read-only; fixed existing endpoints',
                                  branch_hops=3,violation_tolerance=TOL,float_absolute_tolerance=5e-4))
    data=load_training_data(DATA);graph,starts=graph_inputs(data)
    src,dst,cost=data.src.numpy(),data.dst.numpy(),data.cost.double().numpy()
    physical=data.physical.numpy();rows=data.original.numpy();hashes={}
    for file in [DATA/'verified_replay.pt',DATA/'manifest.json',DATA/'scenario.json',REFERENCE]:
        hashes[str(file.relative_to(ROOT))]=digest(file)
    with np.load(REFERENCE) as arrays:
        ref=arrays['distance'].copy();nxt=arrays['next_edge'].copy()
        np.testing.assert_array_equal(arrays['source'],src);np.testing.assert_array_equal(arrays['target'],dst)
    validate_reference(data,ref)
    reps,row_edge=edge_index(src,dst,data.raw.actions.numpy(),cost,physical)
    assert len(reps)==6958 and len(np.unique(starts))==165
    degree,near=branch_neighborhood(src,dst,reps,len(ref),3)
    counts=np.bincount(src[rows],minlength=len(ref));unique=np.flatnonzero(counts)
    copies=np.bincount(row_edge[physical],minlength=len(reps))
    far=unique[ref[unique]>=np.quantile(ref[unique],.75)]
    reference_exposure=shortest_exposure(far,nxt,src,dst,row_edge,ref)
    masks={'all':np.ones(len(reps),bool),'far_source':np.isin(src[reps],far),
           'far_shortest_paths':reference_exposure>0,'branch_outgoing':degree[src[reps]]>1,
           'near_branch3':(near[src[reps]]>=0)|(near[dst[reps]]>=0)}
    masks['away_branch3']=~masks['near_branch3']
    for name,low,high in BINS:masks['multiplicity_'+name]=(counts[src[reps]]>=low)&(counts[src[reps]]<=high)
    for name,low,high in BINS:masks['edge_copies_'+name]=(copies>=low)&(copies<=high)
    summaries={};exposures={};residuals={};checkpoints={};verifications={}
    for name,run in inputs().items():
        print('Reading '+name,flush=True)
        folder=run['folder'];comp=run['comparison']
        for file in [folder/'config.json',folder/'COMPLETE',comp/(run['arm']+'_decisions.json'),comp/'arrays.npz',folder/'final_constraints.npz']:
            hashes[str(file.relative_to(ROOT))]=digest(file)
        ck0,_,h0=checkpoint(folder,'phr.pt',data,hashes)
        ck1,prediction,h1=checkpoint(folder,'final.pt',data,hashes)
        assert ck1['step']==json.loads((folder/'COMPLETE').read_text())['step']==50000
        assert digest(folder/'final.pt')==json.loads((folder/'COMPLETE').read_text())['final_sha256']
        with np.load(comp/'arrays.npz') as saved:
            # All existing comparisons use identical node/query batching.
            pdiff=float(np.max(abs(prediction-saved[run['arm']+'_prediction'])))
        with np.load(folder/'final_constraints.npz') as saved:
            hdiff=float(np.max(abs(h1-saved['residual'])))
            np.testing.assert_array_equal(ck1['dual'].numpy(),saved['dual'])
        assert max(pdiff,hdiff)<=5e-4,(name,pdiff,hdiff)
        decision=json.loads((comp/(run['arm']+'_decisions.json')).read_text())
        summaries[name],exposures[name],_=localize(data,ref,reps,row_edge,masks,run,h0,h1,ck0,ck1,
                                                  decision,prediction,graph,hashes)
        residuals[name]=(h0,h1);checkpoints[name]=(ck0,ck1)
        hi=np.full(len(reps),-np.inf);lo=np.full(len(reps),np.inf)
        np.maximum.at(hi,row_edge[physical],h1[physical]);np.minimum.at(lo,row_edge[physical],h1[physical])
        verifications[name]=dict(prediction_max_difference=pdiff,scan_residual_max_difference=hdiff,
                                  checkpoint_fingerprint=True,path_cost_and_regret_identity=True,
                                  duplicate_residual_max_range=float((hi-lo).max()))
    comparisons={};newsets={}
    for seed in (606,707,808):
        for stage in (0,1):
            a=checkpoints[f'{seed}_R0'][stage];b=checkpoints[f'{seed}_R1'][stage]
            assert torch.equal(a['visits'],b['visits'])
            assert a['sampling_chain_sha256']==b['sampling_chain_sha256']
        label=str(seed);comparisons[label],new=compare(label,data,reps,row_edge,masks,f'{seed}_R0',f'{seed}_R1',summaries,exposures,residuals,checkpoints)
        newsets[label]=set(np.flatnonzero(new).tolist())
    comparisons['606half'],_=compare('606half',data,reps,row_edge,masks,'606_R0','606_Rhalf',summaries,exposures,residuals,checkpoints)
    # Full detail is local only, not staged with the report.
    write(out/'summary.json',dict(runs=summaries,comparisons=comparisons,verification=verifications,
                                 shared_new_edges=len(set.intersection(*newsets.values())),
                                 union_new_edges=len(set.union(*newsets.values())),
                                 group_definitions={k:int(v.sum()) for k,v in masks.items()},
                                 physical_records=len(rows),unique_edges=len(reps),branch_nodes=int((degree>1).sum()),
                                 fingerprints=data.fingerprint))
    np.savez_compressed(out/'edges.npz',representatives=reps,row_edge=row_edge,source=src,target=dst,
                        source_counts=counts,branch_distance=near,far_reference_exposure=reference_exposure,
                        **{name+'_h'+str(step):h for name,hs in residuals.items() for step,h in zip((45,50),hs)},
                        **{name+'_exposure165':e for name,e in exposures.items()})
    plots(out,data,reps,summaries,comparisons,residuals)
    for path,before in hashes.items():
        if digest(ROOT/path)!=before:raise ValueError('input changed: '+path)
    for path,before in source_hashes.items():
        if digest(ROOT/path)!=before:raise ValueError('analysis source changed: '+path)
    write(out/'integrity.json',dict(passed=True,inputs_unchanged=True,sources_unchanged=True,
                                  input_sha256=hashes,source_sha256=source_hashes,
                                  evaluation_only=True,training_updates=0))
    print(json.dumps(dict(done=str(out),shared_new_edges=len(set.intersection(*newsets.values())))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
