"""Read-only signed shortest-path slack and full-objective stationarity audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.connected_expert_utrap.compare import graph_inputs,write_json
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.four_path_factorial.run import IndexedIQE,file_digest
from .objective import full_gradients,all_distances


def decompose(reference,next_edge,dst,cost,prediction,edge):
    length=np.zeros(len(reference));edge_slack=np.zeros(len(reference));hops=np.zeros(len(reference),int)
    for n in np.argsort(reference):
        if reference[n]==0:continue
        r=int(next_edge[n]);assert r>=0
        t=int(dst[r]);assert reference[t]<reference[n]
        assert abs(cost[r]+reference[t]-reference[n])<1e-8
        length[n]=edge[r]+length[t];edge_slack[n]=cost[r]-edge[r]+edge_slack[t];hops[n]=hops[t]+1
    triangle=length-prediction;gap=reference-prediction
    assert np.max(abs(gap-edge_slack-triangle))<1e-7
    return dict(gap=gap,edge_slack=edge_slack,triangle_slack=triangle,hops=hops)


def audit_checkpoint(data,reference,next_edge,checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['fingerprints']==data.fingerprint and saved['training_signature']['push'].endswith('0.4 fixed B1')
    model=IndexedIQE(data.small,'encoder',0,initialization_seed=saved['initialization_seed'])
    model.load_state_dict(saved['model']);model.eval()
    prediction,edge=all_distances(model,data)
    pieces=decompose(reference,next_edge,data.dst.numpy(),data.cost.double().numpy(),prediction,edge)
    _,starts=graph_inputs(data);unique=data.src[data.original].unique().numpy()
    groups=dict(records=data.src[data.original].numpy(),unique_sources=unique,starts=starts,
                unique_starts=np.unique(starts),longest_quartile=unique[reference[unique]>=np.quantile(reference[unique],.75)])
    aggregates={name:{key:float(values[ids].mean()) for key,values in pieces.items()} for name,ids in groups.items()}
    metrics,vectors,h=full_gradients(model,data,saved['dual'])
    for name,ids in groups.items():
        aggregates[name]['negative_edge_slack_fraction']=float((pieces['edge_slack'][ids]<0).mean())
    return dict(step=saved['step'],checkpoint_sha256=file_digest(checkpoint),groups=aggregates,
                full_objective=metrics,minimum_triangle_slack=float(pieces['triangle_slack'].min()),
                identity_max_error=float(abs(pieces['gap']-pieces['edge_slack']-pieces['triangle_slack']).max())),pieces


def run(args):
    torch.set_num_threads(1);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    data=load_training_data(args.data)
    for key in ('global_push_task_mask','global_push_pair_mask'):
        assert data.raw.transition_infos[key][data.original].bool().all()
    refroot=Path(args.reference);info=json.loads((refroot/'summary.json').read_text())
    assert info['fingerprints']==data.fingerprint
    assert info['reference_sha256']==file_digest(refroot/'graph_reference_audit_only.npz')
    with np.load(refroot/'graph_reference_audit_only.npz') as a:
        reference=a['distance'].copy();next_edge=a['next_edge'].copy()
        assert np.array_equal(a['source'],data.src.numpy()) and np.array_equal(a['target'],data.dst.numpy())
    validate_reference(data,reference);results={};arrays={}
    for name,file in [('45k','phr.pt'),('50k','final.pt')]:
        checkpoint=Path(args.run)/file;before=file_digest(checkpoint)
        results[name],pieces=audit_checkpoint(data,reference,next_edge,checkpoint)
        assert file_digest(checkpoint)==before
        arrays.update({name+'_'+k:v for k,v in pieces.items()})
        print(json.dumps(dict(stage=name,**results[name]),ensure_ascii=False),flush=True)
    np.savez_compressed(out/'decomposition.npz',reference=reference,**arrays)
    write_json(out/'summary.json',dict(fingerprints=data.fingerprint,reference_sha256=info['reference_sha256'],
        results=results,source_sha256=file_digest(__file__),scope='Read-only original task+B1 objective; no training or constraint replacement'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label,color in [('edge_slack','tab:blue'),('triangle_slack','tab:orange')]:
        axes[0].scatter(reference,arrays['50k_'+label],s=3,alpha=.25,label=label,color=color)
    axes[0].axhline(0,color='k',lw=.7);axes[0].set(xlabel='Reference task distance',ylabel='Signed decomposition');axes[0].legend()
    keys=['task','b1','phr','augmented','lagrangian'];x=np.arange(len(keys))
    for j,stage in enumerate(results):
        axes[1].bar(x+(j-.5)*.35,[results[stage]['full_objective']['gradient_norms'][k] for k in keys],width=.35,label=stage)
    axes[1].set(xticks=x,xticklabels=keys,ylabel='Full-record gradient norm');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'diagnostic.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','reference','run','output'):p.add_argument('--'+key,required=True)
    run(p.parse_args())
