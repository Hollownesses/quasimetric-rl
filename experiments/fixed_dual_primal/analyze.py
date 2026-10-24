"""Post-budget pairing, optional prespecified probes, and graph evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from experiments.connected_expert_utrap.compare import evaluate,graph_inputs,equality,write_json
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.connected_phr_solver.diagnostic import decompose
from experiments.connected_phr_solver.objective import all_distances
from experiments.connected_phr_solver.train import SOURCE,SOURCE_SHA,DATA
from experiments.four_path_factorial.run import IndexedIQE,file_digest,tensor_digest
from .probe import probe_state


def read_history(path):
    """A boundary replay may repeat audit rows; require identical scientific data."""
    rows={};timing={'training_seconds','audit_seconds','elapsed_seconds'}
    for line in Path(path).read_text().splitlines():
        rec=json.loads(line);p=rec['completed_passes']
        if p in rows:
            assert {k:v for k,v in rec.items() if k not in timing}=={
                k:v for k,v in rows[p].items() if k not in timing},'non-deterministic duplicate pass'
        rows[p]=rec
    return [rows[p] for p in sorted(rows)]


def regret_exposure(decision):
    """Exact graph-path regret accounting: local regret times visits."""
    visits=Counter(r for ep in decision['episodes'] for r in ep['rows'])
    count=len(decision['episodes'])
    contributions=[dict(source=b['source'],visits=visits[b['chosen_row']],regret=b['regret'],
        mean_contribution=visits[b['chosen_row']]*b['regret']/count) for b in decision['branches']]
    mean=sum(b['mean_contribution'] for b in contributions)
    assert abs(mean-np.mean([ep['regret'] for ep in decision['episodes']]))<1e-7
    return contributions


def paired(parent,initials,finals,logs,n):
    for arm in ('mini','full'):
        for key in ('model','optimizer','dual','visits','rng','push_counters'):
            assert equality(parent[key],initials[arm][key]),(arm,'initial',key)
        final=finals[arm]
        assert final['completed_passes']==20 and final['processed_records']==20*n
        assert torch.equal(final['dual'],parent['dual'])
        assert torch.equal(final['visits']-parent['visits'],torch.full_like(final['visits'],20))
        updates=20 if arm=='full' else 20*((n+255)//256)
        assert final['primal_updates']==updates and final['step']==parent['step']+updates
        assert [r['completed_passes'] for r in logs[arm]]==list(range(21))
        for rec in logs[arm]:
            assert rec['dual_sha256']==tensor_digest(parent['dual'])
            assert rec['processed_records']==rec['completed_passes']*n and rec['anchor_max']==0.
    for key in ('sampler','visits','rng','push_counters','processed_records','dual'):
        assert equality(finals['mini'][key],finals['full'][key]),key
    assert [r['order_sha256'] for r in logs['mini']]==[r['order_sha256'] for r in logs['full']]
    assert logs['mini'][0]['full_objective']==logs['full'][0]['full_objective']
    return dict(initial_model_adam_dual_rng_exact=True,all_duals_frozen=True,
        all_epoch_permutations_equal=True,each_record_visits=20,train_forward_records=20*n,
        train_backward_records=20*n,mini_updates=finals['mini']['primal_updates'],full_updates=20)


def run(args):
    torch.set_num_threads(1);root=Path(args.root);out=root/'comparison';out.mkdir(exist_ok=False)
    data=load_training_data(DATA);assert len(data.original)==42536
    assert file_digest(SOURCE)==SOURCE_SHA;parent=torch.load(SOURCE,weights_only=True)
    initial={};final={};logs={};cfgs={};complete={}
    for arm in ('mini','full'):
        folder=root/arm;complete[arm]=json.loads((folder/'COMPLETE').read_text())
        assert complete[arm]['passes']==20 and complete[arm]['dual_unchanged'] and not complete[arm]['reference_loaded']
        cfgs[arm]=json.loads((folder/'config.json').read_text())
        assert cfgs[arm]['source_sha256']==SOURCE_SHA
        for name,sha in cfgs[arm]['source_files'].items():assert file_digest(name)==sha,'frozen source changed'
        initial[arm]=torch.load(folder/'initial.pt',weights_only=True)
        final[arm]=torch.load(folder/'final.pt',weights_only=True)
        logs[arm]=read_history(folder/'training.jsonl')
        for p in (0,1,5,10,15,20):assert json.loads((folder/f'supervision_{p:02d}.json').read_text())['passed']
    for key in cfgs['mini']:
        if key!='arm':assert cfgs['mini'][key]==cfgs['full'][key]
    pairing=paired(parent,initial,final,logs,len(data.original))
    hashes={arm:file_digest(root/arm/'final.pt') for arm in ('mini','full')}
    # Evaluate the prespecified objective trigger BEFORE reading any graph reference.
    trigger=logs['full'][-1]['augmented_loss']>=logs['full'][0]['augmented_loss']-1e-4
    probes=dict(triggered=trigger,criterion='full final loss >= parent loss - 1e-4',results={})
    if trigger:
        for name,state in [('parent',parent),('full',final['full'])]:
            probes['results'][name]=probe_state(data,state)
            write_json(out/'probes.json',probes)
            print(json.dumps(dict(probe_state=name,**probes['results'][name])),flush=True)
    else:write_json(out/'probes.json',probes)
    # All training and optional objective-only probes are now complete.
    refroot=Path('results/connected_reference_fix_20261024/main')
    refmeta=json.loads((refroot/'summary.json').read_text())
    assert refmeta['fingerprints']==data.fingerprint
    assert file_digest(refroot/'graph_reference_audit_only.npz')==refmeta['reference_sha256']
    with np.load(refroot/'graph_reference_audit_only.npz') as a:
        ref=a['distance'].copy();next_edge=a['next_edge'].copy()
        assert np.array_equal(a['source'],data.src.numpy()) and np.array_equal(a['target'],data.dst.numpy())
    validate_reference(data,ref);graph,starts=graph_inputs(data)
    unique=data.src[data.original].unique().numpy()
    groups=dict(records=data.src[data.original].numpy(),unique_sources=unique,starts=starts,
        unique_starts=np.unique(starts),longest_quartile=unique[ref[unique]>=np.quantile(ref[unique],.75)])
    results={};decisions={};arrays={}
    for name,path in [('parent',SOURCE),('mini',root/'mini/final.pt'),('full',root/'full/final.pt')]:
        saved=parent if name=='parent' else final[name]
        result,pred,decision=evaluate(data,graph,ref,starts,path)
        model=IndexedIQE(data.small,'encoder',0,initialization_seed=saved['initialization_seed'])
        model.load_state_dict(saved['model']);again,edges=all_distances(model,data)
        assert np.array_equal(pred,again)
        parts=decompose(ref,next_edge,data.dst.numpy(),data.cost.double().numpy(),pred,edges)
        result['decomposition']={g:{k:float(v[ids].mean()) for k,v in parts.items()} for g,ids in groups.items()}
        result['full_objective']=logs['mini'][0]['full_objective'] if name=='parent' else logs[name][-1]['full_objective']
        result['branch_exposure']=regret_exposure(decision)
        result['model_displacement_norm']=float(sum((v.double()-parent['model'][k].double()).square().sum()
            for k,v in saved['model'].items() if torch.is_floating_point(v)).sqrt())
        results[name]=result;decisions[name]=decision;arrays[name+'_prediction']=pred
        arrays.update({name+'_'+k:v for k,v in parts.items()})
        write_json(out/(name+'_decisions.json'),decision)
        print(json.dumps(dict(arm=name,distance=result['distance']['unique_sources'],
            local=result['local']['unique_edges'],decisions=result['decisions'])),flush=True)
    changes={}
    for left,right in [('parent','mini'),('parent','full'),('mini','full')]:
        x={b['source']:b for b in decisions[left]['branches']};y={b['source']:b for b in decisions[right]['branches']}
        changes[left+'_'+right]=[dict(source=s,left_regret=x[s]['regret'],right_regret=y[s]['regret'],
            left_choice=x[s]['chosen_row'],right_choice=y[s]['chosen_row']) for s in x if x[s]['chosen_row']!=y[s]['chosen_row']]
    for arm,sha in hashes.items():assert file_digest(root/arm/'final.pt')==sha
    assert file_digest(SOURCE)==SOURCE_SHA
    summary=dict(pairing=pairing,results=results,history=logs,complete=complete,probes=probes,
        branch_changes=changes,reference_sha256=refmeta['reference_sha256'],fingerprints=data.fingerprint,
        analysis_source_sha256=file_digest(__file__),probe_source_sha256=file_digest(Path(__file__).with_name('probe.py')))
    write_json(out/'summary.json',summary);np.savez_compressed(out/'arrays.npz',reference=ref,**arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for arm in ('mini','full'):
        h=logs[arm];x=[r['processed_records']/1000 for r in h]
        axes[0,0].plot(x,[r['augmented_loss']-h[0]['augmented_loss'] for r in h],'-o',ms=3,label=arm)
        gh=[r for r in h if 'full_objective' in r]
        axes[0,1].plot([r['processed_records']/1000 for r in gh],
            [r['full_objective']['gradient_norms']['augmented'] for r in gh],'-o',label=arm)
        axes[1,0].plot(x,[r['max_violation'] for r in h],'-o',ms=3,label=arm)
    axes[0,0].axhline(0,color='k',lw=.6)
    for ax,y in [(axes[0,0],'Full objective change (fixed dual)'),(axes[0,1],'Full augmented gradient norm'),(axes[1,0],'Maximum edge violation')]:
        ax.set(xlabel='Training records processed (thousands)',ylabel=y);ax.legend()
    names=['parent','mini','full'];x=np.arange(3)
    for j,key in enumerate(('unique_sources','longest_quartile','unique_starts')):
        axes[1,1].bar(x+(j-1)*.24,[results[a]['distance'][key]['nrmse_percent'] for a in names],width=.24,label=key)
    axes[1,1].set(xticks=x,xticklabels=names,ylabel='Task NRMSE (%)',ylim=(0,20))
    axes[1,1].legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    write_json(out/'COMPLETE',dict(passed=True,pairing=pairing))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/fixed_dual_primal_20261024/main')
    run(p.parse_args())
