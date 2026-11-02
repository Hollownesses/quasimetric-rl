"""Frozen native U-trap models; no inferred or interpolated edge exposure."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'experiments/native_margin')]
from shared import rows,sha,write
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.r_series_mppi.common import adapter
from experiments.rare_edge_attribution.core import select_edges,counterfactual

def run(raw,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    here=Path(__file__).resolve().parent
    (out/'PROTOCOL.md').write_bytes((here/'PROTOCOL.md').read_bytes())
    manifest=json.loads((REPO/'results/unified_mechanism_20261102/audit.json').read_text())['verified_sha256']
    dpath=raw/'results/connected_expert_utrap_20261021/main';refpath=raw/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz'
    for p in [dpath/'verified_replay.pt',refpath]:assert sha(p)==manifest[str(p.relative_to(raw))]
    data=load_training_data(dpath);rd=np.load(refpath);v,nxt=rd['distance'],rd['next_edge'];goal,=validate_reference(data,v)
    src,dst,cost=(x.numpy() for x in (data.src,data.dst,data.cost));physical=data.original.numpy()
    _,ix,inverse,count=np.unique(np.column_stack((src[physical],dst[physical],cost[physical])),axis=0,return_index=True,return_inverse=True,return_counts=True)
    representatives=physical[ix];rep=np.full(len(src),-1,int);rep[physical]=representatives[inverse]
    copies=np.zeros(len(src),int);copies[representatives]=count
    records=defaultdict(list)
    for k in physical:records[rep[k]].append(k)
    probes=list(csv.DictReader((REPO/'results/native_margin_20261102_verified/u/probes.csv').open()))
    def route(node):
        result=[]
        while node!=goal:
            k=int(nxt[node]);assert k>=0 and v[dst[k]]<v[node]
            result.append(int(rep[k]));node=dst[k]
        return result
    fixed=[]
    for p in probes:
        a,b=int(p['good_edge']),int(p['bad_edge']);good,bad=route(dst[a]),route(dst[b])
        r,c=select_edges(good,copies,cost)
        fixed.append((p,good,bad,r,c))
    output=[];edge_rows=[];maxclosure=0.;digests={}
    for arm in ['R0','R1']:
        ck=raw/f'results/task_source_balance_20261025/main/{arm}/final.pt';digest=sha(ck)
        assert digest==manifest[str(ck.relative_to(raw))];digests[arm]=digest
        state=torch.load(ck,weights_only=True);value,model=adapter(data.small,ck)
        nodes=data.nodes.numpy()
        def score(x,y):
            y=np.broadcast_to(y,x.shape).copy()
            return np.concatenate([value.batch_value(x[i:i+512],y[i:i+512]) for i in range(0,len(x),512)]).astype(float)
        j=score(nodes,nodes[goal]);d=score(nodes[src],nodes[dst]);h=d-cost
        tau=d+j[dst]-j[src]
        for p,good,bad,rare,control in fixed:
            a,b=int(p['good_edge']),int(p['bad_edge']);gap=float(cost[b]+v[dst[b]]-cost[a]-v[dst[a]])
            m=float(cost[b]+j[dst[b]]-cost[a]-j[dst[a]])
            dh=float(h[bad].sum()-h[good].sum());dt=float(tau[bad].sum()-tau[good].sum())
            error=abs(m-gap-dh+dt);maxclosure=max(maxclosure,error);assert error<1e-8
            rec=dict(arm=arm,case=p['case'],margin=m,ref_gap=gap,delta_h=dh,delta_tau=dt,
                     good_length=len(good),bad_length=len(bad),rare=rare,control=control)
            for name,e in [('rare',rare),('control',control)]:
                if e is None:continue
                cf,dm=counterfactual(m,good,bad,h,[e]);pos,_=counterfactual(m,good,bad,h,[e],True)
                rec[name+'_margin']=cf;rec[name+'_positive_only_margin']=pos;rec[name+'_delta']=dm
                lam=state['dual'][records[e]].numpy().astype(float)
                edge_rows.append(dict(arm=arm,case=p['case'],kind=name,edge=e,source=int(src[e]),target=int(dst[e]),
                    copies=int(copies[e]),empirical_mass=copies[e]/len(physical),cost=float(cost[e]),h=float(h[e]),
                    shared=int(e in bad),lambda_mean=float(lam.mean()),lambda_min=float(lam.min()),lambda_max=float(lam.max()),
                    effective_phr_record_mean_coefficient=float(np.maximum(lam+100*h[e],0).sum()/len(physical))))
            rec['single_copy_margin']=counterfactual(m,good,bad,h,[e for e in set(good+bad) if copies[e]==1])[0]
            rec['zero_all_h_margin']=m-dh;rec['zero_all_tau_margin']=m+dt
            # Upper bound for any one-edge h removal along either suffix (posthoc descriptive).
            rec['best_single_edge_margin']=max([m]+[counterfactual(m,good,bad,h,[e])[0] for e in set(good+bad)])
            output.append(rec)
        assert sha(ck)==digest
    rows(out/'pairs.csv',output);rows(out/'edges.csv',edge_rows)
    summary=[]
    for arm in ['R0','R1']:
        for group in ['all','wrong','matched_wrong']:
            rr=[r for r in output if r['arm']==arm and (group=='all' or r['margin']< -1e-4) and (group!='matched_wrong' or r['control'] is not None)]
            rec=dict(arm=arm,group=group,n=len(rr),wrong=sum(r['margin']< -1e-4 for r in rr),matched=sum(r['control'] is not None for r in rr))
            for name in ['rare','control','single_copy','zero_all_h','zero_all_tau','best_single_edge']:
                vv=[r for r in rr if name+'_margin' in r]
                rec[name+'_n']=len(vv);rec[name+'_repaired']=sum(r['margin']< -1e-4 and r[name+'_margin']>1e-4 for r in vv)
                rec[name+'_new_wrong']=sum(r['margin']>1e-4 and r[name+'_margin']< -1e-4 for r in vv)
                rec[name+'_tied']=sum(abs(r[name+'_margin'])<=1e-4 for r in vv)
                rec[name+'_mean_delta']=float(np.mean([r[name+'_margin']-r['margin'] for r in vv])) if vv else None
            summary.append(rec)
    rows(out/'summary.csv',summary);write(out/'summary.json',summary)
    write(out/'audit.json',dict(checkpoints=digests,pairs=58,unique_edges=len(count),physical_records=len(physical),
           closure_max=maxclosure,reference_sha256=sha(refpath),checkpoints_unchanged=True,
           sources={p.name:sha(p) for p in here.glob('*.py')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,arm in zip(axes,['R0','R1']):
        rr=[r for r in output if r['arm']==arm and r['margin']< -1e-4]
        for name,marker,label in [('rare','o','preselected lowest-copy edge'),('control','x','matched higher-copy edge')]:
            vv=[r for r in rr if name+'_margin' in r]
            ax.scatter([r['margin'] for r in vv],[r[name+'_margin'] for r in vv],marker=marker,label=label,s=28)
        ax.axhline(0,c='#777',lw=.7);ax.set_xlabel('original bad-minus-good score');ax.set_ylabel('component-neutralized score')
        ax.set_title(arm+'   Initially reversed pairs');ax.legend(fontsize=7,frameon=False)
    fig.savefig(out/'counterfactual.svg');fig.savefig(out/'counterfactual.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
