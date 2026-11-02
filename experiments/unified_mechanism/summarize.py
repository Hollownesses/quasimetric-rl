"""Summarize the two read-only runs, with explicit weighting and sample scopes."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from metrics import read, stats, write_rows


def load(path):
    with path.open() as f: return list(csv.DictReader(f))


def group(rows, *keys):
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    return groups


def values(rows, key):
    return np.array([float(r[key]) for r in rows if r.get(key) not in ('',None)])


def run(raw,out):
    metrics=[]; decisions=[]; paths=[]; outcomes=[]
    for sub,scene in [('u','U'),('og','OG')]:
        metrics.extend(load(out/sub/'metrics.csv'))
        for name,rows in [('paths',paths),('decisions',decisions)]:
            rows.extend(dict(scene=scene,**r) for r in load(out/sub/(name+'.csv')))
    for (scene,model,scope),rr in group(paths,'scene','model','scope').items():
        for k in ['steps','cost','end','start','error_vs_path','sum_h','sum_tau','h_over_cost','tau_over_cost',
                  'tau_first_half','tau_min','prefix_identity_max','last100_span']:
            x=values(rr,k)
            if len(x): metrics.append(dict(scene=scene,model=model,scope=scope,metric=k,**stats(x)))
    for (scene,model,scope),rr in group(decisions,'scene','model','scope').items():
        for k in ['b','b_norm','b_set','wrong','inversion','reference_margin','ref_best_second','model_best_second',
                  'chosen_model_gap','chosen_error_difference','dyn_best_true_gap','actor_true_gap','latent_score_mae',
                  'actor_ref_excess','latent_ref_regret','action_absdiff','snap_max','ref_progress','finite','legal']:
            x=values(rr,k)
            if len(x): metrics.append(dict(scene=scene,model=model,scope=scope,metric=k,**stats(x)))
    rankrows=[]
    for r in decisions:
        if not r.get('candidates'): continue
        rankrows.append(dict(r,split='all'))
        if r.get('outcome'):rankrows.append(dict(r,split=r['outcome']))
    ranks=[]
    for (scene,model,scope,split),rr in group(rankrows,'scene','model','scope','split').items():
        pairs=values(rr,'pairs').sum(); strict=values(rr,'strict_pairs').sum()
        reversed_pairs=values(rr,'reversed_pairs').sum()
        wrong=[r for r in rr if int(r['wrong'])]
        rec=dict(scene=scene,model=model,scope=scope,split=split,states=len(rr),wrong=len(wrong),pairs=int(pairs),
                 strict_pairs=int(strict),reversed_pairs=int(reversed_pairs),inverse_pooled=reversed_pairs/strict if strict else '',
                 reference_tied_pairs=int(pairs-strict),model_tied_strict=int(values(rr,'model_tied_strict').sum()))
        for k in ['reference_margin','chosen_error_difference','chosen_model_gap']:
            if wrong:rec[k+'_wrong_median']=np.median(values(wrong,k))
        for k in ['b','b_norm','action_absdiff','actor_true_gap','dyn_best_true_gap','latent_score_mae',
                  'snap_max','ref_best_second','model_best_second','inversion','ref_progress','actor_ref_excess']:
            x=values(rr,k)
            if len(x):rec[k+'_mean']=x.mean()
        ranks.append(rec)
    write_rows(out/'unified_metrics.csv',metrics)
    write_rows(out/'action_order.csv',ranks)
    og=raw/'results/ogbench_critic_20261029'
    for model,file in [('100k','baseline_greedy/summary.json'),('E6','tail_1000/mean_greedy/summary.json'),
                       ('E16','hard_goal_1000/neural_evaluation/analysis.json')]:
        rec=read(og/file);rec=rec.get('cases',rec)
        for task in range(1,6):
            cases=[v for k,v in rec.items() if k.startswith(f'task{task}_')]
            assert len(cases)==5
            outcomes.append(dict(scene='OG',model=model,controller='true_grid_bank25',task=task,
                                 n=5,success=sum(c['success'] for c in cases)))
    official=load(out/'og/official.csv')
    for (m,t),rr in group(official,'model','task').items():
        outcomes.append(dict(scene='OG',model=m,controller='extracted_actor',task=t,n=len(rr),
                             success=int(values(rr,'success').sum())))
    cf=read(og/'official_counterfactual_completion/analysis.json')
    for model,tag in [('100k','original'),('E16','improved')]:
        rr=list(cf['initials'][tag].values())
        outcomes.append(dict(scene='OG',model=model,controller='true_grid_official',task=3,n=len(rr),
                             success=sum(r['success'] for r in rr)))
    rr=list(cf['actor_timeout_counterfactual'].values())
    outcomes.append(dict(scene='OG',model='E16',controller='true_grid_restart',task=3,n=len(rr),
                         success=sum(r['success'] for r in rr)))
    import json
    for arm in ['none','R0','R1','oracle']:
        rr=[json.loads(l) for l in (raw/f'results/r_series_mppi_20261026/main/arms/{arm}/episodes.jsonl').read_text().splitlines()]
        assert len(rr)==64
        for region in ['all','inside','west']:
            sub=[r for r in rr if region=='all' or r['stratum']==region]
            outcomes.append(dict(scene='U',model=arm,controller='MPPI',task=region,n=len(sub),
                                 success=sum(r['success'] for r in sub),collision=sum(r['collision'] for r in sub)))
    write_rows(out/'outcomes.csv',outcomes)
    plot(metrics,ranks,outcomes,out/'mechanism.svg')


def plot(metrics,ranks,outcomes,path):
    def select(rows,**keys):
        r,=[x for x in rows if all(str(x.get(k))==str(v) for k,v in keys.items())]
        return r
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(11,6.7),layout='constrained')
    arms=['R0','R1','100k','E6','E16']
    def bar(ax,series,labels,names):
        x=np.arange(len(names));w=.78/len(series)
        for i,(y,label) in enumerate(zip(series,labels)):
            ax.bar(x+(i-(len(series)-1)/2)*w,y,width=w,label=label,color=['#246378','#d79b41','#5b896a'][i])
        ax.set_xticks(x,names);ax.axhline(0,color='#555',lw=.6);ax.legend(frameon=False,fontsize=8)
        ax.spines[['right','top']].set_visible(False)
    def metric(m,s,k):return float(select(metrics,model=m,scope=s,metric=k)['mean'])
    scopes=['graph_shortest']*2+['geometry_shortest']*3
    ax=axs[0,0]
    bar(ax,[[metric(m,s,k) for m,s in zip(arms,scopes)] for k in ['h_over_cost','tau_over_cost']],
        [r'$\Sigma h/C$',r'$\Sigma\tau/C$'],arms)
    ax.set_title('A   Along reference shortest paths',loc='left')
    ax.set_ylabel('fraction of path cost')
    ax=axs[0,1]
    scopes=['start_1']*2+['start']*3
    bar(ax,[[metric(m,s,'b_norm') for m,s in zip(arms,scopes)]],['mean at frozen starts'],arms)
    ax.set_title('B   True-successor Bellman residual',loc='left');ax.set_ylabel('step-cost units')
    ax=axs[1,0]
    scopes=['mppi_bank']*2+['start']*3
    bar(ax,[[select(ranks,model=m,scope=s,split='all')['inverse_pooled'] for m,s in zip(arms,scopes)]],
        ['strict pairs, pooled'],arms)
    ax.set_title('C   Reversed order vs. environment grid',loc='left');ax.set_ylabel('fraction reversed')
    ax=axs[1,1]
    labels=['R0','R1','Oracle','100k / A','E6 / C','E16 / B']
    suc=[select(outcomes,model=m,controller='MPPI',task='all')['success']/64 for m in ['R0','R1','oracle']]
    suc.extend(select(outcomes,model=m,controller='extracted_actor',task=3)['success']/50 for m in arms[2:])
    bar(ax,[suc],['MPPI (first 3) / task3 actor (last 3)'],labels)
    ax.set_ylim(0,1);ax.set_title('D   Recorded closed-loop success',loc='left');ax.set_ylabel('success fraction')
    fig.savefig(path);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();run(args.raw,args.out)
