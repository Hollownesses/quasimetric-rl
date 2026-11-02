"""Paired native-ablation summaries; episodes, rather than action pairs, are units."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'experiments/native_margin'))
from shared import rows,write

def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def success(value):
    if str(value) in ('True','1'):return 1
    if str(value) in ('False','0'):return 0
    raise ValueError('unrecognized success flag: '+str(value))

def paired_counts(current,baseline):
    if current.keys()!=baseline.keys():raise ValueError('unmatched episode identities')
    gained=sum(success(r['success']) and not success(baseline[k]['success']) for k,r in current.items())
    lost=sum(not success(r['success']) and success(baseline[k]['success']) for k,r in current.items())
    return gained,lost

def run(out):
    allrows=[];pairs=[]
    for scene,folder in [('U','u'),('OG','og')]:
        margins=read(out/folder/'margins.csv');ds=read(out/folder/'decisions.csv');eps=read(out/folder/'episodes.csv')
        def key(r):return (r['state_id'],r['planner_seed']) if scene=='U' else r['case']
        by={a:{key(r):r for r in eps if r['arm']==a} for a in ['parent','native','no_push','no_local']}
        for arm,ee in by.items():
            assert len(ee)==(64 if scene=='U' else 50)
            assert len(ee)==sum(r['arm']==arm for r in eps),'duplicate episode identity'
        for arm in ['parent','native','no_push','no_local']:
            mm=[r for r in margins if r['arm']==arm];dd=[r for r in ds if r['arm']==arm];ee=by[arm]
            rec=dict(scene=scene,arm=arm,probe_count=len(mm),reversed_probes=sum(float(r['margin'])< -1e-4 for r in mm),
                tied_probes=sum(abs(float(r['margin']))<=1e-4 for r in mm),
                probe_repaired=sum(float(r['parent_margin'])< -1e-4 and float(r['margin'])>1e-4 for r in mm),
                probe_new_wrong=sum(float(r['parent_margin'])>1e-4 and float(r['margin'])< -1e-4 for r in mm),
                wrong_probe_mean_delta=np.mean([float(r['margin'])-float(r['parent_margin']) for r in mm if float(r['parent_margin'])< -1e-4]),
                strict_pairs=sum(int(r['strict_pairs']) for r in dd),reversed_pairs=sum(int(r['reversed_pairs']) for r in dd),
                model_tied_strict=sum(int(r['model_tied_strict']) for r in dd),
                suboptimal_first=sum(int(r['wrong']) for r in dd),first_regret_mean=np.mean([float(r['reference_margin']) for r in dd]),
                episodes=len(ee),successes=sum(success(r['success']) for r in ee.values()))
            if scene=='U':rec['graph_regret_mean']=np.mean([float(r['regret']) for r in read(out/folder/'graph.csv') if r['arm']==arm])
            for compare in ['parent','native']:
                rec['gained_vs_'+compare],rec['lost_vs_'+compare]=paired_counts(ee,by[compare])
            allrows.append(rec)
        for k in by['parent']:
            pairs.append(dict(scene=scene,case=str(k),**{a:success(by[a][k]['success']) for a in by}))
    rows(out/'summary.csv',allrows);rows(out/'episode_pairs.csv',pairs);write(out/'summary.json',allrows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
    colors=['#71828a','#346f8a','#c48b35','#589273'];labels=['parent','native','no push','no local']
    for scene,marker in [('U','o'),('OG','s')]:
        rr=[r for r in allrows if r['scene']==scene]
        axes[0].plot(labels,[r['reversed_pairs']/r['strict_pairs'] for r in rr],marker=marker,label=scene+' reversed')
        axes[0].plot(labels,[r['model_tied_strict']/r['strict_pairs'] for r in rr],marker=marker,linestyle=':',label=scene+' tied')
        axes[1].plot(labels,[r['suboptimal_first']/(58 if scene=='U' else 50) for r in rr],marker=marker,label=scene)
    axes[0].set_ylabel('strict action-pair reversal fraction');axes[1].set_ylabel('suboptimal first-choice fraction')
    x=np.arange(4);w=.36
    for j,scene in enumerate(['U','OG']):
        rr=[r for r in allrows if r['scene']==scene]
        axes[2].bar(x+(j-.5)*w,[r['successes']/r['episodes'] for r in rr],width=w,label=scene+' (MPPI)' if scene=='U' else 'OG (true greedy)')
    axes[2].set_xticks(x,labels);axes[2].set_ylabel('closed-loop success fraction');axes[2].set_ylim(0,1)
    for ax in axes:ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'comparison.svg');fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);run(p.parse_args().out)
