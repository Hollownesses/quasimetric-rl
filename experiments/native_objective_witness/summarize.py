"""Original objective preference and control, reported within each environment."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.native_margin.shared import rows,write

def read_csv(p):
    with p.open() as f:return list(csv.DictReader(f))

def run(raw,out):
    u=read_csv(out/'u/points.csv');o=json.loads((out/'og/audit.json').read_text())['points']
    eps=read_csv(out/'og/episodes.csv');table=[]
    history=json.loads((raw/'results/ogbench_critic_20261029/official_counterfactual_completion/analysis.json').read_text())['initials']
    for r in u:table.append(dict(scene='U',point=r['point'],scale=r['scale'],feasible=int(r['feasible']),alpha=float(r['alpha']),
        original_min_objective=-float(r['F0']),source_min_objective=-float(r['F1']),successes=int(r['successes']),episodes=int(r['episodes'])))
    for key,r in o.items():
        name=r['critic'];scale=r['scale']
        rr=[x for x in eps if x['critic']==name]
        count=sum(int(x['success']) for x in rr) if scale=='boundary' else sum(x['success'] for x in history['original' if name=='original' else 'improved'].values())
        table.append(dict(scene='OG',point=key,scale=scale,feasible=r['feasible'],alpha=r['alpha'],P=r['P'],
            original_min_objective=r['P_plus_D_raw'],extracted_min_objective=r['P_plus_D_extracted'],successes=count,episodes=50))
    rows(out/'summary.csv',table);write(out/'summary.json',table)
    ue=read_csv(out/'u/episodes.csv');pairs=[]
    by={key:{(r['case'],r['seed']):int(r['success']) for r in ue if r['point']==key} for key in ['R0_boundary','R1_boundary']}
    assert by['R0_boundary'].keys()==by['R1_boundary'].keys() and len(by['R0_boundary'])==64
    for k in by['R0_boundary']:pairs.append(dict(scene='U',case=str(k),original=by['R0_boundary'][k],improved=by['R1_boundary'][k]))
    for r in eps:pairs.append(dict(scene='OG',case=r['case'],critic=r['critic'],raw=int(r['raw_success']),boundary=int(r['success'])))
    rows(out/'control_pairs.csv',pairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.8,7),layout='constrained')
    ub=[next(r for r in u if r['point']==p) for p in ['R0_boundary','R1_boundary']]
    x=np.arange(2);w=.32
    for j,(key,label) in enumerate([('F0','original record objective'),('F1','source-weighted objective')]):
        bars=axes[0,0].bar(x+(j-.5)*w,[float(r[key]) for r in ub],w,label=label)
        axes[0,0].bar_label(bars,fmt='%.2f',fontsize=8)
    axes[0,0].set_xticks(x,['R0 feasible','R1 feasible']);axes[0,0].set_ylabel('push functional (higher is better)')
    axes[0,0].set_ylim(0,110)
    axes[0,0].set_title('U: exact native query functionals',loc='left');axes[0,0].legend(frameon=False,fontsize=8)
    comp=read_csv(out/'og/comparisons.csv');width=.24
    for j,(key,label) in enumerate([('P','value only'),('raw','value + raw dynamics'),('extracted','value + extracted dynamics')]):
        vals=[];err=[]
        for scale in ['raw','boundary']:
            r=next(r for r in comp if r['scale']==scale and r['dynamics']==('raw' if key=='P' else key))
            vals.append(float(r['delta_P'] if key=='P' else r['delta_primal']))
            err.append((float(r['hoeffding_upper'])-float(r['hoeffding_lower']))/2)
        axes[0,1].bar(x+(j-1)*width,vals,width,yerr=err,capsize=3,label=label)
    axes[0,1].axhline(0,color='#666',lw=.7);axes[0,1].set_xticks(x,['raw functions','feasible ray boundary'])
    axes[0,1].set_ylabel('E16 minus original (negative favors E16)');axes[0,1].set_title('OG: native primal difference',loc='left')
    axes[0,1].legend(frameon=False,fontsize=7)
    labels=['R0 raw\n(reused)','R1 raw\n(reused)','R0 feasible\n(reused)','R1 feasible\n(new)']
    uu=[next(r for r in u if r['point']==p) for p in ['R0_raw','R1_raw','R0_boundary','R1_boundary']]
    axes[1,0].bar(labels,[int(r['successes'])/64 for r in uu],color=['#a3afb5','#a3afb5','#397f8c','#397f8c'])
    axes[1,0].set_ylim(0,1);axes[1,0].set_ylabel('MPPI success / 64');axes[1,0].set_title('U: same original controller',loc='left')
    for j,name in enumerate(['original','E16']):
        vv=[next(r for r in table if r['scene']=='OG' and r['point']==name+'_'+scale) for scale in ['raw','boundary']]
        axes[1,1].bar(x+(j-.5)*w,[r['successes']/50 for r in vv],w,label=name)
    axes[1,1].set_xticks(x,['raw critic\n(reused)','actual boundary critic\n(new)']);axes[1,1].set_ylim(0,1)
    axes[1,1].set_ylabel('true-successor greedy success / 50');axes[1,1].set_title('OG: same 50 task3 starts',loc='left')
    axes[1,1].legend(frameon=False,fontsize=8)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'objective_control.svg');fig.savefig(out/'objective_control.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
