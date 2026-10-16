"""Read-only plots and machine-readable comparison; no optimizer calls."""
from pathlib import Path
import argparse
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/qrl-factorial-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'encoder_barrier':'#d9822b','encoder_phr':'#14858c','latent_barrier':'#9063af','latent_phr':'#3268b2'}


def summarize(root):
    runs={}
    for path in sorted((root/'main').glob('*')):
        if not (path/'curve.jsonl').exists(): continue
        curve=[json.loads(x) for x in (path/'curve.jsonl').read_text().splitlines()]
        if not curve: continue
        cfg=json.loads((path/'config.json').read_text())
        runs[path.name]=dict(config=cfg,curve=curve,complete=(path/'COMPLETE').exists())
    if not runs: raise ValueError('no runs')
    fingerprints=[v['config']['fingerprints'] for v in runs.values()]
    assert all(x==fingerprints[0] for x in fingerprints)
    for ci in (0,1):
        selected=[v for k,v in runs.items() if k.endswith(f'_c{ci}')]
        if not selected: continue
        assert len({v['config']['initial_projected_sha256'] for v in selected})==1
        for param in ('encoder','latent'):
            init=[v['config']['initialization_sha256'] for v in selected if v['config']['parameterization']==param]
            assert len(set(init))<=1
    out=root/'comparison'; out.mkdir(exist_ok=True)
    summary={name:dict(config=r['config'],complete=r['complete'],final=r['curve'][-1],
        minimum_violation=min(x['max_violation'] for x in r['curve']),
        best_feasible_push=max((x['push'] for x in r['curve'] if x['max_violation']<=1e-5),default=None)) for name,r in runs.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'summary.csv').open('w') as f:
        columns=['run','step','push','push_gap_percent','feasible_push_gap_percent','task_rmse','task_nrmse_percent','max_violation','violation_count','seconds','train_seconds']
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader()
        for name,r in summary.items(): w.writerow({'run':name,**{k:r['final'].get(k) for k in columns[1:]}})
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':140})
    for ci in (0,1):
        selected={k:v for k,v in runs.items() if k.endswith(f'_c{ci}')}
        if not selected: continue
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        specs=[('push','Full-support push',False),('task_rmse','Task suffix RMSE',False),
               ('max_violation','Maximum positive residual',True),('push_gap_percent','Raw push gap (%)',False),
               ('feasible_push_gap_percent','Feasible rescaled push gap (%)',False),('violation_count','Violating physical edges / 335',False)]
        for ax,(metric,label,log) in zip(axes.flat,specs):
            for name,r in selected.items():
                key=name.rsplit('_c',1)[0]; cc=r['curve']
                ax.plot([x['step'] for x in cc],[max(1e-7,x[metric]) if log else x[metric] for x in cc],label=key,color=COLORS[key])
            ax.set(title=label,xlabel='Primal updates')
            if log: ax.set_yscale('log'); ax.axhline(1e-5,color='gray',ls=':')
            if metric=='push': ax.axhline(next(iter(selected.values()))['curve'][0]['push_reference'],color='black',ls='--',lw=1)
        axes[0,0].legend(fontsize=8); fig.suptitle(f'Four-path factorial — initialization {ci}'); fig.tight_layout()
        fig.savefig(out/f'curves_c{ci}.png'); plt.close(fig)
        fig,axes=plt.subplots(1,4,figsize=(16,3.8))
        for name,r in selected.items():
            key=name.rsplit('_c',1)[0]
            rows=list(csv.DictReader((root/'main'/name/'values.csv').open()))
            final=r['curve'][-1]['step']; rows=[x for x in rows if int(x['step'])==final]
            for p,ax in enumerate(axes):
                rr=[x for x in rows if int(x['path'])==p]
                t=[int(x['timestep']) for x in rr]
                if name==next(iter(selected)): ax.plot(t,[float(x['reference']) for x in rr],color='black',ls='--',label='Recorded graph distance')
                ax.plot(t,[float(x['value']) for x in rr],color=COLORS[key],label=key)
                ax.set(title=f'Path {p}',xlabel='Recorded timestep',ylabel='Remaining cost')
        axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/f'paths_c{ci}.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,r in runs.items():
        if not name.endswith('_c0'): continue
        key=name[:-3]; cc=r['curve']
        axes[0].plot([x['seconds'] for x in cc],[x['push'] for x in cc],label=key,color=COLORS[key])
        arr=np.load(root/'main'/name/'latest_edges.npz')['residual']
        support=np.load(root/'main'/name/'support.npz')
        axes[1].plot(np.sort(arr[support['cost']>0]),label=key,color=COLORS[key])
    axes[0].set(xlabel='Measured elapsed seconds (concurrent CPU jobs)',ylabel='Full-support push')
    axes[1].set(xlabel='Physical edge rank by residual',ylabel='Final edge residual')
    axes[1].axhline(0,color='black',ls='--'); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'time_and_residuals.png'); plt.close(fig)
    print((out/'summary.csv').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); args=p.parse_args(); summarize(args.root)
