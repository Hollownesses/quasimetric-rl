"""Fixed-point noise comparison and plots; no optimization or training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.gap_stochastic.core import QUERIES
from experiments.gap_stochastic.diagnose import route_gradient_noise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.folder/'manifest.json').read_text())
    baseline=Path(manifest['baseline'])
    noise={}
    for name,mu in QUERIES.items():
        cases={}
        for at,path in (('old_endpoint',baseline/(manifest['runs'][name]['baseline_key']+'.npz')),
                        ('new_endpoint',args.folder/(name+'.npz'))):
            x=np.load(path)['x'][-1]
            n=route_gradient_noise(x,mu)
            cases[at]=dict(raw_route_mean=n['total']['mean'],
                           original_route_std=n['gradient_std'],
                           projected_route_std=float(np.sqrt(n['gap']['variance'])),
                           relative_route_variance_reduction=1-n['gap']['variance']/n['total']['variance'])
        a=np.load(args.folder/(name+'.npz'))['gradient_audit']
        cases['adam_logging']=dict(maximum_cons_moment_route=float(np.max(abs(a[:,7]))),
                                  maximum_adam_scaled_cons_route=float(np.max(abs(a[:,8]))),
                                  last_500_adam_scaled_cons_route_rms=float(np.sqrt(np.mean(a[-500:,8]**2))))
        noise[name]=cases
    (args.folder/'noise_diagnostics.json').write_text(json.dumps(noise,indent=2,allow_nan=False)+'\n')
    history=[json.loads(line) for line in (args.folder/'history.jsonl').read_text().splitlines()]
    fig,axs=plt.subplots(2,3,figsize=(12.5,7))
    for r,name in enumerate(QUERIES):
        for method,color in (('unbiased','#64748b'),('projected_conservation','#2563eb')):
            rows=[x for x in history if x['query']==name and x['method']==method]
            steps=[x['step'] for x in rows]
            data=([x['policy']['source_cost'] for x in rows],
                  [x['residual_linf'] for x in rows], [abs(x['gap']) for x in rows])
            for c,ys in enumerate(data):
                ys=[np.nan if y is None else max(y,1e-8) if c else y for y in ys]
                axs[r,c].plot(steps,ys,color=color,lw=1.5,label='Original unbiased' if method=='unbiased' else 'Projected raw conservation')
        for c,limit in enumerate((3.025,.005,.005)):
            axs[r,c].axhline(limit,color='#b45309',ls=':',lw=1,label='Frozen threshold')
            axs[r,c].grid(alpha=.2)
            axs[r,c].set_xlabel('Updates')
            if c:
                axs[r,c].set_yscale('log')
                axs[r,c].set_ylim(1e-5,2)
            else:
                axs[r,c].set_ylim(2.99,3.26)
        axs[r,0].set_ylabel('Cheap suffix heavy' if r==0 else 'Expensive suffix heavy')
    for ax,title in zip(axs[0],('Executed cost from s','True conservation L-inf','Absolute cost-distance gap')):
        ax.set_title(title,fontsize=11)
    fig.subplots_adjust(left=.07,right=.99,bottom=.15,top=.86,hspace=.33,wspace=.29)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,fontsize=9)
    fig.suptitle('Single pre-Adam conservation-gradient intervention\nSame seed 2001, batch 256, 20,000 updates; fixed endpoints, no checkpoint selection',fontsize=12)
    fig.savefig(args.folder/'comparison.png',dpi=180)
    fig.savefig(args.folder/'comparison.pdf')
    plt.close(fig)
    print(json.dumps(noise,indent=2))


if __name__=='__main__':
    main()
