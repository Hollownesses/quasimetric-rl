"""Export fixed-endpoint summaries and independent evaluation learning curves."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    evaluation = json.loads((args.folder/'evaluation.json').read_text())
    history = [json.loads(line) for line in (args.folder/'history.jsonl').read_text().splitlines()]
    names = ('cheap_suffix_queries', 'expensive_suffix_queries')
    colors = dict(qrl='#64748b', naive='#d97706', unbiased='#2563eb')
    fig, axs = plt.subplots(2,4, figsize=(15,8))
    for r,name in enumerate(names):
        for method in colors:
            for mode in ('stochastic', 'population'):
                if method == 'qrl' and mode == 'population':
                    continue
                key = f'{mode}__{name}__{method}'
                rows = [row for row in history if row['key'] == key]
                t = [row['step'] for row in rows]
                style = dict(color=colors[method], ls='-' if mode == 'stochastic' else '--',
                             lw=1.7 if mode == 'stochastic' else 1.,
                             alpha=1 if mode == 'stochastic' else .65,
                             label=method+(' (sampled)' if mode == 'stochastic' else ' (expectation)'))
                values = ([row['policy']['source_cost'] for row in rows],
                          [row.get('residual_linf') for row in rows],
                          [row['risk'] for row in rows],
                          [abs(row['gap']) if 'gap' in row else None for row in rows])
                for c, ys in enumerate(values):
                    if method == 'qrl' and c in (1,3):
                        continue
                    ys = [np.nan if y is None else max(y,1e-8) if c else y for y in ys]
                    axs[r,c].plot(t,ys,**style)
        for c in range(4):
            axs[r,c].grid(alpha=.2)
            axs[r,c].set_xlabel('Updates')
            if c:
                axs[r,c].set_yscale('log')
                axs[r,c].set_ylim(1e-8,2)
        axs[r,0].set_ylim(2.99,3.26)
        axs[r,0].axhline(3.,color='black',ls=':',lw=.8)
        axs[r,0].set_ylabel('Cheap suffix heavy' if r == 0 else 'Expensive suffix heavy')
    for ax,title in zip(axs[0],('Executed cost from s','True conservation L-inf','Local squared risk','Absolute cost-distance gap')):
        ax.set_title(title,fontsize=10)
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.subplots_adjust(left=.055,right=.99,top=.87,bottom=.15,wspace=.28,hspace=.34)
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=5,fontsize=9,
               title='Naive endpoints fail from one queried state. Gaps mean failed arrival from s; log display floor = 1e-8.',
               title_fontsize=9)
    fig.suptitle('Free arrival-flow learning: fixed seed 2001, batch 256, 20,000 updates\n'
                 'Dashed: same optimizer with exact expected gradients. Curves are not checkpoint selection.',fontsize=12)
    fig.savefig(args.folder/'learning_curves.png',dpi=180)
    fig.savefig(args.folder/'learning_curves.pdf')
    plt.close(fig)
    fields = ['key','gap','risk','residual_linf','flow_cost','source_cost','query_cost','proper',
              'cheap_probability','critic_source_cost','joint_pass','multiplier']
    with (args.folder/'summary.csv').open('w') as output:
        writer=csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        for key,row in evaluation['runs'].items():
            record={field:row.get(field) for field in fields}
            for field in ('source_cost','query_cost','proper','cheap_probability'):
                record[field]=row['policy'][field]
            for field in ('source_cost','query_cost'):
                if record[field] is None:
                    record[field]='inf'
            record['critic_source_cost']=row['critic_policy']['source_cost']
            writer.writerow(record)


if __name__ == '__main__':
    main()
