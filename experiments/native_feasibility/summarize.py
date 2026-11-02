"""Keep native constraint definitions on separate axes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.native_margin.shared import rows,write

def run(out):
    u=json.loads((out/'u/analysis.json').read_text());o=json.loads((out/'og/analysis.json').read_text())
    table=[dict(scene='U',alpha=u['alpha'],criterion='max physical edge h <= 0',
                old=u['old_max_h'],new=u['new_max_h'],threshold=0.,records=u['training_records'],
                success=u['successes'],episodes=u['episodes'],reused=0),
           dict(scene='OG',alpha=o['alpha'],criterion='mean positive h squared <= eps',
                old=o['old_risk'],new=o['new_risk'],threshold=o['eps'],records=o['training_records'],
                success=o['successes'],episodes=o['episodes'],reused=int(o['reused_parent_episodes']))]
    rows(out/'summary.csv',table);write(out/'summary.json',table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.4),layout='constrained')
    axes[0].bar(['original','projected'],[u['old_max_h'],max(u['new_max_h'],0)],color=['#788b95','#357a86'])
    axes[0].axhline(0,c='#555',lw=.7);axes[0].set_ylabel('max positive local violation')
    axes[0].set_title('U: original hard edge constraints',loc='left')
    axes[1].bar(['original','same agent'],[o['old_risk'],o['new_risk']],color=['#788b95','#357a86'])
    axes[1].axhline(o['eps'],color='#b4752a',ls='--',label='original eps = 0.05')
    axes[1].set_ylabel('full-training mean squared positive h');axes[1].set_title('OG: original mean constraint',loc='left')
    axes[1].legend(fontsize=8,frameon=False)
    x=np.arange(2);w=.34
    axes[2].bar(x-w/2,[28/64,0],w,label='original')
    axes[2].bar(x+w/2,[u['successes']/64,o['successes']/50],w,label='native-feasible function')
    axes[2].set_xticks(x,['U MPPI','OG true greedy']);axes[2].set_ylim(0,1)
    axes[2].set_ylabel('closed-loop success fraction');axes[2].legend(fontsize=8,frameon=False)
    axes[2].set_title('Current model class, no training',loc='left')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'feasibility.svg');fig.savefig(out/'feasibility.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);run(p.parse_args().out)
