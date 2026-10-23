"""Post-training branch changes and exact rollout-regret decomposition."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from .compare import load_training_data,graph_inputs,evaluate,write_json


def contributions(decision):
    visits=Counter(r for episode in decision['episodes'] for r in episode['rows'])
    result=[dict(node=b['source'],visits=visits[b['chosen_row']],regret=b['regret'],
                 mean_contribution=visits[b['chosen_row']]*b['regret']/len(decision['episodes']))
            for b in decision['branches']]
    # On this graph nonbranch edges have zero reference advantage. Telescoping
    # sums must reproduce each complete recorded policy's mean cost regret.
    assert abs(sum(r['mean_contribution'] for r in result)-
               decision['summary']['mean_successful_regret'])<1e-7
    return sorted(result,key=lambda x:x['mean_contribution'],reverse=True)


def run(args):
    torch.set_num_threads(1);out=Path(args.comparison)
    summary=json.loads((out/'summary.json').read_text())
    data=load_training_data(args.data);graph,starts=graph_inputs(data)
    reference=np.load(out/'distances.npz')['reference']
    _,_,before=evaluate(data,graph,reference,starts,Path(summary['runs']['qrl_task'])/'phr.pt')
    final={arm:json.loads((out/(arm+'_decisions.json')).read_text()) for arm in summary['runs']}
    bm={b['source']:b for b in before['branches']};am={b['source']:b for b in final['qrl_task']['branches']}
    changes=[]
    for n,b in bm.items():
        a=am[n]
        if b['chosen_row']==a['chosen_row']:continue
        change=dict(node=n,before_regret=b['regret'],after_regret=a['regret'])
        for name,branch in (('before',b),('after',a)):
            scores=sorted(x['score'] for x in branch['alternatives'])
            change[name+'_score_margin']=scores[1]-scores[0]
            change[name+'_row']=branch['chosen_row']
        changes.append(change)
    write_json(out/'branch_diagnosis.json',dict(task_45k_to_50k_changes=changes,
        task45_contributions=contributions(before),
        final_contributions={arm:contributions(d) for arm,d in final.items()}))
    print(json.dumps(dict(changed_branches=len(changes),regret_decomposition_verified=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--comparison',required=True)
    run(p.parse_args())
