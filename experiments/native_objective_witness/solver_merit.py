"""Posthoc algebra on retained predictions; no new data/model queries or updates."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.native_margin.shared import rows,write,sha

def run(raw,out,mode):
    if mode=='u':
        import torch
        from experiments.connected_expert_utrap.data import load_training_data
        from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms
        torch.set_num_threads(1);data=load_training_data(raw/'results/connected_expert_utrap_20261021/main')
        with (out/'u/points.csv').open() as f:points=list(csv.DictReader(f))
        local=np.load(out/'u/local.npz');result=[];inputs={}
        for owner in ['R0','R1']:
            p=raw/f'results/task_source_balance_20261025/main/{owner}/final.pt';inputs[owner]=sha(p)
            state=torch.load(p,weights_only=True);lam=state['dual'][data.original]
            for r in points:
                h=torch.from_numpy(local[r['point']]).float()-data.cost[data.original]
                phr=float(phr_terms(h,lam,100.)[0]);functional=float(r['F0' if owner=='R0' else 'F1'])
                result.append(dict(multiplier_state=owner,point=r['point'],feasible=r['feasible'],push=functional,PHR=phr,merit=-functional+phr,
                    objective='F0' if owner=='R0' else 'F1',rho=100))
        rows(out/'u/frozen_merit.csv',result);write(out/'u/frozen_merit_audit.json',dict(inputs=inputs,training_updates=0,new_network_queries=0,posthoc=True))
    else:
        sys.path[:0]=[str(raw/'experiments/ogbench_critic'),str(raw)]
        from common import load_base
        base,_=load_base();lam=float(base.network.select('lam')())
        audit=json.loads((out/'og/audit.json').read_text());points=audit['points'];result=[]
        for r in audit['comparisons']:
            a,b=points['original_'+r['scale']],points['E16_'+r['scale']]
            shift=lam*(b['risk']-a['risk'])
            result.append(dict(scale=r['scale'],dynamics=r['dynamics'],lambda_fixed=lam,delta_primal=r['delta_primal'],
                lambda_risk_shift=shift,delta_merit=r['delta_primal']+shift,
                hoeffding_lower=r['hoeffding_lower']+shift,hoeffding_upper=r['hoeffding_upper']+shift))
        rows(out/'og/frozen_merit.csv',result);write(out/'og/frozen_merit_audit.json',dict(lambda_fixed=lam,training_updates=0,new_network_queries=0,posthoc=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);p.add_argument('--mode',choices=['u','og'],required=True)
    a=p.parse_args();run(a.raw.resolve(),a.out,a.mode)
