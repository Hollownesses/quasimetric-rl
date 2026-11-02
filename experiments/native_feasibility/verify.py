"""Replay every new physical trajectory and independently check full-population certificates."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.native_margin.shared import sha,write
from experiments.native_feasibility.core import hard_scale,risk
from experiments.connected_expert_utrap.data import load_training_data
from experiments.r_series_mppi.common import adapter

def run(raw,out):
    torch.set_num_threads(1)
    u=json.loads((out/'u/analysis.json').read_text());o=json.loads((out/'og/analysis.json').read_text())
    local=np.load(out/'u/local.npz');assert len(local['distance'])==42536
    upper,alpha=hard_scale(local['distance'],local['cost']);assert alpha==u['alpha'] and upper==u['alpha_upper']
    assert np.max(local['projected']-local['cost'])<=1e-5
    ol=np.load(out/'og/local.npz');assert len(ol['distance'])==1000000
    np.testing.assert_allclose(risk(ol['distance']),o['old_risk'],atol=1e-14,rtol=0)
    assert risk(ol['projected'])<=o['eps']+1e-6
    data=load_training_data(raw/'results/connected_expert_utrap_20261021/main')
    np.testing.assert_array_equal(local['cost'],data.cost[data.original].numpy())
    value,_=adapter(data.small,raw/'results/task_source_balance_20261025/main/R0/final.pt')
    layer=value.qrl_agent.critics[0].quasimetric_model.projector.module[-1]
    with torch.no_grad():layer.weight.mul_(alpha);layer.bias.mul_(alpha)
    st=json.loads((raw/'results/r_series_mppi_20261026/main/states.json').read_text())['records'];starts={s['id']:s['pose'] for s in st}
    with (out/'u/episodes.csv').open() as f:ee=list(csv.DictReader(f))
    assert len(ee)==64 and len({(r['case'],r['seed']) for r in ee})==64
    xx=[];yy=[];cost=[];cached=[];traces={}
    for r in ee:
        p=out/'u/mppi'/f"{r['case']}_{r['seed']}.npz";tr=np.load(p);n=len(tr['actions'])
        ob,_=data.env.reset(seed=int(r['seed']),options=dict(device_id='u_trap_target',start=np.asarray(starts[r['case']],np.float32)))
        assert n==int(r['steps']) and n<=300
        for k,action in enumerate(tr['actions']):
            xx.append(ob)
            ob,reward,term,trunc,info=data.env.step(action);yy.append(ob);cost.append(-reward)
            np.testing.assert_array_equal(data.env.state,tr['states'][k+1])
            np.testing.assert_allclose(reward,tr['rewards'][k],atol=1e-12,rtol=0)
            assert (term or trunc)==(k==n-1)
        assert bool(info['success'])==bool(int(r['success']))
        cached.extend(tr['h']);traces[str(p.relative_to(out))]=sha(p)
    x,y=np.asarray(xx),np.asarray(yy)
    direct=np.concatenate([value.batch_value(x[i:i+512],y[i:i+512]) for i in range(0,len(x),512)]).astype(float)-np.asarray(cost)
    np.testing.assert_allclose(direct,cached,atol=5e-4,rtol=5e-4)
    for rel,digest in u['inputs'].items():assert sha(raw/rel)==digest
    write(out/'verification.json',dict(physical_trajectories_replayed=64,physical_steps=len(cost),all_states_rewards_termination_exact=True,
        actual_h_batch_max_difference=float(np.max(abs(direct-cached))),full_U_hard_population=True,full_OG_mean_population=True,
        input_sha256_unchanged=True,trace_sha256=traces,sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('VERIFIED',len(cost),'physical replay steps and both native empirical constraints')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
