"""Independent native U forward/physical replay and cross-environment delivery checks."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.connected_expert_utrap.data import load_training_data
from experiments.full_replay_four_path.train import distances
from experiments.four_path_factorial.run import IndexedIQE
from experiments.task_source_balance.train import task_weights
from experiments.native_margin.shared import sha,write
from experiments.native_objective_witness.core import hard_boundary

def run(raw,out,u_only=False):
    torch.set_num_threads(1);data=load_training_data(raw/'results/connected_expert_utrap_20261021/main')
    with (out/'u/points.csv').open() as f:points={r['point']:r for r in csv.DictReader(f)}
    costs=data.cost[data.original].numpy();local=np.load(out/'u/local.npz');w,_=task_weights(data)
    proof={}
    for arm in ['R0','R1']:
        state=torch.load(raw/f'results/task_source_balance_20261025/main/{arm}/final.pt',weights_only=True)
        upper,alpha=hard_boundary(local[arm+'_raw'],costs)
        assert alpha==float(points[arm+'_boundary']['alpha'])
        assert (local[arm+'_boundary']-costs).max()<=1e-5
        model=IndexedIQE(data.small,'encoder',0,initialization_seed=606);model.load_state_dict(state['model']);model.eval()
        for scale in ['raw','boundary']:
            if scale=='boundary':
                last=model.projector.module[-1]
                with torch.no_grad():last.weight.mul_(alpha);last.bias.mul_(alpha)
            all_edge=[];tasks=[];weighted=[];pairs=[]
            with torch.no_grad():
                for rr in data.original.split(512):
                    edge,task,pair=distances(model,data,rr);all_edge.append(edge.double().numpy())
                    tasks.append(task.double().numpy());weighted.append((w[rr].double()*task.double()).numpy());pairs.append(pair.double().numpy())
            edge,task,weight,pair=(np.concatenate(z) for z in (all_edge,tasks,weighted,pairs))
            f0=.6*task.mean()+.4*pair.mean();f1=.6*weight.mean()+.4*pair.mean();r=points[arm+'_'+scale]
            np.testing.assert_allclose([f0,f1],[float(r['F0']),float(r['F1'])],atol=1e-3,rtol=1e-5)
            if scale=='boundary':assert (edge-costs).max()<=1e-5
            proof[arm+'_'+scale]=dict(F0_native_forward=float(f0),F1_native_forward=float(f1),
                F0_difference=float(f0-float(r['F0'])),F1_difference=float(f1-float(r['F1'])),max_h_native=float((edge-costs).max()))
    with (out/'u/episodes.csv').open() as f:ee=list(csv.DictReader(f))
    fresh=[r for r in ee if r['reused']=='0'];assert len(fresh)==64
    starts={r['id']:r['pose'] for r in json.loads((raw/'results/r_series_mppi_20261026/main/states.json').read_text())['records']}
    traces={};steps=0
    for r in fresh:
        p=out/'u/R1_boundary'/f"{r['case']}_{r['seed']}.npz";tr=np.load(p)
        data.env.reset(seed=int(r['seed']),options=dict(device_id='u_trap_target',start=np.asarray(starts[r['case']],np.float32)))
        n=len(tr['actions']);assert n==int(r['steps']) and n<=300
        for k,a in enumerate(tr['actions']):
            _,reward,term,trunc,info=data.env.step(a)
            np.testing.assert_array_equal(data.env.state,tr['states'][k+1])
            np.testing.assert_allclose(reward,tr['rewards'][k],atol=1e-12,rtol=0)
            assert (term or trunc)==(k==n-1)
        assert bool(info['success'])==bool(int(r['success']));steps+=n;traces[str(p.relative_to(out))]=sha(p)
    audit=json.loads((out/'u/audit.json').read_text())
    for rel,digest in audit['checkpoints'].items():assert sha(raw/f'results/task_source_balance_20261025/main/{rel}/final.pt')==digest
    write(out/'u/verification.json',dict(native_forward=proof,complete_fresh_episodes=64,physical_steps_replayed=steps,
          all_states_rewards_termination_exact=True,traces=traces,source_inputs_unchanged=True))
    if not u_only:
        og=json.loads((out/'og/verification.json').read_text());assert og['full_traces']==100 and og['new_queries']==524288
        write(out/'verification.json',dict(U_native_forward_verified=True,new_full_episodes=164,OG_official_terms_verified=True,
            original_network_structures=True,training_updates=0,sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('VERIFIED U original query functionals and',steps,'new physical replay steps')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);p.add_argument('--u-only',action='store_true')
    a=p.parse_args();run(a.raw.resolve(),a.out,a.u_only)
