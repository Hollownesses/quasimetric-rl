"""Hard-feasible function in the same IQE architecture, no optimization."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'experiments/native_margin'),str(REPO/'experiments/unified_mechanism')]
from shared import rows,write,sha,verify_inputs
from metrics import ranking,stats
from experiments.native_feasibility.core import hard_scale
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.r_series_mppi.common import adapter,CONFIG,SEEDS
from minimal_qrl.baselines.mppi import MPPIController
from minimal_qrl.baselines.base import rollout_controller_episode
from minimal_qrl.eval.task_path_support_audit import physical_episodes

def run(raw,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);inputs=verify_inputs(raw)
    data=load_training_data(raw/'results/connected_expert_utrap_20261021/main')
    ck=raw/'results/task_source_balance_20261025/main/R0/final.pt';value,_=adapter(data.small,ck)
    original=copy.deepcopy(value.state_dict());physical=data.original.numpy()
    nodes=data.nodes.numpy();src,dst,cost=(x.numpy() for x in (data.src,data.dst,data.cost))
    def score(x,y):
        y=np.broadcast_to(y,x.shape).copy()
        return np.concatenate([value.batch_value(x[i:i+512],y[i:i+512]) for i in range(0,len(x),512)]).astype(float)
    d0=score(nodes[src[physical]],nodes[dst[physical]])
    upper,alpha=hard_scale(d0,cost[physical])
    final=value.qrl_agent.critics[0].quasimetric_model.projector.module[-1]
    assert isinstance(final,torch.nn.Linear)
    with torch.no_grad():final.weight.mul_(alpha);final.bias.mul_(alpha)
    changed=[k for k,v in value.state_dict().items() if not torch.equal(v,original[k])]
    assert len(changed)==(0 if alpha==1 else 2)
    d1=score(nodes[src[physical]],nodes[dst[physical]])
    h0=d0-cost[physical];h1=d1-cost[physical]
    assert h1.max()<=1e-5
    homogeneous=float(abs(d1-alpha*d0).max())
    relative_l2=float(np.linalg.norm(d1-alpha*d0)/np.linalg.norm(alpha*d0))
    assert relative_l2<1e-4,'FP32 homogeneous realization deviated materially'
    np.savez_compressed(out/'local.npz',distance=d0,projected=d1,cost=cost[physical])
    # Reference is read only after the scale is fixed and feasibility is checked.
    refpath=raw/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz'
    ref=np.load(refpath)['distance'];goal,=validate_reference(data,ref)
    j=score(nodes,nodes[goal]);assert j[goal]==0
    _,ix=np.unique(np.column_stack((src[physical],dst[physical],cost[physical])),axis=0,return_index=True)
    by=defaultdict(list)
    for e in physical[ix]:by[src[e]].append(e)
    decisions=[]
    for s,ee in by.items():
        if len(ee)>1:decisions.append(dict(case=int(s),**ranking(cost[ee],j[dst[ee]],ref[dst[ee]])))
    pe,_,_=physical_episodes(data.raw);starts=np.unique(data.obs_ids[[e['obs_ids'][0] for e in pe]].numpy());assert len(starts)==165
    graphrows=[]
    for s in starts:
        node=int(s);total=0.;n=0
        while node!=goal:
            ee=np.array(by[node]);e=ee[np.argmin(cost[ee]+j[dst[ee]])]
            total+=float(cost[e]);node=int(dst[e]);n+=1;assert n<=len(nodes)
        graphrows.append(dict(case=int(s),cost=total,regret=total-ref[s]))
    old=raw/'results/r_series_mppi_20261026/main'
    inherited=[json.loads(x) for x in (old/'arms/R0/episodes.jsonl').read_text().splitlines()]
    oldrows={(r['state_id'],r['planner_seed']):r for r in inherited}
    states=json.loads((old/'states.json').read_text())['records'];episodes=[];edge_h=[]
    folder=out/'mppi';folder.mkdir()
    class Controller(MPPIController):
        def sample_candidates(self,env):
            bank=super().sample_candidates(env)
            if not hasattr(self,'first_hash'):self.first_hash=hashlib.sha256(bank.tobytes()).hexdigest()
            return bank
    for st in states:
        for seed in SEEDS:
            c=Controller(CONFIG,terminal_mode='qrl',qrl_agent=value)
            tr=rollout_controller_episode(c,data.env,episode_seed=seed,device_id='u_trap_target',start_state=np.array(st['pose'],np.float32))
            assert c.first_hash==oldrows[st['id'],seed]['first_candidate_sha256']
            obs=np.array([data.env.state_to_observation(s) for s in tr['states']])
            hh=score(obs[:-1],obs[1:])+np.asarray(tr['rewards'])
            edge_h.extend(hh);n=len(tr['actions']);assert n<=300
            rec=dict(case=st['id'],seed=seed,success=int(tr['success']),parent_success=int(oldrows[st['id'],seed]['success']),
                     collision=int(tr['collision']),steps=n,cost=float(-np.sum(tr['rewards'])),max_h=float(hh.max()),positive_h=int((hh>1e-4).sum()))
            episodes.append(rec)
            np.savez_compressed(folder/f"{st['id']}_{seed}.npz",states=tr['states'],actions=tr['actions'],rewards=tr['rewards'],h=hh)
        rows(out/'episodes.csv',episodes);print('feasible U',st['id'],flush=True)
    rows(out/'decisions.csv',decisions);rows(out/'graph.csv',graphrows)
    summary=dict(scene='U',alpha_upper=upper,alpha=alpha,training_records=len(physical),
        old_max_h=float(h0.max()),new_max_h=float(h1.max()),old_positive_fraction=float((h0>1e-4).mean()),
        new_positive_fraction=float((h1>1e-4).mean()),homogeneity_max_error=homogeneous,homogeneity_relative_l2=relative_l2,changed_leaves=changed,
        reference_j_bias=float((j-ref).mean()),j_above_reference_max=float((j-ref).max()),
        reversed_pairs=sum(r['reversed_pairs'] for r in decisions),strict_pairs=sum(r['strict_pairs'] for r in decisions),
        wrong_first=sum(r['wrong'] for r in decisions),graph_regret_mean=float(np.mean([r['regret'] for r in graphrows])),
        successes=sum(r['success'] for r in episodes),episodes=64,
        gained=sum(r['success'] and not r['parent_success'] for r in episodes),lost=sum(not r['success'] and r['parent_success'] for r in episodes),
        actual_path_h=stats(edge_h),inputs=inputs,sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    for rel,digest in inputs.items():assert sha(raw/rel)==digest
    write(out/'analysis.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
