"""Evaluation-only graph and original closed-loop MPPI comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO/'experiments/native_margin'),str(REPO)]
from shared import rows,write,sha
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.r_series_mppi.common import CONFIG,SEEDS,adapter,stage_metrics
from minimal_qrl.baselines.mppi import MPPIController
from minimal_qrl.baselines.base import rollout_controller_episode
from minimal_qrl.eval.task_path_support_audit import physical_episodes
sys.path.insert(0,str(REPO/'experiments/unified_mechanism'))
from metrics import ranking,stats

def run(raw,out):
    torch.set_num_threads(1);data=load_training_data(raw/'results/connected_expert_utrap_20261021/main')
    ref=np.load(raw/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz')['distance'];goal,=validate_reference(data,ref)
    probes=list(csv.DictReader((REPO/'results/native_margin_20261102_verified/u/probes.csv').open()))
    nodes=data.nodes.numpy();src,dst,cost=(x.numpy() for x in (data.src,data.dst,data.cost));physical=data.original.numpy()
    _,ix=np.unique(np.column_stack((src[physical],dst[physical],cost[physical])),axis=0,return_index=True);edges=physical[ix]
    by=defaultdict(list)
    for e in edges:by[src[e]].append(e)
    # All historical physical episode starts, deduplicated by canonical node.
    physical_eps,_,_=physical_episodes(data.raw)
    starts=np.unique(data.obs_ids[[e['obs_ids'][0] for e in physical_eps]].numpy());assert len(starts)==165
    old=raw/'results/r_series_mppi_20261026/main'
    states=json.loads((old/'states.json').read_text())['records']
    historical=[json.loads(l) for l in (old/'arms/R0/episodes.jsonl').read_text().splitlines()]
    inherited={(r['state_id'],r['planner_seed']):r for r in historical};assert len(inherited)==64
    decisions=[];local=[];margins=[];graphrows=[];episodes=[];value_stats=[];batch_errors={}
    for arm in ['parent','native','no_push','no_local']:
        ck=raw/'results/task_source_balance_20261025/main/R0/final.pt' if arm=='parent' else out/arm/'final.pt'
        digest=sha(ck);value,model=adapter(data.small,ck)
        def score(x,y):
            y=np.broadcast_to(y,np.shape(x))
            return np.concatenate([value.batch_value(x[i:i+512],y[i:i+512]) for i in range(0,len(x),512)]).astype(float)
        j=score(nodes,nodes[goal]);assert j[goal]==0
        ii=np.flatnonzero(np.arange(len(nodes))!=goal)
        value_stats.append(dict(arm=arm,j_mean=float(j[ii].mean()),j_min=float(j[ii].min()),j_max=float(j[ii].max()),
                                bias=float((j[ii]-ref[ii]).mean()),rmse=float(np.sqrt(np.mean((j[ii]-ref[ii])**2)))))
        h=score(nodes[src],nodes[dst])-cost
        for scope,ii in [('records',physical),('unique',edges)]:local.append(dict(arm=arm,scope=scope,**stats(h[ii])))
        # Match the diagnostic's forward batch and FP32 cost-gap arithmetic.
        ends=torch.tensor([[int(p['good_node']),int(p['bad_node'])] for p in probes])
        with torch.no_grad():
            z=model.projector(model.encoder(data.nodes[ends.reshape(-1)]))
            zg=model.projector(model.encoder(data.nodes[goal:goal+1])).expand_as(z)
            pj=model.head(z,zg).reshape(-1,2)
            pm=(pj[:,1]-pj[:,0]+torch.tensor([float(p['cost_gap']) for p in probes])).numpy().astype(float)
        batch_errors[arm]=float(np.max(abs(pj.numpy().astype(float)-j[ends.numpy()])))
        for i,p in enumerate(probes):
            a,b=int(p['good_edge']),int(p['bad_edge'])
            m=pm[i]
            if arm=='parent':np.testing.assert_allclose(m,float(p['margin']),atol=1e-6,rtol=0)
            margins.append(dict(arm=arm,case=p['case'],margin=m,parent_margin=float(p['margin']),ref_gap=p['ref_gap']))
        for s,ee in by.items():
            if len(ee)>1:decisions.append(dict(arm=arm,case=int(s),**ranking(cost[ee],j[dst[ee]],ref[dst[ee]])))
        for s in starts:
            node=int(s);total=0.;n=0
            while node!=goal:
                ee=np.array(by[node]);e=ee[np.argmin(cost[ee]+j[dst[ee]])]
                total+=cost[e];node=dst[e];n+=1;assert n<=len(nodes)
            graphrows.append(dict(arm=arm,case=int(s),cost=total,regret=total-ref[s],steps=n))
        if arm=='parent':
            episodes.extend(dict(r,arm='parent') for r in historical);continue
        folder=out/arm/'mppi';folder.mkdir(exist_ok=False)
        class Controller(MPPIController):
            def sample_candidates(self,env):
                bank=super().sample_candidates(env)
                if not hasattr(self,'first_hash'):self.first_hash=hashlib.sha256(bank.tobytes()).hexdigest()
                return bank
        for st in states:
            for seed in SEEDS:
                c=Controller(CONFIG,terminal_mode='qrl',qrl_agent=value)
                trace=rollout_controller_episode(c,data.env,episode_seed=seed,device_id='u_trap_target',start_state=np.array(st['pose'],np.float32))
                np.testing.assert_array_equal(trace['states'][0],np.array(st['pose'],np.float32))
                assert c.first_hash==inherited[st['id'],seed]['first_candidate_sha256']
                n=len(trace['actions']);assert n<=300
                if trace['success']:assert data.env.is_task_feasible(data.env.state)
                rec=dict(arm=arm,state_id=st['id'],planner_seed=seed,stratum=st['stratum'],success=int(trace['success']),
                         collision=int(trace['collision']),out_of_bounds=int(trace['out_of_bounds']),
                         timeout=int(trace['truncated'] and not trace['success']),steps=n,cost=float(-np.sum(trace['rewards'])),
                         **stage_metrics(np.asarray(trace['states']),st))
                assert sum(rec[k] for k in ['success','collision','out_of_bounds','timeout'])==1
                episodes.append(rec)
                np.savez_compressed(folder/f"{st['id']}_{seed}.npz",states=trace['states'],actions=trace['actions'],rewards=trace['rewards'])
            print('U eval',arm,st['id'],flush=True)
        assert sha(ck)==digest
        rows(out/'episodes.csv',episodes)
    rows(out/'margins.csv',margins);rows(out/'local.csv',local);rows(out/'decisions.csv',decisions);rows(out/'graph.csv',graphrows)
    rows(out/'values.csv',value_stats)
    write(out/'evaluation.json',dict(starts=len(starts),branches=58,mppi_per_arm=64,all_first_banks_match=True,probe_vs_adapter_batch_max_error=batch_errors,
           parent_episodes_sha256=sha(old/'arms/R0/episodes.jsonl'),reference_sha256=sha(raw/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
