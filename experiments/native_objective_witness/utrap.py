"""Existing U critics, original complete query functionals, and a hard-feasible ray."""
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
REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO),str(REPO/'experiments/unified_mechanism')]
from experiments.native_margin.shared import rows,write,sha,verify_inputs
from experiments.native_objective_witness.core import hard_boundary
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.task_source_balance.train import task_weights
from experiments.r_series_mppi.common import adapter,CONFIG,SEEDS
from experiments.four_path_factorial.run import tensor_digest
from minimal_qrl.baselines.mppi import MPPIController
from minimal_qrl.baselines.base import rollout_controller_episode
from minimal_qrl.eval.task_path_support_audit import physical_episodes
from metrics import ranking

def run(raw,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);inputs=verify_inputs(raw)
    manifest=json.loads((REPO/'results/unified_mechanism_20261102/audit.json').read_text())['verified_sha256']
    data=load_training_data(raw/'results/connected_expert_utrap_20261021/main')
    physical=data.original.numpy();nodes=data.nodes.numpy();src,dst,cost=(x.numpy() for x in (data.src,data.dst,data.cost))
    for k in ['global_push_task_mask','global_push_pair_mask']:assert data.raw.transition_infos[k][data.original].bool().all()
    weights,_=task_weights(data);w=weights[data.original].numpy().astype(float)
    points={};jcache={};models={};alphas={};local={};digests={}
    def score(value,x,y):
        y=np.broadcast_to(y,x.shape).copy()
        return np.concatenate([value.batch_value(x[i:i+512],y[i:i+512]) for i in range(0,len(x),512)]).astype(float)
    for arm in ['R0','R1']:
        ck=raw/f'results/task_source_balance_20261025/main/{arm}/final.pt';digests[arm]=sha(ck)
        assert digests[arm]==manifest[str(ck.relative_to(raw))]
        value,_=adapter(data.small,ck);original=copy.deepcopy(value.state_dict())
        d0=score(value,nodes[src[physical]],nodes[dst[physical]])
        upper,alpha=hard_boundary(d0,cost[physical]);alphas[arm]=alpha
        for scale in ['raw','boundary']:
            if scale=='boundary':
                layer=value.qrl_agent.critics[0].quasimetric_model.projector.module[-1]
                with torch.no_grad():layer.weight.mul_(alpha);layer.bias.mul_(alpha)
            d=score(value,nodes[src[physical]],nodes[dst[physical]])
            h=d-cost[physical]
            if scale=='boundary':assert h.max()<=1e-5
            task=score(value,nodes[src[physical]],nodes[data.goal[data.original].numpy()])
            pair=score(value,nodes[src[physical]],nodes[data.pair[data.original].numpy()])
            key=arm+'_'+scale
            points[key]=dict(arm=arm,scale=scale,alpha=1. if scale=='raw' else alpha,alpha_upper=upper,
                F0=float(.6*task.mean()+.4*pair.mean()),F1=float(.6*np.mean(w*task)+.4*pair.mean()),
                task_record_mean=float(task.mean()),task_source_weighted_mean=float(np.mean(w*task)),pair_mean=float(pair.mean()),
                max_h=float(h.max()),feasible=int(h.max()<=1e-5),params_sha256=tensor_digest(*value.state_dict().values()))
            goal=int(data.goal[data.original][0]);jcache[key]=score(value,nodes,nodes[goal])
            local[key]=d
        changed=[k for k,v in value.state_dict().items() if not torch.equal(v,original[k])];assert len(changed)==2
        points[arm+'_boundary']['changed_leaves']=changed
        models[arm]=value
    # Only after all scales and original functionals are fixed, load the evaluator.
    ref=np.load(raw/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz')['distance'];goal,=validate_reference(data,ref)
    _,ix=np.unique(np.column_stack((src[physical],dst[physical],cost[physical])),axis=0,return_index=True)
    by=defaultdict(list)
    for e in physical[ix]:by[src[e]].append(e)
    pe,_,_=physical_episodes(data.raw);starts=np.unique(data.obs_ids[[e['obs_ids'][0] for e in pe]].numpy());assert len(starts)==165
    decisions=[];graphrows=[]
    for key,j in jcache.items():
        for s,ee in by.items():
            if len(ee)>1:decisions.append(dict(point=key,case=int(s),**ranking(cost[ee],j[dst[ee]],ref[dst[ee]])))
        for s in starts:
            node=int(s);total=0.;n=0
            while node!=goal:
                ee=np.asarray(by[node]);e=ee[np.argmin(cost[ee]+j[dst[ee]])];total+=float(cost[e]);node=int(dst[e]);n+=1;assert n<=len(nodes)
            graphrows.append(dict(point=key,case=int(s),regret=total-ref[s]))
    inherited=raw/'results/r_series_mppi_20261026/main';episodes=[]
    for arm in ['R0','R1']:
        for line in (inherited/f'arms/{arm}/episodes.jsonl').read_text().splitlines():
            r=json.loads(line);episodes.append(dict(point=arm+'_raw',case=r['state_id'],seed=r['planner_seed'],success=int(r['success']),steps=r['steps'],reused=1))
    previous=REPO/'results/native_feasibility_20261103/u'
    old=json.loads((previous/'analysis.json').read_text());assert old['alpha']==alphas['R0']
    assert old['inputs']['results/task_source_balance_20261025/main/R0/final.pt']==digests['R0']
    with (previous/'episodes.csv').open() as f:
        for r in csv.DictReader(f):episodes.append(dict(point='R0_boundary',case=r['case'],seed=int(r['seed']),success=int(r['success']),steps=int(r['steps']),reused=1))
    folder=out/'R1_boundary';folder.mkdir();value=models['R1'];state_rows=json.loads((inherited/'states.json').read_text())['records']
    hist={(r['state_id'],r['planner_seed']):r for r in [json.loads(l) for l in (inherited/'arms/R1/episodes.jsonl').read_text().splitlines()]}
    class Controller(MPPIController):
        def sample_candidates(self,env):
            bank=super().sample_candidates(env)
            if not hasattr(self,'first_hash'):self.first_hash=hashlib.sha256(bank.tobytes()).hexdigest()
            return bank
    for st in state_rows:
        for seed in SEEDS:
            c=Controller(CONFIG,terminal_mode='qrl',qrl_agent=value)
            tr=rollout_controller_episode(c,data.env,episode_seed=seed,device_id='u_trap_target',start_state=np.asarray(st['pose'],np.float32))
            assert c.first_hash==hist[st['id'],seed]['first_candidate_sha256'];assert len(tr['actions'])<=300
            if tr['success']:assert data.env.is_task_feasible(data.env.state)
            episodes.append(dict(point='R1_boundary',case=st['id'],seed=seed,success=int(tr['success']),steps=len(tr['actions']),reused=0))
            np.savez_compressed(folder/f"{st['id']}_{seed}.npz",states=tr['states'],actions=tr['actions'],rewards=tr['rewards'])
        rows(out/'episodes.csv',episodes);print('witness U',st['id'],flush=True)
    for key,r in points.items():
        dd=[x for x in decisions if x['point']==key];ee=[x for x in episodes if x['point']==key]
        r.update(point=key,reversed_pairs=sum(x['reversed_pairs'] for x in dd),wrong_first=sum(x['wrong'] for x in dd),
            graph_regret=float(np.mean([x['regret'] for x in graphrows if x['point']==key])),successes=sum(x['success'] for x in ee),episodes=len(ee))
    rows(out/'points.csv',list(points.values()));rows(out/'decisions.csv',decisions);rows(out/'graph.csv',graphrows)
    np.savez_compressed(out/'local.npz',**local)
    for arm,digest in digests.items():assert sha(raw/f'results/task_source_balance_20261025/main/{arm}/final.pt')==digest
    write(out/'audit.json',dict(inputs=inputs,checkpoints=digests,scales=alphas,physical_rows=len(physical),
          query_objectives_exact=True,training_updates=0,R0_boundary_alpha_exact_to_previous=True,
          points=points,sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
