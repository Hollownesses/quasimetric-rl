"""Short native PHR loss ablations. No reference or evaluation imports."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.connected_expert_utrap.data import load_training_data
from experiments.four_path_factorial.run import IndexedIQE,file_digest
from experiments.full_replay_four_path.train import EpochSampler,distances,advance
from experiments.task_source_balance.train import equal,write
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms

ARMS=('native','no_push','no_local')

def update(model,data,dual,opt,rr,arm):
    if arm=='native':return advance(model,data,dual,opt,rr,'task-b1')
    assert arm in ARMS
    opt.zero_grad(set_to_none=True)
    edge,task,pair=distances(model,data,rr);h=edge-data.cost[rr]
    push=.6*task.mean()+.4*pair.mean();local=phr_terms(h,dual[rr],100.)[0]
    loss=local if arm=='no_push' else -push
    loss.backward();opt.step()
    with torch.no_grad():dual[rr]=(dual[rr]+h.detach()).relu()
    return dict(loss=float(loss.detach()))

def run(raw,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    data=load_training_data(raw/'results/connected_expert_utrap_20261021/main')
    for k in ['global_push_task_mask','global_push_pair_mask']:assert data.raw.transition_infos[k][data.original].bool().all()
    ck=raw/'results/task_source_balance_20261025/main/R0/final.pt';digest=file_digest(ck)
    assert digest=='4225998ede46ac2cb34ab4c6114d9989b413f682863ebc498fa9e709db772c44'
    parent=torch.load(ck,map_location='cpu',weights_only=True)
    def start():
        m=IndexedIQE(data.small,'encoder',0,initialization_seed=606);m.load_state_dict(parent['model'])
        o=torch.optim.AdamW(m.parameters(),lr=2e-7,weight_decay=0);o.load_state_dict(copy.deepcopy(parent['optimizer']))
        assert o.param_groups[0]['lr']==2e-7
        return m,o,parent['dual'].clone()
    sm=EpochSampler(len(data.original),2307);sm.restore(copy.deepcopy(parent['sampler']))
    rr=data.original[sm.next(256)];m,o,dd=start();m2,o2,d2=start()
    advance(m,data,dd,o,rr,'task-b1');update(m2,data,d2,o2,rr,'native')
    assert equal(m.state_dict(),m2.state_dict()) and equal(o.state_dict(),o2.state_dict()) and torch.equal(dd,d2)
    for arm in ['no_push','no_local']:
        ma,oa,da=start();update(ma,data,da,oa,rr,arm)
        assert torch.equal(da,dd),'first-step dual clock must be unchanged by primal ablation'
    smoke={}
    for arm in ARMS:
        m,o,dd=start();sam=EpochSampler(len(data.original),2307);sam.restore(copy.deepcopy(parent['sampler']))
        for i in range(2):update(m,data,dd,o,data.original[sam.next(256)],arm)
        state=dict(model=m.state_dict(),optimizer=o.state_dict(),dual=dd,sampler=sam.state_dict())
        p=out/(arm+'_smoke.pt');torch.save(state,p);assert equal(state,torch.load(p,weights_only=True))
        smoke[arm]=dict(roundtrip=True,finite=all(torch.isfinite(x).all().item() for x in m.parameters()))
    models={};opts={};duals={}
    for arm in ARMS:models[arm],opts[arm],duals[arm]=start()
    sampler=EpochSampler(len(data.original),2307);sampler.restore(copy.deepcopy(parent['sampler']))
    torch.set_rng_state(parent['rng'])
    stream=hashlib.sha256();visits=parent['visits'].clone();first32=None
    chain=parent['sampling_chain_sha256'];counts=copy.deepcopy(parent['push_counters'])
    for i in range(512):
        ind=sampler.next(256);rr=data.original[ind];visits[ind]+=1;stream.update(rr.numpy().tobytes())
        chain=hashlib.sha256(bytes.fromhex(chain)+rr.numpy().tobytes()).hexdigest()
        counts['connected_expert_task_visits']+=len(rr);counts['connected_expert_pair_visits']+=len(rr)
        if i==31:first32=stream.hexdigest()
        for arm in ARMS:update(models[arm],data,duals[arm],opts[arm],rr,arm)
        if (i+1)%32==0:
            for arm in ARMS:assert all(torch.isfinite(p).all() for p in models[arm].parameters()) and torch.isfinite(duals[arm]).all()
            print('U trained',i+1,flush=True)
    expected=json.loads((REPO/'results/native_margin_20261102_verified/u/audit.json').read_text())['stream_sha256'];assert first32==expected
    final={}
    for arm in ARMS:
        dest=out/arm;dest.mkdir()
        state=dict(parent,step=50512,model=models[arm].state_dict(),optimizer=opts[arm].state_dict(),
                   dual=duals[arm],sampler=sampler.state_dict(),visits=visits,diagnostic_arm=arm,
                   rng=torch.get_rng_state(),sampling_chain_sha256=chain,push_counters=counts)
        torch.save(state,dest/'final.pt');reload=torch.load(dest/'final.pt',weights_only=True)
        assert equal(state,reload)
        assert all(int(v['step'])==50512 for v in opts[arm].state.values())
        final[arm]=dict(checkpoint=str(dest/'final.pt'),sha256=file_digest(dest/'final.pt'),roundtrip=True)
    assert file_digest(ck)==digest
    write(out/'training.json',dict(parent_sha256=digest,steps=512,lr=2e-7,first32_sha256=first32,
          stream_sha256=stream.hexdigest(),smoke=smoke,native_matches_original=True,checkpoints=final,
          visits_added=int((visits-parent['visits']).sum()),sources={p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
