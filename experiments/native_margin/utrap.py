"""Native PHR/Adam pressure on reference-selected physical graph branches."""
import argparse
import copy
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
from shared import REPO, rows, select_pair, sha, source_weights, verify_inputs, write
sys.path.insert(0, str(REPO))
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.full_replay_four_path.train import EpochSampler, distances, advance
from experiments.four_path_factorial.run import IndexedIQE
from experiments.task_source_balance.train import equal
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms

def run(raw, out):
    out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    ids = verify_inputs(raw)
    data = load_training_data(raw/'results/connected_expert_utrap_20261021/main')
    ck = raw/'results/task_source_balance_20261025/main/R0/final.pt'
    state = torch.load(ck, map_location='cpu', weights_only=True)
    model = IndexedIQE(data.small, 'encoder', 0, initialization_seed=606)
    model.load_state_dict(state['model']); initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
    opt.load_state_dict(state['optimizer']); lr = opt.param_groups[0]['lr']
    params = list(model.parameters())
    def flat(gg): return torch.cat([g.reshape(-1) for g in gg]).double()
    def grad(loss, retain=True): return flat(torch.autograd.grad(loss, params, retain_graph=retain))
    ref = np.load(raw/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz')
    v, nxt = ref['distance'], ref['next_edge']; terminal, = validate_reference(data, v)
    physical = data.original.numpy(); src, dst, cost = (x.numpy() for x in (data.src,data.dst,data.cost))
    _, ix = np.unique(np.column_stack((src[physical],dst[physical],cost[physical])),axis=0,return_index=True)
    by = defaultdict(list)
    for i in physical[np.sort(ix)]: by[src[i]].append(i)
    copies = Counter(zip(src[physical], dst[physical], cost[physical]))
    support = Counter(src[physical]); probes = []; ends = []
    for s, ee in sorted(by.items()):
        if len(ee)<2: continue
        pair = select_pair(cost[ee]+v[dst[ee]])
        if pair is None: continue
        a,b = (ee[i] for i in pair); ends.append([dst[a],dst[b]])
        p = dict(case=int(s),good_edge=int(a),bad_edge=int(b),good_node=int(dst[a]),bad_node=int(dst[b]),
                 ref_gap=float(cost[b]+v[dst[b]]-cost[a]-v[dst[a]]),cost_gap=float(cost[b]-cost[a]))
        for name,e in [('good',a),('bad',b)]:
            node=dst[e]; counts=[]
            while node!=terminal:
                k=nxt[node]; assert k>=0; counts.append(copies[src[k],dst[k],cost[k]]); node=dst[k]
            p[name+'_source_records']=support[dst[e]]
            p[name+'_suffix_min_copies']=min(counts) if counts else 0
        probes.append(p)
    ends=torch.tensor(ends)
    def margins():
        z=model.projector(model.encoder(data.nodes[ends.reshape(-1)]))
        zg=model.projector(model.encoder(data.nodes[terminal:terminal+1])).expand_as(z)
        j=model.head(z,zg).reshape(-1,2)
        return j[:,1]-j[:,0]+torch.tensor([p['cost_gap'] for p in probes])
    mm=margins(); base=mm.detach().numpy().astype(float)
    jac=torch.stack([grad(m) for m in mm]); del mm
    weights,_,_=source_weights(src[physical,None]); w=torch.zeros(len(src));w[physical]=torch.tensor(weights,dtype=torch.float32)
    sampler=EpochSampler(len(physical),2307);sampler.restore(copy.deepcopy(state['sampler']))
    stream=hashlib.sha256(); output=[]; visits=Counter(); active=0; nrows=0; max_linear=0.; split_error=0.; split_relative=0.
    # Pure inherited Adam motion, with explicit zero gradients for all active parameters.
    model.load_state_dict(initial);opt.load_state_dict(copy.deepcopy(state['optimizer']))
    for p in params:p.grad=torch.zeros_like(p)
    opt.step();zero=margins().detach().numpy().astype(float)-base
    for batch in range(32):
        model.load_state_dict(initial);opt.load_state_dict(copy.deepcopy(state['optimizer']))
        rr=data.original[sampler.next(256)]
        stream.update(rr.numpy().tobytes());visits.update(src[rr.numpy()])
        edge,task,pair=distances(model,data,rr);h=edge-data.cost[rr]
        local=phr_terms(h,state['dual'][rr],100.)[0]
        losses=dict(task=-.6*task.mean(),b1=-.4*pair.mean(),local=local,
                    local_equal=(((state['dual'][rr].detach()+100*h).relu().square()-state['dual'][rr].detach().square())/(200)*w[rr]).mean())
        gg={k:grad(loss) for k,loss in losses.items()}
        total=grad(losses['task']+losses['b1']+losses['local'],False)
        summed=gg['task']+gg['b1']+gg['local']
        split_error=max(split_error,float((summed-total).abs().max()))
        # Cancellation makes an elementwise absolute floor scale-dependent.
        relative=float(torch.linalg.vector_norm(summed-total)/torch.linalg.vector_norm(total))
        split_relative=max(split_relative,relative);assert relative<1e-5
        projections={k:(-lr*jac@g).numpy() for k,g in gg.items()}
        projections['total']=(-lr*jac@total).numpy()
        active+=int(((state['dual'][rr]+100*h.detach())>0).sum());nrows+=len(rr)
        before=flat([p.detach() for p in params])
        advance(model,data,state['dual'].clone(),opt,rr,'task-b1')
        delta=flat([p.detach() for p in params])-before
        linear=(jac@delta).numpy(); actual=margins().detach().numpy().astype(float)-base
        max_linear=max(max_linear,float(abs(linear-actual).max()))
        for i,p in enumerate(probes):
            output.append(dict(case=p['case'],batch=batch,**{k:float(x[i]) for k,x in projections.items()},
                               adam=float(actual[i]),adam_linear=float(linear[i])))
        print('U batch',batch,flush=True)
    model.load_state_dict(initial)
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    assert equal(model.state_dict(),initial) and equal(opt.state_dict(),state['optimizer'])
    assert sha(ck)==ids[str(ck.relative_to(raw))]
    for i,p in enumerate(probes):
        p.update(margin=float(base[i]),zero_adam=float(zero[i]),
                 good_sample_visits=visits[p['good_node']],bad_sample_visits=visits[p['bad_node']])
    rows(out/'probes.csv',probes);rows(out/'pressure.csv',output)
    write(out/'audit.json',dict(inputs=ids,lr=lr,probes=len(probes),batch_rows=nrows,
          stream_sha256=stream.hexdigest(),local_active_fraction=active/nrows,
          split_gradient_max_error=split_error,split_gradient_relative_l2_max=split_relative,adam_linear_max_error=max_linear,
          checkpoint_unchanged=True,model_restored=True,source_weights_mean=float(weights.mean()),
          source_files={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.raw.resolve(),a.out)
