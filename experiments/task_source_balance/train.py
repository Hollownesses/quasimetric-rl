"""Independent entry point; no reference/graph-distance imports or queries.

R0 delegates its update to the unchanged historical trainer. R1 changes only
its task mean. All weights are computed globally from existing canonical IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.connected_expert_utrap.data import load_training_data
from experiments.full_replay_four_path.train import EpochSampler, distances, scan, advance as original_advance
from experiments.four_path_factorial.run import IndexedIQE, tensor_digest, file_digest
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def task_weights(data):
    rows=data.original
    rows=rows[data.physical[rows] & data.raw.transition_infos['global_push_task_mask'][rows].bool()]
    sources,inverse,counts=torch.unique(data.src[rows],return_inverse=True,return_counts=True)
    if not len(rows): raise ValueError('empty task population')
    values=len(rows)/(len(sources)*counts[inverse].double())
    weights=torch.zeros(len(data.src));weights[rows]=values.float()
    totals=torch.zeros(len(sources),dtype=torch.float64).scatter_add_(0,inverse,weights[rows].double())
    assert torch.allclose(totals,torch.full_like(totals,len(rows)/len(sources)),atol=1e-6,rtol=1e-7)
    assert abs(float(weights[rows].double().mean())-1)<1e-7
    return weights,dict(N=len(rows),M=len(sources),mean=float(weights[rows].double().mean()),
        min=float(values.min()),max=float(values.max()),max_source_count=int(counts.max()),
        source_total_min=float(totals.min()),source_total_max=float(totals.max()),
        weights_sha256=tensor_digest(weights),identity='existing full-observation/task canonical node ID',
        normalization='global N/(M*n(s)); no minibatch weight-sum normalization')


def weighted_push(data,rows,task,pair,weights):
    physical=rows[data.physical[rows]]
    tm=data.raw.transition_infos['global_push_task_mask'][physical].bool()
    pm=data.raw.transition_infos['global_push_pair_mask'][physical].bool()
    t=(weights[physical][tm]*task[tm]);p=pair[pm]
    tv=t.mean() if t.numel() else t.sum();pv=p.mean() if p.numel() else p.sum()
    return .6*tv+.4*pv,dict(task_rows=int(tm.sum()),pair_rows=int(pm.sum()),
        optimized_pair_rows=int(pm.sum()),empty_task_batch=not bool(tm.any()),
        task_mean=float(tv.detach()),pair_mean=float(pv.detach()))


def advance(model,data,dual,optimizer,rows,mode,weights):
    if mode=='record': return original_advance(model,data,dual,optimizer,rows,'task-b1')
    if mode!='source': raise ValueError('unknown task weight mode')
    optimizer.zero_grad(set_to_none=True)
    edge,task,pair=distances(model,data,rows);h=edge-data.cost[rows]
    push,component=weighted_push(data,rows,task,pair,weights)
    loss=-push+phr_terms(h,dual[rows],100.)[0]
    residual=h.detach()
    if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
    loss.backward();optimizer.step()
    with torch.no_grad(): dual[rows]=(dual[rows]+residual).relu()
    return dict(loss=float(loss.detach()),connected_expert_push=float(push.detach()),connected_expert_components=component)


def learning_rate(step,main,decay):
    return 2e-5 if step<=main else 2e-6*.1**((step-main-1)/max(1,decay-1))


def initial_state(data,seed,sampler_seed=2307):
    torch.manual_seed(seed)
    model=IndexedIQE(data.small,'encoder',0,initialization_seed=seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.)
    dual=torch.zeros(len(data.src));visits=torch.zeros(len(data.original),dtype=torch.long)
    sampler=EpochSampler(len(data.original),sampler_seed)
    return model,optimizer,dual,visits,sampler


def initial_fields(model,optimizer,dual,visits,sampler):
    return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),dual=dual,visits=visits,
        sampler=sampler.state_dict(),rng=torch.get_rng_state(),
        push_counters=dict(connected_expert_empty_task_batches=0,
            connected_expert_task_visits=0,connected_expert_pair_visits=0))


def load_initial(protocol):
    # Protocols from the original seed606 experiment retain their old identity.
    path=protocol.get('initial_checkpoint',protocol['historical_initial'])
    digest=protocol.get('initial_checkpoint_sha256',protocol['historical_initial_sha256'])
    assert file_digest(path)==digest,('initial checkpoint changed',path)
    initial=torch.load(path,map_location='cpu',weights_only=True)
    assert initial['step']==0 and initial['fingerprints']==protocol['fingerprints']
    assert initial['initialization_seed']==protocol['seed'], 'initialization seed mismatch'
    return initial


def restore(saved,signature,model,optimizer,sampler):
    if saved['training_signature']!=signature: raise ValueError('checkpoint configuration/weight mode mismatch')
    model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
    sampler.restore(saved['sampler']);torch.set_rng_state(saved['rng'])
    return saved['dual'].clone(),saved['visits'].clone()


def run(args):
    protocol=json.loads(Path(args.protocol).read_text());torch.set_num_threads(protocol['threads'])
    for path,digest in protocol['source_files'].items():
        assert file_digest(path)==digest,('source changed after freeze',path)
    data=load_training_data(protocol['data'],task_mode='all');weights,stats=task_weights(data)
    assert data.fingerprint==protocol['fingerprints'] and stats==protocol['weights']
    # This experiment's all-task connected dataset has every physical row active.
    for key in ('global_push_task_mask','global_push_pair_mask'):
        assert data.raw.transition_infos[key][data.original].bool().all()
    signature=dict(protocol_sha256=file_digest(args.protocol),task_weight_mode=args.mode,
        weights_sha256=stats['weights_sha256'],fingerprints=data.fingerprint)
    out=Path(args.output);model,optimizer,dual,visits,sampler=initial_state(
        data,protocol['seed'],protocol.get('sampler_seed',2307))
    initial=load_initial(protocol)
    current=initial_fields(model,optimizer,dual,visits,sampler)
    for key,value in current.items(): assert equal(value,initial[key]),('paired initial mismatch',key)
    offset=0;counters=dict(connected_expert_empty_task_batches=0,connected_expert_task_visits=0,connected_expert_pair_visits=0)
    # Persist a hash chain of the ordered row sequence, resumable without serializing a hashlib object.
    sequence='0'*64
    if args.resume:
        assert json.loads((out/'config.json').read_text())['training_signature']==signature
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        dual,visits=restore(saved,signature,model,optimizer,sampler)
        offset=saved['step'];counters=saved['push_counters'].copy();sequence=saved['sampling_chain_sha256']
    else:
        out.mkdir(parents=True,exist_ok=False)
        write(out/'config.json',dict(protocol,training_signature=signature,task_weight_mode=args.mode,
            initial_model_sha256=tensor_digest(*model.state_dict().values())))
        write(out/'initial_verification.json',dict(paired_initial_exact=True,
            historical_initial_exact=protocol.get('initial_exact_to_history',True),
            seed=protocol['seed'],sampler_seed=protocol.get('sampler_seed',2307),
            fields=list(current),zero_dual=True))
        torch.save(weights,out/'task_weights.pt')
    def save(step,name):
        state=dict(step=step,initialization_seed=protocol['seed'],model=model.state_dict(),optimizer=optimizer.state_dict(),
            dual=dual,visits=visits,sampler=sampler.state_dict(),rng=torch.get_rng_state(),
            fingerprints=data.fingerprint,task_mode='all',push_counters=counters,
            dataset_kind=data.dataset_kind,training_signature=signature,sampling_chain_sha256=sequence,
            task_weights=weights,task_weight_mode=args.mode)
        temporary=out/(name+'.tmp');torch.save(state,temporary);temporary.replace(out/name)
    if not args.resume:save(0,'initial.pt')
    total=protocol['phr_steps']+protocol['decay_steps'];assert 0<=offset<=total
    started=time.perf_counter()
    with (out/'training.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(offset,total+1):
            if step%protocol['audit_interval']==0 or step in (protocol['phr_steps'],total):
                rec,h=scan(model,data,dual,visits)
                assert torch.isfinite(h).all() and torch.isfinite(dual).all()
                rec.update(step=step,seconds=time.perf_counter()-started,epochs=sampler.epochs,
                    push_counters=counters.copy(),sampling_chain_sha256=sequence)
                log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True)
                save(step,'latest.pt')
                if step==protocol['phr_steps']:save(step,'phr.pt')
                if step==total:
                    save(step,'final.pt');np.savez_compressed(out/'final_constraints.npz',residual=h.numpy(),dual=dual.numpy());break
            j=step+1;lr=learning_rate(j,protocol['phr_steps'],protocol['decay_steps'])
            for group in optimizer.param_groups:group['lr']=lr
            indices=sampler.next(protocol['batch']);rows=data.original[indices];visits[indices]+=1
            sequence=hashlib.sha256(bytes.fromhex(sequence)+rows.numpy().tobytes()).hexdigest()
            result=advance(model,data,dual,optimizer,rows,args.mode,weights)
            c=result['connected_expert_components']
            counters['connected_expert_empty_task_batches']+=int(c['empty_task_batch'])
            counters['connected_expert_task_visits']+=c['task_rows'];counters['connected_expert_pair_visits']+=c['pair_rows']
            if j%100==0:
                log.write(json.dumps(dict(step=j,lr=lr,**result))+'\n');log.flush()
    write(out/'COMPLETE',dict(step=total,final_sha256=file_digest(out/'final.pt'),
        training_signature=signature,sampling_chain_sha256=sequence,reference_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['record','source'],required=True);p.add_argument('--resume');run(p.parse_args())
