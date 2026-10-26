"""Training-only R2 entry. Every record and every multiplier remains separate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.connected_expert_utrap.data import load_training_data
from experiments.full_replay_four_path.train import distances,scan
from experiments.four_path_factorial.run import tensor_digest,file_digest
from experiments.task_source_balance.train import (task_weights,weighted_push,initial_state,initial_fields,
    load_initial,learning_rate,equal,write,restore as original_restore)


def edge_weights(data):
    """Exact existing canonical IDs + complete action bytes + original cost.

    The representatives identify classes only; they NEVER replace record losses
    or multipliers. No distances, paths, graph reference or Oracle are queried.
    """
    rows=data.physical.nonzero().flatten()
    assert torch.equal(rows,data.original)
    ids=torch.full((len(data.src),),-1,dtype=torch.long);lookup={};representatives=[]
    for row in rows.tolist():
        key=(int(data.src[row]),int(data.dst[row]),data.raw.actions[row].numpy().tobytes(),float(data.cost[row]))
        if key not in lookup:
            lookup[key]=len(representatives);representatives.append(row)
        ids[row]=lookup[key]
    reps=torch.tensor(representatives,dtype=torch.long)
    counts=torch.bincount(ids[rows],minlength=len(reps))
    if not len(rows) or not len(reps):raise ValueError('empty physical population')
    weights=torch.zeros(len(ids));weights[rows]=(len(rows)/(len(reps)*counts[ids[rows]].double())).float()
    totals=torch.zeros(len(reps),dtype=torch.float64).scatter_add_(0,ids[rows],weights[rows].double())
    assert abs(float(weights[rows].double().mean())-1)<1e-7
    assert torch.allclose(totals,torch.full_like(totals,len(rows)/len(reps)),rtol=1e-7,atol=1e-6)
    stats=dict(N=len(rows),K=len(reps),mean=float(weights[rows].double().mean()),
        min=float(weights[rows].min()),max=float(weights[rows].max()),max_copies=int(counts.max()),
        edge_total_min=float(totals.min()),edge_total_max=float(totals.max()),
        weights_sha256=tensor_digest(weights),partition_sha256=tensor_digest(ids,reps,counts),
        identity='exact canonical source, target, complete action bytes, original scalar cost',
        normalization='global N/(K*m(e)); before PHR mean; no batch weight-sum normalization')
    return weights,ids,reps,counts,stats


def weighted_phr(residual,multiplier,weights,rho=100.):
    if residual.shape!=multiplier.shape or residual.shape!=weights.shape:raise ValueError('PHR shape mismatch')
    if rho<0:raise ValueError('rho must be nonnegative')
    lam=multiplier.detach()
    if rho==0:return (weights*lam*residual).mean()
    numerator=(lam+rho*residual).relu().square()-lam.square()
    # Weight each record's numerator BEFORE averaging. Division by the common
    # 2*rho follows the mean, preserving original rounding when weights==1.
    return (weights*numerator).mean()/(2*rho)


def advance(model,data,dual,optimizer,rows,task_weight,edge_weight):
    optimizer.zero_grad(set_to_none=True)
    edge,task,pair=distances(model,data,rows);h=edge-data.cost[rows]
    push,component=weighted_push(data,rows,task,pair,task_weight)
    phr=weighted_phr(h,dual[rows],edge_weight[rows],100.)
    loss=-push+phr;residual=h.detach()
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    loss.backward();optimizer.step()
    # Intentionally NOT edge-weighted; every physical record owns its multiplier.
    with torch.no_grad():dual[rows]=(dual[rows]+residual).relu()
    return dict(loss=float(loss.detach()),connected_expert_push=float(push.detach()),
        connected_expert_components=component,weighted_phr=float(phr.detach()))


def restore(saved,signature,model,optimizer,sampler):
    if saved.get('phr_weight_mode')!='unique-edge' or saved.get('task_weight_mode')!='source':
        raise ValueError('checkpoint experiment mode mismatch')
    for key,label in [('task_weights','weights_sha256'),('edge_weights','edge_weights_sha256')]:
        if tensor_digest(saved[key])!=signature[label]:raise ValueError('checkpoint weights mismatch')
    if tensor_digest(saved['edge_ids'],saved['edge_representatives'],saved['edge_counts'])!=signature['edge_partition_sha256']:
        raise ValueError('checkpoint edge partition mismatch')
    return original_restore(saved,signature,model,optimizer,sampler)


def run(args):
    protocol=json.loads(Path(args.protocol).read_text());torch.set_num_threads(protocol['threads'])
    for path,digest in protocol['source_files'].items():
        assert file_digest(path)==digest,('source changed after freeze',path)
    data=load_training_data(protocol['data'],task_mode='all');weights,stats=task_weights(data)
    assert data.fingerprint==protocol['fingerprints'] and stats==protocol['weights']
    v,edge_ids,reps,counts,edge_stats=edge_weights(data)
    assert protocol['phr_weight_mode']=='unique-edge' and edge_stats==protocol['edge_weights']
    # This experiment's all-task connected dataset has every physical row active.
    for key in ('global_push_task_mask','global_push_pair_mask'):
        assert data.raw.transition_infos[key][data.original].bool().all()
    signature=dict(protocol_sha256=file_digest(args.protocol),task_weight_mode='source',phr_weight_mode='unique-edge',
        edge_weights_sha256=edge_stats['weights_sha256'],edge_partition_sha256=edge_stats['partition_sha256'],
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
        write(out/'config.json',dict(protocol,training_signature=signature,task_weight_mode='source',phr_weight_mode='unique-edge',
        edge_weights_sha256=edge_stats['weights_sha256'],edge_partition_sha256=edge_stats['partition_sha256'],
            initial_model_sha256=tensor_digest(*model.state_dict().values())))
        write(out/'initial_verification.json',dict(paired_initial_exact=True,
            historical_initial_exact=protocol.get('initial_exact_to_history',True),
            seed=protocol['seed'],sampler_seed=protocol.get('sampler_seed',2307),
            fields=list(current),zero_dual=True))
        torch.save(weights,out/'task_weights.pt')
        torch.save(dict(weights=v,ids=edge_ids,representatives=reps,counts=counts),out/'edge_weights.pt')
    def save(step,name):
        state=dict(step=step,initialization_seed=protocol['seed'],model=model.state_dict(),optimizer=optimizer.state_dict(),
            dual=dual,visits=visits,sampler=sampler.state_dict(),rng=torch.get_rng_state(),
            fingerprints=data.fingerprint,task_mode='all',push_counters=counters,
            dataset_kind=data.dataset_kind,training_signature=signature,sampling_chain_sha256=sequence,
            task_weights=weights,task_weight_mode='source',phr_weight_mode='unique-edge',
            edge_weights=v,edge_ids=edge_ids,edge_representatives=reps,edge_counts=counts)
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
            result=advance(model,data,dual,optimizer,rows,weights,v)
            c=result['connected_expert_components']
            counters['connected_expert_empty_task_batches']+=int(c['empty_task_batch'])
            counters['connected_expert_task_visits']+=c['task_rows'];counters['connected_expert_pair_visits']+=c['pair_rows']
            if j%100==0:
                log.write(json.dumps(dict(step=j,lr=lr,**result))+'\n');log.flush()
    write(out/'COMPLETE',dict(step=total,final_sha256=file_digest(out/'final.pt'),
        training_signature=signature,sampling_chain_sha256=sequence,reference_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');run(p.parse_args())
