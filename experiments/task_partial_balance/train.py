"""Partial task-source balance; original trainer and artifacts remain unchanged.

Only task weights change. Endpoint alpha=0/1 calls the existing R0/R1 update.
No reference/Oracle data are read here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.connected_expert_utrap.data import load_training_data
from experiments.full_replay_four_path.train import scan
from experiments.four_path_factorial.run import tensor_digest, file_digest
from experiments.task_source_balance.train import (
    task_weights, initial_state, initial_fields, load_initial, learning_rate,
    equal, write, advance as original_advance, restore as original_restore)


def mixed_weights(data,alpha):
    if alpha not in (0.,.5,1.): raise ValueError('only alpha=0, 0.5, 1 are supported; no search')
    source,base=task_weights(data)
    active=data.physical & data.raw.transition_infos['global_push_task_mask'].bool()
    weights=torch.zeros_like(source)
    weights[active]=(1-alpha)+alpha*source[active]
    assert abs(float(weights[active].double().mean())-1)<1e-7
    stats=dict(alpha=alpha,N=base['N'],M=base['M'],
        mean=float(weights[active].double().mean()),min=float(weights[active].min()),
        max=float(weights[active].max()),weights_sha256=tensor_digest(weights),
        source_weights_sha256=base['weights_sha256'],
        formula='(1-alpha)+alpha*N/(M*n(s)); global; no minibatch normalization')
    return weights,stats,base


def advance(model,data,dual,optimizer,rows,alpha,weights):
    if alpha not in (0.,.5,1.): raise ValueError('unsupported alpha')
    # Reuse all B1, PHR, primal and pre-primal-residual dual code unchanged.
    return original_advance(model,data,dual,optimizer,rows,'record' if alpha==0 else 'source',weights)


def restore(saved,signature,model,optimizer,sampler):
    if saved.get('alpha')!=signature['alpha'] or saved.get('task_weight_mode')!='partial-source':
        raise ValueError('checkpoint alpha/mode mismatch')
    if tensor_digest(saved['task_weights'])!=signature['weights_sha256']:
        raise ValueError('checkpoint weights mismatch')
    return original_restore(saved,signature,model,optimizer,sampler)


def run(args):
    protocol=json.loads(Path(args.protocol).read_text());torch.set_num_threads(protocol['threads'])
    for path,digest in protocol['source_files'].items():
        assert file_digest(path)==digest,('source changed after freeze',path)
    assert args.alpha==protocol['alpha'], 'protocol alpha mismatch'
    data=load_training_data(protocol['data'],task_mode='all')
    weights,stats,base=mixed_weights(data,args.alpha)
    assert data.fingerprint==protocol['fingerprints'] and stats==protocol['mixed_weights'] and base==protocol['weights']
    # This experiment's all-task connected dataset has every physical row active.
    for key in ('global_push_task_mask','global_push_pair_mask'):
        assert data.raw.transition_infos[key][data.original].bool().all()
    signature=dict(protocol_sha256=file_digest(args.protocol),task_weight_mode='partial-source',alpha=args.alpha,
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
        write(out/'config.json',dict(protocol,training_signature=signature,task_weight_mode='partial-source',alpha=args.alpha,
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
            task_weights=weights,task_weight_mode='partial-source',alpha=args.alpha)
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
            result=advance(model,data,dual,optimizer,rows,args.alpha,weights)
            c=result['connected_expert_components']
            counters['connected_expert_empty_task_batches']+=int(c['empty_task_batch'])
            counters['connected_expert_task_visits']+=c['task_rows'];counters['connected_expert_pair_visits']+=c['pair_rows']
            if j%100==0:
                log.write(json.dumps(dict(step=j,lr=lr,**result))+'\n');log.flush()
    write(out/'COMPLETE',dict(step=total,final_sha256=file_digest(out/'final.pt'),
        training_signature=signature,sampling_chain_sha256=sequence,reference_loaded=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--alpha',type=float,choices=[0.,.5,1.],default=.5);p.add_argument('--resume');run(p.parse_args())
