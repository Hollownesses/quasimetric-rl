"""Paired 5k continuation: visited duals versus full latest-residual outer PHR."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.four_path_execution.common import read,write,file_digest
from experiments.four_path_factorial.run import IndexedIQE,tensor_digest
from experiments.full_replay_four_path.data import load_union
from experiments.full_replay_four_path.train import EpochSampler,distances,push_objective,scan,advance
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms

SOURCE=Path('results/witnessed_task_push_20261019/main/train/final.pt')
EXPECTED_SOURCE='5d2094784b6ea07ee115970e20a5c950db7fb907ffa4a55b6ced7731809cc7c1'


def advance_outer(model,data,dual,optimizer,rows):
    """Same primal as advance(); only original per-visit dual updates are absent."""
    optimizer.zero_grad(set_to_none=True);loss=0.;pushes=[];components=[]
    for r in (rows,data.paths):
        edge,task,pair=distances(model,data,r);h=edge-data.cost[r]
        push,component=push_objective(data,r,task,pair);components.append(component)
        loss=loss+.5*(-push+phr_terms(h,dual[r],100.)[0]);pushes.append(float(push.detach()))
        if r is data.paths:path_residual=h.detach()
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    loss.backward();optimizer.step()
    with torch.no_grad():dual[data.paths]=(dual[data.paths]+path_residual).relu()
    return dict(loss=float(loss.detach()),original_push=pushes[0],path_push=pushes[1],
                original_components=components[0],path_components=components[1])


@torch.no_grad()
def update_full_original(dual,residual,original,eta):
    """Stable replay-row update; never touch four-path or abstract anchor duals."""
    if eta<=0 or not np.isfinite(eta) or residual.shape!=dual.shape or not torch.isfinite(residual).all():
        raise ValueError('invalid full residual or outer step')
    before=dual[original].clone();after=(before+eta*residual[original]).relu()
    if not torch.isfinite(after).all():raise RuntimeError('nonfinite outer multiplier')
    dual[original]=after
    return dict(eta=eta,rows=len(original),max_change=float((after-before).abs().max()),
                mean_change=float((after-before).mean()),increased=int((after>before).sum()),
                decreased=int((after<before).sum()),dual_max_before=float(before.max()),
                dual_max_after=float(after.max()))


def learning_rate(step):
    # Both arms restart the previously used low-rate decay, keeping Adam moments.
    return 2e-6*.1**((step-1)/4999)


def signature(arm,steps,period,eta,source_sha,mask_sha,fingerprints):
    return dict(arm=arm,steps=steps,outer_period=period,outer_eta=eta,
                source_sha256=source_sha,task_mask_sha256=mask_sha,fingerprints=fingerprints,
                batch=256,threads=4,rho=100.,learning_rate='2e-6 * 0.1**((j-1)/4999)',
                task_mode='witnessed',path_dual_step=1.,original_visit_dual_step=1.)


def run(args):
    torch.set_num_threads(4)
    source_sha=file_digest(SOURCE);assert source_sha==EXPECTED_SOURCE
    d=load_union(task_mode='witnessed');mask_sha=tensor_digest(d.raw.transition_infos['global_push_task_mask'])
    source=torch.load(SOURCE,map_location='cpu',weights_only=True)
    assert source['step']==25000 and source['task_mode']=='witnessed'
    assert source['fingerprints']==d.fingerprint and source['task_mask_sha256']==mask_sha
    if args.steps<=0 or args.steps>5000 or args.outer_period<=0 or args.outer_eta<=0:raise ValueError('invalid budget')
    config=signature(args.arm,args.steps,args.outer_period,args.outer_eta,source_sha,mask_sha,d.fingerprint)
    out=Path(args.output)
    if args.resume:
        assert read(out/'config.json')==config,'resume configuration mismatch'
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        assert state['calibration_config']==config,'checkpoint arm/config mismatch'
        offset=state['continuation_step']
    else:
        out.mkdir(parents=True,exist_ok=False);write(out/'config.json',config)
        state=source;offset=0
    model=IndexedIQE(d.small,'encoder',0,initialization_seed=source['initialization_seed'])
    model.load_state_dict(state['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6,weight_decay=0.)
    optimizer.load_state_dict(state['optimizer'])
    dual=state['dual'].clone();visits=state['visits'].clone()
    sampler=EpochSampler(len(d.original),0);sampler.restore(state['sampler'])
    torch.set_rng_state(state['rng'])
    counters=state['push_counters'].copy();outer_updates=state.get('outer_updates',0)
    initial_visits=source['visits'];initial_counters=source['push_counters']
    probes={k:torch.tensor(v) for k,v in read(SOURCE.parent/'diagnostic_probes.json').items()}
    def save(step,name):
        payload=dict(step=25000+step,continuation_step=step,initialization_seed=source['initialization_seed'],
            model=model.state_dict(),optimizer=optimizer.state_dict(),dual=dual,visits=visits,
            sampler=sampler.state_dict(),rng=torch.get_rng_state(),fingerprints=d.fingerprint,
            task_mode='witnessed',task_mask_sha256=mask_sha,push_counters=counters,
            outer_updates=outer_updates,calibration_config=config)
        temp=out/(name+'.tmp');torch.save(payload,temp);temp.replace(out/name)
    if not args.resume:save(0,'initial.pt')
    started=time.perf_counter()
    with (out/'training.jsonl').open('a' if args.resume else 'w') as log:
        def record_audit(step,update):
            nonlocal outer_updates
            rec,h=scan(model,d,dual,visits,probes)
            if not torch.isfinite(h).all():raise RuntimeError('nonfinite full scan')
            if update:
                rec['outer_update']=update_full_original(dual,h,d.original,args.outer_eta);outer_updates+=1
                rec['original']['dual_max']=float(dual[d.original].max())
                rec['original']['complementarity_mean']=float((h[d.original]*dual[d.original]).abs().mean())
            delta_visits=visits-initial_visits
            rec.update(step=25000+step,continuation_step=step,seconds=time.perf_counter()-started,
                outer_updates=outer_updates,epochs=sampler.epochs,
                extra_visits=dict(min=int(delta_visits.min()),max=int(delta_visits.max()),mean=float(delta_visits.float().mean())),
                extra_push_counters={k:counters[k]-initial_counters[k] for k in counters})
            log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True)
            save(step,'latest.pt')
            if step in (250,2500):save(step,f'at_{step:04d}.pt')
            if step==args.steps:
                save(step,'final.pt');np.savez_compressed(out/'final_constraints.npz',residual=h.numpy(),dual=dual.numpy())
            return rec
        # A restored outer-boundary checkpoint already contains that update.
        if not args.resume:record_audit(0,False)
        for step in range(offset+1,args.steps+1):
            for group in optimizer.param_groups:group['lr']=learning_rate(step)
            indices=sampler.next(256);rows=d.original[indices];visits[indices]+=1
            result=(advance if args.arm=='A' else advance_outer)(model,d,dual,optimizer,rows)
            counters['original_empty_task_batches']+=int(result['original_components']['empty_task_batch'])
            counters['original_task_visits']+=result['original_components']['task_rows']
            counters['original_pair_visits']+=result['original_components']['pair_rows']
            counters['path_task_visits']+=result['path_components']['task_rows']
            counters['path_pair_visits']+=result['path_components']['pair_rows']
            if step%100==0:
                log.write(json.dumps(dict(continuation_step=step,step=25000+step,lr=learning_rate(step),**result))+'\n');log.flush()
            if step%args.outer_period==0 or step==args.steps:
                record_audit(step,args.arm=='B' and step%args.outer_period==0)
    if offset==args.steps:
        assert (out/'final.pt').exists(), 'final checkpoint missing'
    write(out/'COMPLETE',dict(continuation_steps=args.steps,source_sha256=source_sha,
          arm=args.arm,outer_updates=outer_updates,feasibility_or_navigation_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--arm',choices=['A','B'],required=True)
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--outer-period',type=int,default=250)
    p.add_argument('--outer-eta',type=float,default=100.);p.add_argument('--resume');run(p.parse_args())
