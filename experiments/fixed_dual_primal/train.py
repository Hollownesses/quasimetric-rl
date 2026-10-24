"""Original QRL primal objective, frozen dual, equal physical-record budgets."""
import argparse
import json
import time
from pathlib import Path
import torch
from experiments.connected_expert_utrap.data import load_training_data
from experiments.four_path_factorial.run import IndexedIQE,file_digest,tensor_digest
from experiments.full_replay_four_path.train import EpochSampler,distances,push_objective
from experiments.connected_phr_solver.train import SOURCE,SOURCE_SHA,DATA,advance_fixed,forward_audit,write
from experiments.connected_phr_solver.objective import full_gradients
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms


def accumulate(model,data,dual,rows,batch=256,observer=None):
    """One full mean gradient; no parameter/optimizer/dual mutation in chunks."""
    params=[p for p in model.parameters() if p.requires_grad]
    buffers=[torch.zeros_like(p,dtype=torch.float64) for p in params]
    total=0.
    for chunk in rows.split(batch):
        if observer is not None:observer(model)
        edge,task,pair=distances(model,data,chunk)
        push,_=push_objective(data,chunk,task,pair)
        value=-push+phr_terms(edge-data.cost[chunk],dual[chunk],100.)[0]
        weight=len(chunk)/len(rows)
        if not torch.isfinite(value):raise RuntimeError('nonfinite objective')
        grads=torch.autograd.grad(value*weight,params,allow_unused=True)
        for buf,g in zip(buffers,grads):
            if g is not None:buf.add_(g.detach().double())
        total+=float(value.detach())*weight
    for p,g in zip(params,buffers):
        if not torch.isfinite(g).all():raise RuntimeError('nonfinite full gradient')
        p.grad=g.to(p.dtype)
    return dict(loss=total,gradient_norm=float(sum(g.square().sum() for g in buffers).sqrt()))


def make_config(args,data,source_sha):
    files=[Path(__file__),Path(__file__).with_name('PROTOCOL.md'),Path(__file__).with_name('probe.py'),
        Path('experiments/connected_phr_solver/train.py'),Path('experiments/connected_phr_solver/objective.py'),
        Path('experiments/full_replay_four_path/train.py'),Path('experiments/four_path_factorial/run.py'),
        Path('experiments/connected_expert_utrap/data.py'),
        Path('experiments/expert_utrap_network/training_adapter.py'),
        Path('quasimetric_rl/modules/quasimetric_critic/losses/phr.py')]
    return dict(arm=args.arm,passes=args.passes,batch=args.batch,lr=args.lr,sampling_seed=2026102401,
        threads=1,rho=100.,source_sha256=source_sha,fingerprints=data.fingerprint,
        physical_records=len(data.original),train_record_budget=args.passes*len(data.original),
        gradients_at=sorted(set([0,1,5,10,args.passes])&set(range(args.passes+1))),
        objective='-.6 task -.4 B1 + mean PHR; original pairwise edge constraint',
        dual='frozen at original 50k parent for entire experiment',
        source_files={str(p):file_digest(p) for p in files})


def run(args):
    torch.set_num_threads(1)
    if args.passes<2 or args.batch<1 or not 0<args.lr<1:raise ValueError('invalid budget or learning rate')
    path=Path(args.source);sha=file_digest(path)
    if path.resolve()==SOURCE.resolve():assert sha==SOURCE_SHA
    data=load_training_data(args.data);source=torch.load(path,map_location='cpu',weights_only=True)
    assert source['fingerprints']==data.fingerprint
    assert source['training_signature']['push'].endswith('0.4 fixed B1')
    for key in ('global_push_task_mask','global_push_pair_mask'):
        assert data.raw.transition_infos[key][data.original].bool().all()
    cfg=make_config(args,data,sha);out=Path(args.output)
    if args.resume:
        assert json.loads((out/'config.json').read_text())==cfg,'resume configuration mismatch'
        state=torch.load(args.resume,weights_only=True);assert state['primal_config']==cfg
        offset=state['completed_passes']
    else:
        out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg);state=source;offset=0
    model=IndexedIQE(data.small,'encoder',0,initialization_seed=source['initialization_seed'])
    model.load_state_dict(state['model']);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.)
    opt.load_state_dict(state['optimizer']);dual=state['dual'].clone();fixed=source['dual'].clone()
    assert torch.equal(dual,fixed),'multiplier drift on resume'
    visits=state['visits'].clone();counters=state['push_counters'].copy();sampler=EpochSampler(len(data.original),cfg['sampling_seed'])
    if args.resume:sampler.restore(state['sampler'])
    torch.set_rng_state(state['rng']);updates=state.get('primal_updates',0)
    seconds=state.get('timing',dict(training=0.,audit=0. )).copy()
    processed=offset*len(data.original);started=time.perf_counter()
    def save(p,name):
        payload={k:source[k] for k in ('initialization_seed','fingerprints','task_mode','task_mask_sha256','dataset_kind','training_signature')}
        payload.update(step=source['step']+updates,completed_passes=p,primal_updates=updates,
            processed_records=processed,model=model.state_dict(),optimizer=opt.state_dict(),dual=dual,
            visits=visits,push_counters=counters,sampler=sampler.state_dict(),rng=torch.get_rng_state(),
            timing=seconds,primal_config=cfg)
        temp=out/(name+'.tmp');torch.save(payload,temp);temp.replace(out/name)
    with (out/'training.jsonl').open('a' if args.resume else 'w') as log:
        def audit(p,order_sha=None,update_info=None):
            before=time.perf_counter();rec,_=forward_audit(model,data,dual)
            if p in cfg['gradients_at']:rec['full_objective']=full_gradients(model,data,dual)[0]
            seconds['audit']+=time.perf_counter()-before
            assert torch.equal(dual,fixed),'multiplier changed'
            assert torch.equal(visits-source['visits'],torch.full_like(visits,p))
            assert processed==p*len(data.original)
            expected=p if args.arm=='full' else p*((len(data.original)+args.batch-1)//args.batch)
            assert updates==expected
            rec.update(completed_passes=p,primal_updates=updates,processed_records=processed,
                order_sha256=order_sha,dual_sha256=tensor_digest(dual),lr=args.lr,
                training_seconds=seconds['training'],audit_seconds=seconds['audit'],
                elapsed_seconds=time.perf_counter()-started,update_info=update_info)
            log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush();print(json.dumps(rec),flush=True)
            if p in (0,1,5,10,15,args.passes):
                write(out/f'supervision_{p:02d}.json',dict(passed=True,pass_number=p,
                    dual_unchanged=True,exact_record_budget=True,reference_loaded=False,scan=rec))
            return rec
        if not args.resume:
            # Save before setting the new common lr: initial Adam state exactly matches parent.
            audit(0);save(0,'initial.pt')
        for p in range(offset+1,args.passes+1):
            before=time.perf_counter()
            for group in opt.param_groups:group['lr']=args.lr
            assert sampler.cursor==0
            ids=torch.cat([sampler.next(args.batch) for _ in range((len(data.original)+args.batch-1)//args.batch)])
            assert len(ids)==len(data.original) and len(ids.unique())==len(ids)
            rows=data.original[ids];order_sha=tensor_digest(rows)
            if args.arm=='full':
                opt.zero_grad(set_to_none=True);model_sha=tensor_digest(*model.state_dict().values())
                info=accumulate(model,data,dual,rows,args.batch)
                assert tensor_digest(*model.state_dict().values())==model_sha,'parameter changed inside accumulation'
                opt.step();updates+=1
            else:
                for chunk in rows.split(args.batch):
                    info=advance_fixed(model,data,dual,opt,chunk);updates+=1
            visits+=1;processed+=len(rows)
            counters['connected_expert_task_visits']+=len(rows);counters['connected_expert_pair_visits']+=len(rows)
            seconds['training']+=time.perf_counter()-before
            audit(p,order_sha,info);save(p,'latest.pt')
            if p in (1,5,10,args.passes-1):save(p,f'pass_{p:02d}.pt')
            if p==args.passes:save(p,'final.pt')
    assert file_digest(path)==sha
    for name,digest in cfg['source_files'].items():assert file_digest(name)==digest,'source drift'
    write(out/'COMPLETE',dict(arm=args.arm,passes=args.passes,processed_records=processed,
        primal_updates=updates,dual_unchanged=True,reference_loaded=False,timing=seconds,
        training_forward_records=processed,training_backward_records=processed,
        audit_forward_records=(args.passes+1+len(cfg['gradients_at']))*len(data.original),
        audit_backward_record_equivalents=4*len(cfg['gradients_at'])*len(data.original),
        source_sha256=sha,elapsed_this_invocation=time.perf_counter()-started))


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['mini','full'])
    p.add_argument('--output',required=True);p.add_argument('--source',default=str(SOURCE))
    p.add_argument('--data',default=str(DATA));p.add_argument('--passes',type=int,default=20)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--lr',type=float,default=2e-6)
    p.add_argument('--resume');return p.parse_args(argv)


if __name__=='__main__':run(parse_args())
