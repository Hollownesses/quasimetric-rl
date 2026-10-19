"""Stratified stochastic PHR with stable per-row duals and fixed stopping budget."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.four_path_factorial.run import IndexedIQE,tensor_digest,file_digest
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms
from experiments.full_replay_four_path.data import load_union,audit


def distances(model,data,rows,push=True):
    left=data.src[rows];right=data.dst[rows]
    if push:
        physical=data.physical[rows];pr=rows[physical]
        left=torch.cat((left,data.src[pr],data.src[pr]));right=torch.cat((right,data.goal[pr],data.pair[pr]))
    unique,inverse=torch.unique(torch.cat((left,right)),return_inverse=True)
    z=model.projector(model.encoder(data.nodes[unique]));a,b=inverse.chunk(2)
    d=model.head(z[a],z[b])
    return d.split((len(rows),int(physical.sum()),int(physical.sum()))) if push else d


class EpochSampler:
    def __init__(self,size,seed):
        self.generator=torch.Generator().manual_seed(seed);self.size=size
        self.order=torch.randperm(size,generator=self.generator);self.cursor=0;self.epochs=0
    def next(self,batch):
        count=min(batch,self.size-self.cursor)
        result=self.order[self.cursor:self.cursor+count];self.cursor+=count
        if self.cursor==self.size:
            self.order=torch.randperm(self.size,generator=self.generator);self.cursor=0;self.epochs+=1
        return result
    def state_dict(self):
        return dict(order=self.order,cursor=self.cursor,epochs=self.epochs,rng=self.generator.get_state())
    def restore(self,state):
        self.order=state['order'];self.cursor=state['cursor'];self.epochs=state['epochs'];self.generator.set_state(state['rng'])


def masked_mean(values, mask):
    """Mean over retained entries; empty selection is a differentiable exact zero."""
    selected = values[mask]
    return selected.mean() if selected.numel() else selected.sum()


def push_objective(data, rows, task, pair):
    # distances() returns pushes only for physical rows, so align masks likewise.
    physical_rows = rows[data.physical[rows]]
    infos = data.raw.transition_infos
    task_mask = infos['global_push_task_mask'][physical_rows].bool()
    pair_mask = infos['global_push_pair_mask'][physical_rows].bool()
    task_value = masked_mean(task, task_mask)
    pair_value = masked_mean(pair, pair_mask)
    # Do not redistribute the absent task weight to B1, or fall back to all-task.
    return .6 * task_value + .4 * pair_value, dict(
        task_rows=int(task_mask.sum()), pair_rows=int(pair_mask.sum()),
        empty_task_batch=not bool(task_mask.any()),
        task_mean=float(task_value.detach()), pair_mean=float(pair_value.detach()))


def single_source_name(data):
    return {'expert_network':'expert','curriculum_network':'curriculum'}.get(getattr(data,'dataset_kind',None))


def training_strata(data, rows):
    name=single_source_name(data)
    if name:
        return [(name,1.,rows)]
    return [('original',.5,rows),('path',.5,data.paths)]


def advance(model,data,dual,optimizer,rows):
    optimizer.zero_grad(set_to_none=True);loss=0.;residuals=[];result={}
    strata=training_strata(data,rows)
    for name,weight,r in strata:
        edge,task,pair=distances(model,data,r);h=edge-data.cost[r]
        push,component=push_objective(data,r,task,pair)
        loss=loss+weight*(-push+phr_terms(h,dual[r],100.)[0]);residuals.append(h.detach())
        result[name+'_push']=float(push.detach());result[name+'_components']=component
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    loss.backward();optimizer.step()
    with torch.no_grad():
        for (_,_,r),h in zip(strata,residuals):
            # Original rows are without replacement within each batch; each edge owns its own dual.
            dual[r]=(dual[r]+h).relu()
    return dict(loss=float(loss.detach()),**result)


@torch.no_grad()
def scan(model,data,dual,visits,diagnostic_rows=None):
    residual=[]
    for r in torch.arange(len(data.src)).split(1024):residual.append(distances(model,data,r,False)-data.cost[r])
    h=torch.cat(residual);result={}
    single=single_source_name(data)
    groups=[(single,torch.ones(len(h),dtype=torch.bool))] if single else [
        ('original',torch.arange(len(h))<data.nbase),('paths',torch.arange(len(h))>=data.nbase)]
    for name,mask in groups:
        r=h[mask&data.physical];lam=dual[mask&data.physical]
        result[name]=dict(max_violation=float(r.relu().max()),violation_count=int((r>1e-5).sum()),
            mean_positive=float(r.relu().mean()),p99_positive=float(torch.quantile(r.relu(),.99)),
            complementarity_mean=float((r*lam).abs().mean()),dual_max=float(lam.max()))
    result['anchor_max']=float(h[~data.physical].abs().max()) if (~data.physical).any() else 0.
    result[single+'_visits' if single else 'original_visits']=dict(min=int(visits.min()),max=int(visits.max()),mean=float(visits.float().mean()))
    if diagnostic_rows is not None:
        result['model_only_probes']={}
        for name,rows in diagnostic_rows.items():
            _,task,pair=distances(model,data,rows)
            result['model_only_probes'][name]=dict(task_mean=float(task.mean()),
                task_max=float(task.max()),pair_mean=float(pair.mean()))
    return result,h


def run(args):
    torch.set_num_threads(args.threads)
    expert_root=getattr(args,'expert_data',None)
    curriculum_root=getattr(args,'curriculum_data',None)
    if expert_root and curriculum_root:raise ValueError('choose only one standalone dataset')
    if curriculum_root:
        from experiments.curriculum_utrap.data import load_training_data,audit_training_data
        data=load_training_data(curriculum_root,task_mode=args.task_mode);audit_data=audit_training_data
    elif expert_root:
        from experiments.expert_utrap_network.training_adapter import load_training_data,audit_training_data
        data=load_training_data(expert_root,task_mode=args.task_mode);audit_data=audit_training_data
    else:
        data=load_union(task_mode=args.task_mode);audit_data=audit
    dataset_kind='curriculum_network' if curriculum_root else ('expert_network' if expert_root else 'full_replay_four_path')
    single='curriculum' if curriculum_root else ('expert' if expert_root else None)
    out=Path(args.output)
    task_mask_sha256=tensor_digest(data.raw.transition_infos['global_push_task_mask'])
    if args.resume:
        assert out.is_dir();config=json.loads((out/'config.json').read_text())
        assert config['fingerprints']==data.fingerprint
        for key in ('seed','phr_steps','decay_steps','batch','threads'):assert config[key]==getattr(args,key),key
        assert config.get('task_mode','all')==args.task_mode, 'task mode mismatch'
        assert config.get('task_mask_sha256',task_mask_sha256)==task_mask_sha256
        assert config.get('dataset_kind','full_replay_four_path')==dataset_kind,'dataset kind mismatch'
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/'data_audit.json').write_text(json.dumps(audit_data(data),indent=2)+'\n')
    torch.manual_seed(args.seed);model=IndexedIQE(data.small,'encoder',0,initialization_seed=args.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.)
    dual=torch.zeros(len(data.src));visits=torch.zeros(len(data.original),dtype=torch.long)
    sampler=EpochSampler(len(data.original),args.seed+1701);offset=0
    config=dict(vars(args),fingerprints=data.fingerprint,torch=str(torch.__version__),rho=100.,dual_step=1.,
        push=f'0.5 original + 0.5 four-path; each 0.6 task({args.task_mode}) + 0.4 fixed B1',
        task_mask_sha256=task_mask_sha256,
        task_normalization='conditional mean within each source stratum minibatch; empty => zero; no coefficient redistribution',
        constraints='0.5 mean sampled original physical PHR + 0.5 mean all four-path edges PHR; original anchors quotient-identical',
        dual_clock='pre-primal residual, eta=1 per visit; path every update, original each sampled visit',
        stopping='fixed budget; no graph or oracle queries before final checkpoint; no QP',
        initial_model_sha256=tensor_digest(*model.state_dict().values()),script_sha256=file_digest(__file__))
    config['dataset_kind']=dataset_kind
    if single:
        adapter='curriculum_utrap/data.py' if curriculum_root else 'expert_utrap_network/training_adapter.py'
        config.update(push=f'1.0 {single}; 0.6 task({args.task_mode}) + 0.4 fixed B1',
            constraints=f'1.0 mean sampled {single} physical PHR; legal quotient-identical anchors scanned only',
            dual_clock=f'pre-primal residual, eta=1 per sampled {single} record visit',
            sampling=f'{single} physical records without replacement per epoch; '+
                ('all successes and failures retained' if curriculum_root else 'shared-segment duplicates retained'),
            b1_seed=42,adapter_sha256=file_digest(Path(__file__).parents[1]/adapter))
    resume_signature={key:config[key] for key in ('dataset_kind','fingerprints','seed','phr_steps',
        'decay_steps','batch','threads','task_mode','task_mask_sha256','rho','dual_step','push','constraints')}
    if not args.resume:(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        assert saved['fingerprints']==data.fingerprint
        assert saved.get('task_mode','all')==args.task_mode, 'checkpoint task mode mismatch'
        assert saved.get('task_mask_sha256',task_mask_sha256)==task_mask_sha256
        assert saved.get('dataset_kind','full_replay_four_path')==dataset_kind,'checkpoint dataset kind mismatch'
        if single:assert saved['training_signature']==resume_signature,'checkpoint training configuration mismatch'
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);dual=saved['dual'];visits=saved['visits']
        sampler.restore(saved['sampler']);torch.set_rng_state(saved['rng']);offset=saved['step']
    push_counters=dict(original_empty_task_batches=0,original_task_visits=0,original_pair_visits=0,
                       path_task_visits=0,path_pair_visits=0)
    if single:
        push_counters={single+suffix:0 for suffix in ('_empty_task_batches','_task_visits','_pair_visits')}
    if args.resume:push_counters.update(saved.get('push_counters',{}))
    diagnostics=None
    if args.task_mode=='witnessed':
        generator=torch.Generator().manual_seed(2026101901)
        diagnostics={} if single else {'paths':data.paths}
        device=data.raw.transition_infos['device_index']
        for i in range(4):
            rows=data.original[device[data.original]==i]
            if not len(rows):continue
            diagnostics[f'device_{i}']=rows[torch.randperm(len(rows),generator=generator)[:512]]
        (out/'diagnostic_probes.json').write_text(json.dumps({k:v.tolist() for k,v in diagnostics.items()})+'\n')
    def save(step,name):
        state=dict(step=step,initialization_seed=args.seed,model=model.state_dict(),optimizer=optimizer.state_dict(),
            dual=dual,visits=visits,sampler=sampler.state_dict(),rng=torch.get_rng_state(),fingerprints=data.fingerprint,
            task_mode=args.task_mode,task_mask_sha256=task_mask_sha256,push_counters=push_counters,
            dataset_kind=dataset_kind,training_signature=resume_signature)
        temporary=out/(name+'.tmp');torch.save(state,temporary);temporary.replace(out/name)
    if not args.resume:save(0,'initial.pt')
    start=time.perf_counter();total=args.phr_steps+args.decay_steps
    with (out/'training.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(offset,total+1):
            if step%args.audit_interval==0 or step in (args.phr_steps,total):
                rec,h=scan(model,data,dual,visits,diagnostics);rec.update(step=step,seconds=time.perf_counter()-start,epochs=sampler.epochs,push_counters=push_counters.copy())
                log.write(json.dumps(rec)+'\n');log.flush();print(json.dumps(rec),flush=True)
                save(step,'latest.pt')
                if step==args.phr_steps:save(step,'phr.pt')
                if step==total:
                    save(step,'final.pt');np.savez_compressed(out/'final_constraints.npz',residual=h.numpy(),dual=dual.numpy());break
            j=step+1
            lr=2e-5 if j<=args.phr_steps else 2e-6*.1**((j-args.phr_steps-1)/max(1,args.decay_steps-1))
            for group in optimizer.param_groups:group['lr']=lr
            indices=sampler.next(args.batch);rows=data.original[indices];visits[indices]+=1
            update=advance(model,data,dual,optimizer,rows)
            for name,_,_ in training_strata(data,rows):
                component=update[name+'_components']
                if name+'_empty_task_batches' in push_counters:
                    push_counters[name+'_empty_task_batches']+=int(component['empty_task_batch'])
                push_counters[name+'_task_visits']+=component['task_rows']
                push_counters[name+'_pair_visits']+=component['pair_rows']
            if j%100==0:
                log.write(json.dumps(dict(step=j,lr=lr,**update))+'\n');log.flush()
    (out/'COMPLETE').write_text('Fixed training budget completed; no claim of feasibility or navigation success.\n')


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=606)
    p.add_argument('--phr-steps',type=int,default=20000);p.add_argument('--decay-steps',type=int,default=5000)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--audit-interval',type=int,default=1000)
    p.add_argument('--task-mode',choices=['all','witnessed'],default='all')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--resume')
    source=p.add_mutually_exclusive_group()
    source.add_argument('--expert-data',help='Use only this expert network directory instead of the legacy replay union')
    source.add_argument('--curriculum-data',help='Use this curriculum replay, retaining successful and failed transitions')
    return p.parse_args(argv)


if __name__=='__main__':
    run(parse_args())
