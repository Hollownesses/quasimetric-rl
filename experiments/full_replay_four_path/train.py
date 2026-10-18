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


def advance(model,data,dual,optimizer,rows):
    optimizer.zero_grad(set_to_none=True);loss=0.;pushes=[];residuals=[]
    for r in (rows,data.paths):
        edge,task,pair=distances(model,data,r);h=edge-data.cost[r]
        push=.6*task.mean()+.4*pair.mean()
        loss=loss+.5*(-push+phr_terms(h,dual[r],100.)[0]);pushes.append(float(push.detach()));residuals.append(h.detach())
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    loss.backward();optimizer.step()
    with torch.no_grad():
        for r,h in zip((rows,data.paths),residuals):
            # Original rows are without replacement within each batch; each edge owns its own dual.
            dual[r]=(dual[r]+h).relu()
    return dict(loss=float(loss.detach()),original_push=pushes[0],path_push=pushes[1])


@torch.no_grad()
def scan(model,data,dual,visits):
    residual=[]
    for r in torch.arange(len(data.src)).split(1024):residual.append(distances(model,data,r,False)-data.cost[r])
    h=torch.cat(residual);result={}
    for name,mask in [('original',torch.arange(len(h))<data.nbase),('paths',torch.arange(len(h))>=data.nbase)]:
        r=h[mask&data.physical];lam=dual[mask&data.physical]
        result[name]=dict(max_violation=float(r.relu().max()),violation_count=int((r>1e-5).sum()),
            mean_positive=float(r.relu().mean()),p99_positive=float(torch.quantile(r.relu(),.99)),
            complementarity_mean=float((r*lam).abs().mean()),dual_max=float(lam.max()))
    result['anchor_max']=float(h[~data.physical].abs().max())
    result['original_visits']=dict(min=int(visits.min()),max=int(visits.max()),mean=float(visits.float().mean()))
    return result,h


def run(args):
    torch.set_num_threads(args.threads);data=load_union();out=Path(args.output)
    if args.resume:
        assert out.is_dir();config=json.loads((out/'config.json').read_text())
        assert config['fingerprints']==data.fingerprint
        for key in ('seed','phr_steps','decay_steps','batch','threads'):assert config[key]==getattr(args,key),key
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/'data_audit.json').write_text(json.dumps(audit(data),indent=2)+'\n')
    torch.manual_seed(args.seed);model=IndexedIQE(data.small,'encoder',0,initialization_seed=args.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.)
    dual=torch.zeros(len(data.src));visits=torch.zeros(len(data.original),dtype=torch.long)
    sampler=EpochSampler(len(data.original),args.seed+1701);offset=0
    config=dict(vars(args),fingerprints=data.fingerprint,torch=str(torch.__version__),rho=100.,dual_step=1.,
        push='0.5 original + 0.5 four-path; each 0.6 task(all) + 0.4 fixed B1',
        constraints='0.5 mean sampled original physical PHR + 0.5 mean all four-path edges PHR; original anchors quotient-identical',
        dual_clock='pre-primal residual, eta=1 per visit; path every update, original each sampled visit',
        stopping='fixed budget; no graph or oracle queries before final checkpoint; no QP',
        initial_model_sha256=tensor_digest(*model.state_dict().values()),script_sha256=file_digest(__file__))
    if not args.resume:(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        assert saved['fingerprints']==data.fingerprint
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);dual=saved['dual'];visits=saved['visits']
        sampler.restore(saved['sampler']);torch.set_rng_state(saved['rng']);offset=saved['step']
    def save(step,name):
        state=dict(step=step,initialization_seed=args.seed,model=model.state_dict(),optimizer=optimizer.state_dict(),
            dual=dual,visits=visits,sampler=sampler.state_dict(),rng=torch.get_rng_state(),fingerprints=data.fingerprint)
        temporary=out/(name+'.tmp');torch.save(state,temporary);temporary.replace(out/name)
    if not args.resume:save(0,'initial.pt')
    start=time.perf_counter();total=args.phr_steps+args.decay_steps
    with (out/'training.jsonl').open('a' if args.resume else 'w') as log:
        for step in range(offset,total+1):
            if step%args.audit_interval==0 or step in (args.phr_steps,total):
                rec,h=scan(model,data,dual,visits);rec.update(step=step,seconds=time.perf_counter()-start,epochs=sampler.epochs)
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
            if j%100==0:
                log.write(json.dumps(dict(step=j,lr=lr,**update))+'\n');log.flush()
    (out/'COMPLETE').write_text('Fixed training budget completed; no claim of feasibility or navigation success.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=606)
    p.add_argument('--phr-steps',type=int,default=20000);p.add_argument('--decay-steps',type=int,default=5000)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--audit-interval',type=int,default=1000)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--resume');run(p.parse_args())
