"""Matched visit-step budgets, original QRL constraints and task+B1 objective.

No reference or shortest-path evaluation imports. Outer is a damped PHR clock,
not a claim to solve each augmented subproblem exactly.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from experiments.connected_expert_utrap.data import load_training_data
from experiments.four_path_factorial.run import IndexedIQE,file_digest,tensor_digest
from experiments.full_replay_four_path.train import EpochSampler,advance,distances,push_objective
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms
from .objective import full_gradients

SOURCE=Path('results/connected_expert_qrl_20261023/main/train/final.pt')
SOURCE_SHA='e40171f8c4d23f4d894dc64f37f29ca5d4f9e42a9f5846e4664d197afbe3e37d'
DATA=Path('results/connected_expert_utrap_20261021/main')


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def advance_fixed(model,data,dual,optimizer,rows):
    optimizer.zero_grad(set_to_none=True)
    edge,task,pair=distances(model,data,rows);h=edge-data.cost[rows]
    push,component=push_objective(data,rows,task,pair)
    loss=-push+phr_terms(h,dual[rows],100.)[0]
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    loss.backward();optimizer.step()
    return dict(loss=float(loss.detach()),connected_expert_push=float(push.detach()),
                connected_expert_components=component)


@torch.no_grad()
def outer_update(dual,physical,residual,counts):
    assert counts.shape==physical.shape and (counts>=0).all()
    assert residual.shape==dual.shape and torch.isfinite(residual).all()
    before=dual[physical].clone()
    dual[physical]=(before+counts*residual[physical]).relu()
    assert torch.isfinite(dual).all()
    return dict(max_change=float((dual[physical]-before).abs().max()),
                eta_min=int(counts.min()),eta_max=int(counts.max()),eta_sum=int(counts.sum()))


@torch.no_grad()
def forward_audit(model,data,dual):
    h=torch.zeros_like(dual);task_total=pair_total=0.
    for rows in data.original.split(512):
        edge,task,pair=distances(model,data,rows);h[rows]=edge-data.cost[rows]
        task_total+=float(task.double().sum());pair_total+=float(pair.double().sum())
    # Anchors never enter the objective but must remain quotient-identical.
    anchors=torch.where(~data.physical)[0]
    for rows in anchors.split(512):h[rows]=distances(model,data,rows,False)-data.cost[rows]
    assert torch.isfinite(h).all() and float(h[anchors].abs().max())==0.
    push=(.6*task_total+.4*pair_total)/len(data.original)
    return dict(task_mean=task_total/len(data.original),pair_mean=pair_total/len(data.original),
                push=push,anchor_max=0.,**dual_metrics(h,data.original,dual,push)),h


def dual_metrics(h,rows,dual,push):
    r=h[rows];lam=dual[rows]
    return dict(augmented_loss=float(phr_terms(r,lam,100.)[0])-push,
        max_violation=float(r.relu().max()),violation_count=int((r>1e-5).sum()),
        mean_positive=float(r.relu().mean()),complementarity_mean=float((r*lam).abs().mean()),
        dual_max=float(lam.max()),dual_mean=float(lam.mean()))


def run(args):
    torch.set_num_threads(1)
    if args.steps<1 or args.period<1 or args.batch<1 or args.steps%args.period:
        raise ValueError('budget must end on an outer boundary')
    source_path=Path(args.source);source_sha=file_digest(source_path)
    if source_path.resolve()==SOURCE.resolve():assert source_sha==SOURCE_SHA
    data=load_training_data(args.data);source=torch.load(source_path,map_location='cpu',weights_only=True)
    assert source['fingerprints']==data.fingerprint
    assert source['training_signature']['push'].endswith('0.4 fixed B1')
    for key in ('global_push_task_mask','global_push_pair_mask'):
        assert data.raw.transition_infos[key][data.original].bool().all()
    sources=[Path(__file__),Path(__file__).with_name('objective.py'),Path(__file__).with_name('SOLVER_PROTOCOL.md'),
             Path('experiments/full_replay_four_path/train.py'),Path('quasimetric_rl/modules/quasimetric_critic/losses/phr.py')]
    config=dict(arm=args.arm,steps=args.steps,period=args.period,batch=args.batch,threads=1,rho=100.,
        source_sha256=source_sha,fingerprints=data.fingerprint,
        source_files={str(p):file_digest(p) for p in sources},
        learning_rate='2e-6 * .1**((step-1)/max(steps-1,1))',
        objective='0.6 task + 0.4 B1; original pairwise edge constraints',
        dual_budget='eta1 per visit; outer eta equals actual per-row visits per block')
    out=Path(args.output)
    if args.resume:
        assert json.loads((out/'config.json').read_text())==config,'resume configuration mismatch'
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        assert state['solver_config']==config,'checkpoint configuration mismatch'
        assert state['checkpoint_phase']=='after_outer','resume requires a completed outer boundary'
        offset=state['continuation_step'];assert offset%args.period==0
    else:
        out.mkdir(parents=True,exist_ok=False);write(out/'config.json',config)
        state=source;offset=0
    model=IndexedIQE(data.small,'encoder',0,initialization_seed=source['initialization_seed'])
    model.load_state_dict(state['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6,weight_decay=0.)
    optimizer.load_state_dict(state['optimizer']);dual=state['dual'].clone();visits=state['visits'].clone()
    sampler=EpochSampler(len(data.original),0);sampler.restore(state['sampler']);torch.set_rng_state(state['rng'])
    counters=state['push_counters'].copy();pending=torch.zeros_like(visits)
    nominal=state.get('nominal_dual_steps',torch.zeros_like(visits)).clone()
    boundary_loss=state.get('boundary_loss');started=time.perf_counter()
    def save(step,name,phase='after_outer'):
        payload={k:source[k] for k in ('initialization_seed','fingerprints','task_mode','task_mask_sha256','dataset_kind','training_signature')}
        payload.update(step=source['step']+step,continuation_step=step,model=model.state_dict(),
            optimizer=optimizer.state_dict(),dual=dual,visits=visits,sampler=sampler.state_dict(),
            rng=torch.get_rng_state(),push_counters=counters,nominal_dual_steps=nominal,
            solver_config=config,boundary_loss=boundary_loss,checkpoint_phase=phase)
        temp=out/(name+'.tmp');torch.save(payload,temp);temp.replace(out/name)
    with (out/'training.jsonl').open('a' if args.resume else 'w') as log:
        def emit(rec):
            rec['seconds']=time.perf_counter()-started
            log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush()
            if rec.get('kind')=='boundary':print(json.dumps(rec,allow_nan=False),flush=True)
        if not args.resume:
            rec,_=forward_audit(model,data,dual);boundary_loss=rec['augmented_loss'];save(0,'initial.pt')
            emit(dict(kind='boundary',continuation_step=0,**rec))
            write(out/'supervision_0000.json',dict(passed=True,parent_sha256=source_sha,scan=rec))
        for step in range(offset+1,args.steps+1):
            lr=2e-6*.1**((step-1)/max(args.steps-1,1))
            for group in optimizer.param_groups:group['lr']=lr
            ids=sampler.next(args.batch);rows=data.original[ids];visits[ids]+=1;pending[ids]+=1
            result=(advance if args.arm=='A' else advance_fixed)(model,data,dual,optimizer,rows)
            if args.arm=='A':nominal[ids]+=1
            component=result['connected_expert_components']
            counters['connected_expert_empty_task_batches']+=int(component['empty_task_batch'])
            counters['connected_expert_task_visits']+=component['task_rows']
            counters['connected_expert_pair_visits']+=component['pair_rows']
            if step%100==0:emit(dict(kind='minibatch',continuation_step=step,lr=lr,**result))
            if step%args.period:continue
            rec,h=forward_audit(model,data,dual)
            rec['inner_objective_change']=rec['augmented_loss']-boundary_loss
            if step==args.steps:
                metrics,_,_=full_gradients(model,data,dual)
                rec['full_objective_before_outer']=metrics
                if args.arm=='B':save(step,'final_before_outer.pt',phase='before_outer')
            if args.arm=='B':
                rec['outer_update']=outer_update(dual,data.original,h,pending)
                nominal+=pending
            pending.zero_();assert torch.equal(nominal,visits-source['visits'])
            rec['after_outer']=dual_metrics(h,data.original,dual,rec['push'])
            boundary_loss=rec['after_outer']['augmented_loss']
            if step==args.steps-args.period:
                metrics,_,_=full_gradients(model,data,dual);rec['full_objective_after_outer']=metrics
            rec.update(kind='boundary',continuation_step=step,lr=lr,extra_visits_min=int(nominal.min()),
                extra_visits_max=int(nominal.max()),nominal_dual_steps_sha256=tensor_digest(nominal))
            emit(rec);save(step,'latest.pt')
            if step==args.steps-args.period:save(step,'last_block_start.pt')
            if step==args.steps:save(step,'final.pt')
            if step%5000==0 or step==args.steps:
                assert int(visits.max()-visits.min())<=1
                write(out/f'supervision_{step:04d}.json',dict(passed=True,scan=rec,
                    exact_nominal_budget=True,reference_loaded=False))
    assert file_digest(source_path)==source_sha
    for path,digest in config['source_files'].items():assert file_digest(path)==digest,'source modified during run'
    write(out/'COMPLETE',dict(arm=args.arm,steps=args.steps,total_seconds=time.perf_counter()-started,
        source_sha256=source_sha,reference_loaded=False))


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['A','B'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--data',default=str(DATA));p.add_argument('--source',default=str(SOURCE))
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--period',type=int,default=500)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--resume')
    return p.parse_args(argv)


if __name__=='__main__':run(parse_args())
