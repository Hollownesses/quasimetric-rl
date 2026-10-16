"""Fixed three-stage pipeline; graph references enter only after final selection."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from experiments.four_path_factorial.run import (
    ROOT, IndexedIQE, TabularPHR, load_data, tensor_digest, file_digest,
)
from experiments.four_path_refinement.run import advance, diagnose, restore, optimizer_digest
from experiments.four_path_refinement.repair import actual, run as repair_run

SEEDS = (101, 202, 303, 404, 505)


def fresh(data, seed):
    torch.manual_seed(seed)
    model = IndexedIQE(data,'encoder',0,initialization_seed=seed)
    solver = TabularPHR(len(data.src),rho=100.,dual_step=1.)
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.)
    assert len(optimizer.state) == 0 and not bool(solver.multiplier.any())
    return model,solver,optimizer


def snapshot(model,solver,optimizer,seed,step,stage,path):
    torch.save(dict(step=step,stage=stage,initialization_seed=seed,model=model.state_dict(),
        solver=solver.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),path)


def train_stage(model,data,solver,optimizer,seed,stage,steps,offset,out,trace,updates):
    start=time.perf_counter();train_seconds=0.
    for step in range(steps+1):
        if step%250==0 or step==steps:
            h,push=actual(model,data)
            rec=dict(stage=stage,stage_step=step,total_primal_step=offset+step,
                seconds=time.perf_counter()-start,train_seconds=train_seconds,push=push,
                max_violation=float(h[data.physical].relu().max()),
                violation_count=int((h[data.physical]>1e-5).sum()),
                complementarity=float((h*solver.multiplier).abs().mean()))
            trace.write(json.dumps(rec)+'\n');trace.flush();updates.flush()
            print(json.dumps(dict(seed=seed,**rec)),flush=True)
        if step==steps: break
        tick=time.perf_counter()
        update=advance(model,data,solver,optimizer,'A' if stage=='phr' else 'B',step+1,5000)
        train_seconds+=time.perf_counter()-tick
        updates.write(json.dumps(dict(stage=stage,total_primal_step=offset+step+1,**update))+'\n')
    snapshot(model,solver,optimizer,seed,offset+steps,stage,out/f'{stage}.pt')
    return dict(primal_updates=steps,dual_updates=steps,seconds=time.perf_counter()-start,
                train_seconds=train_seconds,model_sha256=tensor_digest(*model.state_dict().values()),
                adam_sha256=optimizer_digest(optimizer),dual_sha256=tensor_digest(solver.multiplier))


def run(args):
    torch.set_num_threads(2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    data=load_data();data.fingerprint.pop('initial_checkpoint')
    model,solver,optimizer=fresh(data,args.seed)
    config=dict(vars(args),fingerprints=data.fingerprint,initialization='fresh production constructor; no checkpoint weights',
        initial_model_sha256=tensor_digest(*model.state_dict().values()),
        initial_encoder_sha256=tensor_digest(*model.encoder.state_dict().values()),
        initial_projector_sha256=tensor_digest(*model.projector.state_dict().values()),
        initial_adam_states=len(optimizer.state),initial_nonzero_multipliers=int((solver.multiplier!=0).sum()),
        initial_projected_sha256=tensor_digest(model.project()),nodes=len(data.nodes),physical_edges=len(data.rows),edges=len(data.src),
        rho=100.,dual_step=1.,phr_lr=2e-5,decay_initial_lr=2e-6,decay_final_lr=2e-7,decay_schedule_steps=5000,
        qp_delta=.15,qp_push_step=0.,threads=2,torch=str(torch.__version__),python=platform.python_version(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=file_digest(__file__),
        stopping='fixed PHR/decay budgets; model-only repair feasibility/value and iteration/stall budget',
        evaluation='graph references only after repair selected and saved final model')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    snapshot(model,solver,optimizer,args.seed,0,'initial',out/'initial.pt')
    start=time.perf_counter()
    with (out/'training.jsonl').open('w') as trace,(out/'updates.jsonl').open('w') as updates:
        phr=train_stage(model,data,solver,optimizer,args.seed,'phr',args.phr_steps,0,out,trace,updates)
        boundary=dict(model_sha256=tensor_digest(*model.state_dict().values()),
            adam_sha256=optimizer_digest(optimizer),dual_sha256=tensor_digest(solver.multiplier))
        assert all(boundary[key]==phr[key] for key in boundary)
        decay=train_stage(model,data,solver,optimizer,args.seed,'decay',args.decay_steps,args.phr_steps,out,trace,updates)
    # There are deliberately no reference evaluations or value-based branches
    # before this unconditional repair stage.
    repair_run(argparse.Namespace(checkpoint=str(out/'decay.pt'),output=str(out/'repair'),
                                  iterations=args.qp_iterations,delta=.15,push_step=0.))
    evaluations={}
    for stage,path in [('phr',out/'phr.pt'),('decay',out/'decay.pt'),('qp',out/'repair/latest.pt')]:
        loaded,dual,_,saved=restore(data,path)
        rec,values,arrays=diagnose(loaded,data,dual,saved['step'],0.)
        rec['strict_physical_feasible']=rec['max_residual']<=0.
        rec['model_sha256']=tensor_digest(*loaded.state_dict().values())
        rec['checkpoint_sha256']=file_digest(path)
        evaluations[stage]=rec
        (out/f'{stage}_evaluation.json').write_text(json.dumps(rec,indent=2)+'\n')
        (out/f'{stage}_values.json').write_text(json.dumps(values)+'\n')
    repair_evaluation=json.loads((out/'repair/evaluation.json').read_text())
    summary=dict(seed=args.seed,stages=evaluations,timing=dict(phr=phr,decay=decay,total_seconds=time.perf_counter()-start),
        stage_boundary=boundary,repair_stop_reason=repair_evaluation['stop_reason'],
        repair_iterations=repair_evaluation['step'],repair_candidate_evaluations=repair_evaluation['candidate_evaluations'],
        fixed_push_floor=repair_evaluation['push_floor'],final_joint_pass=evaluations['qp']['joint_pass'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'COMPLETE').write_text('complete; success is independently recorded in summary.json\n')
    print(json.dumps(dict(seed=args.seed,complete=True,joint_pass=summary['final_joint_pass'],stop_reason=summary['repair_stop_reason'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--phr-steps',type=int,default=20000)
    p.add_argument('--decay-steps',type=int,default=5000)
    p.add_argument('--qp-iterations',type=int,default=40)
    run(p.parse_args())
