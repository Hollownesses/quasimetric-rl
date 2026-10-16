"""All-edge local restoration with a cumulative model-derived push floor.

Reuse the existing dual-Gram projection QP for the entire final projector layer.
Its exact Jacobian Gram is assembled from endpoint derivatives and shared hidden
features, avoiding a dense 339 x 1050624 Jacobian. Factorizing this Gram supplies
an equivalent small Euclidean projection to the existing solver. Its transient
dual coefficients are pulled back to actual network weight/bias directions.
Every accepted candidate is checked through the actual nonlinear network.
No graph distance, suffix return, or reference multiplier enters the solver.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.four_path_refinement.run import (
    restore, load_data, diagnose, file_digest, tensor_digest,
)
from quasimetric_rl.modules.quasimetric_critic.losses.feasible_direction import solve_direction


@torch.no_grad()
def actual(model, data):
    edge, task, pair = model.distances(data)
    return edge-data.cost, float(.6*task.mean()+.4*pair.mean())


def accept_candidate(old_h, new_h, new_push, floor, physical):
    if not torch.isfinite(new_h).all() or not math.isfinite(new_push) or new_push < floor:
        return False
    if float(new_h[~physical].abs().max()) > 1e-6:
        return False
    a = old_h[physical].relu(); b = new_h[physical].relu()
    if float(b.max()) <= 1e-5:
        return True
    # Restoration may start infeasible. A step need not solve every edge at once.
    return (float(b.square().sum()) < float(a.square().sum())*(1-1e-5)
            and float(b.max()) <= float(a.max())*1.05+1e-7)


def local_linearization(model, data, floor, margin=5e-5):
    module = model.projector.module
    with torch.no_grad():
        hidden = getattr(module,'1')(getattr(module,'0')(model.encoder(model.nodes)))
        features = torch.cat((hidden,torch.ones(len(hidden),1)),dim=1)
        z0 = model.project().detach()
    x = z0[data.src].clone().requires_grad_()
    y = z0[data.dst].clone().requires_grad_()
    edges = model.head(x,y)
    gx,gy = (v.detach().double() for v in torch.autograd.grad(edges.sum(),(x,y)))
    z = z0.clone().requires_grad_()
    task = model.head(z[data.task_src],z[data.goal])
    pair = model.head(z[data.task_src],z[data.pair])
    push = .6*task.mean()+.4*pair.mean()
    features = features.double()
    push_gradient = torch.autograd.grad(push,z)[0].detach().double().T@features
    fs,fd = features[data.src],features[data.dst]
    gram = ((gx@gx.T)*(fs@fs.T)+(gx@gy.T)*(fs@fd.T)
            +(gy@gx.T)*(fd@fs.T)+(gy@gy.T)*(fd@fd.T))
    cross = -((gx*(fs@push_gradient.T)).sum(1)+(gy*(fd@push_gradient.T)).sum(1))
    full = torch.zeros((len(edges)+1,len(edges)+1),dtype=torch.float64)
    full[:-1,:-1]=gram;full[-1,:-1]=cross;full[:-1,-1]=cross
    full[-1,-1]=push_gradient.square().sum()
    eigenvalues,eigenvectors = torch.linalg.eigh((full+full.T)/2)
    # Only numerical negative eigenvalues of a PSD Gram are clipped.
    assert float(eigenvalues.min()) >= -1e-10*float(eigenvalues.max())
    A = eigenvectors*eigenvalues.clamp_min(0).sqrt()[None,:]
    h = torch.cat((edges.detach()-data.cost+data.physical*margin,
                   torch.tensor([floor+margin-float(push.detach())])))
    return A,h,features,gx,gy,push_gradient


def qp_direction(linearization, data, elastic_penalty=None, push_step=0.):
    A,h,features,gx,gy,push_gradient = linearization
    coefficient = push_step/max(float(push_gradient.norm()),1e-30)
    # The last constraint row is -grad(push), so this factor-space vector
    # corresponds exactly to a normalized ascent direction in real parameters.
    desired = -coefficient*A[-1]
    _,alpha,_ = solve_direction(desired,A,h,
        kappa=1.,elastic_penalty=elastic_penalty,sweeps=2000,tolerance=1e-10)
    direction = (-(gx*alpha[:-1,None]).T@features[data.src]
                 -(gy*alpha[:-1,None]).T@features[data.dst]+(alpha[-1]+coefficient)*push_gradient)
    dz = features@direction.T
    slopes = (gx*dz[data.src]).sum(1)+(gy*dz[data.dst]).sum(1)
    slopes = torch.cat((slopes,-(push_gradient*direction).sum().reshape(1)))
    # Check slack in the original parameter space, not just the Gram factor.
    return direction.float().flatten(), alpha, (slopes+h.double()).relu()


@torch.no_grad()
def apply_delta(model, base_weight, base_bias, delta, scale):
    layer = getattr(model.projector.module,'2')
    d = delta.reshape(2048,513)*scale
    layer.weight.copy_(base_weight+d[:,:512])
    layer.bias.copy_(base_bias+d[:,512])


def run(args):
    torch.set_num_threads(2)
    data = load_data()
    model, solver, optimizer, saved = restore(data,Path(args.checkpoint))
    out = Path(args.output); out.mkdir(parents=True,exist_ok=False)
    initial_h, initial_push = actual(model,data)
    floor = initial_push-args.delta
    config = dict(vars(args), checkpoint_sha256=file_digest(args.checkpoint),
        fingerprints=data.fingerprint, initial_push=initial_push, fixed_push_floor=floor,
        initial_model_sha256=tensor_digest(*model.state_dict().values()),
        subspace='entire final projector: 2048 x (512 hidden+bias); 1050624 existing parameters',
        constraints='all 339 edges + model-derived global push floor',
        interior_margin=5e-5, trust_radius=.1, qp_sweeps=2000,
        script_sha256=file_digest(__file__),
        qp_script_sha256=file_digest('quasimetric_rl/modules/quasimetric_critic/losses/feasible_direction.py'))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    curve = (out/'curve.jsonl').open('w'); trace = (out/'qp.jsonl').open('w')
    start = time.perf_counter(); stalls = 0; candidate_evaluations = 0
    for iteration in range(args.iterations+1):
        rec, values, arrays = diagnose(model,data,solver,iteration,time.perf_counter()-start)
        rec['push_floor']=floor; rec['push_loss_from_start']=initial_push-rec['push']
        rec['candidate_evaluations']=candidate_evaluations
        curve.write(json.dumps(rec)+'\n'); curve.flush()
        np.savez(out/'latest_edges.npz',**arrays)
        torch.save(dict(step=iteration,model=model.state_dict(),solver=solver.state_dict(),
            optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),optimizer_stale_after_repair=True),out/'latest.pt')
        (out/'values.json').write_text(json.dumps(values))
        print(json.dumps({k:rec[k] for k in ('step','seconds','push','max_violation','violation_count','joint_pass')}),flush=True)
        if rec['joint_pass'] or iteration == args.iterations or stalls >= 3: break
        old_h, old_push = actual(model,data)
        linearization = local_linearization(model,data,floor)
        direction,alpha,slack = qp_direction(linearization,data,push_step=args.push_step)
        mode = 'hard'
        if not torch.isfinite(direction).all(): raise RuntimeError('nonfinite QP direction')
        layer = getattr(model.projector.module,'2')
        weight,bias = layer.weight.detach().clone(),layer.bias.detach().clone()
        accepted = False; attempts = []
        for fallback in range(2):
            if fallback:
                mode = 'elastic'
                direction,alpha,slack = qp_direction(linearization,data,elastic_penalty=100.,push_step=args.push_step)
            norm = float(direction.norm())
            scale = min(1.,.1/max(norm,1e-30))
            for backtrack in range(13):
                apply_delta(model,weight,bias,direction,scale)
                new_h,new_push = actual(model,data); candidate_evaluations += 1
                accepted = accept_candidate(old_h,new_h,new_push,floor,data.physical)
                attempts.append(dict(mode=mode,scale=scale,push=new_push,max_violation=float(new_h.relu().max()),accepted=accepted))
                if accepted: break
                scale *= .5
            if accepted: break
        if accepted:
            stalls = 0
        else:
            apply_delta(model,weight,bias,direction,0.)
            stalls += 1
        trace.write(json.dumps(dict(iteration=iteration,accepted=accepted,mode=mode,
            qp_linear_slack_max=float(slack.max()),qp_direction_norm=norm,
            actual_parameter_update_l2=float(torch.cat(((layer.weight.detach()-weight).flatten(),(layer.bias.detach()-bias).flatten())).norm()),
            attempts=attempts))+'\n'); trace.flush()
    curve.close();trace.close();(out/'COMPLETE').write_text('completed diagnostic budget\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--iterations',type=int,default=40)
    p.add_argument('--delta',type=float,default=.1)
    p.add_argument('--push-step',type=float,default=0.)
    run(p.parse_args())
