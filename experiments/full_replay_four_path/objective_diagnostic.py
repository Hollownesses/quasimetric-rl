"""Evaluation-only decomposition at the frozen final checkpoint; no optimizer steps."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.full_replay_four_path.data import load_union
from experiments.full_replay_four_path.train import distances
from experiments.four_path_factorial.run import IndexedIQE,file_digest
from experiments.four_path_execution.common import write
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms


def run(root):
    torch.set_num_threads(4);assert (root/'train/COMPLETE').exists()
    path=root/'train/final.pt';before=file_digest(path);saved=torch.load(path,map_location='cpu',weights_only=True)
    d=load_union();m=IndexedIQE(d.small,'encoder',0,initialization_seed=606);m.load_state_dict(saved['model'])
    rng=np.random.default_rng(2026101801);device=d.raw.transition_infos['device_index'].numpy()
    arrays=np.load(root/'value_analysis/recorded_values.npz');rows=arrays['rows'];original=rows<d.nbase
    groups={};gradients={}
    for dev in range(4):
        mask=original&(device[rows]==dev);task=arrays['task'][mask];pair=arrays['pair'][mask]
        groups[f'device_{dev}']=dict(rows=int(mask.sum()),task_mean=float(task.mean()),pair_mean=float(pair.mean()),
            push_mean=float(.6*task.mean()+.4*pair.mean()),task_quantiles=np.quantile(task,[0,.25,.5,.75,1]).tolist())
    for name,r in [('paths',d.paths)]+[(f'device_{i}',torch.from_numpy(rng.choice(d.original.numpy()[device[d.original.numpy()]==i],512,replace=False))) for i in range(4)]:
        edge,task,pair=distances(m,d,r);push=.6*task.mean()+.4*pair.mean()
        constraint=phr_terms(edge-d.cost[r],saved['dual'][r],100.)[0]
        gp=torch.autograd.grad(push,tuple(m.parameters()),retain_graph=True)
        gc=torch.autograd.grad(constraint,tuple(m.parameters()))
        gp=torch.cat([x.reshape(-1) for x in gp]);gc=torch.cat([x.reshape(-1) for x in gc])
        gradients[name]=(gp,gc)
        groups.setdefault(name,{})['gradient_sample']=dict(rows=len(r),push=float(push.detach()),phr=float(constraint.detach()),
            push_gradient_norm=float(gp.norm()),phr_gradient_norm=float(gc.norm()),
            objective_descent_norm=float((gp-gc).norm()))
    path_grad=gradients['paths'][0];path_descent=path_grad-gradients['paths'][1]
    base_grad=sum(gradients[f'device_{i}'][0] for i in range(4))/4
    base_descent=sum(gradients[f'device_{i}'][0]-gradients[f'device_{i}'][1] for i in range(4))/4
    def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))
    assert file_digest(path)==before
    write(root/'value_analysis/objective_diagnostic.json',dict(checkpoint_sha256=before,groups=groups,
        push_gradient_cosine_path_original=cosine(path_grad,base_grad),
        objective_descent_cosine_path_original=cosine(path_descent,base_descent),
        caution='post-run local Euclidean gradient diagnostic, fixed 512 rows/device; not Adam update or causal identification; no parameter mutation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
