"""Full physical-record objective derivatives; no reference or graph labels."""
import numpy as np
import torch
from experiments.full_replay_four_path.train import distances
from quasimetric_rl.modules.quasimetric_critic.losses.phr import phr_terms


def full_gradients(model,data,dual,batch=512,rho=100.):
    rows=data.original;params=[p for p in model.parameters() if p.requires_grad]
    size=sum(p.numel() for p in params)
    vectors={k:torch.zeros(size,dtype=torch.float64) for k in ('task','b1','phr','lagrange')}
    totals={k:0. for k in vectors};hs=[]
    for r in rows.split(batch):
        edge,task,pair=distances(model,data,r)
        h=edge-data.cost[r];hs.append(h.detach())
        fraction=len(r)/len(rows)
        terms={'task':-.6*task.mean(),'b1':-.4*pair.mean(),
               'phr':phr_terms(h,dual[r],rho)[0],'lagrange':(dual[r].detach()*h).mean()}
        for j,(name,value) in enumerate(terms.items()):
            grads=torch.autograd.grad(value*fraction,params,retain_graph=j<3,allow_unused=True)
            vectors[name]+=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().double()
                                      for p,g in zip(params,grads)])
            totals[name]+=float(value.detach())*fraction
    h=torch.cat(hs).double();lam=dual[rows].double()
    push=vectors['task']+vectors['b1'];aug=push+vectors['phr'];lag=push+vectors['lagrange']
    norm=lambda x:float(x.norm())
    cosine=lambda x,y:float(torch.dot(x,y)/(x.norm()*y.norm())) if norm(x)*norm(y)>0 else None
    effective=(lam+rho*h).clamp_min(0)
    metrics=dict(losses=totals,augmented_loss=totals['task']+totals['b1']+totals['phr'],
        gradient_norms={**{k:norm(v) for k,v in vectors.items()},'push':norm(push),'augmented':norm(aug),'lagrangian':norm(lag)},
        cos_push_phr=cosine(push,vectors['phr']),cos_task_b1=cosine(vectors['task'],vectors['b1']),
        normalized_augmented_gradient=norm(aug)/(norm(push)+norm(vectors['phr'])+1e-30),
        normalized_lagrangian_gradient=norm(lag)/(norm(push)+norm(vectors['lagrange'])+1e-30),
        max_violation=float(h.clamp_min(0).max()),violation_count=int((h>1e-5).sum()),
        mean_positive=float(h.clamp_min(0).mean()),dual_max=float(lam.max()),dual_mean=float(lam.mean()),
        complementarity_mean=float((lam*h).abs().mean()),complementarity_max=float((lam*h).abs().max()),
        positive_dual_on_strictly_slack_rows=int(((lam>1e-5)&(h < -1e-3)).sum()),
        phr_active_rows=int((effective>0).sum()),
        projected_dual_residual_eta1_rms=float(((lam+h).clamp_min(0)-lam).square().mean().sqrt()),
        conventions='Uniform physical-record mean, original .6 task + .4 B1, rho100; saved dual fixed; parameter-coordinate gradient norms are not coordinate-invariant KKT certificates')
    assert all(torch.isfinite(v).all() for v in vectors.values())
    return metrics,vectors,h.numpy()


@torch.no_grad()
def all_distances(model,data,batch=512):
    z=torch.cat([model.projector(model.encoder(x)) for x in data.nodes.split(batch)])
    goal=int(data.goal[data.original].unique().item())
    task=torch.cat([model.head(z[i],z[goal].expand(len(i),-1)) for i in torch.arange(len(z)).split(batch)])
    edges=torch.cat([model.head(z[data.src[r]],z[data.dst[r]]) for r in torch.arange(len(data.src)).split(batch)])
    return task.double().numpy(),edges.double().numpy()
