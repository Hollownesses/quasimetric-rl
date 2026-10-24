"""Prespecified independent one-step objective probes, never training selection."""
import copy
import time
import torch
from experiments.four_path_factorial.run import IndexedIQE
from experiments.connected_phr_solver.train import forward_audit
from .train import accumulate


def probe_state(data,state):
    started=time.perf_counter()
    base=IndexedIQE(data.small,'encoder',0,initialization_seed=state['initialization_seed'])
    base.load_state_dict(state['model']);dual=state['dual'].clone()
    initial=accumulate(base,data,dual,data.original)
    grads=[p.grad.clone() for p in base.parameters() if p.requires_grad]
    params=[p.detach().clone() for p in base.parameters() if p.requires_grad]
    records=[]
    for mode in ('inherited','zero_first_moment'):
        for lr in (2e-6,2e-7,2e-8):
            model=IndexedIQE(data.small,'encoder',0,initialization_seed=state['initialization_seed'])
            model.load_state_dict(state['model'])
            opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
            opt.load_state_dict(copy.deepcopy(state['optimizer']))
            for group in opt.param_groups:group['lr']=lr
            if mode=='zero_first_moment':
                for entry in opt.state.values():
                    if 'exp_avg' in entry:entry['exp_avg'].zero_()
            active=[p for p in model.parameters() if p.requires_grad]
            for p,g in zip(active,grads):p.grad=g.clone()
            opt.step();after,_=forward_audit(model,data,dual)
            delta=[p.detach().double()-v.double() for p,v in zip(active,params)]
            directional=float(sum((g.double()*v).sum() for g,v in zip(grads,delta)))
            records.append(dict(moment_mode=mode,lr=lr,before_loss=initial['loss'],
                after_loss=after['augmented_loss'],change=after['augmented_loss']-initial['loss'],
                gradient_dot_displacement=directional,
                displacement_norm=float(sum(v.square().sum() for v in delta).sqrt())))
    return dict(probes=records,gradient_norm=initial['gradient_norm'],seconds=time.perf_counter()-started,
        forward_records=7*len(data.original),backward_records=len(data.original),
        affects_main_checkpoints=False,uses_graph_reference=False)
