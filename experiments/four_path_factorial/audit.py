"""Independent checkpoint reload, full constraint and gradient audit (no training)."""
from pathlib import Path
import copy
import json
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as f


def norms(grads):
    return float(sum(g.double().square().sum() for g in grads if g is not None).sqrt())


def main(root):
    torch.set_num_threads(2)
    data=f.load_data(); result={}
    for directory in sorted((root/'main').glob('*')):
        if not (directory/'COMPLETE').exists(): continue
        cfg=json.loads((directory/'config.json').read_text())
        model=f.IndexedIQE(data,cfg['parameterization'],cfg['critic'])
        ck=torch.load(directory/'latest.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(ck['model'])
        solver=f.TabularPHR(len(data.src),cfg['rho'],cfg['dual_step']) if cfg['method']=='phr' else f.CachedBarrier(f.GoalSetBarrierConf())
        solver.load_state_dict(ck['solver'])
        saved=json.loads((directory/'curve.jsonl').read_text().splitlines()[-1])
        current,_,arrays=f.evaluate(model,data,solver,ck['step'],0)
        for key in ('push','task_rmse','max_violation','violation_count','terminal_max','anchor_max'):
            assert abs(current[key]-saved[key])<1e-5,(directory,key)
        prior=np.load(directory/'latest_edges.npz')
        for key,value in arrays.items(): np.testing.assert_allclose(value,prior[key],atol=1e-5,rtol=1e-6)
        edge,task,pair=model.distances(data); residual=edge-data.cost
        push=.6*task.mean()+.4*pair.mean()
        if cfg['method']=='phr':
            penalty=solver(residual)
        else:
            penalty=500*(.01*torch.nn.functional.softplus((residual[data.physical]+.01)/.01)).topk(8).values.mean()
        params=list(model.parameters())
        gp=torch.autograd.grad(-push,params,retain_graph=True,allow_unused=True)
        gc=torch.autograd.grad(penalty,params,allow_unused=True)
        gp=[torch.zeros_like(p) if g is None else g for g,p in zip(gp,params)]
        gc=[torch.zeros_like(p) if g is None else g for g,p in zip(gc,params)]
        dot=float(sum((a.double()*b.double()).sum() for a,b in zip(gp,gc)))
        record=dict(checkpoint_reload_matches=True,step=ck['step'],metrics=current,
            optimizer_group_lrs=[g['lr'] for g in ck['optimizer']['param_groups']],
            gradient_probe=dict(push_norm=norms(gp),constraint_norm=norms(gc),total_norm=norms([a+b for a,b in zip(gp,gc)]),
                cosine=dot/max(norms(gp)*norms(gc),1e-30),
                caveat='PHR full current primal gradient; barrier full-replay top8 surrogate, not stochastic-workset stationarity. IQE is nonsmooth. Norms across parameterizations are not comparable.'))
        record['path_violation_budgets']=[]
        for path in range(4):
            mask=(data.raw.transition_infos['context_id']==path)&data.physical
            record['path_violation_budgets'].append(dict(path=path,
                sum_positive_residual=float(residual[mask].detach().relu().sum()),
                max_positive_residual=float(residual[mask].detach().relu().max()),
                recorded_cost=float(data.cost[mask].sum())))
        if cfg['method']=='phr':
            lam=solver.multiplier
            record['dual']=dict(nonzero=int((lam>0).sum()),mean=float(lam.mean()),max=float(lam.max()),
                complementarity_mean=float((lam*residual.detach()).abs().mean()),
                projected_residual_max=float(((lam+cfg['dual_step']*residual.detach()).relu()-lam).abs().max()),
                anchor_max=float(lam[~data.physical].max()),sum_coefficient_max=float(lam.max()/len(lam)))
            worst=torch.argsort(residual.detach(),descending=True)[:10]
            record['worst_edges']=[dict(row=int(i),source_node=int(data.src[i]),target_node=int(data.dst[i]),
                cost=float(data.cost[i]),distance=float(edge[i].detach()),residual=float(residual[i].detach()),
                multiplier=float(lam[i])) for i in worst]
            np.savez(directory/'audited_dual.npz',multiplier=lam.numpy(),residual=residual.detach().numpy(),src=data.src.numpy(),dst=data.dst.numpy())
        # Check exact homogeneity of the feasible-rescaled diagnostic, through
        # the actual IQE head, instead of merely multiplying predictions.
        with torch.no_grad():
            z=model.project()*current['feasible_scale']
            scaled_edge=model.head(z[data.src],z[data.dst])
            record['rescaled_actual_max_violation']=float((scaled_edge-data.cost).relu().max())
            assert record['rescaled_actual_max_violation']<2e-5
        result[directory.name]=record
    # Recompute historic baseline on precisely the same frozen support.
    old=torch.load(f.OLD/'checkpoint_final.pth',map_location='cpu',weights_only=True)['agent']
    baseline=[]
    for ci in (0,1):
        model=f.IndexedIQE(data,'encoder',ci)
        state=model.state_dict()
        for k in state:
            if k=='nodes':continue
            oldkey=(f'critics.{ci}.quasimetric_model.quasimetric_head.'+k[5:] if k.startswith('head.')
                else f'critics.{ci}.quasimetric_model.'+k if k.startswith('projector.') else f'critics.{ci}.'+k)
            state[k]=old[oldkey]
        model.load_state_dict(state)
        baseline.append(f.evaluate(model,data,None,20000,0)[0])
    out=root/'comparison'; out.mkdir(exist_ok=True)
    (out/'independent_audit.json').write_text(json.dumps(dict(runs=result,historical_stochastic_baseline=baseline),indent=2)+'\n')
    reference=f.graph_reference(data)
    head=copy.deepcopy(model.head).double()
    with torch.no_grad():
        z=-reference[:,None].expand(-1,2048)
        edge=head(z[data.src],z[data.dst]); task=head(z[data.task_src],z[data.goal]); pair=head(z[data.task_src],z[data.pair])
        witness=dict(scope='Evaluation-only representation witness; never training labels or initialization',
            edge_max_error=float((edge-data.cost).abs().max()),
            task_max_error=float((task-reference[data.task_src]).abs().max()),
            pair_max_error=float((pair-(reference[data.task_src]-reference[data.pair])).abs().max()),
            push=float(.6*task.mean()+.4*pair.mean()))
    (out/'representation_witness.json').write_text(json.dumps(witness,indent=2)+'\n')
    print(json.dumps(dict(audited_runs=list(result),historical_push=[x['push'] for x in baseline]),indent=2))

if __name__=='__main__': main(Path(sys.argv[1]))
