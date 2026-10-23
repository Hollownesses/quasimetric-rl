"""Post-budget comparison on the corrected connected expert graph.

No training module imports this evaluator. Reference values never select stopping
steps. Decision/statistics definitions are shared with the existing TD evaluator.
"""
import argparse
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .data import load_training_data
from .reference import validate_reference
from experiments.four_path_factorial.run import IndexedIQE, file_digest, tensor_digest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def statistics(prediction, reference):
    error = np.asarray(prediction, np.float64) - reference
    if not len(error):
        return dict(count=0)
    rmse = float(np.sqrt(np.mean(error**2)))
    scale = float(np.sqrt(np.mean(np.asarray(reference)**2)))
    return dict(count=len(error), mae=float(np.abs(error).mean()), rmse=rmse,
                nrmse_percent=100 * rmse / scale if scale else None,
                bias=float(error.mean()), mean_prediction=float(np.mean(prediction)),
                mean_reference=float(np.mean(reference)))


def decisions(data, graph, reference, prediction, starts, max_steps):
    """Same recorded action set for every arm; reference only scores decisions."""
    scores = data.cost.double().numpy() + prediction[data.dst.numpy()]
    truth = data.cost.double().numpy() + reference[data.dst.numpy()]
    chosen = np.full(len(data.nodes), -1, dtype=np.int64)
    branches = []
    correct_pairs = total_pairs = 0
    for node, rows in enumerate(graph.outgoing):
        if not rows:
            continue
        choice = int(rows[int(np.argmin(scores[rows]))])
        chosen[node] = choice
        if len(rows) <= 1:
            continue
        good = pairs = 0
        for i, row in enumerate(rows):
            for other in rows[i+1:]:
                gap = truth[row] - truth[other]
                if abs(gap) > 1e-6:
                    pairs += 1
                    good += int((scores[row] - scores[other]) * gap > 0)
        correct_pairs += good
        total_pairs += pairs
        branches.append(dict(
            source=node, observation=data.nodes[node].tolist(), chosen_row=choice,
            regret=float(max(0., truth[choice]-reference[node])),
            optimal_choice=bool(truth[choice]-reference[node] <= 1e-6),
            correctly_ordered_pairs=good, strict_reference_pairs=pairs,
            alternatives=[dict(row=r, target=int(data.dst[r]), action=data.raw.actions[r].tolist(),
                               step_cost=float(data.cost[r]), score=float(scores[r]),
                               reference_cost=float(truth[r])) for r in rows],
        ))
    episodes = []
    for index, start in enumerate(starts):
        node = int(start)
        seen, path, cost = set(), [], 0.
        status = "step_limit"
        for _ in range(max_steps):
            if node == graph.goal:
                status = "success"
                break
            if node in seen:
                status = "cycle"
                break
            seen.add(node)
            row = int(chosen[node])
            if row < 0:
                status = "unsupported"
                break
            path.append(row)
            cost += float(data.cost[row])
            node = int(data.dst[row])
        if node == graph.goal:
            status = "success"
        episodes.append(dict(
            episode=index, start=int(start), status=status, steps=len(path), rows=path,
            cost=cost, optimal_cost=float(reference[start]),
            regret=float(max(0., cost-reference[start])) if status == "success" else None,
        ))
    successes = [r for r in episodes if r["status"] == "success"]
    return dict(
        branches=branches, episodes=episodes,
        summary=dict(
            branch_nodes=len(branches), optimal_branch_choices=sum(b["optimal_choice"] for b in branches),
            pairwise_order_accuracy=correct_pairs / total_pairs if total_pairs else None,
            mean_branch_regret=float(np.mean([b["regret"] for b in branches])) if branches else None,
            graph_rollouts=len(episodes), graph_successes=len(successes),
            graph_success_rate=len(successes) / len(episodes),
            optimal_rollouts=sum(r["regret"] is not None and r["regret"] <= 1e-6 for r in episodes),
            mean_successful_cost=float(np.mean([r["cost"] for r in successes])) if successes else None,
            mean_successful_regret=float(np.mean([r["regret"] for r in successes])) if successes else None,
        ),
    ), scores


def graph_inputs(data):
    goal=int(data.goal[data.original].unique().item())
    outgoing=[[] for _ in data.nodes];seen=set()
    for r in data.original.tolist():
        key=(int(data.src[r]),int(data.dst[r]),data.raw.actions[r].numpy().tobytes(),float(data.cost[r]))
        if key not in seen:
            seen.add(key);outgoing[key[0]].append(r)
    assert all(rr or n==goal for n,rr in enumerate(outgoing))
    degree=torch.tensor([len(rr) for rr in outgoing])
    edge_rows=torch.full((len(outgoing),int(degree.max())),-1,dtype=torch.long)
    for n,rr in enumerate(outgoing):
        edge_rows[n,:len(rr)]=torch.tensor(rr,dtype=torch.long)
    fingerprint=tensor_digest(edge_rows,degree,torch.tensor([goal]))
    starts_rows=torch.cat((torch.zeros(1,dtype=torch.long),data.raw.episode_lengths.cumsum(0)[:-1]))
    starts=data.src[starts_rows[data.physical[starts_rows]]].numpy()
    return SimpleNamespace(goal=goal,outgoing=outgoing,fingerprint=fingerprint),starts


def equality(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equality(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equality(x,y) for x,y in zip(a,b))
    return a==b


def graph_structure(data,graph,starts):
    indegree=np.zeros(len(data.nodes),int);dst=data.dst.numpy()
    for rows in graph.outgoing:
        for r in rows:indegree[dst[r]]+=1
    queue=deque(np.flatnonzero(indegree==0));order=[]
    while queue:
        node=queue.popleft();order.append(int(node))
        for r in graph.outgoing[node]:
            indegree[dst[r]]-=1
            if indegree[dst[r]]==0:queue.append(int(dst[r]))
    acyclic=len(order)==len(data.nodes);longest=np.zeros(len(data.nodes),int)
    if acyclic:
        for n in reversed(order):
            if graph.outgoing[n]:longest[n]=1+max(longest[dst[r]] for r in graph.outgoing[n])
    return dict(nodes=len(data.nodes),physical_records=len(data.original),
                unique_edges=sum(map(len,graph.outgoing)),
                branches=sum(len(r)>1 for r in graph.outgoing),starts=len(starts),
                unique_starts=len(np.unique(starts)),acyclic=acyclic,
                longest_possible_recorded_rollout=int(longest[starts].max()) if acyclic else None)


def check_pairing(roots,data,graph):
    configurations={k:json.loads((r/'config.json').read_text()) for k,r in roots.items()}
    initial={k:torch.load(r/'initial.pt',weights_only=True) for k,r in roots.items()}
    final={k:torch.load(r/'final.pt',weights_only=True) for k,r in roots.items()}
    base=initial['qrl_b1'];last=final['qrl_b1'];cfg=configurations['qrl_task']
    for arm in roots:
        assert (roots[arm]/'COMPLETE').exists(),arm
        for key in ('model','optimizer','sampler','visits','rng'):
            assert equality(initial[arm][key],base[key]),(arm,'initial',key)
        for key in ('sampler','visits'):
            assert equality(final[arm][key],last[key]),(arm,'final',key)
        assert final[arm]['step']==50000 and final[arm]['fingerprints']==data.fingerprint
        assert final[arm]['dataset_kind']==data.dataset_kind
        assert torch.equal(final[arm]['model']['nodes'],data.nodes)
    for key in ('seed','phr_steps','decay_steps','batch','threads','task_mode','rho','dual_step',
                'fingerprints','constraints','dual_clock','sampling','b1_seed','initial_model_sha256'):
        assert configurations['qrl_b1'][key]==cfg[key],key
    assert cfg['push_mode']=='task-only' and cfg['phr_steps']==45000 and cfg['decay_steps']==5000
    signature=configurations['td']['training_signature']
    assert signature==final['td']['training_signature']
    assert signature['qrl_config_sha256']==file_digest(roots['qrl_task']/'config.json')
    assert signature['qrl_initial_sha256']==file_digest(roots['qrl_task']/'initial.pt')
    assert signature['graph_sha256']==graph.fingerprint
    assert signature['main_steps']==45000 and signature['decay_steps']==5000
    for key in ('seed','batch','threads','task_mode'):assert signature[key]==cfg[key]
    assert equality(initial['td']['target'],base['model'])
    assert json.loads((roots['td']/'COMPLETE').read_text())['final_sha256']==file_digest(roots['td']/'final.pt')
    return configurations,dict(initial_model_sha256=cfg['initial_model_sha256'],
        identical_initial_model_optimizer_sampler_rng=True,identical_final_sampler_visits=True,
        final_steps=50000,physical_record_visits=int(last['visits'].sum()),
        visits_min=int(last['visits'].min()),visits_max=int(last['visits'].max()),
        graph_candidate_sha256=graph.fingerprint)


@torch.no_grad()
def evaluate(data,graph,reference,starts,path):
    saved=torch.load(path,weights_only=True)
    assert saved['fingerprints']==data.fingerprint
    model=IndexedIQE(data.small,'encoder',0,initialization_seed=saved['initialization_seed'])
    model.load_state_dict(saved['model'],strict=True);model.eval()
    # Use identical node/query batching for all three arms.
    z=torch.cat([model.projector(model.encoder(x)) for x in data.nodes.split(512)])
    prediction=torch.cat([model.head(z[i],z[graph.goal].expand(len(i),-1))
                          for i in torch.arange(len(z)).split(512)]).numpy().astype(float)
    edges=torch.cat([model.head(z[data.src[r]],z[data.dst[r]])
                     for r in torch.arange(len(data.src)).split(512)]).numpy().astype(float)
    assert np.isfinite(prediction).all() and np.isfinite(edges).all()
    assert prediction[graph.goal]==0
    rows=data.original.numpy();src=data.src.numpy();dst=data.dst.numpy();cost=data.cost.double().numpy()
    unique=np.unique(src[rows]);long=unique[reference[unique]>=np.quantile(reference[unique],.75)]
    groups=dict(records=src[rows],unique_sources=unique,starts=starts,
                unique_starts=np.unique(starts),longest_quartile=long)
    distance={k:statistics(prediction[ids],reference[ids]) for k,ids in groups.items()}
    h=edges-cost
    selected=np.array([r for rr in graph.outgoing for r in rr])
    def residual_stats(v):
        return dict(count=len(v),violations=int((v>1e-5).sum()),max_positive=float(np.maximum(v,0).max()),
                    mean_positive=float(np.maximum(v,0).mean()),mae=float(np.abs(v).mean()),
                    rmse=float(np.sqrt(np.mean(v*v))),mean=float(np.mean(v)))
    local=dict(records=residual_stats(h[rows]),unique_edges=residual_stats(h[selected]),
               anchor_max=float(np.abs(h[~data.physical.numpy()]).max()))
    task_h=prediction[src]-cost-prediction[dst]
    minimum=np.full(len(prediction),np.inf)
    np.minimum.at(minimum,src[rows],cost[rows]+prediction[dst[rows]])
    bellman={k:residual_stats(prediction[ids]-minimum[ids]) for k,ids in groups.items()}
    # Triangle slack quantifies the difference between pairwise-edge feasibility
    # and the weaker task-potential inequality; it is not a distance target.
    geometry=dict(task_inequality=residual_stats(task_h[rows]),
                  triangle_slack_mean=float(np.mean(h[rows]-task_h[rows])),
                  minimum_triangle_slack=float(np.min(h[rows]-task_h[rows])))
    decision,scores=decisions(data,graph,reference,prediction,starts,300)
    unique_decision,_=decisions(data,graph,reference,prediction,np.unique(starts),300)
    decision['summary']['unique_start_rollouts']=unique_decision['summary']
    for limit in (1,5,10):
        decision['summary'][f'branch_regret_over_{limit}']=sum(b['regret']>limit for b in decision['branches'])
    result=dict(step=saved['step'],checkpoint_sha256=file_digest(path),distance=distance,local=local,
                bellman=bellman,geometry=geometry,decisions=decision['summary'])
    return result,prediction,decision


def plot(out,reference,predictions,results,logs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    labels={'qrl_b1':'QRL task+B1','qrl_task':'QRL task-only','td':'Optimality TD'}
    for arm,pred in predictions.items():
        axes[0].scatter(reference,pred,s=3,alpha=.25,label=labels[arm])
    axes[0].plot([0,reference.max()],[0,reference.max()],'k--',lw=1)
    axes[0].set(xlabel='Recorded-graph distance',ylabel='Predicted task distance');axes[0].legend(fontsize=8)
    x=np.arange(3)
    axes[1].bar(x,[results[k]['decisions']['optimal_branch_choices'] for k in labels])
    axes[1].set(xticks=x,xticklabels=['QRL+B1','QRL task','TD'],ylabel='Optimal branch choices / 58',ylim=(0,58))
    axes[2].bar(x,[results[k]['decisions']['mean_successful_regret'] for k in labels])
    axes[2].set(xticks=x,xticklabels=['QRL+B1','QRL task','TD'],ylabel='Mean recorded-rollout regret')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm in ('qrl_b1','qrl_task'):
        scans=[r for r in logs[arm] if 'connected_expert' in r]
        for ax,key in zip(axes[:2],('max_violation','mean_positive')):
            ax.plot([r['step'] for r in scans],[r['connected_expert'][key] for r in scans],label=labels[arm])
            ax.set(xlabel='Update',ylabel=key);ax.axvline(45000,c='k',ls=':',lw=.8);ax.legend(fontsize=8)
    scans=[r for r in logs['td'] if r.get('kind')=='scan']
    for name in ('online_bellman','target_network'):
        axes[2].plot([r['step'] for r in scans],[r[name]['physical_records']['rmse'] for r in scans],label=name)
    axes[2].set(xlabel='Update',ylabel='TD record residual RMSE');axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'training.png',dpi=180);plt.close(fig)


def run(args):
    torch.set_num_threads(1)
    data=load_training_data(args.data);graph,starts=graph_inputs(data)
    refroot=Path(args.reference);manifest=json.loads((refroot/'summary.json').read_text())
    assert manifest['fingerprints']==data.fingerprint and manifest['passed']
    assert manifest['reference_sha256']==file_digest(refroot/'graph_reference_audit_only.npz')
    with np.load(refroot/'graph_reference_audit_only.npz') as f:
        for key,expected in (('source',data.src),('target',data.dst),('physical',data.physical)):
            assert np.array_equal(f[key],expected.numpy())
        reference=f['distance'].copy()
    validate_reference(data,reference)
    roots={k:Path(getattr(args,k)).resolve() for k in ('qrl_b1','qrl_task','td')}
    configs,parity=check_pairing(roots,data,graph)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    results={};predictions={};stages={};logs={}
    for arm,root in roots.items():
        result,pred,decision=evaluate(data,graph,reference,starts,root/'final.pt')
        results[arm]=result;predictions[arm]=pred;write_json(out/(arm+'_decisions.json'),decision)
        stage=root/('main.pt' if arm=='td' else 'phr.pt')
        stages[arm]=evaluate(data,graph,reference,starts,stage)[0]
        logs[arm]=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
    # Reference greedy policy is an evaluation sanity check, never a learner.
    oracle,_=decisions(data,graph,reference,reference,starts,300)
    assert oracle['summary']['optimal_branch_choices']==58
    assert oracle['summary']['graph_successes']==307 and oracle['summary']['mean_successful_regret']<1e-8
    summary=dict(parity=parity,graph_structure=graph_structure(data,graph,starts),
                 fingerprints=data.fingerprint,reference_sha256=manifest['reference_sha256'],
                 reference_correction=manifest['changed'],results=results,stage45000=stages,
                 oracle=oracle['summary'],runs={k:str(r) for k,r in roots.items()},
                 evaluator_sha256=file_digest(__file__),
                 definition='NRMSE=RMSE/RMS(reference); strict violation >1e-5; branch tie tolerance 1e-6',
                 scope='Single seed, fixed finite expert graph; no unseen-state or closed-loop navigation test')
    write_json(out/'summary.json',summary)
    np.savez_compressed(out/'distances.npz',reference=reference,starts=starts,**predictions)
    plot(out,reference,predictions,results,logs)
    write_json(out/'COMPLETE',dict(summary_sha256=file_digest(out/'summary.json')))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('data','reference','qrl-b1','qrl-task','td','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
