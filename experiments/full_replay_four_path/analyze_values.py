"""Post-training graph, value and support analysis; never imported by training."""
import argparse
import heapq
from pathlib import Path
import numpy as np
import torch
from experiments.full_replay_four_path.data import load_union
from experiments.full_replay_four_path.train import distances
from experiments.full_replay_four_path.evaluate import ARMS,make_controller,model_path
from experiments.four_path_execution.common import adapter,write,read,file_digest
from experiments.four_path_factorial.run import evaluate


def reverse_graph(data):
    rev=[[] for _ in range(len(data.nodes))]
    for a,b,c in zip(data.src.tolist(),data.dst.tolist(),data.cost.tolist()):rev[b].append((a,c))
    ref=np.full(len(rev),np.inf);queue=[]
    for g in data.goal.unique().tolist():ref[g]=0.;heapq.heappush(queue,(0.,g))
    while queue:
        cost,node=heapq.heappop(queue)
        if cost!=ref[node]:continue
        for before,c in rev[node]:
            new=cost+c
            if new<ref[before]:ref[before]=new;heapq.heappush(queue,(new,before))
    return ref


def stats(pred,ref):
    finite=np.isfinite(ref);p=np.asarray(pred)[finite];r=np.asarray(ref)[finite]
    return dict(total=len(ref),finite=int(finite.sum()),unreachable=int((~finite).sum()),
        mae=float(abs(p-r).mean()) if len(p) else None,rmse=float(np.square(p-r).mean()**.5) if len(p) else None,
        mean_prediction=float(p.mean()) if len(p) else None,mean_reference=float(r.mean()) if len(p) else None,
        nrmse_percent=float(100*np.sqrt(np.square(p-r).mean()/np.square(r).mean())) if len(p) and np.any(r) else None)


@torch.no_grad()
def run(root):
    assert (root/'train/COMPLETE').exists();out=root/'value_analysis';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);d=load_union();adapted,model=adapter(d.small,root/'train/final.pt')
    small_rec,small_values,small_arrays=evaluate(model,d.small,None,25000,0.)
    write(out/'four_path.json',small_rec);write(out/'four_path_values.json',small_values)
    residual=np.load(root/'train/final_constraints.npz')['residual']
    visits=torch.load(root/'train/final.pt',weights_only=True,map_location='cpu')['visits'].numpy()
    physical=d.physical.numpy();entry=physical&(d.dst.numpy()==d.goal.numpy())&(d.src.numpy()!=d.dst.numpy())
    local_groups={}
    for name,mask in [('original',physical&(np.arange(len(physical))<d.nbase)),
                      ('paths',physical&(np.arange(len(physical))>=d.nbase)),
                      ('original_terminal_entry',entry&(np.arange(len(entry))<d.nbase)),
                      ('path_terminal_entry',entry&(np.arange(len(entry))>=d.nbase))]:
        h=residual[mask]
        local_groups[name]=dict(count=int(mask.sum()),violations=int((h>1e-5).sum()),
            max_violation=float(np.maximum(h,0).max()) if len(h) else None,
            mean_positive=float(np.maximum(h,0).mean()) if len(h) else None)
    for device in range(4):
        mask=physical&(d.raw.transition_infos['device_index'].numpy()==device)&(np.arange(len(physical))<d.nbase)
        h=residual[mask];local_groups[f'original_device_{device}']=dict(count=int(mask.sum()),violations=int((h>1e-5).sum()),
            max_violation=float(np.maximum(h,0).max()),mean_positive=float(np.maximum(h,0).mean()))
    worst=[]
    for row in np.argsort(-residual)[:50]:
        worst.append(dict(row=int(row),source='original' if row<d.nbase else 'paths',
            device=int(d.raw.transition_infos['device_index'][row]),cost=float(d.cost[row]),residual=float(residual[row]),
            terminal_entry=bool(entry[row]),source_observation=d.raw.all_observations[d.src_obs[row]].tolist(),
            target_observation=d.raw.all_observations[d.src_obs[row]+1].tolist()))
    edges=residual+d.cost.numpy()
    feasible_scale=min(1.,float(np.min(d.cost.numpy()[physical]/np.maximum(edges[physical],1e-30))))
    write(out/'local_constraint_groups.json',dict(groups=local_groups,worst_edges=worst,
        conservative_full_feasible_scale=feasible_scale,
        scale_usage='diagnostic only; not applied to checkpoint or controller'))
    predictions=[]
    for r in torch.arange(len(d.src)).split(1024):predictions.append(torch.cat(distances(model,d,r)[1:]).reshape(2,-1).numpy())
    # Abstract rows excluded from push outputs; concatenate each physical block in replay order.
    predictions=np.concatenate(predictions,axis=1);task,pair=predictions
    rows=d.physical.nonzero().flatten().numpy();reference=reverse_graph(d);ref=reference[d.src[rows].numpy()]
    obs=d.raw.all_observations[d.src_obs[rows]].numpy();dev=d.raw.transition_infos['device_index'][rows].numpy()
    u=(dev==0)&(obs[:,0]>2.8)&(obs[:,0]<5.2)&(obs[:,1]>1.9)&(obs[:,1]<5.4)
    origin=rows<d.nbase
    groups={}
    for name,mask in [('original',origin),('paths',~origin),('original_U',origin&u),('paths_U',~origin&u)]:
        groups[name]=stats(task[mask],ref[mask])
        groups[name]['predicted_task_quantiles']=np.quantile(task[mask],[0,.25,.5,.75,1]).tolist()
    for device in range(4):groups[f'original_device_{device}']=stats(task[origin&(dev==device)],ref[origin&(dev==device)])
    spans=d.raw.transition_infos['global_push_pair_steps'][rows].numpy()
    prefix=np.r_[0.,np.cumsum(d.cost.numpy(),dtype=np.float64)]
    upper=prefix[rows+spans]-prefix[rows]
    pair_report={}
    for name,mask in [('original',origin),('paths',~origin),('original_U',origin&u)]:
        pair_report[name]=dict(rows=int(mask.sum()),mean_recorded_chain_cost=float(upper[mask].mean()),
            mean_prediction=float(pair[mask].mean()),mean_chain_gap=float((upper[mask]-pair[mask]).mean()),
            above_chain_bound_count=int((pair[mask]-upper[mask]>1e-5).sum()),
            max_above_chain_bound=float(np.maximum(pair[mask]-upper[mask],0).max()))
    write(out/'recorded_graph.json',dict(groups=groups,b1_chain_bounds=pair_report,
        semantics='exact float32 canonical nodes, directed recorded edges only; infinity means no recorded chain, not environmental impossibility',
        b1_semantics='recorded forward segment cost is an upper bound, not assumed shortest'))
    np.savez_compressed(out/'recorded_values.npz',rows=rows,task=task,pair=pair,graph_task=ref,b1_chain_upper=upper)
    # Adapter equality across all four-path pairs and independently sampled original rows.
    select=np.random.default_rng(1704).choice(len(rows),2048,replace=False);r=rows[select]
    actual=adapted.batch_value(d.nodes[d.src[r]].numpy(),d.nodes[d.goal[r]].numpy())
    np.testing.assert_allclose(actual,task[select],rtol=5e-6,atol=1e-4)
    write(out/'adapter_check.json',dict(sampled_original_pairs=len(r),max_error=float(abs(actual-task[select]).max()),
        checkpoint_sha256=file_digest(root/'train/final.pt')))
    # Evaluation-only Oracle over original replay U states, plus both evaluation cohorts.
    controls={a:make_controller(root)(d.small,a) for a in ARMS if a!='none'}
    env=d.env;env.reset(seed=0,options={'device_id':'u_trap_target','start':[4.,3.65,3.14]})
    for control in controls.values():control.begin_episode(env,env.abstract_goal_observation(),0)
    state_records=read(root/'protocol.json')['original_states']+read(root/'protocol.json')['independent_states']
    poses=np.asarray([r['pose'] for r in state_records],np.float32)
    observations=np.asarray([env.state_to_observation(p) for p in poses]);goals=np.repeat(env.abstract_goal_observation()[None],len(poses),axis=0)
    oracle=controls['oracle'].terminal_value_provider
    oracle_values=oracle.batch_value(env,poses)
    values={a:control.qrl_agent.batch_value(observations,goals) for a,control in controls.items() if a!='oracle'}
    for arm,pred in values.items():
        np.savez_compressed(out/f'{arm}_evaluation_states.npz',poses=poses,value=pred,oracle=oracle_values)
    comparisons={}
    for stratum in ('support','perturbed','holdout','independent'):
        mask=np.array([r['stratum']==stratum for r in state_records])
        comparisons[stratum]={a:stats(pred[mask],np.where(oracle_values[mask]<1e6,oracle_values[mask],np.inf)) for a,pred in values.items()}
    write(out/'oracle_state_comparison.json',comparisons)
    uposes=np.column_stack((obs[origin&u,:2],np.arctan2(obs[origin&u,3],obs[origin&u,2])))
    oracle_u=oracle.batch_value(env,uposes)
    write(out/'oracle_original_U.json',stats(task[origin&u],np.where(oracle_u<1e6,oracle_u,np.inf)))
    # Empirical support distances: replay coverage is broad, hence "independent" != far from all samples.
    from scipy.spatial import cKDTree
    trainobs=d.nodes.numpy();trainpose=np.column_stack((trainobs[:,:2],np.arctan2(trainobs[:,3],trainobs[:,2])))
    original_task_obs=obs[origin&(dev==0)]
    original_task_poses=np.column_stack((original_task_obs[:,:2],np.arctan2(original_task_obs[:,3],original_task_obs[:,2])))
    tree=cKDTree(original_task_poses[:,:2]);nearest=tree.query(poses[:,:2])[0]
    keys={x.tobytes() for x in d.nodes.numpy()};novel=[]
    for rec,o,dist in zip(state_records,observations,nearest):
        pose=np.array(rec['pose'])
        spatial=np.linalg.norm(original_task_poses[:,:2]-pose[:2],axis=1)
        angle=np.abs((original_task_poses[:,2]-pose[2]+np.pi)%(2*np.pi)-np.pi)
        nearby=spatial<=.25
        novel.append(dict(id=rec['id'],stratum=rec['stratum'],exact_replay_node=o.tobytes() in keys,
            nearest_original_position=float(dist),within_025_count=int(nearby.sum()),
            min_heading_error_within_025=float(angle[nearby].min()) if nearby.any() else None,
            nearest_joint_distance=float(np.sqrt(spatial**2+(.3*angle)**2).min())))
    write(out/'state_support.json',novel)
    write(out/'COMPLETE',dict(checkpoint_sha256=file_digest(root/'train/final.pt'),graph_evaluation_only=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
