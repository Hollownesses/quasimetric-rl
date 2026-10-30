"""Bounded read-only actor/Q/V comparison of the predeclared midpoint and endpoint."""
import argparse
import json
from pathlib import Path
import time

from diagnostics import core,load_checkpoint,value_components,aggregate,sha,tree_digest,UNIT
import jax
import numpy as np
from utils.env_utils import make_env_and_datasets

EPOCHS=(50000,100000)


def budget(deadline):
    if time.monotonic()>deadline:raise TimeoutError('360-second diagnostic budget exhausted; no continuation or retraining.')


def q_values(agent,observations,goals,actions):
    assert agent.config['use_action_for_distance']
    return aggregate(agent.get_distance(observations,goals,actions),'max')


def grid_q(agent,ob,goal):
    return q_values(agent,np.broadcast_to(ob,core.ACTIONS.shape),np.broadcast_to(goal,core.ACTIONS.shape),core.ACTIONS)


def run(root,output):
    start=time.monotonic();deadline=start+360;root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    old=json.loads((root/'mqe_100k_diagnostics/summary.json').read_text());checkpoint=Path(old['protocol']['checkpoint'])
    bank=root/'evaluation_bank.npz';assert sha(bank)==old['protocol']['bank_sha256'];cases=core.read_bank(bank);names=[k for k in cases if k.startswith('task3_')]
    assert len(cases)==25 and len(names)==5
    models={};metadata={}
    for epoch in EPOCHS:models[epoch],_,metadata[epoch]=load_checkpoint(checkpoint,epoch)
    assert metadata[100000]['checkpoint_sha256']==old['protocol']['checkpoint_sha256']
    frozen={e:tree_digest(a.network.params) for e,a in models.items()}
    protocol=dict(epochs=list(EPOCHS),formal_endpoint_unchanged=100000,time_limit_seconds=360,
        bank_sha256=sha(bank),models=metadata,source_sha256=sha(__file__),
        spec_sha256=sha(Path(__file__).parent/'TEMPORAL_PROTOCOL.md'),training_updates=0)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env,_,_=make_env_and_datasets('pointmaze-medium-navigate-v0');results={}
    for epoch,controller,selected in [(50000,'actor',list(cases)),(50000,'V-greedy',names),(50000,'Q-greedy',names),(100000,'Q-greedy',names)]:
        key=f'{epoch}_{controller}';directory=output/key;directory.mkdir();result={};agent=models[epoch]
        for name in selected:
            budget(deadline);case=cases[name];scores=[]
            if controller=='V-greedy':
                def score(obs,goals):
                    budget(deadline);return aggregate(value_components(agent,obs,goals),'max')
                info,traj=core.rollout(env,case,score=score)
            else:
                def actor(ob,goal,step):
                    budget(deadline)
                    if controller=='actor':return agent.sample_actions(ob,goal,seed=jax.random.PRNGKey(step),temperature=0)
                    values=grid_q(agent,ob,goal);index=int(np.argmin(values));scores.append(float(values[index]));return core.ACTIONS[index]
                info,traj=core.rollout(env,case,actor=actor)
            if scores:traj['q_selected']=np.asarray(scores)
            np.savez_compressed(directory/f'{name}.npz',**traj);result[name]=info
            (directory/'progress.json').write_text(json.dumps(result,indent=2)+'\n');print(key,name,info,flush=True)
        results[key]=result;(directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # Identical observation/goal probes; no simulation or actor/critic mixing in execution.
    probes={}
    for name in names:
        case=cases[name];trace=np.load(root/f'mqe_100k_diagnostics/actor_rollout/{name}.npz')
        assert len(trace['observations'])>100
        for tag,point in [('start',case['observation']),('actor100_step100',trace['observations'][100])]:
            probes[f'{name}_{tag}']=(point,case['goal'])
    scores={}
    for critic_epoch,agent in models.items():
        for actor_epoch,policy in models.items():
            row={}
            for name,(ob,goal) in probes.items():
                budget(deadline);actions=np.asarray(policy.sample_actions(ob,goal,seed=jax.random.PRNGKey(0),temperature=0))
                dq=float(q_values(agent,ob[None],goal[None],actions[None])[0]);grid=grid_q(agent,ob,goal)
                dv=float(aggregate(value_components(agent,ob[None],goal[None]),'max')[0])
                row[name]=dict(actor_q_distance=dq,grid_min_q_distance=float(grid.min()),actor_minus_grid=dq-float(grid.min()),
                    v_distance=dv,grid_min_q_minus_v=float(grid.min())-dv,actor_action=actions.tolist(),grid_action=core.ACTIONS[int(np.argmin(grid))].tolist())
            scores[f'critic{critic_epoch}_actor{actor_epoch}']=row
    walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2);fields={}
    for name in names:
        case=cases[name];by_epoch={}
        for epoch,agent in models.items():
            budget(deadline)
            if epoch==100000:
                with np.load(root/f'mqe_endpoint_analysis/{name}_field.npz') as data:
                    assert np.array_equal(data['nodes'],nodes) and np.array_equal(data['goal'],case['goal']);values=data['values'].copy()
            else:
                components=np.concatenate([np.asarray(value_components(agent,x,np.broadcast_to(case['goal'],x.shape))) for x in np.array_split(nodes,int(np.ceil(len(nodes)/1024)))],axis=1)
                values=aggregate(components,'max')
                np.savez_compressed(output/f'{name}_{epoch}_field.npz',nodes=nodes,values=values,goal=case['goal'],walls=walls)
            stats,_=core.graph_metrics(nodes,graph,values,goal=case['goal'],anchor=case['observation'],tol=1e-6*UNIT)
            by_epoch[str(epoch)]=stats
        fields[name]=by_epoch
    env.close();budget(deadline)
    assert all(tree_digest(a.network.params)==frozen[e] for e,a in models.items())
    assert all(sha(checkpoint/f'params_{e}.pkl')==metadata[e]['checkpoint_sha256'] for e in EPOCHS)
    result=dict(protocol=protocol,results=results,cross_scores=scores,start_fields=fields,complete=True,elapsed_seconds=time.monotonic()-start,
                existing_100k_actor=old['rollouts']['actor']['cases'],existing_100k_v_greedy=old['rollouts']['max']['cases'])
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',output,result['elapsed_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
