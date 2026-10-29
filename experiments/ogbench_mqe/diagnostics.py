"""Native MQE inference with the same fixed true-successor/graph diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from runtime import prepare,COMMIT,BENCHMARK_COMMIT
prepare()
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.ogbench_diagnostics import core
import jax
import numpy as np
from agents.mqe import MQEAgent
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent

MODES=('ensemble0','ensemble1','max')
UNIT=-np.log(.995)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(tree):
    h=hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a=np.asarray(x);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()


@jax.jit
def value_components(agent,observations,goals):
    psi=agent.network.select('psi')(observations);target=agent.network.select('psi')(goals)
    return agent.distance(psi,target)


def aggregate(values,mode):
    values=np.asarray(values);assert values.ndim==2 and values.shape[0]==2
    assert np.isfinite(values).all()
    if mode=='max':return values.max(axis=0)
    if mode=='ensemble0':return values[0]
    if mode=='ensemble1':return values[1]
    raise ValueError(mode)


def load_checkpoint(checkpoint,epoch):
    checkpoint=Path(checkpoint).resolve();flags=json.loads((checkpoint/'flags.json').read_text());cfg=flags['agent']
    assert cfg['agent_name']=='mqe' and cfg['discount']==.995
    for key in ['actor_hidden_dims','value_hidden_dims']:cfg[key]=tuple(cfg[key])
    dummy=np.zeros((1,2),np.float32);template=MQEAgent.create(flags['seed'],dummy,dummy,cfg)
    agent=restore_agent(template,str(checkpoint),epoch)
    assert int(agent.network.opt_state[0].count)==epoch and int(agent.network.step)==epoch+1
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
    metadata=dict(checkpoint=str(checkpoint),epoch=epoch,checkpoint_kind='native-mqe',
        checkpoint_sha256=sha(checkpoint/f'params_{epoch}.pkl'),network_digest=tree_digest(agent.network.params),
        psi_digest=tree_digest(agent.network.params['modules_psi']),
        phi_digest=tree_digest(agent.network.params['modules_phi']),mqe_commit=COMMIT,benchmark_commit=BENCHMARK_COMMIT,
        unit=UNIT,distance_unit='log-discount',primary_value_aggregation='max',
        note='State-state V from native psi; actor remains native. No Q-based action search used.')
    return agent,cfg,metadata


def run(checkpoint,epoch,bank,output,smoke=False):
    assert epoch in ([2] if smoke else [100000,1000000]),'Only fixed declared endpoints or the explicit two-step smoke.'
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    agent,cfg,metadata=load_checkpoint(checkpoint,epoch);bank=Path(bank).resolve();bank_sha=sha(bank)
    cases=core.read_bank(bank);assert len(cases)==25
    if smoke:cases={name:c for name,c in cases.items() if name=='task3_seed3100'}
    env,_,_=make_env_and_datasets('pointmaze-medium-navigate-v0')
    manifest=dict(**metadata,bank_sha256=bank_sha,action_grid_sha256=tree_digest(core.ACTIONS),
        purpose='I/O smoke; not learning performance' if smoke else 'Fixed endpoint method reference',
        cases=list(cases),source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        core_sha256=sha(core.__file__),protocol_sha256=sha(Path(__file__).parent/'PROTOCOL.md'),
        devices=[str(d) for d in jax.devices()])
    (output/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results={}
    for mode in [*MODES,'actor']:
        directory=output/f'{mode}_rollout';directory.mkdir();result=dict(**metadata,bank_sha256=bank_sha,
            action_grid_sha256=tree_digest(core.ACTIONS),controller='native-actor' if mode=='actor' else 'real-greedy',
            aggregation=None if mode=='actor' else mode,cases={},tasks={})
        for name,case in cases.items():
            if mode=='actor':
                info,traj=core.rollout(env,case,actor=lambda ob,g,step:agent.sample_actions(ob,g,seed=jax.random.PRNGKey(step),temperature=0))
            else:
                info,traj=core.rollout(env,case,score=lambda ob,g:aggregate(value_components(agent,ob,g),mode))
            np.savez_compressed(directory/f'{name}.npz',**traj);result['cases'][name]=info
            (directory/'progress.json').write_text(json.dumps(result,indent=2)+'\n');print(mode,name,info,flush=True)
        for task in range(1,6):
            entries=[v for k,v in result['cases'].items() if k.startswith(f'task{task}_')]
            if entries:result['tasks'][f'task{task}']=dict(successes=sum(v['success'] for v in entries),episodes=len(entries),mean_steps=float(np.mean([v['steps'] for v in entries])))
        (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n');results[mode]=result
    walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2)
    components=np.concatenate([np.asarray(value_components(agent,chunk,np.broadcast_to(core.GOAL,chunk.shape)))
        for chunk in np.array_split(nodes,int(np.ceil(len(nodes)/1024)))],axis=1)
    assert components.shape==(2,len(nodes)) and np.isfinite(components).all()
    fields={}
    for mode in MODES:
        directory=output/f'{mode}_field';directory.mkdir();values=aggregate(components,mode)
        stats,data=core.graph_metrics(nodes,graph,values,tol=1e-6*UNIT)
        replay,trace=core.replay_reference(env,data['path']);data['walls']=walls;data['component_values']=components
        np.savez_compressed(directory/'field.npz',**data)
        np.savez_compressed(directory/'reference_replay.npz',observations=trace,goal=core.GOAL)
        stats.update(**metadata,**replay,spacing=.2,radius=radius,aggregation=mode,
                     descent_tolerance=1e-6*UNIT,goal=core.GOAL.tolist(),anchor=core.ANCHOR.tolist())
        (directory/'summary.json').write_text(json.dumps(stats,indent=2)+'\n');fields[mode]=stats
    assert sha(bank)==bank_sha
    assert sha(Path(checkpoint)/f'params_{epoch}.pkl')==metadata['checkpoint_sha256']
    env.close();summary=dict(protocol=manifest,rollouts=results,fields=fields,complete=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',output,flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--bank',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.checkpoint,a.epoch,a.bank,a.output,a.smoke)
