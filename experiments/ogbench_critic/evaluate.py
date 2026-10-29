"""Fixed true-successor greedy evaluation, with exact in-episode state restore."""
import argparse
import copy
import json
from pathlib import Path
import time

from common import load_base, load_data, distances, tree_digest, sha256
import jax
import mujoco
import numpy as np
from utils.flax_utils import restore_agent

STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION
ACTIONS=np.asarray([(x,y) for x in np.linspace(-1,1,11) for y in np.linspace(-1,1,11)],np.float64)


def wrappers(env):
    layers=[]
    while True:
        if '_elapsed_steps' in env.__dict__:
            layers.append(env)
        if not hasattr(env,'env'):
            return layers
        env=env.env


def capture(env):
    return copy.copy(env.unwrapped.data),[w._elapsed_steps for w in wrappers(env)]


def restore(env,state):
    mujoco.mj_copyData(env.unwrapped.data,env.unwrapped.model,state[0])
    for w,n in zip(wrappers(env),state[1]):
        w._elapsed_steps=n


def save_integration(env):
    state=np.empty(mujoco.mj_stateSize(env.unwrapped.model,STATE_SPEC))
    mujoco.mj_getState(env.unwrapped.model,env.unwrapped.data,state,STATE_SPEC)
    return state


def load_case(env,case):
    env.reset(seed=int(case['seed']),options=dict(task_id=int(case['task'])))
    env.unwrapped.set_goal(goal_xy=case['goal'])
    mujoco.mj_setState(env.unwrapped.model,env.unwrapped.data,case['integration'],STATE_SPEC)
    mujoco.mj_forward(env.unwrapped.model,env.unwrapped.data)
    for w in wrappers(env): w._elapsed_steps=0
    assert np.max(np.abs(env.unwrapped.get_ob()-case['observation']))<1e-10


def make_bank(path):
    base,config=load_base()
    env,_,_=load_data(config)
    cases={}
    for task in range(1,6):
        for seed in range(3100,3105):
            np.random.seed(seed)
            env.action_space.seed(seed)
            ob,info=env.reset(seed=seed,options=dict(task_id=task))
            name=f'task{task}_seed{seed}'
            cases[name]=dict(seed=seed,task=task,observation=ob.copy(),goal=info['goal'].copy(),integration=save_integration(env))
    np.savez(path,**{f'{name}__{key}':value for name,c in cases.items() for key,value in c.items()})
    env.close()
    return cases


def read_bank(path):
    data=np.load(path)
    names=sorted({k.split('__')[0] for k in data.files})
    return {name:{k.split('__')[1]:data[k] for k in data.files if k.startswith(name+'__')} for name in names}


def candidates(env,actions=ACTIONS):
    state=capture(env)
    observations=[]
    for action in actions:
        restore(env,state)
        observations.append(env.step(action)[0])
    restore(env,state)
    return np.array(observations),state


def rollout(agent,env,case,controller):
    load_case(env,case)
    goal=case['goal']
    positions=[env.unwrapped.get_ob().copy()]
    actions=[]
    goal_distances=[]
    success=False
    start=time.time()
    for step in range(1000):
        if controller=='real-greedy':
            next_obs,state=candidates(env)
            scores=np.asarray(distances(agent,next_obs,np.broadcast_to(goal,next_obs.shape)))
            index=int(np.argmin(scores))
            action=ACTIONS[index]
            obs,reward,term,trunc,info=env.step(action)
            assert np.array_equal(obs,next_obs[index]), 'State restoration changed successor'
            goal_distances.append(float(scores[index]))
        elif controller=='actor':
            action=np.asarray(agent.sample_actions(positions[-1],goal,seed=jax.random.PRNGKey(step),temperature=0))
            obs,reward,term,trunc,info=env.step(action)
        else:
            raise ValueError(controller)
        actions.append(action.copy())
        positions.append(obs.copy())
        if term or trunc:
            success=bool(info['success'])
            break
    assert term or trunc
    return dict(success=success,steps=len(actions),seconds=time.time()-start,
        final_xy=positions[-1].tolist(),max_last100_displacement=float(np.ptp(np.asarray(positions[-100:]),axis=0).max())),dict(
        observations=np.asarray(positions),actions=np.asarray(actions),goal=np.asarray(goal),distances=np.asarray(goal_distances))


def run(bank,output,checkpoint=None,epoch=None,controller='real-greedy'):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    agent,config=load_base()
    if checkpoint:
        agent=restore_agent(agent,str(checkpoint),epoch)
    env,_,_=load_data(config)
    cases=read_bank(bank)
    result=dict(bank_sha256=sha256(bank),controller=controller,action_grid_sha256=tree_digest(ACTIONS),
                checkpoint=str(checkpoint) if checkpoint else 'official100k',cases={},tasks={})
    for name,case in cases.items():
        info,traj=rollout(agent,env,case,controller)
        np.savez_compressed(output/f'{name}.npz',**traj)
        result['cases'][name]=info
        (output/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,info,flush=True)
    for task in range(1,6):
        values=[v for k,v in result['cases'].items() if k.startswith(f'task{task}_')]
        result['tasks'][f'task{task}']=dict(successes=sum(v['success'] for v in values),episodes=len(values),
            mean_steps=float(np.mean([v['steps'] for v in values])))
    env.close()
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bank',required=True)
    p.add_argument('--make-bank',action='store_true')
    p.add_argument('--output')
    p.add_argument('--checkpoint')
    p.add_argument('--epoch',type=int)
    p.add_argument('--controller',choices=['actor','real-greedy'],default='real-greedy')
    args=p.parse_args()
    if args.make_bank:
        assert not Path(args.bank).exists()
        make_bank(args.bank)
    if args.output:
        run(args.bank,args.output,args.checkpoint,args.epoch,args.controller)
