"""Expert-designed controls; exact float32 shooting; all accepted edges use env.step.

No model, training, Oracle value or graph distance is used by this builder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs
from minimal_qrl.eval.comm_inspection_oracle_bank import environment_signature
from minimal_qrl.dataset import (_goal_set_transition_infos, _make_goal_set_abstract_edge,
                                 EXPLORE_OUTCOME_SUCCESS)
from minimal_qrl.path_support import replay_actions
from minimal_qrl.global_push_sampling import replay_fingerprint
from quasimetric_rl.data import EpisodeData, MultiEpisodeData

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / 'results/full_replay_four_path_20261017/main/scenario.json'
DEVICE = 'u_trap_target'
SEED = 20261019
EXIT = np.array([2.0, 3.65, 3.0], np.float32)
JOIN = np.array([8.4, 3.65, 3.0], np.float32)
FIELDS = ('episode_lengths', 'all_observations', 'actions', 'rewards', 'terminals',
          'timeouts', 'observation_infos', 'transition_infos')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_env(scenario):
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.set_task_by_device_id(DEVICE)
    return env


def angle_delta(a, b):
    return (a-b+np.pi) % (2*np.pi)-np.pi


def proposal_kinematics(start, headings, env):
    """Proposal only: reproduce float32 rounding, never insert these as data edges."""
    state = np.asarray(start, np.float32).copy()
    states, actions = [state.copy()], []
    for heading in headings:
        action = np.float32(angle_delta(heading, float(state[2])) / env.dt)
        if abs(action) > env.omega_max:
            raise ValueError('proposal exceeds turn bound')
        theta = angle_delta(float(state[2])+float(action)*env.dt, 0.)
        state = np.array([float(state[0])+env.v*env.dt*math.cos(theta),
                          float(state[1])+env.v*env.dt*math.sin(theta), theta], np.float32)
        states.append(state.copy()); actions.append([action])
    return np.asarray(states), np.asarray(actions, np.float32)


def execute_segment(env, start, actions, *, success=False):
    """No set_state, snapping, resets inside an edge sequence, or tolerance joins."""
    obs, _ = env.reset(seed=SEED, options={'device_id': DEVICE, 'start': start})
    states, observations, rewards = [env.state.copy()], [obs.copy()], []
    if env.is_task_feasible(env.state):
        raise ValueError('terminal segment start')
    for i, action in enumerate(actions):
        obs, reward, done, timeout, info = env.step(action)
        if (info['collision'] or info['out_of_bounds'] or timeout or
                (done and (not success or i != len(actions)-1))):
            raise ValueError('invalid/prematurely terminated segment')
        states.append(env.state.copy()); observations.append(obs.copy()); rewards.append(reward)
    if bool(done) != success or bool(info['success']) != success:
        raise ValueError('wrong segment terminal status')
    return dict(start=states[0].tolist(), states=np.asarray(states).tolist(),
                observations=np.asarray(observations).tolist(),
                actions=np.asarray(actions,np.float32).tolist(),
                rewards=np.asarray(rewards,np.float32).tolist())


def shooting(env, start, target, n):
    """Two heading basis corrections, finite differences through rounded dynamics.

    Matching is exact, including heading; numerical proximity is insufficient.
    This optimizes endpoint geometry only, not costs or value targets.
    """
    bearing = math.atan2(target[1]-start[1], target[0]-start[0])
    h = np.unwrap([start[2], bearing, target[2]])
    angles = np.interp(np.arange(1,n+1), [0,n*.3,n*.7,n], [h[0],h[1],h[1],h[2]])
    t = np.arange(1,n+1)/n
    basis = np.stack([np.sin(np.pi*t),np.sin(2*np.pi*t)],axis=1)
    basis[-1] = 0
    def run(z):
        return proposal_kinematics(start, angles+basis@z, env)
    def f(z):
        return run(z)[0][-1,:2].astype(float)-target[:2]
    z = np.zeros(2)
    for iteration in range(120):
        states, actions = run(z)
        if np.array_equal(states[-1], target):
            actual = execute_segment(env, start, actions)
            if not np.array_equal(np.asarray(actual['states'],np.float32)[-1],target):
                raise ValueError('proposal/environment mismatch')
            return actual, iteration+1
        error = states[-1,:2].astype(float)-target[:2]
        eps = .002
        jac = np.column_stack([(f(z+np.eye(2)[j]*eps)-f(z-np.eye(2)[j]*eps))/(2*eps)
                               for j in range(2)])
        delta = np.linalg.lstsq(jac,-error,rcond=None)[0]
        z += np.clip(delta,-.2,.2)
    raise ValueError('no bit-exact endpoint within shooting budget')


def connector(env,start,target,lengths,attempts,name):
    for n in lengths:
        try:
            record, iterations = shooting(env,np.asarray(start,np.float32),target,n)
            attempts.append(dict(name=name,steps=n,accepted=True,iterations=iterations))
            return record
        except ValueError as exc:
            attempts.append(dict(name=name,steps=n,accepted=False,reason=str(exc)))
    raise RuntimeError(f'Cannot construct {name}; retain attempts and inspect geometry')


def guided(env,start,points):
    env.reset(seed=SEED,options={'device_id':DEVICE,'start':start})
    actions=[]
    for x,y in points:
        for _ in range(150):
            if np.linalg.norm(env.state[:2]-[x,y]) < .14:
                break
            err=angle_delta(math.atan2(y-env.state[1],x-env.state[0]),env.state[2])
            action=np.array([np.clip(3*err,-2.6,2.6)],np.float32)
            _,_,done,timeout,_=env.step(action)
            if done or timeout:
                raise ValueError('guide failed')
            actions.append(action)
        else:
            raise ValueError('guide waypoint budget exhausted')
    return execute_segment(env,start,np.asarray(actions))


def terminal_tail(env,sign):
    env.reset(seed=SEED,options={'device_id':DEVICE,'start':JOIN})
    actions=[]
    for t in range(40):
        err=angle_delta(math.atan2(3.65-env.state[1],6.7-env.state[0]),env.state[2])
        omega=sign*.65 if t < 4 else np.clip(3*err,-2.6,2.6)
        action=np.array([omega],np.float32)
        _,_,done,timeout,info=env.step(action);actions.append(action)
        if done:
            if not info['success']:raise ValueError('tail failed')
            return execute_segment(env,JOIN,np.asarray(actions),success=True)
        if timeout:break
    raise ValueError('tail did not succeed')


def episode_pair(env, trajectory, context):
    obs=np.asarray(trajectory['observations'],np.float32)
    actions=np.asarray(trajectory['actions'],np.float32);n=len(actions)
    goal=env.abstract_goal_observation().astype(np.float32)
    infos=_goal_set_transition_infos(n=n,task_goal_obs=goal,context_id=context,env=env,
        teacher_guided=True,planner_path_support=True,task_success_episode=True,
        exploration_outcome=EXPLORE_OUTCOME_SUCCESS,include_global_push_pairs=False)
    infos['goal_return_mask'][:]=False
    terminals=np.zeros(n,bool);terminals[-1]=True
    physical=EpisodeData.from_simple_trajectory(observations=obs[:-1],next_observations=obs[1:],
        actions=actions,rewards=np.asarray(trajectory['rewards'],np.float32),terminals=terminals,
        timeouts=np.zeros(n,bool),transition_infos=infos)
    anchor=_make_goal_set_abstract_edge(env,terminal_obs=obs[-1],task_goal_obs=goal,
        context_id=context,action_template=actions[-1],action_dtype=np.float32)
    anchor.transition_infos['teacher_guided'].fill_(True)
    anchor.transition_infos['planner_path_support'].fill_(True)
    return [physical,anchor]


def require_exact_join(left, right):
    for field in ('states', 'observations'):
        if not np.array_equal(np.asarray(left[field][-1],np.float32),
                              np.asarray(right[field][0],np.float32)):
            raise ValueError(f'non-exact segment join: {field}')


def build(output, scenario_path=SCENARIO):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    scenario=json.loads(Path(scenario_path).read_text());write_json(output/'scenario.json',scenario)
    env=make_env(scenario);attempts=[];segments={}
    try:
        # Four distinct routes at exactly the same full physical exit state.
        for name,y,via in [('upper_inner',6.4,4.4),('upper_outer',7.0,4.4),
                           ('lower_inner',.9,2.9),('lower_outer',.45,2.9)]:
            final_y=5.3 if name.startswith('upper') else 2.0
            guide=guided(env,EXIT,[(1.4,via),(1.6,y),(7.9,y),(9.2,final_y)])
            end=np.asarray(guide['states'][-1],np.float32)
            extra=connector(env,end,JOIN,range(32,49,2),attempts,name)
            actions=guide['actions']+extra['actions']
            segments[name]=execute_segment(env,EXIT,actions)
            print(name,len(actions),flush=True)
        segments['tail_left']=terminal_tail(env,-1)
        segments['tail_right']=terminal_tail(env,1)
        # 64 independent physical prefixes, all ending at EXIT exactly.
        for iy,y in enumerate(np.linspace(2.55,4.75,8)):
            for ix,x in enumerate(np.linspace(3.7,4.85,8)):
                name=f'prefix_{iy*8+ix:02d}'
                heading=(3.0 if y < 3.65 else -3.0)+(ix%3-1)*.12
                start=np.array([x,y,heading],np.float32)
                base=int(np.ceil(np.linalg.norm(start[:2]-EXIT[:2])/(env.v*env.dt)))
                segments[name]=connector(env,start,EXIT,range(base+3,base+38),attempts,name)
        write_json(output/'segments.json',segments)
    finally:
        write_json(output/'construction_attempts.json',attempts)
    names=['upper_inner','upper_outer','lower_inner','lower_outer']
    episodes=[];records=[]
    for i in range(64):
        route=names[i%4];tail=['tail_left','tail_right'][(i//4)%2]
        path=[f'prefix_{i:02d}',route,tail]
        for left,right in zip(path[:-1],path[1:]):
            require_exact_join(segments[left],segments[right])
        actions=sum([segments[name]['actions'] for name in path],[])
        start=segments[path[0]]['start']
        # Full uninterrupted execution verifies joins and unchanged 300-step limit.
        trajectory=replay_actions(env,start,actions,device_id=DEVICE,seed=SEED+i)
        expected=np.concatenate([np.asarray(segments[name]['states'],np.float32)[:-1] for name in path]
                                +[np.asarray(segments[path[-1]]['states'],np.float32)[-1:]])
        assert np.array_equal(expected,np.asarray(trajectory['states'],np.float32))
        episodes.extend(episode_pair(env,trajectory,i))
        records.append(dict(context=i,seed=SEED+i,segments=path,steps=len(actions)))
    raw=MultiEpisodeData.cat(episodes)
    # Normal replay schema only. Graph labels are saved later by a separate auditor.
    torch.save(dict(schema_version=1,sha256=replay_fingerprint(raw),
                    raw={name:getattr(raw,name) for name in FIELDS}),output/'verified_replay.pt')
    write_json(output/'trajectories.json',records)
    dependencies=[SCENARIO,ROOT/'experiments/four_path_factorial/PROTOCOL.md',
        ROOT/'experiments/full_replay_four_path/README.md',ROOT/'experiments/witnessed_task_push/README.md',
        ROOT/'experiments/large_phr_calibration/README.md',
        ROOT/'results/witnessed_task_push_20261019/main/comparison/report.md',
        ROOT/'results/large_phr_calibration_20261019/main/comparison/report.md',
        ROOT/'minimal_qrl/envs/comm_inspection_dubins_uav_2d.py',ROOT/'minimal_qrl/dataset.py',
        ROOT/'minimal_qrl/global_push_sampling.py',ROOT/'minimal_qrl/path_support.py',
        ROOT/'minimal_qrl/eval/task_path_support_audit.py',
        ROOT/'quasimetric_rl/modules/quasimetric_critic/models/encoder.py',
        ROOT/'minimal_qrl/baselines/mppi.py',ROOT/'minimal_qrl/industry_exp/scalability_scenarios.py',
        ROOT/'experiments/four_path_factorial/run.py']
    snapshot=output/'source_snapshot';snapshot.mkdir()
    sources=dependencies+list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.md'))
    hashes={}
    for source in sources:
        relative=source.resolve().relative_to(ROOT)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);hashes[str(relative)]=sha256(source)
    manifest=dict(schema_version=1,expert_free=False,expert_source='assistant-authored explicit waypoints and controls + endpoint-only float32 shooting; no human demonstrations',
        training_started=False,oracle_or_shortest_path_labels_in_replay=False,seed=SEED,
        environment_signature=environment_signature(env),python=platform.python_version(),numpy=np.__version__,
        torch=torch.__version__,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources=hashes,shared_states=dict(exit=EXIT.tolist(),east_join=JOIN.tolist()),
        replay_sha256=replay_fingerprint(raw),
        files={p.name:sha256(p) for p in output.iterdir() if p.is_file()},
        graph_identity='exact full float32 observation + fixed task; no rounded or spatial edges',
        trajectory_design='64 prefixes x 4 main routes x 2 tails; 64 recorded combinations; 512 possible recombinations')
    write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(output=str(output),physical=sum(r['steps'] for r in records),
                          replay_sha256=manifest['replay_sha256'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--scenario',default=str(SCENARIO))
    a=p.parse_args();build(a.output,a.scenario)
