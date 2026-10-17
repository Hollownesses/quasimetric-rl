"""Closed-loop execution of an immutable arm/state/RNG matrix; no training."""
import argparse
import time

from experiments.four_path_execution.common import *
from minimal_qrl.baselines.base import rollout_controller_episode


def evaluate_arm(root,arm):
    torch.set_num_threads(1);manifest=read(root/'manifest.json')
    assert arm in manifest['arms'] and asdict(CONFIG)==manifest['config']
    for name,sha in manifest['sources'].items():assert file_digest(ROOT/name)==sha,name
    assert file_digest(root/'states.json')==manifest['states_sha256']
    if arm in manifest['checkpoints']:assert file_digest(checkpoint(arm))==manifest['checkpoints'][arm]['sha256']
    out=root/'arms'/arm;out.mkdir(parents=True,exist_ok=False)
    data=load_data();env=data.env;control=controller(data,arm)
    before=tensor_digest(*control.qrl_agent.state_dict().values()) if control.qrl_agent else None
    started=time.perf_counter();records=read(root/'states.json')['records']
    # Warm inference on a training state, not a held-out outcome.
    if control.qrl_agent:
        env.reset(seed=0,options={'device_id':'u_trap_target','start':records[0]['pose']})
        obs=env.state_to_observation(env.state);goal=env.abstract_goal_observation()
        control.qrl_agent.batch_value(np.repeat(obs[None],256,axis=0),np.repeat(goal[None],256,axis=0))
    with (out/'episodes.jsonl').open('w') as stream:
        for record in records:
            for planner_seed in manifest['planner_seeds']:
                rollout=rollout_controller_episode(control,env,episode_seed=planner_seed,
                    device_id='u_trap_target',start_state=np.asarray(record['pose'],dtype=np.float32))
                assert np.array_equal(rollout['states'][0],np.asarray(record['pose'],dtype=np.float32))
                if rollout['success']:assert env.is_task_feasible(env.state) and not rollout['collision'] and not rollout['out_of_bounds']
                diag=rollout['action_diagnostics'];count=rollout['num_steps']
                assert sum(r['model_rollouts'] for r in diag)==count*CONFIG.num_samples
                row=dict(arm=arm,state_id=record['id'],stratum=record['stratum'],planner_seed=planner_seed,
                    pose=record['pose'],success=rollout['success'],collision=rollout['collision'],
                    out_of_bounds=rollout['out_of_bounds'],timeout=rollout['truncated'] and not rollout['success'],
                    cost=float(-sum(rollout['rewards'])),steps=count,**stage_metrics(rollout['states'],record),
                    planning_seconds=rollout['controller_diagnostics']['total_planning_time_sec'],
                    planning_p95_seconds=rollout['controller_diagnostics']['planning_time_p95_sec'],
                    model_rollouts=sum(r['model_rollouts'] for r in diag),model_transitions=sum(r['model_transitions'] for r in diag))
                assert sum(row[k] for k in ('success','collision','out_of_bounds','timeout'))==1
                np.savez_compressed(out/f"{record['id']}_{planner_seed}.npz",states=rollout['states'],
                    actions=rollout['actions'],rewards=rollout['rewards'],
                    planning_seconds=[d['planning_time_sec'] for d in diag],
                    model_transitions=[d['model_transitions'] for d in diag],
                    invalid_candidate_fraction=[d['invalid_rollout_ratio'] for d in diag])
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps(dict(arm=arm,state=record['id'],planner_seed=planner_seed,success=row['success'],steps=count,elapsed=time.perf_counter()-started)),flush=True)
    if control.qrl_agent:assert tensor_digest(*control.qrl_agent.state_dict().values())==before
    if arm in manifest['checkpoints']:assert file_digest(checkpoint(arm))==manifest['checkpoints'][arm]['sha256']
    write(out/'COMPLETE',dict(seconds=time.perf_counter()-started,episodes=64,parameters_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',required=True)
    args=p.parse_args();evaluate_arm(Path(args.root),args.arm)
