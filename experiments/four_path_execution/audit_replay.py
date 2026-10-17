"""Replay every saved real action sequence in the unchanged environment."""
import argparse
import time
from experiments.four_path_execution.common import *


def audit(root):
    output=root/'replay_audit.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);data=load_data();env=data.env;counts={};started=time.perf_counter()
    for arm in ARMS:
        directory=root/'arms'/arm;rows=[json.loads(x) for x in (directory/'episodes.jsonl').read_text().splitlines()]
        steps=0;state_error=0.;reward_error=0.
        for row in rows:
            stored=np.load(directory/f"{row['state_id']}_{row['planner_seed']}.npz")
            env.reset(seed=row['planner_seed'],options={'device_id':'u_trap_target','start':row['pose']})
            for i,action in enumerate(stored['actions']):
                _,reward,done,truncated,info=env.step(action);steps+=1
                state_error=max(state_error,float(np.abs(env.state-stored['states'][i+1]).max()))
                reward_error=max(reward_error,abs(float(reward)-float(stored['rewards'][i])))
                assert not(done or truncated) or i==len(stored['actions'])-1
            assert done or truncated
            assert bool(info['success'])==row['success'] and bool(info['collision'])==row['collision']
            assert bool(info['out_of_bounds'])==row['out_of_bounds'] and bool(truncated and not info['success'])==row['timeout']
            if row['success']:assert env.is_task_feasible(env.state)
        assert state_error==0. and reward_error==0.
        counts[arm]=dict(episodes=len(rows),real_steps_replayed=steps,max_state_error=state_error,max_reward_error=reward_error)
        print(json.dumps(dict(arm=arm,**counts[arm])),flush=True)
    write(output,dict(arms=counts,seconds=time.perf_counter()-started,episodes=sum(r['episodes'] for r in counts.values()),
                     total_real_steps=sum(r['real_steps_replayed'] for r in counts.values())))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);audit(Path(p.parse_args().root))
