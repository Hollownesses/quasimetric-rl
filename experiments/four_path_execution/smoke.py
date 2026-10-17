"""Three real steps per arm on a development pose; never touches holdouts."""
import argparse
import time
from experiments.four_path_execution.common import *


def smoke(output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);data=load_data();rows=[]
    for arm in ARMS:
        env=data.env;obs,_=env.reset(seed=9001,options={'device_id':'u_trap_target','start':[4.,3.65,np.pi]})
        control=controller(data,arm);control.begin_episode(env,env.abstract_goal_observation(),9001)
        times=[]
        for step in range(3):
            action,diag=control.act(obs,env);times.append(diag['planning_time_sec'])
            assert np.isfinite(action).all() and abs(action[0])<=env.omega_max
            assert diag['model_rollouts']==256
            obs,_,done,timeout,_=env.step(action)
            if done or timeout:break
        rows.append(dict(arm=arm,steps=step+1,seconds_per_plan=float(np.mean(times))))
    write(output/'smoke.json',rows);print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);smoke(Path(p.parse_args().output))
