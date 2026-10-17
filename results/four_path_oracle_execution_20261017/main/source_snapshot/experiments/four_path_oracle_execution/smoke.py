"""Three real transitions per arm at the frozen development pose; no tuning."""
import tempfile
from pathlib import Path
from experiments.four_path_execution.common import load_data,torch,write
from experiments.four_path_oracle_execution.protocol import ARMS,controller_factory


def smoke():
    torch.set_num_threads(1);data=load_data();env=data.env;rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        make=controller_factory(Path(tmp))
        for arm in ARMS:
            obs,_=env.reset(seed=314159,options={'device_id':'u_trap_target','start':[4.,3.65,3.14159265]})
            control=make(data,arm);begin=control.begin_episode(env,env.abstract_goal_observation(),314159)
            steps=[]
            for _ in range(3):
                action,diag=control.act(obs,env);obs,reward,done,trunc,info=env.step(action)
                assert not done and not trunc
                assert diag['model_rollouts']==128
                steps.append(dict(action=action.tolist(),reward=reward,model_transitions=diag['model_transitions']))
            rows.append(dict(arm=arm,begin=begin,steps=steps,end=control.end_episode()))
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();write(args.output,smoke())
