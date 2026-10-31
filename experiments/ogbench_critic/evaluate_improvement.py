"""Standalone checkpoint evaluation using only the official actor and evaluator."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from common import ROOT,QRLAgent,load_data,tree_digest,sha256
from actor_recording import EpisodeRecorder
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics.core import save_integration
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
import numpy as np


def run(model,epoch,config_path,episodes,output,limit):
    started=time.monotonic();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);env=None
    def budget():
        if time.monotonic()-started>limit:raise TimeoutError('Standalone official evaluation deadline')
    try:
        assert episodes in [1,50] and 'hard_goal' not in sys.modules and 'roadmap_metric' not in sys.modules and 'data_graph' not in sys.modules
        config=json.loads(Path(config_path).read_text());template=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),config)
        agent=restore_agent(template,str(model),epoch);assert int(agent.network.opt_state[0].count)==epoch and int(agent.network.step)==epoch+1
        frozen=tree_digest(agent);env,_,_=load_data(config);np.random.seed(0);env.reset(seed=0);np.random.seed(0)
        recorded=EpisodeRecorder(env,check=budget,integration=save_integration);tasks={};cases={}
        for task in range(1,6):
            budget();offset=len(recorded.starts)
            stats,trajs,video=evaluate(agent,recorded,task_id=task,config=config,num_eval_episodes=episodes,num_video_episodes=0,eval_temperature=0,eval_gaussian=None)
            assert len(trajs)==episodes and not video
            for i,t in enumerate(trajs):
                name=f'task{task}_episode{i:03d}';ob=np.concatenate([np.asarray(t['observation'])[:1],np.asarray(t['next_observation'])]);actions=np.asarray(t['action'])
                assert 0<len(actions)<=1000 and t['done'][-1] and all(np.isfinite(x).all() for x in [ob,actions])
                np.savez_compressed(output/f'{name}.npz',observations=ob,actions=actions,goal=recorded.starts[offset+i]['goal'],rewards=np.asarray(t['reward']),done=np.asarray(t['done']),final_integration=recorded.ends[offset+i])
                cases[name]=dict(success=bool(t['info'][-1]['success']),steps=len(actions))
            tasks[str(task)]={k:float(v) for k,v in stats.items()};assert np.isclose(tasks[str(task)]['success'],np.mean([cases[f'task{task}_episode{i:03d}']['success'] for i in range(episodes)]))
            (output/'progress.json').write_text(json.dumps(dict(tasks=tasks,cases=cases),indent=2)+'\n')
        budget();assert tree_digest(agent)==frozen and 'hard_goal' not in sys.modules and 'roadmap_metric' not in sys.modules
        result=dict(complete=True,only_official_neural_actor=True,no_teacher_imports=True,episodes_per_task=episodes,tasks=tasks,cases=cases,
            checkpoint_sha256=sha256(Path(model)/f'params_{epoch}.pkl'),elapsed_seconds=time.monotonic()-started)
        (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    except BaseException as e:
        (output/'failure.json').write_text(json.dumps(dict(complete=False,error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n');raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--epoch',type=int,required=True);p.add_argument('--config',required=True)
    p.add_argument('--episodes',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--limit-seconds',type=float,required=True)
    a=p.parse_args();run(a.model,a.epoch,a.config,a.episodes,a.output,a.limit_seconds)
