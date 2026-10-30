"""Fresh-process official 50/task actor evaluator plus the existing fixed bank."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from common import ROOT,UPSTREAM,load_base,load_data,tree_digest,sha256
from actor_recording import EpisodeRecorder
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_critic.evaluate import rollout
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
import numpy as np


def run(root,checkpoint,epoch,critic_digest,output,limit):
    started=time.monotonic();deadline=started+limit;root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);env=None
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('Fixed extraction evaluation deadline exhausted')
    try:
        assert 'roadmap_metric' not in sys.modules and 'hard_goal' not in sys.modules and 'data_graph' not in sys.modules
        base,config=load_base();agent=restore_agent(base,str(checkpoint),epoch)
        assert int(agent.network.opt_state[0].count)==epoch and int(agent.network.step)==epoch+1
        assert tree_digest(agent.network.params['modules_value'])==critic_digest
        digest=tree_digest(agent);env,_,_=load_data(config)
        assert len(env.unwrapped.task_infos)==5
        np.random.seed(0);env.reset(seed=0);np.random.seed(0)
        def progress(count):
            (output/'official_progress.json').write_text(json.dumps(dict(completed_episodes=count,elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
        recorded=EpisodeRecorder(env,check=budget,integration=core.save_integration,on_end=progress)
        official=output/'official';official.mkdir();tasks={};episodes={}
        for task in range(1,6):
            budget();offset=len(recorded.starts)
            stats,trajs,renders=evaluate(agent,recorded,task_id=task,config=config,num_eval_episodes=50,num_video_episodes=0,eval_temperature=0,eval_gaussian=None)
            assert len(trajs)==50 and len(renders)==0 and len(recorded.starts)==len(recorded.ends)==offset+50
            for i,traj in enumerate(trajs):
                name=f'task{task}_episode{i:03d}';start=recorded.starts[offset+i];end=recorded.ends[offset+i]
                observations=np.concatenate([np.asarray(traj['observation'])[:1],np.asarray(traj['next_observation'])])
                actions=np.asarray(traj['action']);steps=len(actions);last=traj['info'][-1]
                assert 0<steps<=1000 and len(traj['done'])==steps and traj['done'][-1]
                np.testing.assert_array_equal(observations[0],start['observation'])
                assert np.isfinite(observations).all() and np.isfinite(actions).all()
                np.savez_compressed(official/f'{name}.npz',observations=observations,actions=actions,goal=start['goal'],rewards=np.asarray(traj['reward']),done=np.asarray(traj['done']),final_integration=end)
                episodes[name]=dict(success=bool(last['success']),steps=steps,task=task,episode=i,initial_xy=observations[0].tolist(),goal=start['goal'].tolist(),final_xy=observations[-1].tolist())
            successes=sum(v['success'] for v in episodes.values() if v['task']==task)
            assert np.isclose(float(stats['success']),successes/50)
            tasks[str(task)]=dict(successes=successes,episodes=50,success_rate=successes/50,official_statistics={k:float(v) for k,v in stats.items()})
            (official/'summary.json').write_text(json.dumps(dict(tasks=tasks,episodes=episodes),indent=2)+'\n');print('OFFICIAL_TASK',task,tasks[str(task)],flush=True)
        budget();fixed_dir=output/'fixed_bank';fixed_dir.mkdir();fixed={};guard=EpisodeRecorder(env,check=budget)
        for name,case in core.read_bank(root/'evaluation_bank.npz').items():
            budget();info,trace=rollout(agent,guard,case,'actor');trace['final_integration']=core.save_integration(env)
            np.savez_compressed(fixed_dir/f'{name}.npz',**trace);fixed[name]=info
            (fixed_dir/'summary.json').write_text(json.dumps(dict(cases=fixed),indent=2)+'\n');print('FIXED_CASE',name,info,flush=True)
        budget();assert tree_digest(agent)==digest
        assert 'roadmap_metric' not in sys.modules and 'hard_goal' not in sys.modules and 'data_graph' not in sys.modules
        a=dict(complete=True,epoch=epoch,checkpoint_sha256=sha256(Path(checkpoint)/f'params_{epoch}.pkl'),critic_digest=critic_digest,
            official=tasks,official_episodes=episodes,fixed_cases=fixed,only_neural_actor=True,teacher_modules_imported=False,
            official_evaluation_sha256=sha256(UPSTREAM/'impls/utils/evaluation.py'),official_qrl_sha256=sha256(UPSTREAM/'impls/agents/qrl.py'),
            evaluation_rng=0,bank_sha256=sha256(root/'evaluation_bank.npz'),elapsed_seconds=time.monotonic()-started,
            source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ['evaluate_extraction.py','actor_recording.py','common.py']})
        (output/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');print('EVALUATION_COMPLETE',a['elapsed_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(complete=False,error=repr(error),elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()),indent=2)+'\n');raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--checkpoint',required=True);p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--critic-digest',required=True);p.add_argument('--output',required=True);p.add_argument('--limit-seconds',type=float,required=True)
    a=p.parse_args();run(a.root,a.checkpoint,a.epoch,a.critic_digest,a.output,a.limit_seconds)
