"""Post-hoc fixed-start/end one-step teacher/student comparison; no episodes."""
import argparse
import json
from pathlib import Path
import sys
import time

from common import ROOT,load_base,load_data,distances,sha256
from roadmap_metric import RoadmapMetric
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
from utils.flax_utils import restore_agent
import numpy as np
from scipy.stats import spearmanr


def run(root,output):
    started=time.monotonic();root=Path(root).resolve();directory=root/'iqe_distillation_1000'
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=json.loads((directory/'analysis.json').read_text());assert result['complete']
    base,config=load_base();agent=restore_agent(base,str(directory/'model'),101000)
    assert sha256(directory/'model/params_101000.pkl')==result['training']['checkpoint_sha256']
    teacher=RoadmapMetric.load(root/'roadmap_metric/graph.npz',root/'roadmap_metric/distances.npy')
    env,_,_=load_data(config);engine=FastPointmaze(env,nthread=4);bank=core.read_bank(root/'evaluation_bank.npz');probes={}
    try:
        for name,case in bank.items():
            if not name.startswith('task3_'):continue
            with np.load(directory/'neural_evaluation'/f'{name}.npz') as trace:
                end={**case,'observation':trace['observations'][-1],'integration':trace['final_integration']}
            for label,position in [('start',case),('end',end)]:
                assert time.monotonic()-started<30,'30-second probe budget exhausted'
                core.load_case(env,position);next_obs,_=engine.candidates(env);goals=np.broadcast_to(case['goal'],next_obs.shape)
                t=teacher(next_obs,goals);s=np.asarray(distances(agent,next_obs,goals))
                ti,si=int(np.argmin(t)),int(np.argmin(s));key=f'{name}_{label}'
                current_t=float(teacher(position['observation'][None],case['goal'][None])[0])
                current_s=float(np.asarray(distances(agent,position['observation'][None],case['goal'][None]))[0])
                probes[key]=dict(student_episode_success=result['neural']['cases'][name]['success'],
                    rmse=float(np.sqrt(np.square(s-t).mean())),mean_error=float((s-t).mean()),
                    spearman=float(spearmanr(s,t).statistic),teacher_best_action=core.ACTIONS[ti].tolist(),
                    student_best_action=core.ACTIONS[si].tolist(),teacher_regret_of_student=float(t[si]-t[ti]),
                    teacher_improvement_best=current_t-float(t[ti]),teacher_improvement_student=current_t-float(t[si]),
                    student_improvement_selected=current_s-float(s[si]),
                    teacher_best_displacement=float(np.max(np.abs(next_obs[ti]-position['observation']))),
                    student_best_displacement=float(np.max(np.abs(next_obs[si]-position['observation']))))
                np.savez_compressed(output/f'{key}.npz',observations=next_obs,teacher=t,student=s,actions=core.ACTIONS)
                print(key,probes[key],flush=True)
        (output/'analysis.json').write_text(json.dumps(dict(probes=probes,post_hoc=True,new_episodes=0,
            seconds=time.monotonic()-started,source_sha256=sha256(__file__)),indent=2)+'\n')
    finally:engine.close();env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
