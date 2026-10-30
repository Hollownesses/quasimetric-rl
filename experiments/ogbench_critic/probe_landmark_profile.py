"""Post-hoc all failed endpoints, no new episodes or network training."""
import argparse
import json
from pathlib import Path
import sys
import time
from common import ROOT,BASE,load_data,sha256
from roadmap_metric import RoadmapMetric
from landmark_profile import LandmarkProfile
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import numpy as np


def run(root,output):
    started=time.monotonic();root=Path(root).resolve();directory=root/'landmark_profile';output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=json.loads((directory/'analysis.json').read_text());assert result['complete']
    assert sha256(directory/'profile.npz')==result['protocol']['profile_sha256']
    teacher=RoadmapMetric.load(root/'roadmap_metric/graph.npz',root/'roadmap_metric/distances.npy')
    metric=LandmarkProfile.load(teacher,directory/'profile.npz');config=json.loads((BASE/'flags.json').read_text())['agent']
    env,_,_=load_data(config);engine=FastPointmaze(env,nthread=4);bank=core.read_bank(root/'evaluation_bank.npz');probes={}
    try:
        for name,info in result['cases'].items():
            if info['success']:continue
            assert time.monotonic()-started<30,'30-second endpoint probe exhausted'
            with np.load(directory/'greedy'/f'{name}.npz') as f:
                case={**bank[name],'observation':f['observations'][-1],'integration':f['final_integration']}
            core.load_case(env,case);observations,_=engine.candidates(env);goals=np.broadcast_to(case['goal'],observations.shape)
            t=teacher(observations,goals);p=metric(observations,goals);ti,pi=int(np.argmin(t)),int(np.argmin(p))
            current_t=float(teacher(case['observation'][None],case['goal'][None])[0]);current_p=float(metric(case['observation'][None],case['goal'][None])[0])
            h=metric.profile(observations);hg=metric.profile(case['goal'][None]);active=np.argmax(np.abs(h-hg),axis=1)
            probes[name]=dict(teacher_current=current_t,profile_current=current_p,teacher_minus_profile=current_t-current_p,
                teacher_candidate_range=float(np.ptp(t)),profile_candidate_range=float(np.ptp(p)),
                teacher_improvement_best=current_t-float(t[ti]),teacher_improvement_profile_action=current_t-float(t[pi]),
                profile_improvement_best=current_p-float(p[pi]),teacher_regret_of_profile_action=float(t[pi]-t[ti]),
                teacher_best_displacement=float(np.max(np.abs(observations[ti]-case['observation']))),
                profile_best_displacement=float(np.max(np.abs(observations[pi]-case['observation']))),
                profile_at_teacher_action=float(p[ti]),active_landmarks=int(len(np.unique(active))),
                source_correction=float(teacher.attachments(case['observation'][None])[2][0]),goal_correction=float(teacher.attachments(case['goal'][None])[2][0]))
            np.savez_compressed(output/f'{name}.npz',observations=observations,teacher=t,profile=p,active_landmark=active,actions=core.ACTIONS)
            print(name,probes[name],flush=True)
        (output/'analysis.json').write_text(json.dumps(dict(post_hoc=True,new_episodes=0,probes=probes,seconds=time.monotonic()-started,source_sha256=sha256(__file__)),indent=2)+'\n')
    finally:engine.close();env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
