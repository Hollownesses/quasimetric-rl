"""Post-gate file/state review: fixed E15 starts/ends and five actual fields."""
import argparse
import json
from pathlib import Path
import sys
import time
from common import ROOT,load_base,load_data,distances,sha256,tree_digest
from roadmap_metric import RoadmapMetric
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
from utils.flax_utils import restore_agent
import numpy as np
from scipy.stats import spearmanr


def read(path):return json.loads(Path(path).read_text())


def run(root,output):
    started=time.monotonic();root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=read(root/'hard_goal_1000/analysis.json');previous=read(root/'goal_support_1000/analysis.json')
    assert result['complete'] and all(v['success'] for v in result['neural']['cases'].values())
    base,cfg=load_base();models={}
    for label,directory,a in [('mean','goal_support_1000',previous),('hard','hard_goal_1000',result)]:
        assert sha256(root/directory/'model/params_101000.pkl')==a['training']['checkpoint_sha256']
        models[label]=restore_agent(base,str(root/directory/'model'),101000)
    frozen={k:tree_digest(v) for k,v in models.items()}
    teacher=RoadmapMetric.load(root/'roadmap_metric/graph.npz',root/'roadmap_metric/distances.npy')
    env,_,_=load_data(cfg);engine=FastPointmaze(env,nthread=4);bank=core.read_bank(root/'evaluation_bank.npz');probes={};fields={}
    def budget():
        if time.monotonic()-started>60:raise TimeoutError('60-second fixed post-gate review exhausted')
    try:
        for name,case in bank.items():
            if not name.startswith('task3_'):continue
            with np.load(root/'goal_support_1000/neural_evaluation'/f'{name}.npz') as f:
                end={**case,'observation':f['observations'][-1].copy(),'integration':f['final_integration'].copy()}
            for where,position in [('start',case),('mean_end',end)]:
                budget();core.load_case(env,position);successors,_=engine.candidates(env);goals=np.broadcast_to(case['goal'],successors.shape)
                target=teacher(successors,goals);ti=int(np.argmin(target));current_t=float(teacher(position['observation'][None],case['goal'][None])[0])
                info=dict(teacher_best_improvement=current_t-float(target[ti]),teacher_best_displacement=float(np.max(np.abs(successors[ti]-position['observation']))))
                arrays=dict(observations=successors,teacher=target,actions=core.ACTIONS)
                for label,agent in models.items():
                    values=np.asarray(distances(agent,successors,goals));si=int(np.argmin(values));current=float(np.asarray(distances(agent,position['observation'][None],case['goal'][None]))[0])
                    arrays[label]=values;info[label]=dict(rmse=float(np.sqrt(np.square(values-target).mean())),spearman=float(spearmanr(values,target).statistic),
                        teacher_regret=float(target[si]-target[ti]),teacher_improvement_selected=current_t-float(target[si]),
                        selected_displacement=float(np.max(np.abs(successors[si]-position['observation']))),current_error=current-current_t,
                        predicted_improvement_selected=current-float(values[si]))
                key=name+'_'+where;probes[key]=info;np.savez_compressed(output/f'{key}.npz',**arrays);print(key,info,flush=True)
        budget();walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2)
        with np.load(root/'baseline_field/field.npz') as f:np.testing.assert_array_equal(nodes,f['nodes'])
        for name,case in bank.items():
            if not name.startswith('task3_'):continue
            budget();values=np.asarray(distances(models['hard'],nodes,np.broadcast_to(case['goal'],nodes.shape)))
            info,field=core.graph_metrics(nodes,graph,values,goal=case['goal'],anchor=case['observation']);fields[name]=info
            np.savez_compressed(output/f'{name}_field.npz',**{**field,'walls':walls,'goal':case['goal']})
        budget();assert {k:tree_digest(v) for k,v in models.items()}==frozen
        a=dict(complete=True,post_hoc=True,new_episodes=0,training_updates=0,probes=probes,fields=fields,elapsed_seconds=time.monotonic()-started,
            source_sha256=sha256(__file__),hard_analysis_sha256=sha256(root/'hard_goal_1000/analysis.json'),mean_analysis_sha256=sha256(root/'goal_support_1000/analysis.json'))
        (output/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');print('REVIEW_COMPLETE',a['elapsed_seconds'],flush=True)
    finally:engine.close();env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
