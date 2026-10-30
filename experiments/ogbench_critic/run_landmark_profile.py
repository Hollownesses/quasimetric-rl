"""Fixed 128-landmark coordinate pilot; no neural training or geometry input."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from common import ROOT,BASE,BASE_SHA,load_data,sha256
from roadmap_metric import RoadmapMetric
from landmark_profile import LandmarkProfile
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def run(root,output):
    started=time.monotonic();deadline=started+120;phase='setup';env=engine=None
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__).with_name(n) for n in ['LANDMARK_PROFILE_PROTOCOL.md','landmark_profile.py','test_landmark_profile.py','run_landmark_profile.py','roadmap_metric.py','common.py']]
    files += [ROOT/'experiments/ogbench_diagnostics'/n for n in ['fast_pointmaze.py','core.py']]
    for path in files:shutil.copy2(path,snapshot/path.name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('120-second landmark profile pilot exhausted; no extension')
    try:
        assert sha256(BASE/'params_100000.pkl')==BASE_SHA
        verified=read(root/'cached_pointmaze_verification/verification.json')
        assert verified['passed'] and len(verified['cases'])==25
        for name in ['fast_pointmaze.py','core.py']:
            assert sha256(ROOT/'experiments/ogbench_diagnostics'/name)==verified['source_sha256'][name]
        config=read(BASE/'flags.json')['agent'];env,train,_=load_data(config)
        phase='representation';prior=read(root/'roadmap_metric/analysis.json');assert prior['complete'] and prior['eligible_for_neural_study']
        tp=prior['protocol'];assert sha256(root/'roadmap_metric/graph.npz')==tp['graph_sha256']
        assert sha256(root/'roadmap_metric/distances.npy')==tp['distance_sha256']
        assert sha256(Path(__file__).with_name('roadmap_metric.py'))==tp['source_sha256']['roadmap_metric.py']
        teacher=RoadmapMetric.load(root/'roadmap_metric/graph.npz',root/'roadmap_metric/distances.npy')
        model=LandmarkProfile.build(teacher,count=128);model.save(output/'profile.npz')
        restored=LandmarkProfile.load(teacher,output/'profile.npz')
        query=teacher.nodes[np.linspace(0,len(teacher.nodes)-1,32,dtype=int)]+np.array([.031,-.017])
        direct=np.column_stack([teacher(query,np.broadcast_to(l,query.shape)) for l in model.landmarks])
        np.testing.assert_allclose(model.profile(query),direct,rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(model.profile(query),restored.profile(query))
        assert np.all(model(query,query[::-1])<=teacher(query,query[::-1])+1e-8)
        radius=float(np.min(teacher.distances[:,model.indices],axis=1).max())
        protocol=dict(training_updates=0,time_limit_seconds=120,landmarks=128,selection='farthest point, raw graph D, start index zero, first argmax',
            teacher_analysis_sha256=sha256(root/'roadmap_metric/analysis.json'),teacher_graph_sha256=tp['graph_sha256'],teacher_distance_sha256=tp['distance_sha256'],
            profile_sha256=sha256(output/'profile.npz'),raw_graph_covering_radius=radius,build_seconds=time.monotonic()-started,
            metric='max absolute differences of 128 fixed teacher distance coordinates; pseudometric',
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},baseline_checkpoint_sha256=BASE_SHA)
        print('PROFILE_READY',protocol,flush=True)
        # Landmark selection and representation are complete before reading any evaluation case.
        phase='evaluation';cases=core.read_bank(root/'evaluation_bank.npz');assert len(cases)==25
        protocol['bank_sha256']=sha256(root/'evaluation_bank.npz')
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        engine=FastPointmaze(env,nthread=4);greedy=output/'greedy';greedy.mkdir();results={}
        for name,case in cases.items():
            budget()
            def score(observations,goals):budget();return model(observations,goals)
            info,trace=engine.rollout(case,score);trace['final_integration']=core.save_integration(env)
            trace['state_distances']=model(trace['observations'],np.broadcast_to(case['goal'],trace['observations'].shape))
            changes=np.diff(trace['state_distances'])
            info['strict_distance_decrease_fraction']=float((changes<-1e-8).mean())
            info['distance_increase_fraction']=float((changes>1e-8).mean())
            np.savez_compressed(greedy/f'{name}.npz',**trace);results[name]=info
            (greedy/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print(name,info,flush=True)
        (greedy/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        phase='fields';budget();walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2)
        with np.load(root/'baseline_field/field.npz') as old:np.testing.assert_array_equal(nodes,old['nodes'])
        fields={};failure_endpoints={}
        targets=[('canonical',core.GOAL,core.ANCHOR)]+[(name,c['goal'],c['observation']) for name,c in cases.items() if name.startswith('task3_')]
        for name,goal,anchor in targets:
            budget();values=model(nodes,np.broadcast_to(goal,nodes.shape))
            stats,field=core.graph_metrics(nodes,graph,values,goal=goal,anchor=anchor);fields[name]=stats
            np.savez_compressed(output/f'{name}_field.npz',**{**field,'walls':walls,'goal':goal})
            if name!='canonical' and not results[name]['success']:
                failure_endpoints[name]=core.graph_metrics(nodes,graph,values,goal=goal,anchor=np.array(results[name]['final_xy']))[0]
        base=read(root/'baseline_greedy/summary.json')['cases']
        teacher_gate=all(results[n]['success'] for n in results if n.startswith('task3_')) and all(results[n]['success'] for n in base if base[n]['success'])
        budget();assert sha256(BASE/'params_100000.pkl')==BASE_SHA
        result=dict(complete=True,protocol=protocol,cases=results,fields=fields,failure_endpoints=failure_endpoints,
            eligible_for_neural_study=teacher_gate,elapsed_seconds=time.monotonic()-started,
            queries=model.calls)
        (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['elapsed_seconds'],'TEACHER_GATE',teacher_gate,flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error),phase=phase,elapsed_seconds=time.monotonic()-started,
            traceback=traceback.format_exc(),complete=False),indent=2)+'\n');raise
    finally:
        if engine is not None:engine.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
