"""Read-only fixed teacher regularity and neural-response probes; no episodes."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from common import ROOT,BASE,BASE_SHA,load_base,load_data,distances,tree_digest,sha256
from roadmap_metric import RoadmapMetric
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from utils.flax_utils import restore_agent
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def run(root,output):
    started=time.monotonic();root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in ['QUERY_REGULARITY_PROTOCOL.md','query_regularity.py','test_query_regularity.py','roadmap_metric.py','common.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copy2(ROOT/'experiments/ogbench_diagnostics/core.py',snapshot/'core.py')
    def budget():
        if time.monotonic()-started>60:raise TimeoutError('60-second fixed regularity probe exhausted')
    try:
        previous=read(root/'local_shape_1000/analysis.json');assert previous['complete']
        assert sha256(root/'local_shape_1000/model/params_101000.pkl')==previous['training']['checkpoint_sha256']
        tp=read(root/'roadmap_metric/analysis.json')['protocol']
        assert sha256(root/'roadmap_metric/graph.npz')==tp['graph_sha256'] and sha256(root/'roadmap_metric/distances.npy')==tp['distance_sha256']
        assert sha256(Path(__file__).with_name('roadmap_metric.py'))==tp['source_sha256']['roadmap_metric.py']
        teacher=RoadmapMetric.load(root/'roadmap_metric/graph.npz',root/'roadmap_metric/distances.npy')
        original,config=load_base();student=restore_agent(original,str(root/'local_shape_1000/model'),101000)
        frozen=[tree_digest(a) for a in [original,student]];env,_,val=load_data(config);env.close()
        np.random.seed(9001);validation=val.sample(2048);assert tree_digest(validation)==previous['protocol']['validation_sha256']
        bank=core.read_bank(root/'evaluation_bank.npz');task3={n:c for n,c in bank.items() if n.startswith('task3_')}
        labels=[f'val_observation_{i}' for i in range(32)]+[f'val_goal_{i}' for i in range(32)]+['canonical_anchor','canonical_goal']
        points=[*validation['observations'][:32],*validation['value_goals'][:32],core.ANCHOR,core.GOAL]
        pairs=[('canonical',core.ANCHOR,core.GOAL)]
        for name,c in task3.items():
            labels.extend([name+'_start',name+'_goal']);points.extend([c['observation'],c['goal']]);pairs.append((name+'_start',c['observation'],c['goal']))
            with np.load(root/'local_shape_1000/neural_evaluation'/f'{name}.npz') as trace:pairs.append((name+'_end',trace['observations'][-1].copy(),c['goal']))
        points=np.array(points,np.float64);assert points.shape==(76,2) and len(pairs)==11
        epsilon=np.logspace(-2,-7,6);direction=np.array([1,1/np.sqrt(2)])
        left=np.repeat(points,len(epsilon),axis=0);right=left+np.tile(epsilon,len(points))[:,None]*direction
        budget();tv=teacher(left,right).reshape(76,6);ov=np.asarray(distances(original,left,right)).reshape(76,6);sv=np.asarray(distances(student,left,right)).reshape(76,6)
        np.testing.assert_array_equal(teacher(points,points),np.zeros(76))
        ids,b,correction=teacher.attachments(points);rids,_,_=teacher.attachments(right.reshape(76,6,2)[:,-1])
        stable=np.all(np.sort(ids,axis=1)==np.sort(rids,axis=1),axis=1);limit=2*b.min(axis=1)
        assert stable.any();np.testing.assert_allclose(tv[stable,-1],limit[stable],rtol=0,atol=1e-4)
        alias=np.all(left.astype(np.float32)==right.astype(np.float32),axis=1).reshape(76,6)
        np.savez_compressed(output/'diagonal.npz',points=points,epsilon=epsilon,teacher=tv,original=ov,student=sv,
            expected_limit=limit,stable_neighbors=stable,float32_alias=alias,correction=correction)
        responses={};directions=np.array([[1,0],[-1,0],[0,1],[0,-1]],np.float64)
        for name,s,g in pairs:
            budget();info={}
            for which in ['source','goal']:
                changes=(directions[:,None,:]*epsilon[None,:,None]).reshape(-1,2)
                ss=np.broadcast_to(s,changes.shape).copy();gg=np.broadcast_to(g,changes.shape).copy()
                if which=='source':ss+=changes
                else:gg+=changes
                t0=float(teacher(s[None],g[None])[0]);o0=float(np.asarray(distances(original,s[None],g[None]))[0]);n0=float(np.asarray(distances(student,s[None],g[None]))[0])
                t=teacher(ss,gg).reshape(4,6);o=np.asarray(distances(original,ss,gg)).reshape(4,6);n=np.asarray(distances(student,ss,gg)).reshape(4,6)
                center=s if which=='source' else g;ci=teacher.attachments(center[None])[0];new=ss if which=='source' else gg
                changed=np.any(np.sort(teacher.attachments(new)[0],axis=1)!=np.sort(ci,axis=1),axis=1).reshape(4,6)
                info[which]=dict(teacher_max_abs_change=np.abs(t-t0).max(axis=0).tolist(),original_max_abs_change=np.abs(o-o0).max(axis=0).tolist(),
                    student_max_abs_change=np.abs(n-n0).max(axis=0).tolist(),changed_neighbors=changed.sum(axis=0).tolist(),teacher_current=t0,student_current=n0)
                np.savez_compressed(output/f'{name}_{which}.npz',sources=ss,goals=gg,teacher=t,original=o,student=n,
                    teacher_current=t0,original_current=o0,student_current=n0,changed_neighbors=changed)
            responses[name]=info
        budget();assert [tree_digest(a) for a in [original,student]]==frozen and sha256(BASE/'params_100000.pkl')==BASE_SHA
        result=dict(complete=True,training_updates=0,new_episodes=0,elapsed_seconds=time.monotonic()-started,time_limit_seconds=60,
            point_labels=labels,epsilon=epsilon.tolist(),positive_limit_points=int((limit>1e-8).sum()),stable_points=int(stable.sum()),
            teacher_smallest_epsilon_range=[float(tv[:,-1].min()),float(tv[:,-1].max())],expected_limit_range=[float(limit.min()),float(limit.max())],
            responses=responses,baseline_sha256=BASE_SHA,student_sha256=previous['training']['checkpoint_sha256'],teacher_graph_sha256=tp['graph_sha256'],
            teacher_distance_sha256=tp['distance_sha256'],bank_sha256=sha256(root/'evaluation_bank.npz'),validation_sha256=tree_digest(validation),
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()})
        (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['elapsed_seconds'],result['expected_limit_range'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error),elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()),indent=2)+'\n');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
