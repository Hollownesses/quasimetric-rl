"""No-training critic counterfactual on the exact E17 official evaluation inputs."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from common import ROOT,UPSTREAM,BASE,BASE_SHA,load_base,load_data,distances,tree_digest,sha256
from official_replay import official_resets
from exact_score_cache import ExactScoreCache
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
from utils.flax_utils import restore_agent
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def trace_values(agent,observations,goal):
    # Logging only: fixed batch avoids recompiling every variable episode length.
    values=[]
    for start in range(0,len(observations),128):
        part=observations[start:start+128];padded=np.pad(part,((0,128-len(part)),(0,0)),mode='edge')
        values.append(np.asarray(distances(agent,padded,np.broadcast_to(goal,padded.shape)))[:len(part)])
    return np.concatenate(values)


def run(root,output):
    started=time.monotonic();root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    env=engine=None;phase='setup';completed={};snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in ['OFFICIAL_COUNTERFACTUAL_PROTOCOL.md','official_replay.py','test_official_replay.py','run_official_counterfactual.py','common.py','exact_score_cache.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    for name in ['core.py','fast_pointmaze.py']:shutil.copy2(ROOT/'experiments/ogbench_diagnostics'/name,snapshot/name)
    shutil.copy2(UPSTREAM/'impls/utils/evaluation.py',snapshot/'official_evaluation.py')
    def budget():
        if time.monotonic()-started>240:raise TimeoutError('240-second fixed counterfactual budget exhausted; no extension')
    try:
        prior=read(root/'extraction_1000/analysis.json');assert prior['complete'] and prior['paired_250_initials_and_goals']
        verified=read(root/'cached_pointmaze_verification/verification.json');assert verified['passed']
        for name in ['core.py','fast_pointmaze.py']:assert sha256(ROOT/'experiments/ogbench_diagnostics'/name)==verified['source_sha256'][name]
        assert sha256(Path(__file__).with_name('exact_score_cache.py'))==verified['source_sha256']['exact_score_cache.py']
        assert sha256(root/'hard_goal_1000/model/params_101000.pkl')==prior['protocol']['improved_sha256']
        assert sha256(UPSTREAM/'impls/utils/evaluation.py')==prior['protocol']['source_sha256']['official_evaluation.py']
        base,config=load_base();models=dict(original=base,improved=restore_agent(base,str(root/'hard_goal_1000/model'),101000))
        for arm,a in models.items():assert tree_digest(a.network.params['modules_value'])==prior['evaluations'][arm]['critic_digest']
        frozen={k:tree_digest(v) for k,v in models.items()};env,_,_=load_data(config)
        assert env.unwrapped._teleport_info is None
        aa=prior['evaluations']['original']['official_episodes'];bb=prior['evaluations']['improved']['official_episodes'];cases={};matched=0;trace_hashes={}
        phase='reconstruct'
        for task,episode,ob,info in official_resets(env):
            budget();name=f'task{task}_episode{episode:03d}'
            for old in [aa[name],bb[name]]:
                np.testing.assert_array_equal(ob,old['initial_xy']);np.testing.assert_array_equal(info['goal'],old['goal'])
            matched+=1
            if task==3:cases[name]=dict(seed=np.array(0),task=np.array(3),observation=ob.copy(),goal=info['goal'].copy(),integration=core.save_integration(env))
        assert matched==250 and len(cases)==50
        np.savez_compressed(output/'official_task3_bank.npz',**{f'{n}__{k}':v for n,c in cases.items() for k,v in c.items()})
        phase='first_step_verification';first_steps=0
        for name,case in cases.items():
            for arm in ['original','improved']:
                budget();path=root/'extraction_1000'/f'{arm}_evaluation/official'/f'{name}.npz';trace_hashes[arm+'/'+name]=sha256(path)
                with np.load(path) as f:
                    core.load_case(env,case);next_ob=env.step(f['actions'][0])[0]
                    np.testing.assert_array_equal(next_ob,f['observations'][1]);first_steps+=1
        assert first_steps==100
        protocol=dict(training_updates=0,time_limit_seconds=240,official_initials=50,matched_reset_inputs=250,first_steps_bitwise_verified=100,
            prior_analysis_sha256=sha256(root/'extraction_1000/analysis.json'),base_checkpoint_sha256=BASE_SHA,improved_checkpoint_sha256=prior['protocol']['improved_sha256'],
            bank_sha256=sha256(output/'official_task3_bank.npz'),trace_sha256=trace_hashes,source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},
            diagnostic_logging='state_values in padded batches of 128, increase/decrease tolerance 1e-4; control scores unchanged',
            build_and_verify_seconds=time.monotonic()-started)
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('RECONSTRUCTION_VERIFIED',protocol['build_and_verify_seconds'],flush=True)
        engine=FastPointmaze(env,nthread=4);caches={k:ExactScoreCache(v,audit=True) for k,v in models.items()};results={}
        def execute(label,agent,cache,bank):
            directory=output/label;directory.mkdir();infos={}
            for name,case in bank.items():
                budget()
                def score(observations,goals):budget();return cache(observations,goals)
                info,trace=engine.rollout(case,score);trace['final_integration']=core.save_integration(env)
                trace['state_values']=trace_values(agent,trace['observations'],case['goal']);change=np.diff(trace['state_values'])
                info['strict_decrease_fraction']=float((change<-1e-4).mean());info['increase_fraction']=float((change>1e-4).mean())
                np.savez_compressed(directory/f'{name}.npz',**trace);infos[name]=info;completed[label]=len(infos)
                (directory/'progress.json').write_text(json.dumps(infos,indent=2)+'\n');print(label,name,info,flush=True)
            return infos
        for arm,agent in models.items():
            phase=arm+'_initials';results[arm]=execute(arm,agent,caches[arm],cases)
        phase='actor_failed_endpoints';end_cases={}
        for name in cases:
            if bb[name]['success']:continue
            path=root/'extraction_1000/improved_evaluation/official'/f'{name}.npz'
            with np.load(path) as f:end_cases[name]={**cases[name],'observation':f['observations'][-1].copy(),'integration':f['final_integration'].copy()}
        assert len(end_cases)==19
        rescues=execute('improved_from_actor_timeout',models['improved'],caches['improved'],end_cases)
        budget();assert {k:tree_digest(v) for k,v in models.items()}==frozen
        assert sha256(BASE/'params_100000.pkl')==BASE_SHA and sha256(root/'hard_goal_1000/model/params_101000.pkl')==protocol['improved_checkpoint_sha256']
        for name,digest in trace_hashes.items():
            arm,case=name.split('/');assert sha256(root/'extraction_1000'/f'{arm}_evaluation/official'/f'{case}.npz')==digest
        assert 'roadmap_metric' not in sys.modules and 'hard_goal' not in sys.modules and 'data_graph' not in sys.modules
        a=dict(complete=True,protocol=protocol,initials=results,actor_timeout_counterfactual=rescues,cache={k:v.statistics() for k,v in caches.items()},
            no_teacher_modules=True,elapsed_seconds=time.monotonic()-started)
        (output/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');print('COMPLETE',a['elapsed_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(complete=False,error=repr(error),phase=phase,completed=completed,elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()),indent=2)+'\n');raise
    finally:
        if engine is not None:engine.close()
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
