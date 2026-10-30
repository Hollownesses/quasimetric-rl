"""Explicit read-only budget amendment; skip verified completed episodes only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from common import ROOT,UPSTREAM,BASE,BASE_SHA,load_base,load_data,tree_digest,sha256
from exact_score_cache import ExactScoreCache
from run_official_counterfactual import trace_values
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
from utils.flax_utils import restore_agent
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def verify_trace(path,info,case):
    with np.load(path) as f:
        assert f['observations'].shape==(info['steps']+1,2) and f['actions'].shape==(info['steps'],2)
        assert f['distances'].shape==(info['steps'],) and f['state_values'].shape==(info['steps']+1,) and f['final_integration'].ndim==1
        assert 0<info['steps']<=1000 and all(np.isfinite(f[k]).all() for k in f.files)
        np.testing.assert_array_equal(f['observations'][0],case['observation']);np.testing.assert_array_equal(f['goal'],case['goal']);np.testing.assert_array_equal(f['observations'][-1],info['final_xy'])


def run(root,output):
    started=time.monotonic();root=Path(root).resolve();source=root/'official_counterfactual';output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir();env=engine=None;phase='verify';completed={};trace_paths={};old_hashes={}
    files=['OFFICIAL_COUNTERFACTUAL_COMPLETION.md','finish_official_counterfactual.py','run_official_counterfactual.py','common.py','exact_score_cache.py']
    for name in files:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    for name in ['core.py','fast_pointmaze.py']:shutil.copy2(ROOT/'experiments/ogbench_diagnostics'/name,snapshot/name)
    def budget():
        if time.monotonic()-started>300:raise TimeoutError('300-second amended read-only completion budget exhausted')
    try:
        failure=read(source/'failure.json');assert failure['complete'] is False and failure['error']=="TimeoutError('240-second fixed counterfactual budget exhausted; no extension')"
        assert failure['completed']==dict(original=40) and failure['phase']=='original_initials' and not (source/'analysis.json').exists()
        p=read(source/'protocol.json');assert p['matched_reset_inputs']==250 and p['first_steps_bitwise_verified']==100
        aliases={'core.py':ROOT/'experiments/ogbench_diagnostics/core.py','fast_pointmaze.py':ROOT/'experiments/ogbench_diagnostics/fast_pointmaze.py',
                 'official_evaluation.py':UPSTREAM/'impls/utils/evaluation.py'}
        for name,digest in p['source_sha256'].items():assert sha256(source/'source_snapshot'/name)==digest==sha256(aliases.get(name,Path(__file__).with_name(name)))
        assert sha256(source/'official_task3_bank.npz')==p['bank_sha256'] and sha256(root/'extraction_1000/analysis.json')==p['prior_analysis_sha256']
        prior=read(root/'extraction_1000/analysis.json');cases=core.read_bank(source/'official_task3_bank.npz');assert len(cases)==50
        old=read(source/'original/progress.json');assert list(old)==list(cases)[:40]
        for name,info in old.items():
            path=source/'original'/f'{name}.npz';verify_trace(path,info,cases[name]);old_hashes[name]=sha256(path)
        assert sha256(root/'hard_goal_1000/model/params_101000.pkl')==p['improved_checkpoint_sha256']
        base,cfg=load_base();models=dict(original=base,improved=restore_agent(base,str(root/'hard_goal_1000/model'),101000))
        frozen={k:tree_digest(v) for k,v in models.items()}
        for arm,agent in models.items():assert tree_digest(agent.network.params['modules_value'])==prior['evaluations'][arm]['critic_digest']
        protocol=dict(original_protocol_sha256=sha256(source/'protocol.json'),original_failure_sha256=sha256(source/'failure.json'),
            original_failed_seconds=failure['elapsed_seconds'],time_limit_seconds=300,training_updates=0,skipped_completed=40,
            completed_trace_sha256=old_hashes,source_sha256={path.name:sha256(path) for path in snapshot.iterdir()})
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        env,_,_=load_data(cfg);engine=FastPointmaze(env,nthread=4);caches={k:ExactScoreCache(v,audit=True) for k,v in models.items()};results={}
        def execute(label,agent,cache,bank,existing=None):
            directory=output/label;directory.mkdir();infos={} if existing is None else dict(existing);trace_paths[label]={}
            for name,info in infos.items():trace_paths[label][name]=str(source/'original'/f'{name}.npz')
            for name,case in bank.items():
                if name in infos:continue
                budget()
                def score(observations,goals):budget();return cache(observations,goals)
                info,trace=engine.rollout(case,score);trace['final_integration']=core.save_integration(env);trace['state_values']=trace_values(agent,trace['observations'],case['goal'])
                delta=np.diff(trace['state_values']);info['strict_decrease_fraction']=float((delta<-1e-4).mean());info['increase_fraction']=float((delta>1e-4).mean())
                path=directory/f'{name}.npz';np.savez_compressed(path,**trace);verify_trace(path,info,case)
                infos[name]=info;trace_paths[label][name]=str(path);completed[label]=len(infos)
                (directory/'progress.json').write_text(json.dumps(infos,indent=2)+'\n');print(label,name,info,flush=True)
            assert set(infos)==set(bank)
            return {name:infos[name] for name in bank}
        phase='original_remaining';results['original']=execute('original',models['original'],caches['original'],cases,old)
        phase='improved_initials';results['improved']=execute('improved',models['improved'],caches['improved'],cases)
        end_cases={};bb=prior['evaluations']['improved']['official_episodes']
        for name in cases:
            if bb[name]['success']:continue
            with np.load(root/'extraction_1000/improved_evaluation/official'/f'{name}.npz') as f:
                end_cases[name]={**cases[name],'observation':f['observations'][-1].copy(),'integration':f['final_integration'].copy()}
        assert len(end_cases)==19;phase='actor_timeout_rescue'
        rescues=execute('improved_from_actor_timeout',models['improved'],caches['improved'],end_cases)
        budget();assert {k:tree_digest(v) for k,v in models.items()}==frozen and sha256(BASE/'params_100000.pkl')==BASE_SHA
        assert sha256(root/'hard_goal_1000/model/params_101000.pkl')==p['improved_checkpoint_sha256']
        for name,digest in old_hashes.items():assert sha256(source/'original'/f'{name}.npz')==digest
        for name,digest in p['trace_sha256'].items():
            arm,case=name.split('/');assert sha256(root/'extraction_1000'/f'{arm}_evaluation/official'/f'{case}.npz')==digest
        assert 'roadmap_metric' not in sys.modules and 'hard_goal' not in sys.modules and 'data_graph' not in sys.modules
        a=dict(complete=True,amended_runtime_budget=True,protocol=protocol,original_protocol=p,initials=results,
            actor_timeout_counterfactual=rescues,trace_paths=trace_paths,new_complete_episodes=79,reused_complete_episodes=40,
            no_teacher_modules=True,cache={k:v.statistics() for k,v in caches.items()},elapsed_seconds=time.monotonic()-started)
        assert sum(len(v) for v in results.values())+len(rescues)==119
        a['combined_seconds']=failure['elapsed_seconds']+a['elapsed_seconds']
        (output/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');print('COMPLETE',a['elapsed_seconds'],'COMBINED',a['combined_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(complete=False,error=repr(error),phase=phase,completed=completed,elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()),indent=2)+'\n');raise
    finally:
        if engine is not None:engine.close()
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
