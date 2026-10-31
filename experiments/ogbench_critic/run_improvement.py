"""History-independent E10+E16+E17-B recipe; plan, smoke, or fixed improve mode."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from common import ROOT,UPSTREAM,BASE,QRLAgent,load_data,tree_digest,tree_equal,sha256
from data_graph import valid_transitions
from roadmap_metric import RoadmapMetric
from hard_goal import HardGoal
from iqe_distill import update as value_update,init_optimizer as value_optimizer
from extraction import update as extract_update,init_optimizer as extract_optimizer,proof,module_key
from utils.flax_utils import save_agent,restore_agent
import jax
import numpy as np


def run(base_checkpoint,mode,output):
    started=time.monotonic();steps=2 if mode=='smoke' else 1000;limit=120 if mode=='smoke' else 600
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);checkpoint=Path(base_checkpoint).resolve();phase='plan';actual={};env=None
    def budget():
        if time.monotonic()-started>limit:raise TimeoutError('Standalone frozen budget exhausted; no extension')
    try:
        flags_path=checkpoint.parent/'flags.json';flags=json.loads(flags_path.read_text());cfg=flags['agent']
        assert flags['env_name']=='pointmaze-medium-navigate-v0' and flags['seed']==0 and flags['train_steps']==100000
        assert checkpoint.name=='params_100000.pkl' and cfg['agent_name']=='qrl' and cfg['quasimetric_type']=='iqe'
        assert cfg['actor_loss']=='ddpgbc' and cfg['alpha']==.0003 and cfg['lr']==.0003 and cfg['const_std']
        assert cfg['actor_p_trajgoal']==1 and cfg['actor_p_randomgoal']==cfg['actor_p_curgoal']==0 and cfg['value_p_randomgoal']==1
        protocol=dict(mode=mode,stage_steps=steps,batch_size=256,sampling_seed=2001,official_evaluation_rng=0,episodes_per_task=1 if mode=='smoke' else 50,
            time_limit_seconds=limit,base_checkpoint=str(checkpoint),base_checkpoint_sha256=sha256(checkpoint),flags_sha256=sha256(flags_path),config=cfg,
            official_commit=(ROOT/'experiments/ogbench_official_qrl/OFFICIAL_COMMIT').read_text().strip(),history_controls_required=False,
            recipe='full valid-data E10 teacher; E16 hard-goal value; official dynamics then DDPG+BC; no intermediate selection')
        snapshot=output/'source_snapshot';snapshot.mkdir()
        for name in ['run_improvement.py','evaluate_improvement.py','actor_recording.py','common.py','data_graph.py','roadmap_metric.py','hard_goal.py','goal_support.py','query_augmentation.py','iqe_distill.py','extraction.py']:
            shutil.copy2(Path(__file__).with_name(name),snapshot/name)
        protocol['source_sha256']={p.name:sha256(p) for p in snapshot.iterdir()}
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(output/'agent_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        if mode=='plan':print('PLAN_ONLY',output,flush=True);return
        budget();phase='load';template=QRLAgent.create(0,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        agent=restore_agent(template,str(checkpoint.parent),100000);assert int(agent.network.opt_state[0].count)==100000
        env,train,_=load_data(cfg);env.close();env=None;phase='teacher';source,target=valid_transitions(train.dataset);assert len(source)==1000000
        teacher=RoadmapMetric.build(source,target,connections=8);teacher.save(output/'graph.npz');teacher.precompute(output/'distances.npy',budget=budget)
        sampler=HardGoal(teacher.nodes,candidates=8);models=output/'model';models.mkdir();proofs={};streams={};stage_times={}
        np.random.seed(2001)
        for stage in ['value','dynamics','actor']:
            phase=stage;stage_start=time.monotonic();before=agent
            active='modules_value' if stage=='value' else module_key(stage);frozen=proof(agent,[k for k in agent.network.params if k!=active])
            opt=value_optimizer(agent) if stage=='value' else extract_optimizer(agent,stage);stream=hashlib.sha256();queries=hashlib.sha256();targets_hash=hashlib.sha256()
            if stage=='dynamics':np.random.seed(2001)
            with (output/f'{stage}.csv').open('w') as handle:
                writer=None
                for step in range(1,steps+1):
                    budget();batch=train.sample(256)
                    for name in ['observations','next_observations','value_goals','actions','actor_goals']:stream.update(batch[name].tobytes())
                    if stage=='value':
                        batch,target,selection,_=sampler.choose(agent,batch,128*(step-1),teacher)
                        for name in ['observations','value_goals']:queries.update(batch[name].tobytes())
                        targets_hash.update(target.tobytes());agent,opt,info=value_update(agent,opt,batch,target)
                    else:agent,opt,info=extract_update(agent,opt,batch,stage)
                    jax.block_until_ready(info);actual[stage]=step;assert all(np.isfinite(float(v)) for v in info.values())
                    if step==1 or step%100==0 or step==steps:
                        row=dict(step=step,elapsed_seconds=time.monotonic()-started,**{k:float(v) for k,v in info.items()})
                        if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);handle.flush();print(stage,row,flush=True)
            assert proof(agent,list(frozen))==frozen and not tree_equal(before.network.params[active],agent.network.params[active])
            epoch=100000+steps*(len(proofs)+1);assert int(agent.network.opt_state[0].count)==epoch and int(agent.network.step)==epoch+1
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
            save_agent(agent,str(models),epoch);assert tree_equal(agent,restore_agent(template,str(models),epoch))
            proofs[stage]=dict(epoch=epoch,frozen=True,full_state_roundtrip=True,checkpoint_sha256=sha256(models/f'params_{epoch}.pkl'))
            streams[stage]=dict(full_actor_fields_sha256=stream.hexdigest(),query_sha256=queries.hexdigest(),target_sha256=targets_hash.hexdigest());stage_times[stage]=time.monotonic()-stage_start
        # Separate process intentionally has no teacher/proposal modules at inference.
        phase='evaluation';budget();remaining=limit-(time.monotonic()-started);epoch=100000+3*steps
        argv=[sys.executable,str(Path(__file__).with_name('evaluate_improvement.py')),'--model',str(models),'--epoch',str(epoch),'--config',str(output/'agent_config.json'),
            '--episodes',str(protocol['episodes_per_task']),'--output',str(output/'evaluation'),'--limit-seconds',str(remaining)]
        (output/'evaluation_argv.json').write_text(json.dumps(argv,indent=2)+'\n')
        with (output/'evaluation.log').open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining,cwd=ROOT)
        ev=json.loads((output/'evaluation/analysis.json').read_text());assert ev['complete'] and ev['no_teacher_imports']
        budget();assert sha256(checkpoint)==protocol['base_checkpoint_sha256']
        a=dict(complete=True,mode=mode,protocol=protocol,actual=actual,stage_proofs=proofs,streams=streams,stage_seconds=stage_times,
            evaluation=ev,elapsed_seconds=time.monotonic()-started,formal_performance_claim=mode=='improve')
        (output/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');print('COMPLETE',mode,a['elapsed_seconds'],flush=True)
    except BaseException as e:
        (output/'failure.json').write_text(json.dumps(dict(complete=False,error=repr(e),phase=phase,actual=actual,elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()),indent=2)+'\n');raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-checkpoint',default=str(BASE/'params_100000.pkl'));p.add_argument('--mode',required=True,choices=['plan','smoke','improve']);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.base_checkpoint,a.mode,a.output)
