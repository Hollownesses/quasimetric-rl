"""Execution-only adapters and immutable inputs; no learner or graph-reference calls."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.four_path_factorial.run import IndexedIQE,load_data,file_digest,tensor_digest
from minimal_qrl.baselines.mppi import MPPIConfig,MPPIController
from minimal_qrl.gc_agents import QRLGoalValueAdapter
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic
from quasimetric_rl.data import EnvSpec

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'results/four_path_reproducibility_20261016'
BANK=ROOT/'results/diagnostic_path_support_only_20k/qrl_path_support_only_b1/path_support_bank.json'
SCENARIO=ROOT/'results/diagnostic_path_support_only/config/diagnostic_scenario.json'
SEEDS=(101,202,303,404,505)
PLANNER_SEEDS=(314159,271828)
ARMS=('none','geometry')+tuple(f'{stage}_{seed}' for seed in SEEDS for stage in ('decay','qp'))
CONFIG=MPPIConfig()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def read(path):return json.loads(Path(path).read_text())


def checkpoint(arm):
    stage,seed=arm.split('_')
    return (RESULT/'main'/f'seed_{seed}'/'decay.pt' if stage=='decay' else
            RESULT/'qp_numerical_fix'/f'seed_{seed}'/'latest.pt')


def adapter(data,path):
    """Strictly export the three learned modules into the production critic."""
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=IndexedIQE(data,'encoder',0,initialization_seed=saved['initialization_seed'])
    model.load_state_dict(saved['model'],strict=True);model.eval()
    critic=QuasimetricCritic.Conf().make(env_spec=EnvSpec.from_env(data.env))
    critic.encoder.load_state_dict(model.encoder.state_dict(),strict=True)
    critic.quasimetric_model.projector.load_state_dict(model.projector.state_dict(),strict=True)
    critic.quasimetric_model.quasimetric_head.load_state_dict(model.head.state_dict(),strict=True)
    agent=nn.Module();agent.critics=nn.ModuleList([critic]);agent.eval();agent.requires_grad_(False)
    adapted=QRLGoalValueAdapter(agent,data.env,torch.device('cpu'),distance_scale=1.0)
    return adapted,model


def controller(data,arm):
    if arm in ('none','geometry'):
        return TimedMPPIController(CONFIG,terminal_mode='none' if arm=='none' else 'model')
    adapted,_=adapter(data,checkpoint(arm))
    return TimedMPPIController(CONFIG,terminal_mode='qrl',qrl_agent=adapted)


def validate_adapter(data,arm):
    adapted,model=adapter(data,checkpoint(arm))
    with torch.no_grad():
        expected=torch.cat(model.distances(data)).numpy()
        source=torch.cat((data.src,data.task_src,data.task_src))
        target=torch.cat((data.dst,data.goal,data.pair))
        observed=adapted.batch_value(data.nodes[source].numpy(),data.nodes[target].numpy())
        np.testing.assert_allclose(observed,expected,rtol=2e-6,atol=5e-5)
        # Also check original full observations against the canonical node path,
        # including physical terminal states, and different inference batch sizes.
        raw=data.raw.all_observations.numpy()
        canonical=data.nodes[data.obs_ids].numpy()
        goal=np.repeat(canonical[-1:],[len(raw)],axis=0)
        left=adapted.batch_value(raw,goal);right=adapted.batch_value(canonical,goal)
        np.testing.assert_allclose(left,right,rtol=0,atol=1e-6)
        chunked=np.concatenate([adapted.batch_value(raw[i:i+37],goal[i:i+37]) for i in range(0,len(raw),37)])
        np.testing.assert_allclose(chunked,left,rtol=3e-6,atol=8e-5)
    return dict(checkpoint=str(checkpoint(arm).relative_to(ROOT)),sha256=file_digest(checkpoint(arm)),
        initialization_seed=int(arm.split('_')[1]),recorded_pairs=len(expected),
        max_absolute_error=float(abs(observed-expected).max()),raw_canonical_error=float(abs(left-right).max()),
        chunk_error=float(abs(left-chunked).max()),distance_scale=adapted._scale,
        model_sha256=tensor_digest(*model.state_dict().values()))


def stage_metrics(states,start):
    """Separate newly attained milestones from stages already passed at reset."""
    s=np.asarray(states);x,y=s[:,0],s[:,1]
    west=(x<=2.8)&(y>=1.9)&(y<=5.4)
    outside=(y<1.4)|(y>5.9);east=x>5.7
    interior=(2.8<x[0]<5.2 and 1.9<y[0]<5.4)
    exit_eligible=bool(interior)
    wrap_eligible=not bool(east[0])
    exit_ids=np.flatnonzero(west)
    # The environment checks the entire linear movement segment for collision.
    # Likewise an exit-region visit can occur between two sampled endpoints.
    # At a west corner a 0.1-length step may have neither endpoint in the box.
    from minimal_qrl.baselines.mppi import _segment_hits_obstacle
    from minimal_qrl.envs.dubins_uav_2d import Obstacle
    crossed_exit = bool(len(s)>1 and _segment_hits_obstacle(
        x[:-1],y[:-1],x[1:],y[1:],Obstacle(0.,2.8,1.9,5.4)).any())
    exited = bool(exit_ids.size or crossed_exit)
    # Wall completion requires an observed outside-envelope state before east.
    outside_ids=np.flatnonzero(outside)
    wrap=bool(len(outside_ids) and np.any(east[int(outside_ids[0]):]))
    prefix=start.get('prefix_stages',{})
    return dict(exit_eligible=exit_eligible,exit_completed=exited if exit_eligible else None,
        wrap_eligible=wrap_eligible,wrap_completed=wrap if wrap_eligible else None,
        west_exit_visited=exited,observed_wrap=wrap,
        cumulative_exit=bool(prefix.get('exit',False) or exited),
        cumulative_wrap=bool(prefix.get('wrap',False) or wrap),
        prefix_stages=prefix)


class TimedMPPIController(MPPIController):
    def begin_episode(self,env,goal_obs,seed):
        from time import perf_counter
        start=perf_counter()
        result=super().begin_episode(env,goal_obs,seed)
        result['initial_planning_time_sec']=perf_counter()-start
        return result
