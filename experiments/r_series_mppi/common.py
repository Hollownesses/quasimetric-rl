"""Frozen controller adapters and passive query instrumentation. No training."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from experiments.connected_expert_utrap.data import load_training_data
from experiments.four_path_execution.common import adapter, TimedMPPIController, stage_metrics, file_digest, tensor_digest, read, write
from experiments.four_path_oracle_execution.protocol import CONFIG, ORACLE_CONFIG
from minimal_qrl.baselines.hybrid_astar_value import HybridAStarValueOracle, _signature_payload, _digest
from minimal_qrl.baselines.mppi import MPPIController

HISTORICAL='e591c088a294c02f1b6f0244802768562dec449f'
DATA=Path('results/connected_expert_utrap_20261021/main')
TRAIN=Path('results/task_source_balance_20261025/main')
OLD=Path('results/four_path_oracle_execution_20261017/main')
OLD_CACHE=OLD/'oracle_cache/hybrid_astar_value_e247996da6d357cf8767.npz'
ARMS=('none','R0','R1','oracle')
SEEDS=(314159,271828)
SAMPLE_SEED=2026102601
COST_KEYS=('cost_time','cost_obs_violation','cost_comm_violation','cost_obs_fail','cost_comm_break','cost_collision','cost_oob')

def load():
    torch.set_num_threads(1)
    return load_training_data(DATA,task_mode='all')

def model_path(arm):return TRAIN/arm/'final.pt'

def poses(observations):
    obs=np.asarray(observations)
    return np.column_stack((obs[:,:2],np.arctan2(obs[:,3],obs[:,2]))).astype(np.float32)

def build_states(env):
    rng=np.random.default_rng(SAMPLE_SEED);records=[]
    # Geometric margins only. Full heading strata; no model/Oracle queries.
    regions={'inside':(24,[2.9,2.0,5.1,5.3]),'west':(8,[1.5,2.0,2.7,5.3])}
    for region,(n,bounds) in regions.items():
        for i in range(n):
            h=i%8;x0,y0,x1,y1=bounds
            p=np.array([rng.uniform(x0,x1),rng.uniform(y0,y1),-np.pi+(h+rng.random())*np.pi/4],np.float32)
            env.reset(seed=0,options={'device_id':'u_trap_target','start':p.copy()})
            assert env.is_valid_state(p) and not env.is_task_feasible(p)
            records.append(dict(id=f'{region}_{i:02d}',stratum=region,heading_bin=h,pose=p.tolist(),
                prefix_stages={'exit':region=='west','wrap':False}))
    return dict(seed=SAMPLE_SEED,regions=regions,heading_edges=np.linspace(-np.pi,np.pi,9).tolist(),
        rule='Uniform independent x/y and uniform heading within 8 equal angular strata; inside 3 draws per stratum, west 1; only validity/nonterminal checks, no score/proximity/outcome rejection',records=records)

class LoggedMPPI(TimedMPPIController):
    """Calls unchanged MPPI scoring; copies diagnostics without changing actions."""
    def begin_episode(self,env,goal_obs,seed):
        self.banks=[];self.query_states=[];self.query_values=[];self.query_counts=[]
        self.endpoint_states=[];self.endpoint_invalid=[];self.endpoint_success=[]
        self.query_invalid=[];self.query_unreachable=[]
        return super().begin_episode(env,goal_obs,seed)
    def sample_candidates(self,env):
        a=super().sample_candidates(env)
        self.banks.append(hashlib.sha256(a.tobytes()).hexdigest())
        return a
    def _terminal_cost(self,env,states):
        v=super()._terminal_cost(env,states)
        self.query_states.append(states.copy());self.query_values.append(v.copy())
        self.query_invalid.extend([not env.is_valid_state(s) for s in states])
        self.query_unreachable.extend((v>=1e6).tolist() if self.terminal_mode=='oracle' else [False]*len(v))
        return v
    def score_candidates(self,env,rollout):
        self.endpoint_states.append(rollout['final_states'].copy())
        self.endpoint_invalid.append(rollout['invalid'].copy());self.endpoint_success.append(rollout['success'].copy())
        n=sum(len(x) for x in self.query_states)
        result=super().score_candidates(env,rollout)
        self.query_counts.append(sum(len(x) for x in self.query_states)-n)
        return result
    def arrays(self):
        return dict(candidate_final_states=np.asarray(self.endpoint_states),candidate_invalid=np.asarray(self.endpoint_invalid),
            candidate_success=np.asarray(self.endpoint_success),query_states=np.concatenate(self.query_states) if self.query_states else np.zeros((0,3),np.float32),
            query_values=np.concatenate(self.query_values) if self.query_values else np.zeros(0,np.float32),query_counts=np.asarray(self.query_counts),
            query_invalid=np.asarray(self.query_invalid,dtype=bool),query_unreachable=np.asarray(self.query_unreachable,dtype=bool),
            candidate_sha256=np.asarray(self.banks))

def make(data,arm,cache_dir,logged=True):
    cls=LoggedMPPI if logged else TimedMPPIController
    if arm=='none':return cls(CONFIG,terminal_mode='none')
    if arm=='oracle':return cls(CONFIG,terminal_mode='oracle',terminal_value_provider=HybridAStarValueOracle(ORACLE_CONFIG,cache_dir=cache_dir))
    assert arm in ('R0','R1')
    value,_=adapter(data.small,model_path(arm))
    return cls(CONFIG,terminal_mode='qrl',qrl_agent=value)

@torch.no_grad()
def validate_model(data,arm):
    value,model=adapter(data.small,model_path(arm));base=data.small
    a=torch.cat((base.src,base.task_src,base.task_src));b=torch.cat((base.dst,base.goal,base.pair))
    z=torch.cat([model.projector(model.encoder(x)) for x in base.nodes.split(512)])
    expected=torch.cat([model.head(z[a[i:i+512]],z[b[i:i+512]]) for i in range(0,len(a),512)]).numpy()
    observed=np.concatenate([value.batch_value(base.nodes[a[i:i+1024]].numpy(),base.nodes[b[i:i+1024]].numpy()) for i in range(0,len(a),1024)])
    np.testing.assert_allclose(observed,expected,rtol=3e-6,atol=8e-5)
    # All graph node-to-task distances, independently through the original index head.
    goal=int(base.goal[0]);expected_task=torch.cat([model.head(x,z[goal].expand_as(x)) for x in z.split(512)]).numpy()
    goals=np.repeat(base.nodes[goal:goal+1].numpy(),len(z),axis=0)
    task=value.batch_value(base.nodes.numpy(),goals)
    np.testing.assert_allclose(task,expected_task,rtol=3e-6,atol=8e-5)
    env=data.env;env.reset(seed=0,options={'device_id':'u_trap_target','start':[4.,3.65,0.]})
    rng=np.random.default_rng(92017);unseen=[]
    while len(unseen)<97:
        p=rng.uniform([.3,.3,-np.pi],[8.5,8.,np.pi]).astype(np.float32)
        if env.is_valid_state(p) and not env.is_task_feasible(p):unseen.append(p)
    observations=np.stack([env.state_to_observation(s) for s in unseen]);g=np.repeat(env.abstract_goal_observation()[None],len(unseen),axis=0)
    full=value.batch_value(observations,g)
    chunks=np.concatenate([value.batch_value(observations[i:i+7],g[i:i+7]) for i in range(0,len(g),7)])
    np.testing.assert_allclose(full,chunks,rtol=3e-6,atol=8e-5)
    assert np.isfinite(full).all() and (full>=0).all() and full.max()>0 and value._scale==1.
    assert not any(np.any(np.all(base.nodes.numpy()==o,axis=1)) for o in observations)
    terminals=np.stack([env.sample_task_terminal_state(seed=8100+i) for i in range(16)])
    terminal_obs=np.stack([env.state_to_observation(s) for s in terminals]);tg=np.repeat(g[:1],16,axis=0)
    terminal_values=value.batch_value(terminal_obs,tg);assert np.max(np.abs(terminal_values))==0
    return dict(checkpoint=str(model_path(arm)),sha256=file_digest(model_path(arm)),recorded_pair_count=len(expected),
        pair_error_max=float(abs(observed-expected).max()),all_node_task_error_max=float(abs(task-expected_task).max()),
        unseen_count=97,unseen_batch_error_max=float(abs(full-chunks).max()),terminal_max=float(abs(terminal_values).max()),
        distance_scale=1.,sign='positive distance added as terminal COST; no sign inversion',
        transfer='strict encoder/projector/IQE state transfer; continuous full observations, no node lookup/snapping',
        model_sha256=tensor_digest(*value.state_dict().values()))
