"""Frozen four-path parameterization x constraint solver experiment.

Graph costs live exclusively in evaluate(); training sees one-step costs and
fixed task/B1 endpoint IDs. Reuses the production quotient, IQE, barrier workset,
and PHR term. Full push expectation removes minibatch sampling as a confound.
"""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import heapq
import json
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quasimetric_rl.data import EnvSpec, MultiEpisodeData
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic
from quasimetric_rl.modules.quasimetric_critic.models.encoder import Encoder
from quasimetric_rl.modules.quasimetric_critic.losses.goal_set_barrier import GoalSetBarrierUpdater, GoalSetBarrierConf
from quasimetric_rl.modules.quasimetric_critic.losses.phr import TabularPHR
from quasimetric_rl.modules.utils import LossResult
from minimal_qrl.global_push_sampling import replay_fingerprint, configure_global_push_sampling
from minimal_qrl.path_support import load_bank
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.scalability_scenarios import load_scenario_config, scenario_to_env_kwargs

BASE = ROOT / 'results/diagnostic_path_support_only'
OLD = ROOT / 'results/diagnostic_path_support_only_20k/qrl_path_support_only_b1'
INIT = OLD / 'checkpoint_00000.pth'


def tensor_digest(*values):
    h = hashlib.sha256()
    for v in values:
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Replay:
    def __init__(self, raw, env):
        self.raw_data, self.env = raw, env
        self.indices_to_episode_indices = torch.repeat_interleave(torch.arange(len(raw.episode_lengths)), raw.episode_lengths)
    def __len__(self):
        return len(self.raw_data.rewards)
    def create_env(self):
        return self.env


def canonical_nodes(obs):
    # The actual production mapping, not a second interpretation of its fields.
    canonical = Encoder._canonicalize_comm_goal_set(None, obs)
    mapping, unique, ids = {}, [], []
    for row in canonical:
        # Full float32 state identity; normalize signed zero, no rounding/binning.
        row = row.clone(); row[row == 0] = 0
        key = row.numpy().tobytes()
        if key not in mapping:
            mapping[key] = len(unique); unique.append(row)
        ids.append(mapping[key])
    return torch.stack(unique), torch.tensor(ids), mapping


def load_data():
    env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(load_scenario_config(BASE/'config/diagnostic_scenario.json')))
    payload = torch.load(BASE/'task_path_support_audit/verified_replay.pt', map_location='cpu', weights_only=True)
    raw = MultiEpisodeData(**payload['raw'])
    assert replay_fingerprint(raw) == payload['sha256']
    bank = load_bank(OLD/'path_support_bank.json')
    assert bank['content_digest'] == '1ed1098b439fb0d7a990ea83c037d3db1cff4be4c58b5a134c10bfbe9c4368aa'
    ds = Replay(raw, env)
    # Cache omits push-only tensors; let the existing sampler reconstruct them.
    for k in ('global_push_source_observations', 'global_push_goal_observations'):
        raw.transition_infos[k] = torch.zeros((len(ds), 42))
    configure_global_push_sampling(ds, state_mode='trajectory', task_mode='all', seed=42)
    nodes, ids, mapping = canonical_nodes(raw.all_observations)
    src_obs = torch.arange(len(ds)) + ds.indices_to_episode_indices
    src, dst = ids[src_obs], ids[src_obs+1]
    physical = ~raw.transition_infos['abstract_goal_edge'].bool()
    rows = physical.nonzero().flatten()
    pair_obs = raw.transition_infos['global_push_pair_goal_obs_index'][rows]
    assert (pair_obs > src_obs[rows]).all()
    goals = Encoder._canonicalize_comm_goal_set(None, raw.transition_infos['task_goal_observations'][rows])
    goal_ids = torch.tensor([mapping[x.numpy().tobytes()] for x in goals])
    assert raw.transition_infos['global_push_task_witness_mask'][rows].all()
    data = SimpleNamespace(nodes=nodes, src=src, dst=dst, cost=-raw.rewards,
        rows=rows, task_src=src[rows], goal=goal_ids, pair=ids[pair_obs], physical=physical,
        raw=raw, src_obs=src_obs, obs_ids=ids, env=env, replay=ds)
    data.fingerprint = dict(bank=bank['content_digest'], replay=payload['sha256'],
        nodes=tensor_digest(nodes, ids), pairs=tensor_digest(data.task_src, data.goal, data.pair),
        costs=tensor_digest(data.cost), initial_checkpoint=file_digest(INIT))
    assert len(nodes) == 336 and len(src) == 339 and len(rows) == 335
    assert (src[~physical] == dst[~physical]).all()
    return data


class IndexedIQE(nn.Module):
    def __init__(self, data, parameterization, critic_index, *, initialization_seed=None):
        super().__init__()
        if initialization_seed is None:
            critic = QuasimetricCritic.Conf().make(env_spec=EnvSpec.from_env(data.env))
            saved = torch.load(INIT, map_location='cpu', weights_only=True)['agent']
            prefix = f'critics.{critic_index}.'
            critic.load_state_dict({k[len(prefix):]: v for k,v in saved.items() if k.startswith(prefix)})
        else:
            # Fresh production initialization of encoder, projector and IQE.
            # No historical critic weights or optimizer state are read.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed)
                critic = QuasimetricCritic.Conf().make(env_spec=EnvSpec.from_env(data.env))
            critic.encoder.enable_comm_goal_set_quotient()
        assert bool(critic.encoder.comm_goal_set_quotient)
        self.register_buffer('nodes', data.nodes.clone())
        self.parameterization = parameterization
        self.head = critic.quasimetric_model.quasimetric_head
        if parameterization == 'encoder':
            self.encoder = critic.encoder
            self.projector = critic.quasimetric_model.projector
        else:
            with torch.no_grad():
                initial = critic.quasimetric_model.projector(critic.encoder(data.nodes))
            self.latent = nn.Parameter(initial.clone())
    def project(self):
        return self.latent if self.parameterization == 'latent' else self.projector(self.encoder(self.nodes))
    def distances(self, data):
        z = self.project()
        a = torch.cat((data.src, data.task_src, data.task_src))
        b = torch.cat((data.dst, data.goal, data.pair))
        values = self.head(z[a], z[b])
        return values.split((len(data.src), len(data.rows), len(data.rows)))


class CachedBarrier(GoalSetBarrierUpdater):
    """Exact production workset/loss on already evaluated full edge distances."""
    def _edge_values(self, critic, local, indices):
        return self.residual[indices], self.cost[indices]


class FixedPush:
    def __init__(self, task, pair):
        self.value = .6 * task.mean() + .4 * pair.mean()
    def __call__(self, data, info):
        return LossResult(loss=-self.value, info={})


def graph_reference(data):
    # Evaluation only. No reference tensors are passed to training_objective.
    n = len(data.nodes)
    rev = [[] for _ in range(n)]
    for a,b,c in zip(data.src.tolist(), data.dst.tolist(), data.cost.tolist()):
        rev[b].append((a,c))
    dist = np.full(n, np.inf)
    queue = []
    for g in data.goal.unique().tolist():
        dist[g]=0; heapq.heappush(queue,(0,g))
    while queue:
        d,u=heapq.heappop(queue)
        if d != dist[u]: continue
        for v,c in rev[u]:
            if d+c < dist[v]:
                dist[v]=d+c; heapq.heappush(queue,(d+c,v))
    assert np.isfinite(dist).all()
    ref = torch.tensor(dist, dtype=torch.float64)
    # Verify every physical edge lies on a graph-shortest suffix. Thus fixed
    # B1 reference is a graph distance, not just an arbitrary path upper bound.
    assert torch.allclose(ref[data.src]-ref[data.dst], data.cost.double(), atol=1e-8, rtol=0)
    return ref


def training_objective(model, data, solver, method):
    edge, task, pair = model.distances(data)
    residual = edge-data.cost
    push = FixedPush(task, pair)
    if method == 'phr':
        loss = -push.value + solver(residual)
    else:
        solver.residual, solver.cost = residual, data.cost
        loss = solver.step(None, SimpleNamespace(critic=model), push, None).loss
    return loss, residual, push.value


@torch.no_grad()
def evaluate(model, data, solver, step, elapsed):
    edge, task, pair = model.distances(data)
    residual = edge-data.cost
    ref = graph_reference(data)
    target = ref[data.task_src]
    pair_ref = ref[data.task_src]-ref[data.pair]
    reference = float(.6*target.mean()+.4*pair_ref.mean())
    push = float(.6*task.mean()+.4*pair.mean())
    real_r = residual[data.physical]
    scale = min(1., float((data.cost[data.physical] / edge[data.physical].clamp_min(1e-30)).min()))
    z = model.project()
    all_task = model.head(z, z[data.goal[0]].expand_as(z))
    record = dict(step=step, seconds=elapsed, push=push, push_reference=reference,
        push_gap=reference-push, push_gap_percent=100*(reference-push)/reference,
        feasible_scale=scale, feasible_push=scale*push,
        feasible_push_gap_percent=100*(reference-scale*push)/reference,
        task_rmse=float(((task.double()-target).square().sum()/(len(task)+4)).sqrt()),
        task_nrmse_percent=float(100*((task.double()-target).square().mean()).sqrt()/target.square().mean().sqrt()),
        pair_rmse=float((pair.double()-pair_ref).square().mean().sqrt()),
        max_residual=float(real_r.max()), max_violation=float(real_r.clamp_min(0).max()),
        violation_count=int((real_r>1e-5).sum()), mean_positive_residual=float(real_r.relu().mean()),
        anchor_max=float(residual[~data.physical].abs().max()), terminal_max=float(all_task[data.goal.unique()].abs().max()),
        full_top8_penalty=float(500*(.01*torch.nn.functional.softplus((real_r+.01)/.01)).topk(8).values.mean()))
    if isinstance(solver, TabularPHR):
        record.update(dual_max=float(solver.multiplier.max()), dual_mean=float(solver.multiplier.mean()),
            complementarity=float((solver.multiplier*residual).abs().mean()))
    paths, values = [], []
    context = data.raw.transition_infos['context_id']
    for path in range(4):
        erows = ((context==path)&data.physical).nonzero().flatten()
        seq = torch.cat((data.src[erows], data.dst[erows[-1:]]))
        pred = all_task[seq]; truth = ref[seq]
        slack = float((data.cost[erows]-edge[erows]).sum())
        triangle = float((edge[erows]+pred[1:]-pred[:-1]).sum())
        assert abs(slack+triangle-float(truth[0]-pred[0])) < .003
        paths.append(dict(path=path, start=float(pred[0]), reference=float(truth[0]), edge_slack=slack, triangle_slack=triangle))
        for t,node in enumerate(seq):
            values.append(dict(step=step,path=path,timestep=t,node=int(node),value=float(pred[t]),reference=float(truth[t])))
    record['paths']=paths
    arrays = dict(edge=edge.numpy(), residual=residual.numpy(), task=task.numpy(), pair=pair.numpy(), all_task=all_task.numpy())
    return record, values, arrays


def make_optimizer(model, lr):
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    return torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in head_ids], 'lr': lr},
        {'params': head_params, 'lr': 2e-5},
    ], weight_decay=0.)


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data = load_data()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    model = IndexedIQE(data, args.parameterization, args.critic)
    lr = args.lr if args.lr is not None else (2e-5 if args.parameterization=='encoder' else .05)
    optimizer = make_optimizer(model, lr)
    solver = TabularPHR(len(data.src), args.rho, args.dual_step) if args.method=='phr' else CachedBarrier(GoalSetBarrierConf())
    if args.method=='barrier': solver.set_replay(data.replay)
    config = dict(vars(args), actual_lr=lr, head_lr=2e-5, fingerprints=data.fingerprint,
        initialization_sha256=tensor_digest(*model.state_dict().values()),
        initial_projected_sha256=tensor_digest(model.project()),
        push='exact uniform physical-source expectation: 0.6 task + 0.4 fixed B1; no auxiliary loss',
        phr_scaling='mean over 339 edges; lambda <- relu(lambda + dual_step * pre-update h), every step',
        barrier='production 64-workedge top8, weight500 margin=temperature=.01 audit25',
        nodes=len(data.nodes), edges=len(data.src), physical_edges=len(data.rows),
        trainable_parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
        python=platform.python_version(), platform=platform.platform(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=file_digest(__file__))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    np.savez(out/'support.npz',nodes=data.nodes.numpy(),src=data.src.numpy(),dst=data.dst.numpy(),
        cost=data.cost.numpy(),task_src=data.task_src.numpy(),goal=data.goal.numpy(),pair=data.pair.numpy())
    torch.save(dict(model=model.state_dict(), fingerprint=data.fingerprint),out/'initial.pt')
    curve=(out/'curve.jsonl').open('w'); vf=(out/'values.csv').open('w'); writer=None
    started=time.perf_counter(); train_seconds=0
    for step in range(args.steps+1):
        if step % args.eval_interval==0 or step==args.steps:
            rec, vals, arrays=evaluate(model,data,solver,step,time.perf_counter()-started)
            rec['train_seconds']=train_seconds
            curve.write(json.dumps(rec)+'\n'); curve.flush()
            if writer is None:
                writer=csv.DictWriter(vf,fieldnames=list(vals[0])); writer.writeheader()
            writer.writerows(vals); vf.flush()
            print(json.dumps({k:rec[k] for k in ('step','seconds','push','push_gap_percent','task_rmse','max_violation','violation_count')}),flush=True)
            np.savez(out/'latest_edges.npz',**arrays)
            torch.save(dict(step=step,model=model.state_dict(),solver=solver.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),out/'latest.pt')
            if step in (2000, 5000, 10000, 20000):
                torch.save(dict(step=step,model=model.state_dict(),solver=solver.state_dict()),out/f'checkpoint_{step:05d}.pt')
        if step==args.steps: break
        tick=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss,residual,push=training_objective(model,data,solver,args.method)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        loss.backward()
        optimizer.step()
        if args.method=='phr': solver.update(residual.detach())
        # Don't retain autograd graphs through cache references across updates.
        if args.method=='barrier': solver.residual=solver.residual.detach()
        train_seconds+=time.perf_counter()-tick
    curve.close(); vf.close()
    (out/'COMPLETE').write_text(str(args.steps)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--parameterization',choices=['latent','encoder'],required=True)
    p.add_argument('--method',choices=['barrier','phr'],required=True)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--critic',type=int,default=0,choices=[0,1])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float)
    p.add_argument('--rho',type=float,default=10.)
    p.add_argument('--dual-step',type=float,default=.1)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--eval-interval',type=int,default=250)
    run(p.parse_args())
