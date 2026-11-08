"""Functional stationarity audit of saved U-trap PHR states; no reference labels."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.connected_expert_utrap.data import load_training_data
from experiments.decision_intervals.audit import checked
from experiments.r_series_mppi.common import adapter
from experiments.task_source_balance.train import task_weights


def run(raw):
    torch.set_num_threads(1)
    replay = checked(raw, 'results/connected_expert_utrap_20261021/main/verified_replay.pt')
    data = load_training_data(replay.parent)
    physical = data.original.numpy()
    nodes = data.nodes.numpy()
    src = data.src[physical].numpy()
    dst = data.dst[physical].numpy()
    cost = data.cost[physical].numpy().astype(float)
    goals = data.goal[physical].numpy()
    pairs = data.pair[physical].numpy()
    weights, _ = task_weights(data)
    source_weights = weights[physical].numpy().astype(float)
    n, count = len(nodes), len(physical)
    for key in ('global_push_task_mask', 'global_push_pair_mask'):
        assert data.raw.transition_infos[key][data.original].bool().all()

    def divergence(values, targets):
        return (np.bincount(src, weights=values, minlength=n)
                - np.bincount(targets, weights=values, minlength=n))

    def score(model, targets):
        return np.concatenate([model.batch_value(nodes[src[i:i+512]], nodes[targets[i:i+512]])
                               for i in range(0, count, 512)]).astype(float)

    with np.load(REPO/'results/native_objective_witness_20261103/u/local.npz') as f:
        cache = {key: f[key].copy() for key in f.files}
    results = {}
    for arm in ('R0', 'R1'):
        path = checked(raw, f'results/task_source_balance_20261025/main/{arm}/final.pt')
        model, _ = adapter(data.small, path)
        state = torch.load(path, map_location='cpu', weights_only=True)
        multiplier = state['dual'][physical].numpy().astype(float)
        edge, task, pair = (score(model, targets) for targets in (dst, goals, pairs))
        np.testing.assert_allclose(edge, cache[arm+'_raw'], rtol=0, atol=0)
        weight = np.ones(count) if arm == 'R0' else source_weights
        demand = .6*divergence(weight/count, goals) + .4*divergence(np.ones(count)/count, pairs)
        effective_flow = np.maximum(multiplier+100*(edge-cost), 0)/count
        residual = demand-divergence(effective_flow, dst)
        phi = residual/np.max(np.abs(residual))

        def merit(step):
            tilted = [np.maximum(values+step*(phi[src]-phi[target]), 0)
                      for values, target in ((edge, dst), (task, goals), (pair, pairs))]
            h = tilted[0]-cost
            phr = np.mean((np.maximum(multiplier+100*h, 0)**2-multiplier**2)/200)
            push = .6*np.mean(weight*tilted[1])+.4*tilted[2].mean()
            return float(-push+phr)

        original = merit(0)
        points = [dict(step=t, merit=merit(t), change=merit(t)-original)
                  for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1)]
        direction = [phi[src]-phi[target] for target in (dst, goals, pairs)]
        derivatives = [np.where(values > 0, v, np.maximum(v, 0))
                       for values, v in zip((edge, task, pair), direction)]
        actual_derivative = (effective_flow@derivatives[0]
                             -.6*np.mean(weight*derivatives[1])-.4*np.mean(derivatives[2]))
        finite_derivative = (merit(1e-6)-original)/1e-6
        assert np.isclose(actual_derivative, finite_derivative, rtol=1e-4, atol=1e-5)
        assert min(p['change'] for p in points) < -1e-5
        results[arm] = dict(
            objective='frozen original PHR minus original task/B1 push',
            frozen_merit=original, aggregate_demand_l1=float(np.abs(demand).sum()),
            aggregate_flow_mass=float(effective_flow.sum()),
            conservation_residual_l1=float(np.abs(residual).sum()),
            conservation_residual_linf=float(np.abs(residual).max()),
            unclipped_directional_derivative=float(-residual@phi),
            actual_right_derivative=float(actual_derivative),
            finite_difference=float(finite_derivative),
            zero_distance_rows={k: int(np.sum(v == 0)) for k, v in zip(('edge', 'task', 'pair'), (edge, task, pair))},
            points=points)
    return dict(training_updates=0, reference_queries=0, physical_rows=count, nodes=n,
                note='Potential-tilt witnesses are valid quasipseudometrics, not certified members of the saved IQE architecture. Aggregate conservation is necessary only in unconstrained potential directions and is not a sufficient goal-wise flow certificate.',
                states=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = run(args.raw.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
