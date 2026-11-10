"""Read-only endpoint diagnostics, including exact U-statistic gradient variance."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import null_space

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gap_stochastic.core import *
from experiments.gap_stochastic.evaluate import oracle, policy_evaluation, policy_from_flow


def kernel_variance(kernel, weights, batch):
    conditional = kernel@weights
    mean = float(weights@conditional)
    first = float(weights@(conditional-mean)**2)
    remainder = kernel-conditional[:, None]-conditional[None, :]+mean
    second = float(np.sum(weights[:, None]*weights[None, :]*remainder**2))
    return dict(mean=mean, variance=4*first/batch+2*second/(batch*(batch-1)),
                first_projection_variance=first, degenerate_variance=second)


def route_gradient_noise(x, mu, batch=256):
    basis = null_space(INC[:, :6])
    assert basis.shape == (6, 1)
    direction = basis[:, 0]
    if COST[:6]@direction > 0:
        direction *= -1
    zdir = np.r_[direction, 0.]
    es = np.repeat(np.arange(7), 5)
    qs = np.tile(np.arange(5), 7)
    weights = P[es]*mu[qs]
    f = np.r_[x[30:], 0.]
    z = COST[es]*f[es]/P[es]-x[QUERY[qs]]
    H = (f[es]/P[es])[:, None]*INC[:, es].T-INJ[qs]
    dz = COST[es]*zdir[es]/P[es]
    dH = (zdir[es]/P[es])[:, None]*INC[:, es].T
    gap_kernel = z[:, None]*dz[None, :]+dz[:, None]*z[None, :]
    flow_kernel = dH@H.T+H@dH.T
    result = {name:kernel_variance(k, weights, batch) for name,k in
              (('gap', gap_kernel), ('conservation', flow_kernel), ('total', gap_kernel+flow_kernel))}
    exact_grad = population(x, mu)[1][30:]@direction
    assert abs(result['total']['mean']-exact_grad) < 1e-10
    assert abs(result['conservation']['mean']) < 1e-10
    result['direction'] = direction.tolist()
    result['cost_derivative'] = float(COST[:6]@direction)
    result['nullspace_error'] = float(np.linalg.norm(INC[:, :6]@direction))
    result['gradient_std'] = float(np.sqrt(result['total']['variance']))
    result['signal_to_noise'] = abs(exact_grad)/max(result['gradient_std'], 1e-15)
    result['naive_expected_directional_gradient'] = float(population(x, mu, diagonal_gap=True, diagonal_flow=True)[1][30:]@direction)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.folder/'manifest.json').read_text())
    evaluation = json.loads((args.folder/'evaluation.json').read_text())
    audit = dict(runs={}, ideal={}, all_primary_sampling_identical=False, policy_checks={})
    hashes = set()
    for key, info in manifest['runs'].items():
        if info['mode'] != 'stochastic':
            continue
        hashes.add(info['sampling_sha256'])
        data = np.load(args.folder/(key+'.npz'))
        x = data['x'][-1]
        mu = QUERIES[key.split('__')[1]]
        row = evaluation['runs'][key]
        pi = row['policy']['edge_probabilities']
        zero_states = [s for s in range(GOAL) if np.sum(np.array(pi)[SRC[:6] == s]) == 0]
        record = dict(zero_action_states=zero_states,
                      snapshot_sha256=hashlib.sha256((args.folder/(key+'.npz')).read_bytes()).hexdigest())
        if info['method'] != 'qrl':
            mode = info['method']
            _, gradient, _ = population(x, mu, diagonal_gap=mode == 'naive', diagonal_flow=mode == 'naive')
            h = 1e-3
            record.update(noise=route_gradient_noise(x, mu),
                          projected_gradient_norm=float(np.linalg.norm((x-project(x-h*gradient))/h)),
                          free_flow=x[30:].tolist(),
                          maximum_relative_source_residual=max(row['residual_relative_to_injection'][:GOAL]),
                          residual=row['residual'])
            if row['policy']['proper']:
                policy_value = np.r_[row['policy']['state_cost'], 0.]
                correction = float(policy_value@np.array(row['residual']))
                mismatch = row['flow_cost']-row['policy']['query_cost']
                assert abs(correction-mismatch) < 1e-12
                record['flow_to_policy_cost_correction'] = correction
                record['cost_correction_identity_error'] = abs(correction-mismatch)
        audit['runs'][key] = record
    audit['all_primary_sampling_identical'] = len(hashes) == 1
    assert audit['all_primary_sampling_identical']
    for name, mu in QUERIES.items():
        truth, flow, _ = oracle(mu)
        xstar = np.r_[[truth[s,t] for s,t in PAIRS], flow]
        audit['ideal'][name] = route_gradient_noise(xstar, mu)
    # Failure is not repaired with a default action. This vector has zero
    # outgoing mass at a, even though the policy from s can reach the goal.
    failure = policy_evaluation(policy_from_flow(np.array([0., 0., .9, .05, .06, .07])), QUERIES['cheap_suffix_queries'])
    assert not failure['proper'] and failure['query_cost'] is None and failure['source_cost'] == 3.25
    audit['policy_checks']['zero_outflow_not_repaired'] = failure
    true_eval = oracle(QUERIES['cheap_suffix_queries'])
    valid = policy_evaluation(policy_from_flow(true_eval[1]), QUERIES['cheap_suffix_queries'])
    assert valid['proper'] and abs(valid['query_cost']-true_eval[2]) < 1e-12
    audit['policy_checks']['oracle_flow_matches_absorbing_chain'] = True
    audit['current_source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (args.folder/'diagnostics.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    for key, record in audit['runs'].items():
        if 'noise' in record:
            n = record['noise']
            print(key, 'grad_mean',n['total']['mean'],'std',n['gradient_std'],
                  'SNR',n['signal_to_noise'],'flow_noise_std',np.sqrt(n['conservation']['variance']),
                  'zero_action_states',record['zero_action_states'])
    for key, n in audit['ideal'].items():
        print('IDEAL',key,'std',n['gradient_std'],'flow_std',np.sqrt(n['conservation']['variance']),
              'naive_direction',n['naive_expected_directional_gradient'])


if __name__ == '__main__':
    main()
