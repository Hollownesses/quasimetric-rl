"""Independent post-training policy evaluation and full-space references."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gap_stochastic.core import *


def oracle(mu):
    adjacency = np.full((N, N), np.inf)
    np.fill_diagonal(adjacency, 0.)
    adjacency[SRC, DST] = COST
    truth = shortest_path(adjacency, directed=True)
    lp = linprog(COST[:6], A_eq=INC[:, :6], b_eq=mu@INJ,
                 bounds=[(0, None)]*6, method='highs')
    assert lp.success
    assert abs(lp.fun-mu@truth[:GOAL, GOAL]) < 1e-10
    return truth, lp.x, float(lp.fun)


def policy_evaluation(edge_probabilities, mu):
    transition = np.zeros((N, N))
    running_cost = np.zeros(N)
    for e in range(6):
        transition[SRC[e], DST[e]] += edge_probabilities[e]
        running_cost[SRC[e]] += edge_probabilities[e]*COST[e]
    # Goal is absorbing, independently of its outgoing data transition.
    matrix = np.eye(GOAL)-transition[:GOAL, :GOAL]
    hitting = np.linalg.solve(matrix, transition[:GOAL, GOAL])
    stopped_costs = np.linalg.solve(matrix, running_cost[:GOAL])
    proper_states = hitting >= 1-1e-10
    occupancy = np.linalg.solve(matrix.T, mu)
    executed_flow = occupancy[SRC[:6]]*edge_probabilities
    proper = bool(np.all(proper_states[mu > 0]))
    return dict(proper=proper, hitting_probability=hitting.tolist(),
                state_cost=[float(c) if ok else None for c, ok in zip(stopped_costs, proper_states)],
                source_cost=float(stopped_costs[0]) if proper_states[0] else None,
                query_cost=float(mu@stopped_costs) if proper else None,
                cheap_probability=float(edge_probabilities[0]),
                edge_probabilities=edge_probabilities.tolist(),
                executed_flow=executed_flow.tolist())


def policy_from_flow(f):
    pi = np.zeros(6)
    for s in range(GOAL):
        mask = SRC[:6] == s
        total = f[mask].sum()
        if total > 0:
            pi[mask] = f[mask]/total
    return pi


def policy_from_distance(d):
    pi = np.zeros(6)
    for s in range(GOAL):
        choices = np.flatnonzero(SRC[:6] == s)
        scores = COST[choices]+d[DST[choices], GOAL]
        selected = choices[scores <= scores.min()+1e-8]
        pi[selected] = 1/len(selected)
    return pi


def metrics(x, mu, method, truth, optimal_cost):
    d = distance_matrix(x)
    _, _, parts = population(x, mu)
    greedy = policy_evaluation(policy_from_distance(d), mu)
    parts.update(critic_policy=greedy,
                 critic_margin=float(d[3, GOAL]-d[1, GOAL]),
                 maximum_goal_distance_error=float(np.max(abs(d[:GOAL, GOAL]-truth[:GOAL, GOAL]))),
                 cone_violation=float(max(0., -np.min(CONE@x[:30]))),
                 optimal_query_cost=optimal_cost,
                 risk_cap_excess=float(parts['risk']-.05))
    if method != 'qrl':
        policy = policy_evaluation(policy_from_flow(x[30:]), mu)
        parts['policy'] = policy
        parts['policy_regret'] = None if policy['query_cost'] is None else policy['query_cost']-optimal_cost
        parts['flow_cost_error_to_execution'] = None if policy['query_cost'] is None else parts['flow_cost']-policy['query_cost']
        parts['learned_to_executed_flow_l1'] = float(np.sum(abs(x[30:]-np.array(policy['executed_flow']))))
        parts['joint_pass'] = bool(policy['proper'] and abs(parts['gap']) <= .005 and parts['risk'] <= 1e-4
                                   and parts['residual_linf'] <= .005 and policy['source_cost'] <= 3.025
                                   and parts['policy_regret'] <= .005)
        parts['residual_relative_to_injection'] = (np.abs(parts['residual'])/np.abs(mu@INJ)).tolist()
    else:
        # Baseline has no learned arrival flow. Never score its untouched dummy
        # coordinates as a learned-flow result, or its evaluation flow as one.
        for name in ('gap', 'flow_cost', 'residual', 'residual_l2', 'residual_linf', 'pure_loss',
                     'gap_sampling_variance', 'conservation_sampling_variance'):
            parts.pop(name)
        parts['policy'] = greedy
        parts['policy_regret'] = greedy['query_cost']-optimal_cost
        parts['executed_cost_minus_distance'] = greedy['query_cost']-parts['value']
        parts['joint_pass'] = None
    return parts


def expected_reference(mu, diagonal_gap, diagonal_flow, batch=256):
    x0 = initial()
    fn = lambda x: population(x, mu, batch=batch, diagonal_gap=diagonal_gap, diagonal_flow=diagonal_flow)
    initial_loss = fn(x0)[0]
    dmax = 5*np.max(COST+np.sqrt(initial_loss/P))
    fmax = (dmax+np.sqrt(initial_loss))/COST.min()
    bounds = [(0., float(dmax))]*30+[(0., float(fmax))]*6
    constraint = np.c_[CONE, np.zeros((len(CONE), 6))]
    result = minimize(lambda x:fn(x)[0], x0, jac=lambda x:fn(x)[1],
                      constraints=[dict(type='ineq', fun=lambda x:constraint@x, jac=lambda x:constraint)],
                      bounds=bounds, method='SLSQP', options=dict(ftol=1e-15, maxiter=4000))
    if not result.success:
        raise ArithmeticError(result.message)
    gradient = fn(result.x)[1]
    scale = max(float(np.max(abs(gradient))), 1e-15)
    linear = linprog(gradient/scale, A_ub=-constraint, b_ub=np.zeros(len(constraint)), bounds=bounds,
                     method='highs', options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    assert linear.success
    linearization_gap = float(gradient@(result.x-linear.x))
    objective_value = fn(result.x)[0]
    # All four expected objectives are nonnegative. For a near-zero solution,
    # this elementary lower bound is tighter than the floating-point LP bound.
    certificate_gap = min(objective_value, max(0., linearization_gap))
    assert np.min(constraint@result.x) >= -1e-8
    assert -1e-8 <= linearization_gap and certificate_gap < 1e-5
    return result.x, dict(expected_objective=objective_value, certificate_gap=certificate_gap,
                         linearization_gap=linearization_gap,
                         iterations=result.nit, diagonal_gap=diagonal_gap, diagonal_flow=diagonal_flow)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.folder/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    start = time.perf_counter()
    results = dict(runs={}, references={}, oracles={})
    histories = []
    for name, mu in QUERIES.items():
        truth, optimal_flow, optimal_cost = oracle(mu)
        results['oracles'][name] = dict(distance=truth.tolist(), flow=optimal_flow.tolist(), cost=optimal_cost)
        for key, info in manifest['runs'].items():
            if key.split('__')[1] != name:
                continue
            data = np.load(args.folder/(key+'.npz'))
            for step, x, multiplier in zip(data['step'], data['x'], data['multiplier']):
                row = metrics(x, mu, info['method'], truth, optimal_cost)
                row.update(key=key, step=int(step), multiplier=float(multiplier))
                histories.append(row)
            row = histories[-1].copy()
            if info['method'] != 'qrl':
                obj, _, _ = population(data['x'][-1], mu, diagonal_gap=info['method'] == 'naive', diagonal_flow=info['method'] == 'naive')
                row['trained_expected_objective'] = obj
            results['runs'][key] = row
        for dg, df, tag in ((False, False, 'unbiased'), (True, True, 'naive'),
                            (True, False, 'gap_diagonal_only'), (False, True, 'flow_diagonal_only')):
            x, meta = expected_reference(mu, dg, df)
            meta.update(metrics(x, mu, tag, truth, optimal_cost))
            meta['x'] = x.tolist()
            results['references'][name+'__'+tag] = meta
    results['evaluation_seconds'] = time.perf_counter()-start
    results['total_seconds_including_training'] = results['evaluation_seconds']+manifest['total_training_seconds']
    assert results['total_seconds_including_training'] < manifest['config']['wall_seconds']
    (args.folder/'evaluation.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    with (args.folder/'history.jsonl').open('w') as output:
        for row in histories:
            output.write(json.dumps(row, allow_nan=False)+'\n')
    for key, row in results['runs'].items():
        print(key, json.dumps({k:row.get(k) for k in ('gap','risk','residual_linf','flow_cost','policy_regret','joint_pass')}),
              'source_cost', row['policy']['source_cost'], 'critic_cost', row['critic_policy']['source_cost'])
    for key, row in results['references'].items():
        print('REFERENCE',key,'cost',row['policy']['source_cost'],'residual',row['residual_linf'],
              'gap',row['gap'],'objective',row['expected_objective'],'cert',row['certificate_gap'])


if __name__ == '__main__':
    main()
