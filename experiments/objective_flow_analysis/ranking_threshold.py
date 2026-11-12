"""Check exact ranking thresholds against globally certified flow solutions.

All problems here are finite synthetic convex problems. No network training,
saved model changes, or environment episodes are performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.sparse.csgraph import shortest_path
from scipy.special import expit


def diamond(demand, p, delta):
    # State order s, a, b, g. Edge order sa, ag, sb, bg, gs.
    r, a, b = np.asarray(demand, float)
    return dict(src=np.array([0, 1, 0, 2, 3]), dst=np.array([1, 3, 2, 3, 0]),
                cost=np.array([1., 1., 1., 1.+delta, 1.]), p=np.asarray(p, float),
                mu=np.array([r, a, b]),
                f0=np.array([0., a, r, b+r, 0.]),
                f1=np.array([r, a+r, 0., b, 0.]),
                first=(0, 2), tail_a=(1,), tail_b=(3,))


def previous_six_node():
    r, a, u, b, v = .05, .01, .92, .01, .01
    return dict(src=np.array([0, 1, 2, 0, 3, 4, 5]),
                dst=np.array([1, 2, 5, 3, 4, 5, 0]),
                cost=np.array([1., 1., 1., 1., 1., 1.25, 1.]),
                p=np.full(7, 1/7), mu=np.array([r, a, u, b, v]),
                f0=np.array([0., a, a+u, r, b+r, v+b+r, 0.]),
                f1=np.array([r, a+r, u+a+r, 0., b, v+b, 0.]),
                first=(0, 3), tail_a=(1, 2), tail_b=(4, 5))


def distances(case, x):
    n = len(case['mu'])+1
    adj = np.full((n, n), np.inf)
    np.fill_diagonal(adj, 0)
    adj[case['src'], case['dst']] = case['cost']+x
    return shortest_path(adj, directed=True)


def threshold(case):
    p, f0, f1 = (case[k] for k in ('p', 'f0', 'f1'))
    ea, eb = case['first']
    slope = f1-f0
    theta = (f0[eb]/p[eb]-f0[ea]/p[ea])/(slope[ea]/p[ea]-slope[eb]/p[eb])
    assert 0 < theta < 1, 'The sign theorem used here requires an interior balance point.'
    flow = f0+theta*(f1-f0)
    norm = np.sqrt(np.sum(flow*flow/p))
    tail = sum(flow[e]/p[e] for e in case['tail_a'])-sum(flow[e]/p[e] for e in case['tail_b'])
    return float(theta), float(tail/norm)


def solve(case, epsilon):
    p, c, mu, f0, f1 = (case[k] for k in ('p', 'cost', 'mu', 'f0', 'f1'))
    slope = f1-f0
    def derivative(theta):
        flow = f0+theta*slope
        return slope@c+epsilon*np.sum(slope*flow/p)/np.sqrt(np.sum(flow*flow/p))
    if derivative(0) >= 0:
        theta = 0.
    elif derivative(1) <= 0:
        theta = 1.
    else:
        theta = brentq(derivative, 0, 1, xtol=1e-14)
    flow = f0+theta*slope
    norm = np.sqrt(np.sum(flow*flow/p))
    extra = epsilon*flow/(p*norm)
    d = distances(case, extra)
    goal = len(mu)
    upper = c@flow+epsilon*norm
    value = np.sum(case['queries']*d) if 'queries' in case else mu@d[:goal, goal]
    risk = p@np.maximum(d[case['src'], case['dst']]-c, 0)**2
    assert abs(value-upper) < 1e-9
    assert abs(risk-epsilon**2) < 1e-10
    assert np.max(d[:, None, :]-d[:, :, None]-d[None, :, :]) < 1e-10
    injection = (case['queries'].sum(axis=1)-case['queries'].sum(axis=0)
                 if 'queries' in case else np.r_[mu, -mu.sum()])
    balance = np.bincount(case['src'], weights=flow, minlength=goal+1)-np.bincount(case['dst'], weights=flow, minlength=goal+1)
    np.testing.assert_allclose(balance, injection, rtol=0, atol=1e-12)
    ea, eb = case['first']
    sa, sb = case['dst'][ea], case['dst'][eb]
    nominal = distances(case, np.zeros_like(c))
    gap = c[eb]+nominal[sb, goal]-c[ea]-nominal[sa, goal]
    model_gap = c[eb]+d[sb, goal]-c[ea]-d[sa, goal]
    theta_equal, kappa = threshold(case)
    predicted = gap-epsilon*kappa
    if abs(predicted) > 1e-10:
        assert np.sign(predicted) == np.sign(model_gap)
    else:
        assert abs(model_gap) < 1e-10
        assert abs(theta-theta_equal) < 1e-10
    return dict(theta=theta, theta_equal=theta_equal, kappa=kappa,
                epsilon=epsilon, risk=float(risk), true_gap=float(gap),
                predicted_sign_quantity=float(predicted), learned_gap=float(model_gap),
                primal_value=float(value), dual_upper=float(upper),
                inflation=extra.tolist(), goal_column=d[:, goal].tolist())


def independent_potential_check(case, epsilon, shaped=False):
    # A convex potential formulation, independent of the scalar flow solver.
    mu, c, p = (case[k] for k in ('mu', 'cost', 'p'))
    n, m = len(mu), len(c)
    incidence = np.zeros((m, n))
    for i, (s, t) in enumerate(zip(case['src'], case['dst'])):
        if s < n:
            incidence[i, s] += 1
        if t < n:
            incidence[i, t] -= 1
    matrix = np.c_[incidence, -np.eye(m)]
    initial = np.r_[distances(case, np.zeros(m))[:n, n], np.full(m, epsilon/4)]
    def utility(j):
        return -100*np.logaddexp(0, 5-j/100) if shaped else j
    def gradient(j):
        return expit(5-j/100) if shaped else np.ones(n)
    result = minimize(lambda z: -mu@utility(z[:n]), initial,
                      jac=lambda z: np.r_[-mu*gradient(z[:n]), np.zeros(m)],
                      bounds=[(0, None)]*(n+m), method='SLSQP',
                      constraints=[dict(type='ineq', fun=lambda z: c-matrix@z, jac=lambda z: -matrix),
                                   dict(type='ineq', fun=lambda z: epsilon**2-p@(z[n:]**2),
                                        jac=lambda z: np.r_[np.zeros(n), -2*p*z[n:]])],
                      options=dict(ftol=1e-12, maxiter=1000))
    assert result.success, result.message
    extra = result.x[n:]
    extra *= min(1., epsilon/np.sqrt(p@(extra**2))) * (1-1e-13)
    d = distances(case, extra)
    objective = mu@utility(d[:n, n])
    tangent = dict(case)
    tangent['mu'] = mu*gradient(d[:n, n])
    assert len(mu) == 3, 'This independent check uses only the diamond.'
    r, a, b = tangent['mu']
    tangent['f0'] = np.array([0., a, r, b+r, 0.])
    tangent['f1'] = np.array([r, a+r, 0., b, 0.])
    certificate = solve(tangent, epsilon)
    upper = objective+certificate['dual_upper']-tangent['mu']@d[:n, n]
    assert -1e-9 <= upper-objective < 1e-7
    ea, eb = case['first']
    gap = c[eb]+d[case['dst'][eb], n]-c[ea]-d[case['dst'][ea], n]
    return dict(shaped=shaped, learned_gap=float(gap), upper_gap=float(upper-objective))


def all_pair_support():
    # Every state-goal query has fixed positive mass, with a product distribution.
    source = np.array([.15, .6, .2, .05])
    goal = np.array([.1, .1, .1, .7])
    r = source[0]*goal[3]
    fixed = np.array([goal[1]*(1-source[1]), source[1]*(1-goal[1]),
                      goal[2]*(1-source[2]), source[2]*(1-goal[2]),
                      source[1]*(goal[0]+goal[2])+source[2]*(goal[0]+goal[1])
                      +source[3]*(1-goal[3])])
    case = diamond(source[:3]*goal[3], np.full(5, .2), .02)
    case['queries'] = np.outer(source, goal)
    case['f0'] = fixed+np.array([0., 0., r, r, 0.])
    case['f1'] = fixed+np.array([r, r, 0., 0., 0.])
    row = solve(case, .05)
    np.testing.assert_allclose(row['theta_equal'], 29/42, rtol=0, atol=1e-14)
    np.testing.assert_allclose(row['kappa'], 2/np.sqrt(381/160), rtol=0, atol=1e-14)
    assert row['learned_gap'] < 0
    return dict(source_distribution=source.tolist(), goal_distribution=goal.tolist(),
                minimum_pair_mass=float(case['queries'].min()), **row)


def run():
    demands = [[.2, .6, .2], [.05, .9, .05], [.4, .2, .4], [.1, .45, .45]]
    frequencies = [np.full(5, .2), [.1, .35, .2, .25, .1],
                   [.225, .1, .225, .225, .225], [.24975, .001, .24975, .24975, .24975]]
    total, reversed_count, ties = 0, 0, 0
    for demand in demands:
        for p in frequencies:
            for epsilon in (.02, .1, np.sqrt(.05), 1.):
                _, kappa = threshold(diamond(demand, p, .2))
                deltas = [.05, .2, .5]
                if kappa > 0:
                    deltas += [.5*epsilon*kappa, epsilon*kappa, 1.5*epsilon*kappa]
                for delta in deltas:
                    row = solve(diamond(demand, p, delta), epsilon)
                    total += 1
                    reversed_count += row['learned_gap'] < -1e-9
                    ties += abs(row['learned_gap']) <= 1e-9
    demand = [.2, .6, .2]
    p = np.full(5, .2)
    delta = .2
    case = diamond(demand, p, delta)
    _, kappa = threshold(case)
    np.testing.assert_allclose(kappa, 2/np.sqrt(3), rtol=0, atol=1e-14)
    critical = delta/kappa
    boundary = [solve(case, factor*critical) for factor in (.5, 1, 1.5)]
    independent = [independent_potential_check(case, factor*critical) for factor in (.5, 1, 1.5)]
    shaped = [dict(delta=gap, **independent_potential_check(diamond(demand, p, gap), np.sqrt(.05), shaped=True))
              for gap in (0., .01, .2)]
    assert all(row['learned_gap'] < 0 for row in shaped)
    old = solve(previous_six_node(), np.sqrt(.05))
    old['critical_budget'] = (old['true_gap']/old['kappa'])**2
    rare = []
    for q in (.2, .1, .01, .001, .0001):
        p_rare = np.full(5, (1-q)/4)
        p_rare[1] = q
        row = solve(diamond(demand, p_rare, 1.), .05)
        rare.append(dict(q=q, kappa=row['kappa'], learned_gap=row['learned_gap']))
    return dict(training_updates=0, grid_cases=total, grid_strict_reversals=int(reversed_count),
                grid_ties=int(ties), all_analytic_signs_match=True,
                diamond=dict(demand=demand, edge_weights=p.tolist(), true_gap=delta,
                             kappa=kappa, critical_epsilon=critical,
                             critical_budget=critical**2, boundary=boundary,
                             independent_linear_checks=independent, shaped_checks=shaped),
                previous_six_node=old, rare_edge_fixed_gap=rare,
                all_pair_product_distribution=all_pair_support())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
