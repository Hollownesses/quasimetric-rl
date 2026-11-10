"""Numerical derivative and exact independent-pair estimator checks."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gap_stochastic.core import *


def finite_difference(fn, x):
    h = 1e-6
    result = np.empty(len(x))
    for j in range(len(x)):
        a, b = x.copy(), x.copy()
        a[j] += h
        b[j] -= h
        result[j] = (fn(a)-fn(b))/(2*h)
    return result


def check():
    x = np.r_[np.linspace(.6, 2., 30), [.12, .27, .83, .05, .21, .34]]
    rng = np.random.default_rng(2001)
    edges = rng.integers(0, 7, 256)
    mu = QUERIES['cheap_suffix_queries']
    queries = np.searchsorted(np.cumsum(mu), rng.random(256))
    checks = {}
    for method in ('qrl', 'naive', 'unbiased'):
        loss, gradient, _ = sampled(x, edges, queries, method)
        fd = finite_difference(lambda z: sampled(z, edges, queries, method)[0], x)
        checks['minibatch_derivative_'+method] = float(np.max(abs(fd-gradient)))
        assert np.max(abs(fd-gradient)) < 1e-7
    for dg, df in ((False, False), (True, True), (True, False), (False, True)):
        _, gradient, _ = population(x, mu, diagonal_gap=dg, diagonal_flow=df)
        fd = finite_difference(lambda z: population(z, mu, diagonal_gap=dg, diagonal_flow=df)[0], x)
        checks[f'population_derivative_{dg}_{df}'] = float(np.max(abs(fd-gradient)))
        assert np.max(abs(fd-gradient)) < 1e-7
    atoms = [(e, q, P[e]*mu[q]) for e in range(7) for q in range(5)]
    for method in ('naive', 'unbiased'):
        loss, gradient = 0., np.zeros(36)
        for e, q, p in atoms:
            for e2, q2, p2 in atoms:
                value, grad, _ = sampled(x, np.array([e, e2]), np.array([q, q2]), method)
                loss += p*p2*value
                gradient += p*p2*grad
        ref, ref_grad, _ = population(x, mu, batch=2, diagonal_gap=method == 'naive', diagonal_flow=method == 'naive')
        checks['pair_loss_'+method] = abs(loss-ref)
        checks['pair_gradient_'+method] = float(np.max(abs(gradient-ref_grad)))
        assert abs(loss-ref) < 1e-10
        assert np.max(abs(gradient-ref_grad)) < 1e-10
    target = rng.standard_normal(30)
    metric = np.exp(np.linspace(-8, 1, 30))
    projected = project_distance(target, metric)
    reference = minimize(lambda z:.5*np.sum(metric*(z-target)**2), np.ones(30),
                         jac=lambda z:metric*(z-target), method='SLSQP',
                         constraints=[dict(type='ineq', fun=lambda z:CONE@z, jac=lambda z:CONE)],
                         options=dict(ftol=1e-13, maxiter=1000))
    assert reference.success
    checks['weighted_projection_objective_difference'] = float(abs(.5*np.sum(metric*(projected-target)**2)-reference.fun))
    checks['weighted_projection_cone_violation'] = float(max(0., -np.min(CONE@projected)))
    assert checks['weighted_projection_objective_difference'] < 1e-10
    assert checks['weighted_projection_cone_violation'] < 1e-8
    # Flow coordinates remain unconstrained apart from elementwise positivity.
    arbitrary = initial()
    arbitrary[30:] = [.2, .7, .4, .9, .01, .3]
    np.testing.assert_array_equal(project(arbitrary)[30:], arbitrary[30:])
    checks['free_flow_residual_not_projected'] = float(np.linalg.norm(INC@np.r_[project(arbitrary)[30:],0.]-mu@INJ))
    assert checks['free_flow_residual_not_projected'] > .5
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = check()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
