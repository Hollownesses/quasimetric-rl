"""Algebraic audits using logged transition rows only; no training claim.

Three logged actions: a->g (cost 1), b->g (cost 3), b->g (cost 4).
The two b actions are distinct actions despite sharing endpoints.
No environment adjacency or unobserved transitions are queried.
"""
import argparse
import json
from pathlib import Path

import numpy as np


SRC = np.array([0, 1, 1])
DST = np.array([2, 2, 2])
COST = np.array([1., 3., 4.])
P = np.full(3, 1/3)
MU = np.array([.5, .5])
PHI = np.array([[1., 0.], [0., 1.], [0., 0.]])
DELTA = PHI[SRC]-PHI[DST]
VALUES = np.array([1., 3.])  # Analytic reference for this audit, never a learned teacher.


def sample(w, values, a, u, edge, query, beta=1.):
    z = w[edge]*COST[edge]-values[query]
    h = w[edge]*DELTA[edge]-PHI[query]
    loss = 2*a*z-a*a+beta*(2*u@h-u@u)
    # Variables ordered as (w[3], values[2], a, u[2]).
    grad = np.zeros(8)
    grad[edge] = 2*a*COST[edge]+2*beta*u@DELTA[edge]
    grad[3+query] = -2*a
    grad[5] = 2*(z-a)
    grad[6:] = 2*beta*(h-u)
    return float(loss), grad, z, h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    flow = np.array([2/3, 0., 1/3])
    w = flow/P
    values = VALUES
    # A complete quasimetric with these goal distances; satisfies all logged costs.
    potential = np.r_[values, 0.]
    distance = np.maximum(potential[:, None]-potential[None, :], 0)
    triangle = distance[:, :, None]+distance[None, :, :]-distance[:, None, :]
    assert triangle.min() >= -1e-12
    risk = float(P@np.maximum(distance[SRC, DST]-COST, 0)**2)
    cost, value = float(flow@COST), float(MU@values)
    residual = (P*w)@DELTA-MU@PHI[:2]
    coarse_residual = float(residual@np.ones(2))
    # The normalized flow picks action cost 1 at a, cost 4 at b.
    policy_values = np.array([1., 4.])
    policy_cost = float(MU@policy_values)
    optimal_cost = float(MU@np.array([1., 3.]))
    assert abs(cost-value) < 1e-12 and risk == 0 and abs(coarse_residual) < 1e-12
    assert policy_cost-optimal_cost == .5
    assert abs(cost-policy_cost-policy_values@residual) < 1e-12
    assert policy_cost-optimal_cost <= abs(cost-value)+np.linalg.norm(policy_values)*np.linalg.norm(residual)+1e-12

    a, u = .37, np.array([.2, -.7])
    expected_loss, expected_gradient = 0., np.zeros(8)
    naive_inner_max = 0.
    for edge, pe in enumerate(P):
        for query, pq in enumerate(MU):
            loss, gradient, z, h = sample(w, values, a, u, edge, query)
            expected_loss += pe*pq*loss
            expected_gradient += pe*pq*gradient
            naive_inner_max += pe*pq*(z*z+h@h)
    gap = cost-value
    population_loss = 2*a*gap-a*a+2*u@residual-u@u
    population_gradient = np.r_[2*a*P*COST+2*P*(DELTA@u),
                                -2*a*MU, 2*(gap-a), 2*(residual-u)]
    error = max(abs(expected_loss-population_loss), float(np.max(abs(expected_gradient-population_gradient))))
    assert error < 1e-12
    assert naive_inner_max > gap*gap+residual@residual

    # A deliberately insufficient test using only the learned critic can hide
    # a zero-outflow source as well. This is a second independent witness.
    collapsed_values = np.array([1., 0.])
    collapsed_flow = np.array([.5, 0., 0.])
    collapsed_residual = collapsed_flow@DELTA-MU@PHI[:2]
    collapsed_gap = collapsed_flow@COST-MU@collapsed_values
    assert abs(collapsed_gap) < 1e-12
    assert abs(collapsed_residual@collapsed_values) < 1e-12

    # At the optimal population saddle, primal samples vanish but dual noise remains.
    optimal_w = np.array([.5, .5, 0.])/P
    ideal_gradients, probs = [], []
    for edge, pe in enumerate(P):
        for query, pq in enumerate(MU):
            ideal_gradients.append(sample(optimal_w, VALUES, 0., np.zeros(2), edge, query)[1])
            probs.append(pe*pq)
    ideal_gradients, probs = np.array(ideal_gradients), np.array(probs)
    ideal_mean = probs@ideal_gradients
    assert np.max(abs(ideal_mean)) < 1e-12
    assert np.max(abs(ideal_gradients[:, :5])) == 0
    dual_second_moment = float(probs@np.sum(ideal_gradients[:, 5:]**2, axis=1))
    assert dual_second_moment > 0

    # Continuous action example: behavior uniform[-1,1], one-step cost 1+a^2.
    # Uniform[-h,h] policies approach the optimal Dirac action but their L2
    # density ratios diverge. This is an analytic integration, not new data.
    nodes, quad_weights = np.polynomial.legendre.leggauss(5)
    continuous = []
    for width in (.5, .25, .1, .05, .01):
        ratio_second_moment = float(.5*width*np.sum(quad_weights)/width**2)
        achieved_cost = float(.5*np.sum(quad_weights*(1+(width*nodes)**2)))
        assert abs(ratio_second_moment-1/width) < 1e-12
        assert abs(achieved_cost-(1+width*width/3)) < 1e-12
        continuous.append(dict(width=width, excess_cost=achieved_cost-1,
                               density_ratio_second_moment=ratio_second_moment,
                               pure_joint_loss=(achieved_cost-1)**2))

    result = dict(training_updates=0, environment_calls=0,
                  insufficient_constant_witness=dict(flow=flow.tolist(), density_ratio=w.tolist(),
                      gap=gap, risk=risk, tested_residual=coarse_residual,
                      flow_cost=cost, actual_policy_cost=policy_cost, optimal_cost=optimal_cost,
                      regret=policy_cost-optimal_cost, full_feature_residual=residual.tolist(),
                      actual_policy_value_residual=float(policy_values@residual)),
                  critic_only_witness=dict(gap=float(collapsed_gap),
                      tested_residual=float(collapsed_residual@collapsed_values), zero_outflow_state='b'),
                  saddle_expectation_error=error,
                  population_gap_and_flow_square=float(gap*gap+residual@residual),
                  expectation_of_per_sample_inner_max=float(naive_inner_max),
                  optimal_saddle_dual_gradient_second_moment=dual_second_moment,
                  continuous_action_singular_limit=continuous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
