"""Exact small problems for the optimization bias of a soft quasimetric objective."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.sparse.csgraph import shortest_path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.edge_weight_graph.graph import SRC, DST, G, A, B, shortest

RISK = 0.05
WEIGHTS = np.full(7, 1 / 7)
COST = np.ones(7)
COST[5] = 1.25
DEMANDS = {
    'cheap_suffix_queries': np.array([.05, .01, .92, .01, .01]),
    'expensive_suffix_queries': np.array([.05, .01, .01, .01, .92]),
}


def edge_flow(demand, fraction):
    s, a, u, b, v = demand
    return np.array([s*fraction, a+s*fraction, u+a+s*fraction,
                     s*(1-fraction), b+s*(1-fraction), v+b+s*(1-fraction), 0.])


def solve_linear(demand):
    slope = edge_flow(demand, 1)-edge_flow(demand, 0)
    def inflation(fraction):
        flow = edge_flow(demand, fraction)
        return np.sqrt(RISK)*flow/WEIGHTS/np.sqrt(np.sum(flow*flow/WEIGHTS))
    def derivative(fraction):
        return float(slope @ (COST+inflation(fraction)))
    if derivative(0) >= 0:
        fraction = 0.
    elif derivative(1) <= 0:
        fraction = 1.
    else:
        fraction = brentq(derivative, 0, 1, xtol=1e-14)
    flow = edge_flow(demand, fraction)
    extra = inflation(fraction)
    distance = shortest(COST+extra)
    primal = demand @ distance[:G, G]
    bound = flow @ COST + np.sqrt(RISK*np.sum(flow*flow/WEIGHTS))
    assert abs(primal-bound) < 1e-10
    return fraction, flow, extra, distance, bound


def shaping(distance):
    return -100*np.logaddexp(0, 5-distance/100)


def solve_shaped(demand):
    _, _, extra, distance, _ = solve_linear(demand)
    initial = np.r_[distance[:G, G], extra]
    incidence = np.zeros((7, 5))
    for row, (src, dst) in enumerate(zip(SRC, DST)):
        if src != G:
            incidence[row, src] += 1
        if dst != G:
            incidence[row, dst] -= 1
    matrix = np.c_[incidence, -np.eye(7)]
    constraints = [
        dict(type='ineq', fun=lambda x: COST-matrix@x, jac=lambda x: -matrix),
        dict(type='ineq', fun=lambda x: RISK-WEIGHTS@(x[5:]**2),
             jac=lambda x: np.r_[np.zeros(5), -2*WEIGHTS*x[5:]]),
    ]
    solution = minimize(lambda x: -demand@shaping(x[:5]), initial,
                        jac=lambda x: np.r_[-demand*expit(5-x[:5]/100), np.zeros(7)],
                        constraints=constraints, bounds=[(0, None)]*12,
                        method='SLSQP', options=dict(ftol=1e-12, maxiter=1000))
    assert solution.success, solution.message
    extra = solution.x[5:]
    # A tiny roundoff excess is removed towards the known feasible zero vector.
    extra *= min(1., np.sqrt(RISK/(WEIGHTS@(extra**2)))) * (1-1e-13)
    distance = shortest(COST+extra)
    values = distance[:G, G]
    objective = demand @ shaping(values)
    # A tangent to concave shaping reduces a global upper certificate to the
    # independently solved linear-demand problem.
    tangent_demand = demand*expit(5-values/100)
    _, _, _, _, linear_bound = solve_linear(tangent_demand)
    upper = objective+linear_bound-tangent_demand@values
    assert -1e-10 <= upper-objective < 1e-7
    return extra, distance, float(upper-objective)


def check_flow():
    truth = shortest(COST)
    incidence = np.zeros((6, 7))
    incidence[SRC, np.arange(7)] = 1
    incidence[DST, np.arange(7)] -= 1
    result = {}
    for name, demand in DEMANDS.items():
        fraction, flow, extra, distance, bound = solve_linear(demand)
        np.testing.assert_allclose(incidence@flow, np.r_[demand, -demand.sum()])
        risk = WEIGHTS@np.maximum(distance[SRC, DST]-COST, 0)**2
        assert risk <= RISK+1e-12
        shaped_extra, shaped_distance, shaped_gap = solve_shaped(demand)
        slope = edge_flow(demand, 1)-edge_flow(demand, 0)
        curvature = np.sum(slope*slope/WEIGHTS)
        step = -(slope@COST)/curvature
        # This update minimizes local-cost flow plus a recentered quadratic.
        # No value or action-order targets occur in the update.
        trajectory = []
        current = fraction
        for iteration in range(11):
            current_flow = edge_flow(demand, current)
            trajectory.append(dict(iteration=iteration, fraction_on_cheap=current,
                                   nominal_cost=float(current_flow@COST),
                                   excess=float((current_flow-edge_flow(demand, 1))@COST)))
            current = float(np.clip(current+step, 0, 1))
        assert trajectory[-1]['excess'] < 1e-12
        result[name] = dict(
            demand=demand.tolist(), local_edge_weights=WEIGHTS.tolist(),
            original_risk=float(risk), original_objective=float(demand@distance[:G, G]),
            dual_upper=float(bound), target_reference_gap=float(truth[B, G]-truth[A, G]),
            target_model_gap=float(distance[B, G]-distance[A, G]),
            original_fraction_on_cheap=float(fraction), inflation=extra.tolist(),
            shaped_model_gap=float(shaped_distance[B, G]-shaped_distance[A, G]),
            shaped_certificate_gap=shaped_gap,
            shaped_risk=float(WEIGHTS@np.maximum(shaped_distance[SRC, DST]-COST, 0)**2),
            proximal_steps=trajectory,
        )
    assert result['cheap_suffix_queries']['target_model_gap'] < -.1
    assert result['expensive_suffix_queries']['target_model_gap'] > .1
    return result


def check_restricted_family():
    costs = np.full((4, 4), 10.)
    np.fill_diagonal(costs, 0.)
    costs[0, 1:3] = 1
    costs[1, 3], costs[2, 3] = 1, 3
    truth = shortest_path(costs, directed=True)
    points = []
    previous = np.zeros_like(truth)
    for theta in (0., .25, .5, .75, 1.):
        learned_edges = costs.copy()
        learned_edges[1, 3] = .2+.8*theta
        learned_edges[2, 3] = .4+.1*theta
        distance = shortest_path(learned_edges, directed=True)
        assert np.max(distance-costs) <= 1e-12
        assert np.min(distance-previous) >= -1e-12
        assert np.max(distance-truth) <= 1e-12
        chosen = 1+np.argmin(costs[0, 1:3]+distance[1:3, 3])
        points.append(dict(theta=theta, task_push=float(distance[:3, 3].mean()),
                           all_pairs_push=float(distance.mean()),
                           shaped_loss=float(-shaping(distance).mean()),
                           learned_goal_column=distance[:, 3].tolist(),
                           physical_cost=float(costs[0, chosen]+truth[chosen, 3]),
                           maximum_hard_violation=float(np.max(distance-costs))))
        previous = distance
    assert points[0]['physical_cost'] == 2 and points[-1]['physical_cost'] == 4
    assert points[-1]['all_pairs_push'] > points[0]['all_pairs_push']
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    results = dict(risk_budget=RISK, flow_duality=check_flow(), restricted_class=check_restricted_family())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write('\n')
