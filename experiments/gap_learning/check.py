"""Convex finite-table audit of a gap objective and its minibatch variance bias.

Distances and a feasible route mixture are optimized jointly from local costs.
The reference shortest path is used only after optimization for evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.edge_weight_graph.graph import SRC, DST, G, A, B, shortest
from experiments.objective_flow_analysis.check import COST, WEIGHTS, DEMANDS, solve_linear, edge_flow

PAIRS = [(i, j) for i in range(6) for j in range(6) if i != j]
INDEX = {pair: k for k, pair in enumerate(PAIRS)}
EDGE = np.zeros((7, 30))
for k, pair in enumerate(zip(SRC, DST)):
    EDGE[k, INDEX[pair]] = 1
QUERY = np.zeros((5, 30))
for s in range(G):
    QUERY[s, INDEX[s, G]] = 1
TRIANGLES = []
for i in range(6):
    for j in range(6):
        for k in range(6):
            if len({i, j, k}) < 3:
                continue
            row = np.zeros(31)
            row[INDEX[i, j]] = -1
            row[INDEX[i, k]] += 1
            row[INDEX[k, j]] += 1
            TRIANGLES.append(row)
TRIANGLES = np.array(TRIANGLES)


def solve(demand, batch=None, local_weight=1.):
    # A constant quasimetric and a half/half route mixture: no distance teacher.
    initial = np.r_[np.ones(30), .5]
    base = edge_flow(demand, 0)
    direction = edge_flow(demand, 1)-base
    query_vector = demand@QUERY
    cost_slope = COST@direction
    gap_gradient = np.r_[-query_vector, cost_slope]

    def values(z):
        flow = base+z[-1]*direction
        value = query_vector@z[:30]
        cost = COST@flow
        local = EDGE@z[:30]-COST
        risk = WEIGHTS@np.maximum(local, 0)**2
        goal_values = QUERY@z[:30]
        edge_variance = np.sum((COST*flow)**2/WEIGHTS)-cost**2
        query_variance = demand@(goal_values**2)-value**2
        return flow, value, cost, local, risk, goal_values, edge_variance, query_variance

    def objective(z):
        _, value, cost, _, risk, _, edge_variance, query_variance = values(z)
        return (cost-value)**2+local_weight*risk+(0 if batch is None else (edge_variance+query_variance)/batch)

    def gradient(z):
        flow, value, cost, local, _, goal_values, _, _ = values(z)
        grad = 2*(cost-value)*gap_gradient
        grad[:30] += 2*local_weight*(WEIGHTS*np.maximum(local, 0))@EDGE
        if batch is not None:
            grad[:30] += 2*((demand*goal_values)@QUERY-value*query_vector)/batch
            grad[-1] += 2*(np.sum(COST**2*flow*direction/WEIGHTS)-cost*cost_slope)/batch
        return grad

    # Every point improving the initial objective has bounded local risk.
    # Triangle inequalities along simple paths then give this table bound;
    # it does not exclude any optimum and uses no shortest-path solution.
    table_bound = 5*np.max(COST+np.sqrt(objective(initial)/(local_weight*WEIGHTS)))
    bounds = [(0, float(table_bound))]*30+[(0, 1)]
    # Scaling leaves the minimizer unchanged and resolves the nearly flat
    # squared-gap direction at small loss under SLSQP's absolute tolerance.
    scale = 1000. if batch is None else 1.
    result = minimize(lambda z: scale*objective(z), initial, jac=lambda z: scale*gradient(z),
                      constraints=[dict(type='ineq', fun=lambda z: TRIANGLES@z, jac=lambda z: TRIANGLES)],
                      bounds=bounds, method='SLSQP',
                      options=dict(ftol=1e-15, maxiter=5000))
    assert result.success, result.message
    flow, value, cost, local, risk, goal_values, edge_var, query_var = values(result.x)
    assert np.min(TRIANGLES@result.x) >= -1e-8
    # Convex first-order global certificate, independently solved as an LP.
    grad = gradient(result.x)
    gradient_scale = max(float(np.max(abs(grad))), 1e-15)
    linear = linprog(grad/gradient_scale, A_ub=-TRIANGLES,
                     b_ub=np.zeros(len(TRIANGLES)), bounds=bounds, method='highs',
                     options=dict(dual_feasibility_tolerance=1e-9,
                                  primal_feasibility_tolerance=1e-9))
    assert linear.success, linear.message
    certificate_gap = float(grad@(result.x-linear.x))
    assert -1e-9 <= certificate_gap < 2e-6
    fraction, _, _, _, _ = solve_linear(demand)  # Baseline evaluation only.
    nominal = float(COST@edge_flow(demand, 1))
    reference = shortest(COST)
    kappa = float(np.sqrt(np.sum(edge_flow(demand, 1)**2/WEIGHTS)))
    pure_loss = (cost-value)**2+local_weight*risk
    calibration_bound = np.sqrt((1+kappa**2/local_weight)*pure_loss)
    assert cost-nominal <= calibration_bound+1e-8
    if batch is None:
        assert pure_loss < 1e-12
        assert result.x[-1] > 1-1e-5
        assert np.max(abs(goal_values-reference[:G,G])) < 1e-4
    # Exact independent-row moments, requiring no Monte Carlo claim.
    probabilities = WEIGHTS[:,None]*demand[None,:]
    random_gap = COST[:,None]*flow[:,None]/WEIGHTS[:,None]-goal_values[None,:]
    mean_gap = float(np.sum(probabilities*random_gap))
    second = float(np.sum(probabilities*random_gap**2))
    assert np.isclose(mean_gap, cost-value, atol=1e-12)
    assert np.isclose(second-mean_gap**2, edge_var+query_var, atol=1e-12)
    # Enumerate two independent rows: at B=2 the unbiased statistic is Z1*Z2.
    probs = probabilities.ravel()
    gaps = random_gap.ravel()
    pair_prob = probs[:, None]*probs[None, :]
    unbiased_pair_square = float(np.sum(pair_prob*gaps[:, None]*gaps[None, :]))
    naive_pair_square = float(np.sum(pair_prob*((gaps[:, None]+gaps[None, :])/2)**2))
    assert np.isclose(unbiased_pair_square, mean_gap**2, atol=1e-12)
    assert np.isclose(naive_pair_square, mean_gap**2+(second-mean_gap**2)/2, atol=1e-12)
    return dict(batch=batch, local_weight=local_weight,
                old_cheap_fraction=float(fraction), cheap_fraction=float(result.x[-1]),
                total_cost=float(cost), optimal_cost=nominal, total_regret=float(cost-nominal),
                fixed_source_cost=float(3.25-.25*result.x[-1]),
                risk=float(risk), mean_gap=float(cost-value), pure_loss=float(pure_loss),
                optimized_objective=float(objective(result.x)),
                convex_global_certificate_gap=certificate_gap,
                edge_sampling_variance=float(edge_var), query_sampling_variance=float(query_var),
                independent_pair_identity_error=abs(unbiased_pair_square-mean_gap**2),
                cost_regret_bound=float(calibration_bound),
                critic_margin=float(goal_values[B]-goal_values[A]),
                max_goal_error=float(np.max(abs(goal_values-reference[:G,G]))),
                optimization_iterations=int(result.nit))


def identity_check():
    signed = np.linspace(-5, 7, 49)
    worst = 0.
    for alpha in (.2, 1., 3.7):
        for cost in (0., .3, 2.):
            corrected = np.maximum(signed, 0)/alpha
            left = np.maximum(signed-alpha*cost, 0)
            right = alpha*np.maximum(corrected-cost, 0)
            worst = max(worst, float(np.max(abs(left-right))))
    assert worst < 1e-12
    return worst


def variance_direction_check(demand):
    """Analytic counterexample with the true critic frozen, for proof only."""
    flow = edge_flow(demand, 1)
    direction = edge_flow(demand, 0)-flow
    matrix = np.diag(COST**2/WEIGHTS)-np.outer(COST, COST)
    delta = float(COST@direction)
    a = float(direction@matrix@flow)
    b = float(direction@matrix@direction)
    assert delta > 0 and b > 0
    return dict(delta=delta, a=a, b=b,
                bad_fraction={str(batch): float(np.clip(-a/(batch*delta**2+b), 0, 1))
                              for batch in (16, 256, 1024, 16384)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    results = dict(coordinate_identity_error=identity_check(), neural_training_updates=0,
                   cases={name: [solve(demand, batch) for batch in (None, 16, 256, 1024, 16384)]
                          for name, demand in DEMANDS.items()},
                   finite_weight_check={str(weight): solve(DEMANDS['cheap_suffix_queries'], local_weight=weight)
                                        for weight in (.1, 10.)},
                   frozen_true_critic_variance={name: variance_direction_check(demand)
                                               for name, demand in DEMANDS.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write('\n')
