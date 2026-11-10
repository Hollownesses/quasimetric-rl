"""Pre-training finite differences, exact expectation and component checks."""
import numpy as np

from experiments.gap_stochastic.core import initial, population, QUERIES, P
from experiments.gap_exact_gradient.estimator import replace_gap, nongap_parts, exact_gap


def check():
    states = [initial(), np.r_[np.linspace(.6, 2., 30), [.12, .27, .83, .05, .21, .34]]]
    results = dict(cases=[], finite_differences=[])
    for name, mu in QUERIES.items():
        atoms = [(e, q, P[e]*mu[q]) for e in range(7) for q in range(5)]
        for ix, x in enumerate(states):
            mean = np.zeros(36)
            mean_loss = 0.
            gap, exact = exact_gap(x, mu)
            worst = np.zeros(4)
            for e, q, p in atoms:
                for e2, q2, p2 in atoms:
                    edges, queries = np.array([e,e2]), np.array([q,q2])
                    loss, gradient, risk, info = replace_gap(x, edges, queries, mu)
                    rest_loss, rest, rest_risk = nongap_parts(x, edges, queries)
                    _, old_loss, old_gap, new_gap, original = info
                    errors = [np.max(abs(original-old_gap-rest)),
                              np.max(abs(gradient-new_gap-rest)),
                              np.max(abs(gradient-rest-exact)),
                              abs(loss-(gap*gap+rest_loss))]
                    worst = np.maximum(worst, errors)
                    assert risk == rest_risk
                    assert np.array_equal(new_gap, exact)
                    mean += p*p2*gradient
                    mean_loss += p*p2*loss
            truth_loss, truth_gradient, _ = population(x, mu)
            item = dict(query=name, state=ix,
                        original_component_error=float(worst[0]),
                        retained_component_error=float(worst[1]),
                        exact_gap_component_error=float(worst[2]),
                        loss_decomposition_error=float(worst[3]),
                        expected_gradient_error=float(np.max(abs(mean-truth_gradient))),
                        expected_loss_error=abs(float(mean_loss-truth_loss)))
            assert max(v for k,v in item.items() if k not in ('query','state')) < 1e-10
            results['cases'].append(item)
        # A fixed batch covers all seven data edges, including the goal outgoing edge.
        edges, queries = np.array([6,0,5,1,5,2,3,4]), np.array([4,0,1,4,3,2,4,0])
        x = states[1]
        gradient = replace_gap(x, edges, queries, mu)[1]
        exact = exact_gap(x, mu)[1]
        numeric, gap_numeric = np.zeros(36), np.zeros(36)
        eps = 1e-6
        for j in range(36):
            plus, minus = x.copy(), x.copy()
            plus[j] += eps
            minus[j] -= eps
            numeric[j] = (replace_gap(plus,edges,queries,mu)[0]-replace_gap(minus,edges,queries,mu)[0])/(2*eps)
            gap_numeric[j] = (exact_gap(plus,mu)[0]**2-exact_gap(minus,mu)[0]**2)/(2*eps)
        item = dict(query=name, gradient_error=float(np.max(abs(numeric-gradient))),
                    exact_gap_distance_error=float(np.max(abs(gap_numeric[:30]-exact[:30]))),
                    exact_gap_flow_error=float(np.max(abs(gap_numeric[30:]-exact[30:]))))
        assert max(v for k,v in item.items() if k != 'query') < 1e-7
        results['finite_differences'].append(item)
    return results
