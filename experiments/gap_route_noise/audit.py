"""Pre-training exact expectation and component-isolation checks."""
import numpy as np
from experiments.gap_stochastic.core import sampled, population, initial, QUERIES, P, INC
from experiments.gap_route_noise.estimator import Z, PROJECTOR, projected_sampled, conservation_gradient


def check():
    states = [initial(), np.r_[np.linspace(.6, 2., 30), [.12, .27, .83, .05, .21, .34]]]
    result = dict(direction=Z.tolist(), norm_error=abs(float(Z@Z)-1),
                  nullspace_error=float(np.linalg.norm(INC[:, :6]@Z)), cases=[])
    for name, mu in QUERIES.items():
        atoms = [(e, q, P[e]*mu[q]) for e in range(7) for q in range(5)]
        for ix, x in enumerate(states):
            mean = np.zeros(36)
            before = np.zeros(6)
            after = np.zeros(6)
            worst_null = worst_decomposition = worst_change = 0.
            for e, q, p in atoms:
                for e2, q2, p2 in atoms:
                    edges, queries = np.array([e,e2]), np.array([q,q2])
                    loss, gradient, risk, raw, filtered, original = projected_sampled(x, edges, queries, 'unbiased')
                    _, no_cons, _ = sampled(x, edges, queries, 'unbiased', beta=0.)
                    worst_decomposition = max(worst_decomposition, float(np.max(abs(original[30:]-no_cons[30:]-raw))))
                    worst_null = max(worst_null, abs(float(Z@filtered)))
                    worst_change = max(worst_change, float(np.max(abs((gradient-original)[30:]+Z*(Z@raw)))))
                    np.testing.assert_array_equal(gradient[:30], original[:30])
                    mean += p*p2*gradient
                    before += p*p2*raw
                    after += p*p2*filtered
            truth = population(x, mu)[1]
            _, base_without_cons, _ = population(x, mu, beta=0.)
            exact_cons = truth[30:]-base_without_cons[30:]
            item = dict(query=name, state=ix, maximum_batch_null_component=worst_null,
                        maximum_decomposition_error=worst_decomposition,
                        maximum_change_error=worst_change,
                        expected_gradient_error=float(np.max(abs(mean-truth))),
                        expected_cons_gradient_error=float(np.max(abs(after-exact_cons))),
                        expected_cons_before_after_error=float(np.max(abs(before-after))),
                        exact_cons_projection_error=float(np.max(abs(PROJECTOR@exact_cons-exact_cons))))
            assert max(item[k] for k in item if k not in ('query','state')) < 1e-10
            result['cases'].append(item)
    return result
