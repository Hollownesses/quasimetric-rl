"""Replace only both blocks of the gap-square gradient by its exact value."""
import numpy as np

from experiments.gap_stochastic.core import (
    sampled, COST, P, EDGE, QUERY, INC, INJ, SRC, DST)


def exact_gap(x, mu):
    gap = float(COST[:6]@x[30:]-mu@x[QUERY])
    gradient = np.zeros(36)
    gradient[QUERY] = -2*gap*mu
    gradient[30:] = 2*gap*COST[:6]
    return gap, gradient


def sampled_gap(x, edges, queries):
    """Original off-diagonal U-statistic and its two gradient blocks."""
    n = len(edges)
    assert n >= 2
    den = n*(n-1)
    f = np.r_[x[30:], 0.]
    z = COST[edges]*f[edges]/P[edges]-x[QUERY[queries]]
    ec = np.bincount(edges, minlength=7)
    qc = np.bincount(queries, minlength=5)
    total = z.sum()
    gradient = np.zeros(36)
    gradient[QUERY] = 2*(-total*qc+np.bincount(queries, weights=z, minlength=5))/den
    gradient[30:] = (2*(total*ec-np.bincount(edges, weights=z, minlength=7))*COST/P/den)[:6]
    return float((total*total-z@z)/den), gradient


def nongap_parts(x, edges, queries, lam=1., beta=1.):
    """Independent audit of original stochastic conservation + local risk."""
    n = len(edges)
    f = np.r_[x[30:], 0.]
    ec = np.bincount(edges, minlength=7)
    h = np.maximum(x[EDGE]-COST, 0)
    risk = float((ec/n)@(h*h))
    H = (f[edges]/P[edges])[:, None]*INC[:, edges].T-INJ[queries]
    total = H.sum(axis=0)
    self_term = (H[np.arange(n), SRC[edges]]-H[np.arange(n), DST[edges]])/P[edges]
    cons = (2*beta*(ec*(INC.T@total)/P-np.bincount(edges, weights=self_term, minlength=7))/(n*(n-1)))[:6]
    gradient = np.zeros(36)
    gradient[EDGE] = 2*lam*(ec/n)*h
    gradient[30:] = cons
    loss = lam*risk+beta*(total@total-np.sum(H*H))/(n*(n-1))
    return float(loss), gradient, risk


def replace_gap(x, edges, queries, mu, *, lam=1., beta=1.):
    loss, original, risk = sampled(x, edges, queries, 'unbiased', lam=lam, beta=beta)
    old_loss, old_gradient = sampled_gap(x, edges, queries)
    gap, gradient = exact_gap(x, mu)
    new_gradient = original-old_gradient+gradient
    return loss-old_loss+gap*gap, new_gradient, risk, (gap, old_loss, old_gradient, gradient, original)


AUDIT_COLUMNS = (
    'step', 'exact_gap', 'original_gap_square_estimate',
    'original_gap_distance_error_l2', 'original_gap_flow_error_l2',
    'exact_gap_distance_error_linf', 'exact_gap_flow_error_linf',
    'conservation_preservation_error', 'local_risk_preservation_error',
    'loss_decomposition_error')


class ExactGapSampler:
    def __init__(self, mu):
        self.mu = np.array(mu, dtype=float, copy=True)
        self.rows = []

    def __call__(self, x, edges, queries, method, *, lam=1., beta=1.):
        assert method == 'unbiased'
        loss, gradient, risk, info = replace_gap(x, edges, queries, self.mu, lam=lam, beta=beta)
        gap, old_loss, old_gap, exact, original = info
        rest_loss, rest_gradient, rest_risk = nongap_parts(x, edges, queries, lam, beta)
        new_gap = gradient-rest_gradient
        old_rest, new_rest = original-old_gap, gradient-exact
        errors = (
            float(np.max(abs(new_gap[:30]-exact[:30]))),
            float(np.max(abs(new_gap[30:]-exact[30:]))),
            max(float(np.max(abs(new_rest[30:]-old_rest[30:]))),
                float(np.max(abs(new_rest[30:]-rest_gradient[30:])))),
            max(float(np.max(abs(new_rest[:30]-old_rest[:30]))),
                float(np.max(abs(new_rest[:30]-rest_gradient[:30])))),
            abs(float(loss-(gap*gap+rest_loss))))
        assert max(errors) < 1e-10
        assert risk == rest_risk
        self.rows.append((len(self.rows)+1, gap, old_loss,
                          float(np.linalg.norm(old_gap[:30]-exact[:30])),
                          float(np.linalg.norm(old_gap[30:]-exact[30:])), *errors))
        return float(loss), gradient, risk

    def arrays(self):
        return np.array(self.rows)
