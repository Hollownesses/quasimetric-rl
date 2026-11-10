"""Only remove the conservation raw gradient's nullspace component."""
import numpy as np
from scipy.linalg import null_space

from experiments.gap_stochastic.core import sampled, INC, INJ, P, SRC, DST

Z = null_space(INC[:, :6])
assert Z.shape == (6, 1)
Z = Z[:, 0]
if Z[np.flatnonzero(abs(Z) > 1e-12)[0]] < 0:
    Z *= -1
PROJECTOR = np.eye(6)-np.outer(Z, Z)
assert np.linalg.norm(INC[:, :6]@Z) < 1e-12
assert abs(Z@Z-1) < 1e-12


def conservation_gradient(x, edges, queries, beta=1.):
    """Same U-statistic conservation derivative as the stage-one estimator."""
    n = len(edges)
    f = np.r_[x[30:], 0.]
    ec = np.bincount(edges, minlength=7).astype(float)
    H = (f[edges]/P[edges])[:, None]*INC[:, edges].T-INJ[queries]
    sh = H.sum(axis=0)
    self_term = (H[np.arange(n), SRC[edges]]-H[np.arange(n), DST[edges]])/P[edges]
    return (2*beta*(ec*(INC.T@sh)/P-np.bincount(edges, weights=self_term, minlength=7))/(n*(n-1)))[:6]


def projected_sampled(x, edges, queries, method, *, lam=1., beta=1.):
    assert method == 'unbiased'
    loss, gradient, risk = sampled(x, edges, queries, method, lam=lam, beta=beta)
    raw = conservation_gradient(x, edges, queries, beta)
    removed = Z*(Z@raw)
    filtered = raw-removed
    changed = gradient.copy()
    changed[30:] -= removed
    return loss, changed, risk, raw, filtered, gradient


class ProjectedSampler:
    """Estimator adapter plus read-only per-update audit before Adam."""
    def __init__(self):
        self.rows = []
        self.conservation_first_moment = np.zeros(6)
        self.flow_second_moment = np.zeros(6)

    def __call__(self, x, edges, queries, method, *, lam=1., beta=1.):
        loss, gradient, risk, raw, filtered, original = projected_sampled(
            x, edges, queries, method, lam=lam, beta=beta)
        step = len(self.rows)+1
        self.conservation_first_moment = .9*self.conservation_first_moment+.1*filtered
        self.flow_second_moment = .999*self.flow_second_moment+.001*gradient[30:]**2
        corrected_cons_m = self.conservation_first_moment/(1-.9**step)
        adam_metric = np.sqrt(self.flow_second_moment/(1-.999**step))+1e-8
        raw_component = float(Z@raw)
        filtered_component = float(Z@filtered)
        gap_preservation_error = abs(float(Z@gradient[30:])-float(Z@(original[30:]-raw)))
        modification_error = float(np.max(abs(gradient[30:]-original[30:]+Z*raw_component)))
        assert np.array_equal(gradient[:30], original[:30])
        assert abs(filtered_component) < 1e-10
        assert gap_preservation_error < 1e-10 and modification_error < 1e-10
        self.rows.append((step, raw_component, filtered_component,
                          float(Z@original[30:]), float(Z@gradient[30:]),
                          gap_preservation_error, modification_error,
                          float(Z@corrected_cons_m), float(Z@(corrected_cons_m/adam_metric))))
        # Only this gradient is consumed by the unchanged stage-one Adam loop.
        return loss, gradient, risk

    def arrays(self):
        return np.array(self.rows)


AUDIT_COLUMNS = ('step', 'raw_cons_route', 'projected_cons_route', 'raw_total_route',
                 'projected_total_route', 'gap_preservation_error', 'modification_error',
                 'cons_first_moment_route', 'cons_adam_scaled_route')
