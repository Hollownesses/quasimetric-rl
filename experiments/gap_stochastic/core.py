"""Training-only finite graph specification, losses, and geometry projection.

No shortest-path solver, optimal flow, path-mixture parameter, or action label.
The six nonterminal edges carry six independent trainable flow variables.
"""
import numpy as np
from scipy.optimize import minimize, nnls

PROJECTION_EVENTS = []

N = 6
GOAL = 5
SRC = np.array([0, 1, 2, 0, 3, 4, 5])
DST = np.array([1, 2, 5, 3, 4, 5, 0])
COST = np.array([1., 1., 1., 1., 1., 1.25, 1.])
P = np.full(7, 1/7)
QUERIES = {
    'cheap_suffix_queries': np.array([.05, .01, .92, .01, .01]),
    'expensive_suffix_queries': np.array([.05, .01, .01, .01, .92]),
}
PAIRS = [(s, t) for s in range(N) for t in range(N) if s != t]
INDEX = {pair: k for k, pair in enumerate(PAIRS)}
EDGE = np.array([INDEX[s, t] for s, t in zip(SRC, DST)])
QUERY = np.array([INDEX[s, GOAL] for s in range(GOAL)])
INC = np.zeros((N, 7))
INC[SRC, np.arange(7)] = 1
INC[DST, np.arange(7)] = -1
INJ = np.eye(N)[:GOAL]-np.eye(N)[GOAL]
ROWS = []
for s in range(N):
    for t in range(N):
        for k in range(N):
            if len({s, t, k}) == 3:
                row = np.zeros(30)
                row[INDEX[s, k]] = row[INDEX[k, t]] = 1
                row[INDEX[s, t]] = -1
                ROWS.append(row)
CONE = np.r_[np.eye(30), np.array(ROWS)]


def initial():
    return np.r_[np.ones(30), np.full(6, .1)]


def project_distance(v, metric=None):
    """Diagonal-metric projection onto {d: CONE @ d >= 0} via Moreau/NNLS.

    The NNLS coefficients are numerical projection multipliers, NOT learned
    arrival flows. No transition costs or queries enter this projection.
    """
    if np.min(CONE@v) >= -1e-12:
        return v.copy()
    scale = np.ones(30) if metric is None else np.sqrt(metric)
    matrix = (CONE/scale).T
    try:
        multipliers, _ = nnls(matrix, -v*scale, maxiter=3000)
        d = v+(matrix@multipliers)/scale
    except (ValueError, RuntimeError) as exc:
        # SciPy 1.13 NNLS may enter an empty active set on a degenerate cone.
        # Solve the SAME weighted projection; no risk or flow repair is added.
        weights = scale*scale
        weights /= weights.max()
        answer = minimize(lambda z:.5*np.sum(weights*(z-v)**2),
                          np.full(30, max(1., float(v.max()))),
                          jac=lambda z:weights*(z-v), method='SLSQP',
                          constraints=[dict(type='ineq', fun=lambda z:CONE@z, jac=lambda z:CONE)],
                          options=dict(ftol=1e-15, maxiter=1000))
        if not answer.success:
            raise ArithmeticError('fallback projection failed: '+answer.message) from exc
        d = answer.x
        PROJECTION_EVENTS.append(dict(error=repr(exc), input_cone_violation=float(-np.min(CONE@v)),
                                      output_cone_violation=float(max(0., -np.min(CONE@d))),
                                      normalized_projection_cost=float(answer.fun)))
    if np.min(CONE@d) < -1e-8:
        raise ArithmeticError('quasimetric projection failed')
    return d


def project(x, metric=None):
    return np.r_[project_distance(x[:30], None if metric is None else metric[:30]),
                 np.maximum(x[30:], 0)]


def distance_matrix(x):
    d = np.zeros((N, N))
    for k, (s, t) in enumerate(PAIRS):
        d[s, t] = x[k]
    return d


def population(x, mu, *, lam=1., beta=1., batch=256,
               diagonal_gap=False, diagonal_flow=False):
    """Exact expected loss and gradient; independent diagnostic, not SGD."""
    d, f = x[:30], np.r_[x[30:], 0.]
    values = d[QUERY]
    value, cost = mu@values, COST@f
    gap = cost-value
    demand = mu@INJ
    transport = INC@f
    residual = transport-demand
    h = np.maximum(d[EDGE]-COST, 0)
    risk = P@(h*h)
    loss = gap*gap+lam*risk+beta*(residual@residual)
    gd = np.zeros(30)
    gd[QUERY] -= 2*gap*mu
    gd[EDGE] += 2*lam*P*h
    gf = 2*gap*COST+2*beta*(INC.T@residual)
    edge_var = np.sum((COST*f)**2/P)-cost*cost
    query_var = mu@(values*values)-value*value
    flow_var = np.sum((f*f/P)*np.sum(INC*INC, axis=0))-transport@transport
    source_var = mu@np.sum(INJ*INJ, axis=1)-demand@demand
    if diagonal_gap:
        loss += (edge_var+query_var)/batch
        gd[QUERY] += 2*(mu*values-value*mu)/batch
        gf += 2*(COST*COST*f/P-cost*COST)/batch
    if diagonal_flow:
        loss += beta*(flow_var+source_var)/batch
        gf += 2*beta*(np.sum(INC*INC, axis=0)*f/P-INC.T@transport)/batch
    parts = dict(gap=float(gap), risk=float(risk), flow_cost=float(cost),
                 value=float(value), residual=residual.tolist(),
                 residual_l2=float(np.linalg.norm(residual)),
                 residual_linf=float(np.max(abs(residual))),
                 pure_loss=float(gap*gap+lam*risk+beta*(residual@residual)),
                 gap_sampling_variance=float(edge_var+query_var),
                 conservation_sampling_variance=float(flow_var+source_var))
    return float(loss), np.r_[gd, gf[:6]], parts


def sampled(x, edges, queries, method, *, lam=1., beta=1.):
    """Analytic minibatch loss/gradient; iid paired transition/query draws."""
    n = len(edges)
    d, f = x[:30], np.r_[x[30:], 0.]
    ec = np.bincount(edges, minlength=7).astype(float)
    qc = np.bincount(queries, minlength=5).astype(float)
    h = np.maximum(d[EDGE]-COST, 0)
    risk = (ec/n)@(h*h)
    gd = np.zeros(30)
    gd[EDGE] += 2*lam*(ec/n)*h
    if method == 'qrl':
        gd[QUERY] -= qc/n
        return float(-(qc/n)@d[QUERY]+lam*risk), np.r_[gd, np.zeros(6)], float(risk)
    z = COST[edges]*f[edges]/P[edges]-d[QUERY[queries]]
    H = (f[edges]/P[edges])[:, None]*INC[:, edges].T-INJ[queries]
    sz, sh = z.sum(), H.sum(axis=0)
    if method == 'naive':
        gap_square = (sz/n)**2
        flow_square = (sh@sh)/(n*n)
        gd[QUERY] -= 2*sz*qc/(n*n)
        gf = 2*sz*ec*COST/P/(n*n)
        gf += 2*beta*ec*(INC.T@sh)/P/(n*n)
    elif method == 'unbiased':
        den = n*(n-1)
        gap_square = (sz*sz-z@z)/den
        flow_square = (sh@sh-np.sum(H*H))/den
        gd[QUERY] += 2*(-sz*qc+np.bincount(queries, weights=z, minlength=5))/den
        gf = 2*(sz*ec-np.bincount(edges, weights=z, minlength=7))*COST/P/den
        self_term = (H[np.arange(n), SRC[edges]]-H[np.arange(n), DST[edges]])/P[edges]
        gf += 2*beta*(ec*(INC.T@sh)/P-np.bincount(edges, weights=self_term, minlength=7))/den
    else:
        raise ValueError(method)
    return float(gap_square+lam*risk+beta*flow_square), np.r_[gd, gf[:6]], float(risk)
