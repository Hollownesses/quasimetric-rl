"""Array-only diagnostics. References score frozen models; never train them."""
from collections import deque
import numpy as np

TOL = 1e-5


def edge_index(src, dst, actions, cost, physical):
    """Retain actions/cost in identity and every record's edge membership."""
    mapping = np.full(len(src), -1, dtype=int)
    representatives, lookup = [], {}
    for row in np.flatnonzero(physical):
        key = (int(src[row]), int(dst[row]), actions[row].tobytes(), float(cost[row]))
        if key not in lookup:
            lookup[key] = len(representatives)
            representatives.append(row)
        mapping[row] = lookup[key]
    return np.asarray(representatives, int), mapping


def branch_neighborhood(src, dst, representatives, nodes, radius=3):
    """Undirected graph-hop proximity; not Euclidean or a causal attribution."""
    degree = np.bincount(src[representatives], minlength=nodes)
    distance = np.full(nodes, -1, dtype=int)
    adjacency = [set() for _ in range(nodes)]
    for row in representatives:
        a, b = int(src[row]), int(dst[row])
        adjacency[a].add(b); adjacency[b].add(a)
    queue = deque(np.flatnonzero(degree > 1).tolist())
    for node in queue:
        distance[node] = 0
    while queue:
        node = queue.popleft()
        if distance[node] >= radius:
            continue
        for other in adjacency[node]:
            if distance[other] < 0:
                distance[other] = distance[node] + 1
                queue.append(other)
    return degree, distance


def shortest_exposure(starts, next_edge, src, dst, row_edge, reference):
    exposure = np.zeros(row_edge.max()+1, int)
    for start in starts:
        node, seen = int(start), set()
        while reference[node] != 0:
            if node in seen:
                raise ValueError('cycle in reference path')
            seen.add(node)
            row = int(next_edge[node])
            if row < 0 or src[row] != node or row_edge[row] < 0:
                raise ValueError('invalid reference edge')
            exposure[row_edge[row]] += 1
            following = int(dst[row])
            if reference[following] >= reference[node]:
                raise ValueError('reference path does not decrease')
            node = following
    return exposure


def rollout_exposure(episodes, src, dst, cost, row_edge, reference, goal, unique=True):
    """Recheck connectivity, total cost and exact advantage/regret identity."""
    exposure = np.zeros(row_edge.max()+1, int)
    seen_starts, regrets, edge_regret = set(), [], np.zeros(len(exposure))
    for episode in episodes:
        start = int(episode['start'])
        if unique and start in seen_starts:
            continue
        seen_starts.add(start)
        node, total, advantage = start, 0., 0.
        if episode['status'] != 'success':
            raise ValueError('expected successful frozen graph rollout')
        for row in episode['rows']:
            if src[row] != node or row_edge[row] < 0:
                raise ValueError('disconnected or nonphysical rollout')
            index = row_edge[row]
            exposure[index] += 1
            delta = float(cost[row]+reference[dst[row]]-reference[node])
            if delta < -1e-7:
                raise ValueError('reference violates edge feasibility')
            edge_regret[index] += delta
            total += float(cost[row]); advantage += delta
            node = int(dst[row])
        regret = total-reference[start]
        if node != goal or abs(total-episode['cost']) > 1e-7:
            raise ValueError('rollout goal/cost mismatch')
        if abs(regret-episode['regret']) > 1e-7 or abs(advantage-regret) > 1e-7:
            raise ValueError('rollout regret identity failed')
        regrets.append(regret)
    if not regrets:
        raise ValueError('empty rollout population')
    if abs(edge_regret.sum()/len(regrets)-np.mean(regrets)) > 1e-7:
        raise ValueError('aggregate regret identity failed')
    return exposure, edge_regret/len(regrets), np.asarray(regrets)


def residual_stats(h, mask):
    values = np.asarray(h)[mask]
    if not len(values):
        return dict(count=0)
    p = np.maximum(values, 0)
    return dict(count=len(values), violations=int((values > TOL).sum()),
                mean_positive=float(p.mean()), sum_positive=float(p.sum()),
                max_positive=float(p.max()), p95_positive=float(np.quantile(p, .95)))


def dual_window(h0, h1, lam0, lam1, visit_delta, mask, rho=100.):
    """Two saved endpoints only. Never equate dual delta to integrated residual."""
    if np.any(np.asarray(visit_delta)[mask] <= 0):
        raise ValueError('window must include visits to each selected record')
    if np.any(np.asarray(lam0) < 0) or np.any(np.asarray(lam1) < 0):
        raise ValueError('negative projected multiplier')
    ids = np.flatnonzero(mask)
    if not len(ids):
        return dict(count=0)
    persistent = (h0[ids] > TOL) & (h1[ids] > TOL)
    grew = lam1[ids]-lam0[ids] > TOL
    worsening = h1[ids]-h0[ids] > TOL
    def group(selected):
        r = ids[selected]
        if not len(r):
            return dict(count=0)
        projection = np.maximum(lam1[r]+h1[r], 0)-lam1[r]
        effective = np.maximum(lam1[r]+rho*h1[r], 0)
        return dict(count=len(r), h45_mean=float(h0[r].mean()), h50_mean=float(h1[r].mean()),
                    lambda45_mean=float(lam0[r].mean()), lambda50_mean=float(lam1[r].mean()),
                    lambda_delta_mean=float((lam1[r]-lam0[r]).mean()),
                    delta_per_visit_mean=float(((lam1[r]-lam0[r])/visit_delta[r]).mean()),
                    visits_min=int(visit_delta[r].min()), visits_max=int(visit_delta[r].max()),
                    effective_phr_coefficient_mean=float(effective.mean()),
                    complementarity_abs_mean=float(np.abs(lam1[r]*h1[r]).mean()),
                    projected_dual_residual_rms=float(np.sqrt(np.mean(projection**2))))
    return dict(count=len(ids), persistent_positive=int(persistent.sum()),
                positive_at50=int((h1[ids] > TOL).sum()),
                persistent_growing_dual=int((persistent & grew).sum()),
                persistent_growing_dual_worsening=int((persistent & grew & worsening).sum()),
                all=group(np.ones(len(ids), bool)), persistent=group(persistent),
                persistent_growing_worsening=group(persistent & grew & worsening))


def score_error_identity(cost, prediction, reference, dst, rows, choice):
    best = int(rows[np.argmin(cost[rows]+reference[dst[rows]])])
    regret = float(cost[choice]+reference[dst[choice]]-cost[best]-reference[dst[best]])
    error = prediction-reference
    contrast = float(error[dst[choice]]-error[dst[best]])
    margin = float(cost[choice]+prediction[dst[choice]]-cost[best]-prediction[dst[best]])
    if abs(margin-regret-contrast) > 1e-7:
        raise ValueError('branch score decomposition failed')
    return dict(best_row=best, regret=regret, successor_error_contrast=contrast,
                predicted_choice_minus_reference_best=margin)
