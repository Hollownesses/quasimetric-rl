"""Freeze a trajectory-supported probe, then run the paired 2x2 continuation."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.native_margin.shared import rows, sha, write

ENV = 'pointmaze-medium-navigate-v0'
DATA = Path('experiments/ogbench_official_qrl/data')
DATA_SHA = {
    ENV + '.npz': '00bc93b9006b4a97cda425b112382960eea87f5f00166be4c485dc391356a443',
    ENV + '-val.npz': '6bcfe4728293b3566ade704175bd144f6937ac8b54bfb6bd8038e3ead951a78e',
}
PROTOCOL = Path(__file__).with_name('PROTOCOL.md')
START_CELL = (3, 2)
ROUTES = {
    'A': [(3, 3), (3, 4), (4, 4), (5, 4), (5, 3), (6, 3)],
    'B': [(4, 2), (4, 1), (5, 1), (6, 1), (6, 2), (6, 3)],
}
ACTIONS = np.array([[1., 0.], [0., 1.]])
CONFIG = dict(steps=512, batch_size=256, sampling_seed=2001,
              budgets=dict(loose=0.05, tight=0.01),
              native_weight=0.50, probe_weight=0.05, background_weight=0.45,
              log_dual_gain=0.01, setpoint_fraction=0.85, seconds_limit=600,
              risk_separation=0.001, direction_tol=0.001,
              scan_batch_size=1024, scale_queries=2048, scale_seed=9001)
GATE_RADIUS = 1.5
GOAL_RADIUS = 1.
HORIZON = 256


def compressed(cells):
    return cells[np.r_[True, np.any(cells[1:] != cells[:-1], axis=1)]]


def eligible_queries(obs, terminals, cells, route, gate, anchor, goal):
    ends = np.flatnonzero(terminals)
    starts = np.r_[0, ends[:-1] + 1]
    reaches_goal = np.linalg.norm(obs - goal, axis=1) <= GOAL_RADIUS
    gate_error = np.linalg.norm(obs - gate, axis=1)
    anchor_error = np.linalg.norm(obs - anchor, axis=1)
    eligible = []
    for episode, (lo, hi) in enumerate(zip(starts, ends)):
        goal_rows = lo + np.flatnonzero(reaches_goal[lo:hi + 1])
        candidates = []
        for i in lo + np.flatnonzero(gate_error[lo:hi + 1] <= GATE_RADIUS):
            k = goal_rows.searchsorted(i + 1)
            if k == len(goal_rows) or goal_rows[k] - i > HORIZON:
                continue
            j = goal_rows[k]
            if not np.array_equal(compressed(cells[i:j + 1]), route):
                continue
            h = i + np.argmin(anchor_error[i:j])
            if anchor_error[h] <= GATE_RADIUS:
                candidates.append((gate_error[i], i, h, j))
        if candidates:
            distance, i, h, j = min(candidates)
            eligible.append(dict(episode=episode, source_row=int(i), anchor_row=int(h),
                                 goal_hit_row=int(j), source=obs[i].tolist(),
                                 steps=int(j - i), suffix_steps=int(j - h),
                                 gate_error=float(distance), anchor_error=float(anchor_error[h]),
                                 goal_hit=obs[j].tolist()))
    return eligible


def match_queries(a, b):
    from scipy.optimize import linear_sum_assignment

    def differences(key):
        x = np.array([p[key] for p in a])
        y = np.array([p[key] for p in b])
        return np.abs(x[:, None] - y[None, :])

    cost = differences('steps') + differences('suffix_steps')
    cost = cost + 0.001 * (differences('gate_error') + differences('anchor_error'))
    ia, ib = linear_sum_assignment(cost)
    assert len(ia) and not {a[i]['episode'] for i in ia} & {b[i]['episode'] for i in ib}
    return dict(A=[a[i] for i in ia], B=[b[i] for i in ib])


def input_manifest(raw, common):
    paths = {DATA / name: digest for name, digest in DATA_SHA.items()}
    parent = common.BASE.relative_to(raw)
    paths[parent / 'params_100000.pkl'] = common.BASE_SHA
    manifest = {str(path): sha(raw / path) for path in paths}
    assert manifest == {str(path): digest for path, digest in paths.items()}
    manifest[str(parent / 'flags.json')] = sha(common.BASE / 'flags.json')
    return manifest


def freeze(raw, out, common):
    from experiments.ogbench_diagnostics import core

    out.mkdir(parents=True, exist_ok=False)
    inputs = input_manifest(raw, common)
    env, _, _ = common.make_env_and_datasets(ENV)
    maze = env.unwrapped
    start = np.array(maze.ij_to_xy(START_CELL), dtype=float)
    goal = np.array(maze.ij_to_xy(ROUTES['A'][-1]), dtype=float)
    assert maze._goal_tol == GOAL_RADIUS and maze._success_timing == 'post'
    assert all(maze.maze_map[cell] == 0 for cell in [START_CELL, *ROUTES['A'], *ROUTES['B']])
    env.action_space.seed(0)
    env.reset(seed=0, options=dict(task_id=3))
    maze.set_goal(goal_xy=goal)
    maze.set_state(start, np.zeros(2))
    case = dict(seed=0, task=3, observation=start.tolist(), goal=goal.tolist(),
                integration=core.save_integration(env).tolist())
    successors, state = core.candidates(env, ACTIONS)
    anchors = {key: list(maze.ij_to_xy(route[1])) for key, route in ROUTES.items()}
    route_steps = {}
    for key, action, successor in zip(ROUTES, ACTIONS, successors):
        core.restore(env, state)
        x, _, done, truncated, _ = env.step(action)
        np.testing.assert_array_equal(x, successor)
        trace = [start, x]
        for cell in ROUTES[key]:
            waypoint = maze.ij_to_xy(cell)
            while not done and np.max(np.abs(x - waypoint)) > 1e-6:
                steer = np.clip((waypoint - x) / 0.2, -1., 1.)
                x, _, done, truncated, _ = env.step(steer)
                trace.append(x)
                assert not truncated and len(trace) <= HORIZON
        cells = np.array([maze.xy_to_ij(x) for x in trace])
        np.testing.assert_array_equal(compressed(cells), [START_CELL, *ROUTES[key]])
        assert done and maze.compute_success()
        assert np.any(np.all(np.isclose(trace, anchors[key], atol=1e-6, rtol=0), axis=1))
        route_steps[key] = len(trace) - 1

    with np.load(raw / DATA / (ENV + '.npz')) as data:
        obs = data['observations']
        offset = np.array([maze._offset_y, maze._offset_x]) + maze._maze_unit / 2
        cells = np.floor((obs[:, ::-1] + offset) / maze._maze_unit).astype(int)
        pools = {key: eligible_queries(obs, data['terminals'], cells, route,
                                       maze.ij_to_xy(route[0]), anchors[key], goal)
                 for key, route in ROUTES.items()}
    matched = match_queries(pools['A'], pools['B'])
    sources = np.array([p['source'] for group in matched.values() for p in group])
    assert len(np.unique(sources, axis=0)) == len(sources)
    probes = np.vstack([start, successors, list(anchors.values())])
    assert not np.any(np.all(sources[:, None] == probes[None, :], axis=-1))
    env.close()
    write(out / 'frozen.json', dict(config=CONFIG, inputs=inputs, case=case,
          routes=ROUTES, actions=ACTIONS.tolist(), true_successors=successors.tolist(),
          suffix_anchors=anchors, physical_route_steps=route_steps,
          selection=dict(gate_radius=GATE_RADIUS, goal_radius=GOAL_RADIUS, horizon=HORIZON,
                         eligible_counts={key: len(pool) for key, pool in pools.items()},
                         matched=matched), protocol_sha256=sha(PROTOCOL),
          script_sha256=sha(__file__)))
    print('Frozen probe and matched backgrounds', {key: len(pool) for key, pool in pools.items()},
          'physical route steps', route_steps, flush=True)


def contrasts(records):
    by_key = {(r['budget'], r['distribution']): r for r in records}
    result = {}
    for budget in CONFIG['budgets']:
        a, b = (by_key[budget, dist] for dist in ROUTES)
        result[budget] = {f'delta_{key}': a[key] - b[key] for key in ('J_A', 'J_B', 'm', 'scaled_m')}
    max_tight = max(r['R'] for r in records if r['budget'] == 'tight')
    min_loose = min(r['R'] for r in records if r['budget'] == 'loose')
    achieved = (all(r['R'] <= r['R_cap'] for r in records)
                and min_loose - max_tight >= CONFIG['risk_separation'])
    loose, tight = result['loose'], result['tight']
    tol = CONFIG['direction_tol']
    direction = loose['delta_J_A'] > tol and loose['delta_m'] < -tol
    attenuation = (tight['delta_J_A'] >= -tol and tight['delta_m'] <= tol
                   and loose['delta_J_A'] - tight['delta_J_A'] >= tol
                   and tight['delta_m'] - loose['delta_m'] >= tol)
    verdict = 'inconclusive_budget' if not achieved else (
        'supports' if direction and attenuation else 'does_not_support')
    return dict(contrasts=result, budget_achieved=achieved, loose_direction=direction,
                attenuation=attenuation, verdict=verdict)


def run(raw, out, common):
    import flax
    import jax
    import jax.numpy as jnp
    import mujoco
    import optax
    from experiments.ogbench_diagnostics import core
    from utils.flax_utils import restore_agent, save_agent

    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > CONFIG['seconds_limit']:
            raise TimeoutError('600-second experiment budget reached')

    frozen_path = out / 'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    assert frozen['config'] == CONFIG and frozen['protocol_sha256'] == sha(PROTOCOL)
    assert frozen['script_sha256'] == sha(__file__) and frozen['inputs'] == input_manifest(raw, common)
    with (out / 'run_started.json').open('x') as file:
        json.dump(dict(frozen_sha256=sha(frozen_path), raw_worktree=str(raw),
                       jax=jax.__version__, flax=flax.__version__, optax=optax.__version__,
                       numpy=np.__version__, mujoco=mujoco.__version__, platform=jax.default_backend(),
                       upstream=(common.UPSTREAM.parents[1] / 'OFFICIAL_COMMIT').read_text().strip()),
                  file, indent=2)
    base, cfg = common.load_base()
    assert cfg['eps'] == CONFIG['budgets']['loose'] and cfg['lr'] == 0.0003
    original = common.tree_digest(base)
    env, train, val = common.load_data(cfg)
    case = {key: np.asarray(x) if isinstance(x, list) else x for key, x in frozen['case'].items()}
    env.action_space.seed(case['seed'])
    core.load_case(env, case)
    successors, _ = core.candidates(env, np.asarray(frozen['actions']))
    np.testing.assert_array_equal(successors, frozen['true_successors'])
    env.close()
    goal = case['goal'].astype(np.float32)
    query_sources = {key: np.asarray([case['observation'], *[p['source'] for p in group]], np.float32)
                     for key, group in frozen['selection']['matched'].items()}
    for key, group in frozen['selection']['matched'].items():
        np.testing.assert_array_equal(train.dataset['observations'][[p['source_row'] for p in group]],
                                      query_sources[key][1:])
    atom_goals = np.broadcast_to(goal, query_sources['A'].shape)
    tx = optax.adam(cfg['lr'])

    @jax.jit
    def update(agent, obs, next_obs, native_goals, sources, cap):
        network = agent.network
        params = network.params
        lam = network.select('lam')()

        def loss(value_params):
            value = partial(network.select('value'), params={'modules_value': value_params})
            native = 100 * jax.nn.softplus(5 - value(obs, native_goals) / 100)
            atoms = 100 * jax.nn.softplus(5 - value(sources, atom_goals) / 100)
            push = (CONFIG['native_weight'] * native.mean() + CONFIG['probe_weight'] * atoms[0]
                    + CONFIG['background_weight'] * atoms[1:].mean())
            risk = (jax.nn.relu(value(obs, next_obs) - 1) ** 2).mean()
            return push + lam * risk, (risk, native.mean(), atoms[0], atoms[1:].mean(), lam)

        value_params = params['modules_value']
        grad, info = jax.grad(loss, has_aux=True)(value_params)
        adam, empty = network.opt_state
        value_opt = (adam._replace(mu=adam.mu['modules_value'], nu=adam.nu['modules_value']), empty)
        delta, (new_adam, _) = tx.update(grad, value_opt, value_params)
        next_adam = adam._replace(count=new_adam.count,
                                  mu={**adam.mu, 'modules_value': new_adam.mu},
                                  nu={**adam.nu, 'modules_value': new_adam.nu})
        error = info[0] / (CONFIG['setpoint_fraction'] * cap) - 1
        log_lam = params['modules_lam']['log_value'] + CONFIG['log_dual_gain'] * jnp.clip(error, -1, 1)
        next_params = {**params, 'modules_value': optax.apply_updates(value_params, delta),
                       'modules_lam': {'log_value': log_lam}}
        return agent.replace(network=network.replace(params=next_params, opt_state=(next_adam, empty),
                                                     step=network.step + 1)), jnp.stack(info)

    models = {(budget, dist): base for budget in CONFIG['budgets'] for dist in ROUTES}
    np.random.seed(CONFIG['sampling_seed'])
    stream = hashlib.sha256()
    logs = []
    batch_size = CONFIG['batch_size']
    for step in range(1, CONFIG['steps'] + 1):
        ids = train.dataset.get_random_idxs(batch_size)
        batch = train.sample(batch_size, idxs=ids)
        fields = [batch[key] for key in ('observations', 'next_observations', 'value_goals')]
        for x in (ids, *fields):
            stream.update(x.tobytes())
        first = []
        for (budget, dist), agent in models.items():
            cap = np.float32(CONFIG['budgets'][budget])
            updated, stats = update(agent, *fields, query_sources[dist], cap)
            risk, native, probe, bg, lam = map(float, np.asarray(stats))
            assert np.isfinite([risk, native, probe, bg, lam]).all()
            models[budget, dist] = updated
            logs.append(dict(step=step, budget=budget, distribution=dist, R_batch=risk,
                             native_push=native, probe_push=probe, background_push=bg, lam=lam))
            first.append([risk, native, probe, lam])
        if step == 1:
            np.testing.assert_array_equal(first, np.tile(first[0], (len(models), 1)))
        if step % 64 == 0:
            rows(out / 'training.csv', logs)
            window = logs[-64 * len(models):]
            mean_risks = [round(np.mean([r['R_batch'] for r in window[i::len(models)]]), 5)
                          for i in range(len(models))]
            print('Completed shared update', step,
                  'mean R over last 64 steps', mean_risks, flush=True)
        check_time()

    assert common.tree_digest(base) == original
    endpoint = 100000 + CONFIG['steps']
    checkpoints = {}
    for (budget, dist), agent in models.items():
        arm = f'{budget}_{dist}'
        path = out / arm
        path.mkdir()
        assert int(agent.network.opt_state[0].count) == endpoint and int(agent.network.step) == endpoint + 1
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
        assert common.tree_equal(agent.rng, base.rng)
        for name in ('modules_actor', 'modules_dynamics'):
            assert common.tree_equal(agent.network.params[name], base.network.params[name])
        for moment in ('mu', 'nu'):
            for name in ('modules_actor', 'modules_dynamics', 'modules_lam'):
                assert common.tree_equal(getattr(agent.network.opt_state[0], moment)[name],
                                         getattr(base.network.opt_state[0], moment)[name])
        save_agent(agent, str(path), endpoint)
        loaded = restore_agent(base, str(path), endpoint)
        assert common.tree_equal(agent, loaded)
        models[budget, dist] = loaded
        checkpoints[arm] = dict(sha256=sha(path / f'params_{endpoint}.pkl'), adam_count=endpoint,
                                frozen_modules_and_moments_verified=True, roundtrip_verified=True)
    training_seconds = time.monotonic() - started

    indices = train.dataset.valid_idxs
    obs = train.dataset['observations']
    assert len(indices) == 1_000_000
    np.random.seed(CONFIG['scale_seed'])
    scale_batch = val.sample(CONFIG['scale_queries'], evaluation=True)
    eval_x = np.vstack([case['observation'], list(frozen['suffix_anchors'].values()), successors]).astype(np.float32)
    eval_g = np.broadcast_to(goal, eval_x.shape)

    def evaluate(agent):
        probe, ja, jb, da, db = map(float, np.asarray(common.distances(agent, eval_x, eval_g)))
        scale = float(np.mean(np.asarray(common.distances(agent, scale_batch['observations'],
                                                         scale_batch['value_goals'])), dtype=np.float64))
        assert np.isfinite([probe, ja, jb, da, db, scale]).all() and scale > 0
        return dict(J_s=probe, J_A=ja, J_B=jb, Q_A=1 + da, Q_B=1 + db,
                    m=db - da, scale=scale, scaled_m=(db - da) / scale)

    records = []
    scan_size = CONFIG['scan_batch_size']
    for (budget, dist), agent in models.items():
        edge_distances = np.empty(len(indices), dtype=np.float32)
        for offset in range(0, len(indices), scan_size):
            ids = indices[offset:offset + scan_size]
            padded = np.pad(ids, (0, scan_size - len(ids)), mode='edge')
            edge_distances[offset:offset + len(ids)] = np.asarray(
                common.distances(agent, obs[padded], obs[padded + 1]))[:len(ids)]
            check_time()
        assert np.isfinite(edge_distances).all()
        np.save(out / f'{budget}_{dist}_edges.npy', edge_distances, allow_pickle=False)
        risk = float(np.mean(np.maximum(edge_distances.astype(np.float64) - 1, 0) ** 2))
        record = dict(budget=budget, R_cap=CONFIG['budgets'][budget], distribution=dist,
                      R=risk, lam=float(agent.network.select('lam')()), **evaluate(agent))
        records.append(record)
        rows(out / 'results.csv', records)
        print('Scanned', budget, dist, 'full R', risk, flush=True)

    assert frozen['inputs'] == input_manifest(raw, common)
    check_time()
    write(out / 'results.json', dict(records=records, **contrasts(records), parent=evaluate(base),
                                    valid_edges=len(indices), shared_stream_sha256=stream.hexdigest(),
                                    initial_agent_sha256=original, checkpoints=checkpoints,
                                    exact_successor_replay=True, frozen_sha256=sha(frozen_path),
                                    training_seconds=training_seconds, total_seconds=time.monotonic() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('raw', type=Path, help='Historical worktree with the pinned runtime and original data')
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    raw = args.raw.resolve()
    sys.path.insert(0, str(raw / 'experiments/ogbench_critic'))
    import common

    (freeze if args.stage == 'freeze' else run)(raw, args.out.resolve(), common)
