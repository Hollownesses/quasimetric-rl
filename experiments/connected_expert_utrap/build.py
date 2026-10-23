"""Expand the old expert backbone using real controls, never snapped graph edges."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.expert_utrap_network.build import (
    ROOT, DEVICE, FIELDS, make_env, execute_segment, shooting, episode_pair,
    sha256, write_json, require_exact_join)
from minimal_qrl.path_support import replay_actions
from minimal_qrl.global_push_sampling import replay_fingerprint
from minimal_qrl.eval.comm_inspection_oracle_bank import environment_signature
from quasimetric_rl.data import MultiEpisodeData

DEFAULT_BACKBONE = ROOT / 'results/expert_utrap_network_20261019/main'
ROUTES = ('upper_inner', 'upper_outer', 'lower_inner', 'lower_outer')
SEED = 20261021


def backbone_paths(env, root):
    manifest = json.loads((root/'manifest.json').read_text())
    if environment_signature(env)['digest'] != manifest['environment_signature']['digest']:
        raise ValueError('backbone environment signature mismatch')
    for name in ('verified_replay.pt', 'scenario.json', 'segments.json', 'trajectories.json'):
        if sha256(root/name) != manifest['files'][name]:
            raise ValueError('changed backbone: '+name)
    segments = json.loads((root/'segments.json').read_text())
    records = json.loads((root/'trajectories.json').read_text())
    paths = {}
    for name in ROUTES:
        # A recorded success suffix, chosen by name, not by cost or graph value.
        names = [name, 'tail_left' if name.startswith('upper') else 'tail_right']
        require_exact_join(segments[names[0]], segments[names[1]])
        actions = sum([segments[n]['actions'] for n in names], [])
        paths[name] = replay_actions(env, segments[name]['start'], actions, device_id=DEVICE, seed=SEED)
    for i, record in enumerate(records):
        names = record['segments']
        actions = sum([segments[n]['actions'] for n in names], [])
        paths[f'original_{i:02d}'] = replay_actions(env, segments[names[0]]['start'], actions,
                                                  device_id=DEVICE, seed=SEED+i)
    return paths, segments, manifest


def target_bank(paths, stride):
    """Existing, exact poses with recorded success suffixes; no graph values."""
    targets = []
    for name, path in paths.items():
        # Original paths contribute their U prefixes; main routes contribute the rest.
        count = len(path['actions']) if name in ROUTES else min(24, len(path['actions']))
        for j in range(0, count, stride):
            targets.append(dict(path=name, index=j, state=np.asarray(path['states'][j], np.float32)))
    return targets


def heading_gap(a, b):
    return abs((a-b+np.pi) % (2*np.pi)-np.pi)


def find_connection(env, start, targets, *, limit=6):
    """Endpoint shooting only; returns actual executed controls and an exact join."""
    ranked = []
    for target in targets:
        distance = float(np.linalg.norm(start[:2]-target['state'][:2]))
        if .45 <= distance <= 3.5:
            bearing=math.atan2(float(target['state'][1]-start[1]),float(target['state'][0]-start[0]))
            # Nearest position alone often proposes a target behind the fixed-speed
            # aircraft. Prefer forward, heading-compatible continuation poses.
            score = (distance + .35*heading_gap(float(start[2]), float(target['state'][2]))
                     + 2.*heading_gap(float(start[2]),bearing)
                     + .7*heading_gap(float(target['state'][2]),bearing))
            ranked.append((score, target, distance))
    ranked.sort(key=lambda item: item[0])
    attempts = []
    for _, target, distance in ranked[:limit]:
        base = int(math.ceil(distance/(env.v*env.dt)))
        for n in sorted(set(max(3, base+k) for k in (0, 1, 2, 4, 8, 15))):
            try:
                connection, iterations = shooting(env, start, target['state'], n)
                if not np.array_equal(np.asarray(connection['states'][-1], np.float32), target['state']):
                    raise ValueError('non-exact connector endpoint')
                attempts.append(dict(target_path=target['path'], target_index=target['index'],
                                     steps=n, accepted=True, iterations=iterations))
                return connection, target, attempts
            except ValueError as error:
                attempts.append(dict(target_path=target['path'], target_index=target['index'],
                                     steps=n, accepted=False, reason=str(error)))
    return None, None, attempts


def compose(env, start, prefix, connection, target, paths):
    suffix = paths[target['path']]
    expected = np.asarray(suffix['states'][target['index']], np.float32)
    if not np.array_equal(np.asarray(connection['states'][-1], np.float32), expected):
        raise ValueError('cannot compose a near endpoint')
    if prefix:
        pre = execute_segment(env, start, prefix)
        require_exact_join(pre, connection)
    elif not np.array_equal(np.asarray(start, np.float32), np.asarray(connection['states'][0], np.float32)):
        raise ValueError('connector starts at a different state')
    actions = prefix + connection['actions'] + suffix['actions'][target['index']:]
    result = replay_actions(env, start, actions, device_id=DEVICE, seed=SEED)
    join_index = len(prefix)+len(connection['actions'])
    if not np.array_equal(np.asarray(result['observations'][join_index:], np.float32),
                          np.asarray(suffix['observations'][target['index']:], np.float32)):
        raise ValueError('connected suffix is not bit-exact')
    return result, join_index


def build(output, backbone=DEFAULT_BACKBONE, *, source_stride=16, target_stride=4,
          heading_grid=3, target_limit=6):
    if min(source_stride, target_stride, heading_grid, target_limit) < 1:
        raise ValueError('all construction sizes must be positive')
    output, backbone = Path(output), Path(backbone)
    output.mkdir(parents=True, exist_ok=False)
    scenario = json.loads((backbone/'scenario.json').read_text())
    write_json(output/'scenario.json', scenario)
    env = make_env(scenario)
    paths, segments, old_manifest = backbone_paths(env, backbone)
    targets = target_bank(paths, target_stride)
    records, trajectories, pairs = [], [], []
    attempt_file = (output/'attempts.jsonl').open('w')

    def add(trajectory, metadata):
        context = len(records)
        pairs.extend(episode_pair(env, trajectory, context))
        records.append(dict(context=context, steps=len(trajectory['actions']), **metadata))
        trajectories.append(trajectory)

    def attempt(start, prefix, candidates, metadata):
        start = np.asarray(start, np.float32)
        entry = dict(start=start.tolist(), prefix=prefix, **metadata)
        try:
            if not env.is_valid_state(start) or env.is_task_feasible(start):
                raise ValueError('invalid or terminal start')
            after = (np.asarray(execute_segment(env, start, prefix)['states'][-1], np.float32)
                     if prefix else start)
            connection, target, tries = find_connection(env, after, candidates, limit=target_limit)
            entry['connector_attempts'] = tries
            if connection is None:
                raise ValueError('bounded connector search exhausted')
            trajectory, join_index = compose(env, start, prefix, connection, target, paths)
            meta = dict(**metadata, target_path=target['path'], target_index=target['index'],
                        join_index=join_index, prefix_steps=len(prefix), connector_steps=len(connection['actions']))
            add(trajectory, meta)
            entry.update(accepted=True, context=len(records)-1, **{k:v for k,v in meta.items() if k not in entry})
        except ValueError as error:
            entry.update(accepted=False, reason=str(error))
        attempt_file.write(json.dumps(entry)+'\n'); attempt_file.flush()
        return entry['accepted']

    try:
        # Preserve every old complete trajectory exactly, with its original route choice.
        for name, path in paths.items():
            if name.startswith('original_'):
                add(path, dict(kind='backbone', backbone_path=name))
        # Explicitly attempt switching both ways shortly after each inner/outer fork.
        for side in ('upper', 'lower'):
            a, b = [side+'_'+v for v in ('inner', 'outer')]
            aa, bb = paths[a]['states'], paths[b]['states']
            fork = next(i for i in range(min(len(aa), len(bb)))
                        if not np.array_equal(np.asarray(aa[i], np.float32), np.asarray(bb[i], np.float32)))
            for source, destination in ((a,b), (b,a)):
                candidates = [dict(path=destination, index=j, state=np.asarray(paths[destination]['states'][j], np.float32))
                              for j in range(fork+5, min(fork+26, len(paths[destination]['actions'])), 3)]
                attempt(paths[source]['states'][fork], [], candidates,
                        dict(kind='cross_route', source_path=source, source_index=fork))
        print(json.dumps(dict(stage='cross_route', accepted=len(records)-64)), flush=True)
        # Local branching on route states and on a fixed subset of the U prefixes.
        sources, seen = [], set()
        for name, path in paths.items():
            if name not in ROUTES and int(name.split('_')[-1]) % 8:
                continue
            stop = len(segments[name]['actions']) if name in ROUTES else min(20, len(path['actions']))
            for j in range(0, stop, source_stride):
                state = np.asarray(path['states'][j], np.float32)
                if state.tobytes() in seen:
                    continue
                seen.add(state.tobytes()); sources.append((name,j,state))
        for k, (name,j,state) in enumerate(sources):
            for omega in (-env.omega_max, -.5*env.omega_max, 0., .5*env.omega_max, env.omega_max):
                attempt(state, [[float(omega)]]*3, targets,
                        dict(kind='local_branch', source_path=name, source_index=j, omega=float(omega)))
            print(json.dumps(dict(stage='local', source=k+1, total=len(sources), trajectories=len(records))), flush=True)
        # New start poses cover eight headings at each fixed U position.
        for ix, x in enumerate(np.linspace(3.35,4.7,heading_grid)):
            for iy, y in enumerate(np.linspace(2.5,4.8,heading_grid)):
                for ih, theta in enumerate(np.arange(8)*np.pi/4-np.pi):
                    start = np.array([x,y,theta], np.float32)
                    # Try direct connection, then sustained legal turns towards west.
                    delta = (np.pi-theta+np.pi) % (2*np.pi)-np.pi
                    n = max(1, int(round(abs(delta)/(env.omega_max*env.dt))))
                    turns = [[], [[float(np.sign(delta) or 1)*env.omega_max]]*n,
                             [[-float(np.sign(delta) or 1)*env.omega_max]]*max(1,21-n)]
                    for proposal, prefix in enumerate(turns):
                        if attempt(start, prefix, targets, dict(kind='heading_start', grid=[ix,iy],
                                                               heading_bin=ih, proposal=proposal)):
                            break
                print(json.dumps(dict(stage='heading', position=[ix,iy], trajectories=len(records))), flush=True)
    finally:
        attempt_file.close()
    raw = MultiEpisodeData.cat(pairs)
    torch.save(dict(schema_version=1, sha256=replay_fingerprint(raw), raw={k:getattr(raw,k) for k in FIELDS}),
               output/'verified_replay.pt')
    write_json(output/'trajectories.json', trajectories)
    write_json(output/'records.json', records)
    sources = {}
    dependencies = list(Path(__file__).parent.glob('*.py')) + [
        ROOT/'experiments/expert_utrap_network/build.py', ROOT/'experiments/expert_utrap_network/data.py',
        ROOT/'experiments/expert_utrap_network/training_adapter.py', ROOT/'experiments/expert_utrap_network/audit.py',
        ROOT/'experiments/full_replay_four_path/train.py', ROOT/'minimal_qrl/path_support.py',
        ROOT/'minimal_qrl/envs/comm_inspection_dubins_uav_2d.py', ROOT/'minimal_qrl/envs/dubins_uav_2d.py',
        ROOT/'minimal_qrl/global_push_sampling.py']
    for source in dependencies:
        relative = source.relative_to(ROOT); dest = output/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source,dest)
        sources[str(relative)] = sha256(source)
    manifest = dict(schema_version=1, dataset_kind='connected_expert_network', seed=SEED,
        config=dict(source_stride=source_stride,target_stride=target_stride,heading_grid=heading_grid,target_limit=target_limit),
        backbone=str(backbone.resolve()), backbone_manifest_sha256=sha256(backbone/'manifest.json'),
        backbone_files=old_manifest['files'], environment_signature=environment_signature(env),
        replay_sha256=replay_fingerprint(raw), sources=sources,
        expert_free=False, oracle_labels=False, graph_labels=False,
        failed_proposals='attempts.jsonl; rejected construction proposals are not replay transitions',
        sampling='uniform physical records; duplicates retained and separately audited',
        files={p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'manifest.json', manifest)
    write_json(output/'BUILD_COMPLETE', dict(trajectories=len(records), physical_edges=int(raw.transition_infos['abstract_goal_edge'].logical_not().sum()),
                                           kinds=dict(Counter(r['kind'] for r in records))))
    print((output/'BUILD_COMPLETE').read_text(), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--backbone', default=str(DEFAULT_BACKBONE))
    p.add_argument('--source-stride', type=int, default=16); p.add_argument('--target-stride', type=int, default=4)
    p.add_argument('--heading-grid', type=int, default=3); p.add_argument('--target-limit', type=int, default=6)
    a=p.parse_args(); build(a.output,a.backbone,source_stride=a.source_stride,target_stride=a.target_stride,
                            heading_grid=a.heading_grid,target_limit=a.target_limit)
