"""Frozen planner-data positive control for goal-set QRL.

The bank contains executed transitions, not oracle cost labels. Planning is a
separate, explicitly supervised data-collection experiment. Training never calls
Hybrid A*, and replay audits reconstruct the same recorded actions and anchors.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from minimal_qrl.baselines.hybrid_astar import HybridAStarConfig, HybridAStarController, HYBRID_ASTAR_ALGORITHM_VERSION
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.comm_inspection_oracle_bank import environment_signature
from minimal_qrl.industry_exp.scalability_scenarios import load_scenario_config, scenario_to_env_kwargs

SCHEMA_VERSION = 1


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save_bank(path, payload):
    body = {k: v for k, v in payload.items() if k != 'content_digest'}
    payload = {**body, 'content_digest': digest(body)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return payload


def load_bank(path, *, require_complete=True):
    bank = json.loads(Path(path).read_text())
    body = {k: v for k, v in bank.items() if k != 'content_digest'}
    if bank.get('content_digest') != digest(body):
        raise ValueError('path-support bank content digest mismatch')
    if bank.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported path-support bank schema')
    if bank.get('purpose') != 'planner_path_support_positive_control':
        raise ValueError('not a path-support positive-control bank')
    if require_complete and (not bank.get('complete') or not bank.get('records')):
        raise ValueError('path-support bank is incomplete; finish collection first')
    if require_complete and len(bank['records']) != bank['collection_config']['paths']:
        raise ValueError('path-support bank path count mismatch')
    indices = [r['candidate_index'] for r in bank['records']]
    if len(indices) != len(set(indices)):
        raise ValueError('duplicated path-support candidate')
    return bank


def excluded_starts(scenario, task_bank):
    xmin, ymin, xmax, ymax = scenario['bounds']
    states = []
    for r in task_bank['records']:
        if 'start' in r or 'start_state' in r:
            states.append(r.get('start', r.get('start_state')))
        else:
            x, y, theta = r['start_normalized']
            states.append([xmin + x * (xmax-xmin), ymin + y * (ymax-ymin), theta])
    # Exclude all local probe positions, regardless of orientation.
    probes = scenario['metadata']['u_trap_local_navigability_probes']
    states.extend([[p['x'], probes['centerline_y'], 0.] for p in probes['positions']])
    return np.asarray(states, dtype=np.float32)


def training_starts(env, scenario, exclusions, *, seed, radius):
    """Independent U grid, alternating upper/lower starts; no oracle labels."""
    xmin, ymin, xmax, ymax = scenario['metadata']['exploration_diagnostic_regions']['u_trap_interior']
    rng = np.random.default_rng(seed)
    groups = [[], []]
    for y in np.arange(ymin + .25, ymax, .5):
        for x in np.arange(xmin + .25, xmax, .5):
            state = np.array([x, y, (np.pi + rng.uniform(-.3, .3) + np.pi) % (2*np.pi)-np.pi], np.float32)
            if (env.is_valid_state(state) and not env.is_task_feasible(state)
                    and np.all(np.linalg.norm(exclusions[:, :2]-state[:2], axis=1) > radius)):
                groups[int(y >= (ymin+ymax)/2)].append(state.tolist())
    # Starts near opposite arms encourage different actual detours without
    # modifying the map, terminal set, costs, or planner's dynamics.
    for group in groups:
        rng.shuffle(group)
        group.sort(key=lambda s: -abs(s[1]-(ymin+ymax)/2))
    result = []
    for i in range(max(map(len, groups), default=0)):
        for group in groups:
            if i < len(group):
                result.append(group[i])
    return result


def replay_actions(env, start, actions, *, device_id, seed):
    """Accept only an uninterrupted physical trajectory ending in real success."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 1 or not len(actions):
        raise ValueError('path-support actions must have shape [T,1], T>0')
    if not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > env.omega_max):
        raise ValueError('invalid path-support actions')
    obs, _ = env.reset(seed=int(seed), options={'device_id':device_id, 'start':start})
    if env.is_task_feasible(env.state):
        raise ValueError('path-support start must be nonterminal')
    observations, states, rewards = [obs.copy()], [env.state.copy()], []
    for i, action in enumerate(actions):
        obs, reward, done, timeout, info = env.step(action)
        observations.append(obs.copy()); states.append(env.state.copy()); rewards.append(reward)
        if (not np.isfinite(reward) or reward > 0 or info['collision'] or info['out_of_bounds']
                or timeout or (done and i != len(actions)-1)):
            raise ValueError('path-support trajectory failed or contains a reset/terminal suffix')
    if not done or not info['success'] or not env.is_task_feasible(env.state):
        raise ValueError('path-support trajectory did not reach a legal terminal')
    return {'observations':np.asarray(observations, np.float32).tolist(),
            'states':np.asarray(states, np.float32).tolist(),
            'rewards':np.asarray(rewards, np.float32).tolist(),
            'actions':actions.tolist()}


def route_summary(states, scenario):
    states = np.asarray(states)
    walls = [o for o in scenario['obstacles'] if o['id'] in ('u_top_wall','u_bottom_wall','u_closed_wall')]
    xmax = max(w['x_max'] for w in walls)
    ymin, ymax = min(w['y_min'] for w in walls), max(w['y_max'] for w in walls)
    east = np.flatnonzero(states[:,0] > xmax)
    before = states[:int(east[0])+1] if len(east) else states
    upper, lower = bool(np.any(before[:,1] > ymax)), bool(np.any(before[:,1] < ymin))
    return 'both' if upper and lower else ('upper' if upper else ('lower' if lower else 'unclassified'))


def collect_bank(env, scenario, task_bank, path, *, paths=4, max_attempts=12, seed=20261014,
                 exclusion_radius=.25, planner_config=None):
    if paths <= 0 or max_attempts < paths or not np.isfinite(exclusion_radius) or exclusion_radius < .25:
        raise ValueError('need positive paths, enough attempts, and exclusion radius >= .25')
    planner_config = planner_config or HybridAStarConfig(timeout_sec=120., max_expansions=200000, terminal_samples=0)
    device = scenario['metadata']['u_trap_local_navigability_probes']['device_id']
    env.set_task_by_device_id(device)
    excluded = excluded_starts(scenario, task_bank)
    candidates = training_starts(env, scenario, excluded, seed=seed, radius=exclusion_radius)
    if len(candidates) < paths:
        raise ValueError('not enough independent training starts')
    config = json.loads(json.dumps({'paths':paths,'max_attempts':max_attempts,'seed':seed,
        'exclusion_radius':exclusion_radius,'planner':asdict(planner_config),
        'planner_algorithm_version':HYBRID_ASTAR_ALGORITHM_VERSION}))
    provenance = {'environment_signature':environment_signature(env), 'scenario':scenario,
                  'exclusion_task_bank_digest':digest(task_bank), 'excluded_starts':excluded.tolist(),
                  'candidate_starts':candidates, 'collection_config':config}
    # Dataclass catalogs contain tuples; normalize once so resumed JSON compares
    # identically without weakening any provenance checks.
    provenance = json.loads(json.dumps(provenance))
    if Path(path).exists():
        bank = load_bank(path, require_complete=False)
        if any(bank.get(k) != v for k,v in provenance.items()):
            raise ValueError('path-support collection provenance changed; use a new bank path')
    else:
        bank = {**provenance,'schema_version':SCHEMA_VERSION,'purpose':'planner_path_support_positive_control',
                'expert_free':False,'path_cost_targets_added':False,'complete':False,'records':[],'attempts':[]}
        bank = save_bank(path, bank)
    if bank['complete']:
        return bank
    for i in range(len(bank['attempts']), min(max_attempts, len(candidates))):
        start, episode_seed = candidates[i], seed + i * 104729
        env.reset(seed=episode_seed, options={'device_id':device,'start':start})
        controller = HybridAStarController(planner_config)
        diagnostics = controller.begin_episode(env, env.abstract_goal_observation(), episode_seed)
        # Failed planner costs are inf; JSON must remain strict and resumable.
        diagnostics = {k:(v if not isinstance(v,float) or np.isfinite(v) else None) for k,v in diagnostics.items()}
        attempt = {'candidate_index':i,'seed':episode_seed,'start':start,'planner':diagnostics,
                   'accepted':False, 'rollout_env_steps':0}
        if diagnostics['planner_success']:
            actions = [a.tolist() for a in controller._actions]
            trajectory = replay_actions(env, start, actions, device_id=device, seed=episode_seed)
            cost = float(-np.asarray(trajectory['rewards'], dtype=float).sum())
            if not np.isclose(cost, diagnostics['planned_cost'], rtol=1e-6, atol=1e-5):
                raise ValueError('planner/replayed cost mismatch')
            record = {'candidate_index':i,'seed':episode_seed,'device_id':device,'start':start,**trajectory,
                'route':route_summary(trajectory['states'],scenario),
                'terminal_heading':trajectory['states'][-1][2],
                'diagnostic_path_cost_only':cost}
            bank['records'].append(record)
            attempt.update(accepted=True, rollout_env_steps=len(actions))
        bank['attempts'].append(attempt)
        bank['complete'] = len(bank['records']) >= paths
        bank['summary'] = {'accepted_paths':len(bank['records']),
            'added_real_transitions':sum(len(r['actions']) for r in bank['records']),
            'added_anchors':len(bank['records']),
            'planner_simulated_env_steps':sum(a['planner']['simulated_env_steps'] for a in bank['attempts']),
            'rollout_env_steps':sum(a['rollout_env_steps'] for a in bank['attempts']),
            'planner_seconds':sum(a['planner']['initial_planning_time_sec'] for a in bank['attempts']),
            'routes':{route:sum(r['route']==route for r in bank['records']) for route in ['upper','lower','both','unclassified']},
            'terminal_headings':[r['terminal_heading'] for r in bank['records']]}
        bank = save_bank(path, bank)
        print(f"path support {len(bank['records'])}/{paths}; candidate {i}; {diagnostics['planner_search_stop_reason']}",flush=True)
        if bank['complete']:
            return bank
    raise RuntimeError(f"path-support bank incomplete ({len(bank['records'])}/{paths}); inspect attempts; do not train on it")


def bank_episodes(env, path, *, context_start, include_global_push_pairs=True):
    from minimal_qrl.dataset import _goal_set_transition_infos, _make_goal_set_abstract_edge, EXPLORE_OUTCOME_SUCCESS
    from quasimetric_rl.data import EpisodeData
    bank = load_bank(path)
    if environment_signature(env)['digest'] != bank['environment_signature']['digest']:
        raise ValueError('path-support bank environment mismatch')
    exclusions = np.asarray(bank['excluded_starts'])
    radius = bank['collection_config']['exclusion_radius']
    for i, r in enumerate(bank['records']):
        if np.any(np.linalg.norm(exclusions[:,:2]-np.asarray(r['start'])[:2],axis=1) <= radius):
            raise ValueError('path-support start overlaps held-out evaluation starts')
        actual = replay_actions(env,r['start'],r['actions'],device_id=r['device_id'],seed=r['seed'])
        for name in ['states','observations','actions','rewards']:
            if not np.array_equal(np.asarray(actual[name],np.float32),np.asarray(r[name],np.float32)):
                raise ValueError(f'path-support replay {name} mismatch')
        n, context = len(r['actions']), context_start+i
        obs = np.asarray(actual['observations'],np.float32)
        actions = np.asarray(actual['actions'],np.float32)
        goal = env.abstract_goal_observation().astype(np.float32)
        infos = _goal_set_transition_infos(n=n,task_goal_obs=goal,context_id=context,env=env,
            teacher_guided=True,planner_path_support=True,task_success_episode=True,
            exploration_outcome=EXPLORE_OUTCOME_SUCCESS,
            global_push_seed=r['seed']+49979687,include_global_push_pairs=include_global_push_pairs)
        # Success masks are for sampling and path witnesses. They must not enable
        # zero-filled return targets or introduce planner cost regression.
        infos['goal_return_mask'][:] = False
        terminals = np.zeros(n,dtype=bool); terminals[-1] = True
        yield EpisodeData.from_simple_trajectory(observations=obs[:-1],next_observations=obs[1:],
            actions=actions,rewards=np.asarray(actual['rewards'],np.float32),terminals=terminals,
            timeouts=np.zeros(n,dtype=bool),transition_infos=infos)
        anchor = _make_goal_set_abstract_edge(env,terminal_obs=obs[-1],task_goal_obs=goal,
            context_id=context,action_template=actions[-1],action_dtype=np.float32)
        anchor.transition_infos['planner_path_support'].fill_(True)
        anchor.transition_infos['teacher_guided'].fill_(True)
        yield anchor


def base_fingerprint(raw, episode_count):
    """Hash the unchanged prefix using views, without copying the large replay."""
    from quasimetric_rl.data import MultiEpisodeData
    from minimal_qrl.global_push_sampling import replay_fingerprint
    n = int(raw.episode_lengths[:episode_count].sum())
    return replay_fingerprint(MultiEpisodeData(
        episode_lengths=raw.episode_lengths[:episode_count],all_observations=raw.all_observations[:n+episode_count],
        actions=raw.actions[:n],rewards=raw.rewards[:n],terminals=raw.terminals[:n],timeouts=raw.timeouts[:n],
        transition_infos={k:v[:n] for k,v in raw.transition_infos.items()}))


def validate_training_protocol(args):
    only = getattr(args, 'comm_dataset_mode', None) == 'path_support_only'
    if only and not getattr(args, 'path_support_bank', None):
        raise ValueError('path_support_only requires --path-support-bank')
    if not getattr(args,'path_support_bank',None):
        return
    expected = {'env_type':'comm_inspection_dubins_uav','comm_dataset_mode':'path_support_only' if only else 'qrl_explore',
        'qrl_local_constraint_mode':'goal_set_barrier','global_push_objective':'linear',
        'global_push_task_sampling':'all','global_push_abstract_goal_ratio':.6,'global_push_state_goal_ratio':.4,
        'qrl_temporal_constraint_weight':0.,'qrl_goal_return_constraint_weight':0.,
        'qrl_nstep_goal_constraint_weight':0.,'qrl_mqe_waypoint_consistency_weight':0.}
    mismatches = [k for k,v in expected.items() if getattr(args,k,None) != v]
    if getattr(args,'init_checkpoint',None):
        mismatches.append('init_checkpoint (fresh initialization required)')
    if args.global_push_state_sampling not in ('environment','trajectory'):
        mismatches.append('global_push_state_sampling')
    if only and args.global_push_state_sampling != 'trajectory':
        mismatches.append('global_push_state_sampling (bank-only diagnostic requires B1)')
    if only and getattr(args, 'path_support_probe_interval', 100) <= 0:
        mismatches.append('path_support_probe_interval (must be positive)')
    if mismatches:
        raise ValueError('path-support positive control requires unchanged barrier/control-B1 protocol: '+', '.join(mismatches))


def validate_path_only_replay(raw, batch_size):
    """Guard the mechanism experiment before optimization, including empty loaders."""
    infos = raw.transition_infos
    real = ~infos['abstract_goal_edge'].bool()
    if not 0 < batch_size <= raw.num_transitions:
        raise ValueError('path_support_only batch size must be positive and no larger than replay')
    if (not real.any() or not infos['planner_path_support'].all()
            or not infos['task_success_episode'][real].all()
            or not infos['global_push_task_witness_mask'][real].all()
            or not infos['global_push_pair_mask'][real].all()
            or infos['goal_return_mask'].any() or infos['goal_return_cost'].any()):
        raise ValueError('path_support_only requires exclusively witnessed physical paths without return targets')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario-config',required=True)
    p.add_argument('--task-bank',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--paths',type=int,default=4)
    p.add_argument('--max-attempts',type=int,default=12)
    p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--exclusion-radius',type=float,default=.25)
    p.add_argument('--timeout-sec',type=float,default=120.)
    p.add_argument('--max-expansions',type=int,default=200000)
    a=p.parse_args()
    scenario=load_scenario_config(a.scenario_config)
    bank=collect_bank(CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario)),scenario,
        json.loads(Path(a.task_bank).read_text()),a.output,paths=a.paths,max_attempts=a.max_attempts,
        seed=a.seed,exclusion_radius=a.exclusion_radius,
        planner_config=HybridAStarConfig(timeout_sec=a.timeout_sec,max_expansions=a.max_expansions,terminal_samples=0))
    print(json.dumps(bank['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
