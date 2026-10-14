"""Locate missing links between recorded U trajectories and legal task anchors.

This is an offline audit. Costs are diagnostic path bounds, never loss targets.
Exact joins preserve the full observation and task goal. Nearby attachments are
reported separately and never promoted to recorded transition witnesses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

import numpy as np
import torch

from minimal_qrl.dataset import EXPLORE_OUTCOME_NAMES
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.eval.replay_constraint_audit import (
    edge_metadata, evaluate_edges, load_models, load_replay, load_run_spec,
    summarize_edges, validate_replay, write_csv,
)
from minimal_qrl.eval.u_trap_local_navigability import build_probe_records
from minimal_qrl.eval.utils import auto_device
from minimal_qrl.train import get_env_kwargs


STAGES = ("west_exit", "outside_wall_envelope", "east_of_closed_wall",
          "within_target_max_distance", "position_and_los_feasible",
          "observation_feasible", "task_feasible", "connected_legal_anchor")


def inside(states, bounds):
    xmin, ymin, xmax, ymax = bounds
    return ((states[:, 0] >= xmin) & (states[:, 0] <= xmax)
            & (states[:, 1] >= ymin) & (states[:, 1] <= ymax))


def ordered_stages(masks, start):
    """Stages must occur in order; different terminal gates must be joint masks."""
    cursor = int(start)
    result = {}
    for name, mask in masks.items():
        found = np.flatnonzero(mask[cursor:]) if cursor >= 0 else []
        cursor = cursor + int(found[0]) if len(found) else -1
        result[name] = cursor if cursor >= 0 else None
    return result


def physical_episodes(raw):
    """Reject ambiguous internal resets rather than invent a continuous chain."""
    _, _, sources = edge_metadata(raw)
    infos = raw.transition_infos
    arrays = {k: v.numpy() for k, v in infos.items()}
    observations = raw.all_observations.numpy()
    episodes, anchors = [], []
    offset = 0
    for episode_id, length in enumerate(raw.episode_lengths.tolist()):
        if length <= 0:
            raise ValueError("Empty replay episode")
        rows = np.arange(offset, offset + length)
        abstract = arrays['abstract_goal_edge'][rows].astype(bool)
        if abstract.any():
            if not abstract.all() or length != 1:
                raise ValueError("Mixed physical/abstract episode has ambiguous continuity")
            anchors.append(int(rows[0]))
        else:
            for name in ['context_id', 'device_index', 'task_success_episode', 'exploration_outcome']:
                if not np.all(arrays[name][rows] == arrays[name][rows[0]]):
                    raise ValueError(f"Internal {name} change has ambiguous continuity")
            goals = arrays['task_goal_observations'][rows]
            if not np.all(goals == goals[0]):
                raise ValueError("Internal task goal change has ambiguous continuity")
            if (raw.terminals[rows[:-1]].any() or raw.timeouts[rows[:-1]].any()):
                raise ValueError("Internal termination/reset has ambiguous continuity")
            episodes.append({'episode_id': episode_id, 'rows': rows,
                             'obs_ids': np.arange(sources[rows[0]], sources[rows[-1]] + 2),
                             'context': int(arrays['context_id'][rows[0]]),
                             'device': int(arrays['device_index'][rows[0]]),
                             'goal': goals[0],
                             'outcome': int(arrays['exploration_outcome'][rows[-1]])})
        offset += length
    if offset != raw.num_transitions or len(observations) != offset + len(raw.episode_lengths):
        raise ValueError("Replay observation/transition indexing mismatch")
    return episodes, anchors, sources


def validate_physical_dynamics(raw, episodes, env, tolerance=1e-5):
    """Check recorded Dubins states/actions, including collision/OOB self-loops."""
    _, _, sources = edge_metadata(raw)
    rows = np.concatenate([e['rows'] for e in episodes])
    obs = raw.all_observations.numpy()
    a, b = obs[sources[rows]], obs[sources[rows] + 1]
    theta = np.arctan2(a[:, 3].astype(float), a[:, 2].astype(float))
    omega = np.clip(raw.actions.numpy()[rows].reshape(-1), -env.omega_max, env.omega_max)
    predicted_theta = theta + omega * env.dt
    predicted = a[:, :2].astype(float) + env.v * env.dt * np.column_stack(
        [np.cos(predicted_theta), np.sin(predicted_theta)])
    blocked = np.zeros(raw.num_transitions, dtype=bool)
    for e in episodes:
        if e['outcome'] in (2, 3):
            blocked[e['rows'][-1]] = True
    predicted[blocked[rows]] = a[blocked[rows], :2]
    predicted_theta[blocked[rows]] = theta[blocked[rows]]
    actual_theta = np.arctan2(b[:, 3].astype(float), b[:, 2].astype(float))
    position_error = np.linalg.norm(predicted - b[:, :2], axis=1)
    heading_error = np.abs(np.arctan2(np.sin(predicted_theta - actual_theta),
                                    np.cos(predicted_theta - actual_theta)))
    bad = (position_error > tolerance) | (heading_error > tolerance)
    if bad.any():
        raise ValueError(f"Recorded physical dynamics mismatch on {bad.sum()} edges")
    return {'checked_edges': len(rows), 'tolerance': tolerance,
            'max_position_error': float(position_error.max()),
            'max_heading_error': float(heading_error.max()), 'mismatched_edges': 0}


def anchor_key(context, device, goal, source):
    return int(context), int(device), goal.tobytes(), source.tobytes()


def valid_anchor_rows(raw, anchors, sources, env):
    infos = raw.transition_infos
    obs = raw.all_observations.numpy()
    valid, records = set(), []
    for row in anchors:
        source, target = obs[sources[row]], obs[sources[row] + 1]
        device = int(infos['device_index'][row])
        legal = bool(env.is_task_feasible_for_task(env.observation_to_state(source), env.device_ids[device]))
        goal = infos['task_goal_observations'][row].numpy()
        ok = bool(legal and infos['source_terminal_goal_state'][row]
                  and float(raw.rewards[row]) == 0 and np.array_equal(target, goal)
                  and raw.terminals[row] and not raw.timeouts[row])
        if ok:
            valid.add(row)
        records.append({'transition_index': row, 'context_id': int(infos['context_id'][row]),
                        'device_index': device, 'legal_source_recomputed': legal,
                        'valid_zero_cost_anchor': ok})
    return valid, records


def exact_reachable(raw, episodes, anchors, sources, *, scope):
    """Reverse reachability with no spatial rounding or unrecorded connectors."""
    if scope not in ('strict_context', 'same_task_goal'):
        raise ValueError(scope)
    obs = raw.all_observations.numpy()
    infos = raw.transition_infos
    reverse = defaultdict(list)
    def key(context, device, goal, observation):
        return (int(context) if scope == 'strict_context' else None,
                int(device), goal.tobytes(), observation.tobytes())
    pending = deque()
    for row in anchors:
        pending.append(key(infos['context_id'][row], infos['device_index'][row],
                           infos['task_goal_observations'][row].numpy(), obs[sources[row]]))
    for e in episodes:
        for a, b in zip(e['obs_ids'][:-1], e['obs_ids'][1:]):
            reverse[key(e['context'], e['device'], e['goal'], obs[b])].append(
                key(e['context'], e['device'], e['goal'], obs[a]))
    reachable = set(pending)
    while pending:
        for predecessor in reverse.get(pending.popleft(), []):
            if predecessor not in reachable:
                reachable.add(predecessor)
                pending.append(predecessor)
    mask = np.zeros(raw.num_transitions, dtype=bool)
    for e in episodes:
        for row, index in zip(e['rows'], e['obs_ids'][:-1]):
            mask[row] = key(e['context'], e['device'], e['goal'], obs[index]) in reachable
    return mask


def nearest_state(query, candidates, *, heading_tolerance=np.pi):
    """Heading-gated nearest physical state; returned proximity is NOT an edge."""
    if not len(candidates):
        return None
    qtheta = np.arctan2(query[3], query[2])
    theta = np.arctan2(candidates[:, 3], candidates[:, 2])
    angles = np.abs(np.arctan2(np.sin(theta-qtheta), np.cos(theta-qtheta)))
    ids = np.flatnonzero(angles <= heading_tolerance + 1e-8)
    if not len(ids):
        return None
    distance = np.linalg.norm(candidates[ids, :2] - query[:2], axis=1)
    i = int(ids[np.argmin(distance)])
    return {'candidate_index': i, 'position_distance': float(distance.min()),
            'heading_difference_rad': float(angles[i]),
            'exact_full_observation_match': bool(np.array_equal(query, candidates[i]))}


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values")
    return {k: float(np.quantile(values, q)) for k, q in
            [('min', 0), ('p25', .25), ('median', .5), ('p75', .75), ('p95', .95), ('max', 1)]}


def build_support(raw, env, scenario, device_id):
    episodes, anchor_rows, sources = physical_episodes(raw)
    dynamics = validate_physical_dynamics(raw, episodes, env)
    valid, anchor_records = valid_anchor_rows(raw, anchor_rows, sources, env)
    target_device = list(env.device_ids).index(device_id)
    selected = [e for e in episodes if e['device'] == target_device]
    if not selected:
        raise ValueError(f'No recorded physical trajectories for {device_id}; no coverage to audit')
    selected_anchors = {r for r in valid if int(raw.transition_infos['device_index'][r]) == target_device}
    obs = raw.all_observations.numpy()
    infos = raw.transition_infos
    features = env.task_context_indices
    regions = scenario['metadata']['exploration_diagnostic_regions']
    walls = {o['id']: o for o in scenario['obstacles']}
    env.set_task_by_device_id(device_id)
    linked = {anchor_key(infos['context_id'][r], infos['device_index'][r],
                         infos['task_goal_observations'][r].numpy(), obs[sources[r]]) for r in valid}
    witnessed = np.zeros(raw.num_transitions, dtype=bool)
    interior = np.zeros_like(witnessed)
    west_exit = np.zeros_like(witnessed)
    costs = (-raw.rewards.numpy()).clip(min=0)
    tail_costs = np.full(raw.num_transitions, np.nan)
    episode_records = []
    for e in selected:
        states = obs[e['obs_ids']]
        rows = e['rows']
        final = states[-1]
        has_anchor = anchor_key(e['context'], e['device'], e['goal'], final) in linked
        same_context_anchors = [r for r in selected_anchors if int(infos['context_id'][r]) == e['context']
                               and np.array_equal(infos['task_goal_observations'][r].numpy(), e['goal'])]
        source_flags = inside(states, regions['u_trap_interior'])
        exit_flags = inside(states, regions['u_trap_west_exit'])
        interior[rows] = source_flags[:-1]
        west_exit[rows] = exit_flags[:-1]
        first_u = int(np.flatnonzero(source_flags)[0]) if source_flags.any() else None
        legal_end = bool(env.is_task_feasible(final[:2].tolist() + [float(np.arctan2(final[3], final[2]))]))
        success = bool(e['outcome'] == 1 and infos['task_success_episode'][rows[-1]]
                       and raw.terminals[rows[-1]] and not raw.timeouts[rows[-1]] and legal_end)
        complete = success and has_anchor
        witnessed[rows] = complete
        tail_costs[rows] = np.cumsum(costs[rows][::-1], dtype=np.float64)[::-1]
        masks = {
            'west_exit': exit_flags,
            'outside_wall_envelope': ((states[:, 1] > walls['u_top_wall']['y_max'])
                                      | (states[:, 1] < walls['u_bottom_wall']['y_min'])),
            'east_of_closed_wall': states[:, 0] > walls['u_closed_wall']['x_max'],
            'within_target_max_distance': states[:, features['target_distance']] <= env.observation_max_distance,
            'position_and_los_feasible': ((states[:, features['distance_margin']] >= 0)
                                         & (states[:, features['sector_margin']] >= 0)
                                         & (states[:, features['target_los']] > .5)
                                         & (states[:, features['valid_state']] > .5)),
            'observation_feasible': states[:, features['observation_feasible']] > .5,
            'task_feasible': states[:, features['task_feasible']] > .5,
            'connected_legal_anchor': np.array([False]*(len(states)-1) + [complete]),
        }
        stages = ordered_stages(masks, first_u) if first_u is not None else {k: None for k in masks}
        after_u = states[first_u:] if first_u is not None else states
        best = after_u[np.argmin(after_u[:, features['target_distance']])]
        anchor_gap = nearest_state(final, obs[sources[same_context_anchors]]) if same_context_anchors else None
        record = {'episode_id': e['episode_id'], 'context_id': e['context'], 'device_id': device_id,
                  'length': len(rows), 'outcome': EXPLORE_OUTCOME_NAMES[e['outcome']],
                  'start_x': float(states[0, 0]), 'start_y': float(states[0, 1]),
                  'start_heading': float(np.arctan2(states[0, 3], states[0, 2])),
                  'starts_in_U': bool(source_flags[0]), 'visits_U': first_u is not None,
                  'first_U_timestep': first_u, 'legal_terminal_recorded': success,
                  'has_valid_anchor_in_context': bool(same_context_anchors),
                  'endpoint_equals_legal_anchor': has_anchor, 'continuous_goal_witness': complete,
                  'end_x': float(final[0]), 'end_y': float(final[1]),
                  'end_heading': float(np.arctan2(final[3], final[2])),
                  'endpoint_to_own_anchor_xy': anchor_gap['position_distance'] if anchor_gap else None,
                  'endpoint_to_own_anchor_heading_rad': anchor_gap['heading_difference_rad'] if anchor_gap else None,
                  'minimum_target_distance_after_U': float(best[features['target_distance']]),
                  'best_observation_margin_after_U': float(after_u[:, features['observation_margin']].max()),
                  'end_task_feasible': bool(final[features['task_feasible']] > .5),
                  'cost_from_first_U_to_end': float(costs[rows[first_u:]].sum(dtype=float)) if first_u is not None else None,
                  'last_ordered_stage': next((k for k in reversed(STAGES) if stages[k] is not None), 'before_west_exit') if first_u is not None else 'not_U',
                  **{f'first_{k}_timestep': t for k, t in stages.items()}}
        candidate_mask = masks['position_and_los_feasible'].copy()
        candidate_mask[:first_u if first_u is not None else len(states)] = False
        candidates = states[candidate_mask]
        record['joint_position_los_candidate_points_after_U'] = len(candidates)
        record['candidate_best_heading_margin'] = float(candidates[:, features['heading_margin']].max()) if len(candidates) else None
        record['candidate_heading_feasible_points'] = int((candidates[:, features['heading_margin']] >= 0).sum())
        record['candidate_communication_feasible_points'] = int((candidates[:, features['communication_feasible']] > .5).sum())
        for name in ['distance_margin', 'sector_margin', 'heading_margin', 'target_los',
                     'observation_feasible', 'communication_feasible', 'task_feasible']:
            record[f'closest_point_{name}'] = float(best[features[name]])
            record[f'ever_{name}_after_U'] = bool((after_u[:, features[name]] > .5).any()) if name.endswith('feasible') or name.endswith('los') else bool((after_u[:, features[name]] >= 0).any())
        episode_records.append(record)
        e['record'] = record
    strict = exact_reachable(raw, selected, selected_anchors, sources, scope='strict_context')
    shared = exact_reachable(raw, selected, selected_anchors, sources, scope='same_task_goal')
    target_rows = np.concatenate([e['rows'] for e in selected])
    counts = []
    for name, group in [('all_target', episode_records),
                        ('starts_in_U', [r for r in episode_records if r['starts_in_U']]),
                        ('visits_U', [r for r in episode_records if r['visits_U']])]:
        counts.append({'group': name, 'episodes': len(group),
                       'outcomes': dict(Counter(r['outcome'] for r in group)),
                       'stages': {k: sum(r[f'first_{k}_timestep'] is not None for r in group) for k in STAGES},
                       'continuous_witnesses': sum(r['continuous_goal_witness'] for r in group),
                       'valid_context_anchors': sum(r['has_valid_anchor_in_context'] for r in group)})
    frontier = []
    for phase in ['before_west_exit', 'after_west_exit']:
        group = [r for r in episode_records if r['visits_U'] and (r['first_west_exit_timestep'] is None) == (phase == 'before_west_exit')]
        frontier.append({'phase': phase, 'episodes': len(group),
                         'outcomes': dict(Counter(r['outcome'] for r in group)),
                         'last_stages': dict(Counter(r['last_ordered_stage'] for r in group))})
    planner_rows = infos.get('planner_path_support', torch.zeros(raw.num_transitions, dtype=torch.bool)).numpy().astype(bool)
    summary = {'device_id': device_id, 'physical_dynamics': dynamics,
               'source_support_by_origin': {
                   origin: {name: {'rows': int((mask & origin_mask).sum()),
                                  'continuous_witness_rows': int((mask & origin_mask & witnessed).sum()),
                                  'same_task_goal_exact_graph_reachable_rows': int((mask & origin_mask & shared).sum())}
                            for name, mask in [('U_interior', interior), ('U_west_exit', west_exit)]}
                   for origin, origin_mask in [('base_replay', ~planner_rows), ('planner_added', planner_rows)]},
               'physical_episodes_all_tasks': len(episodes), 'abstract_edges_all_tasks': len(anchor_rows),
               'valid_anchors_all_tasks': len(valid), 'invalid_anchor_rows': [r for r in anchor_records if not r['valid_zero_cost_anchor']],
               'groups': counts, 'ordered_funnel_origin': 'first_U_visit; non_U_episodes_not_in_funnel',
               'failure_frontiers': frontier,
               'terminal_position_candidates': [
                   {k: r[k] for k in ['context_id', 'outcome', 'length',
                    'joint_position_los_candidate_points_after_U', 'candidate_best_heading_margin',
                    'candidate_heading_feasible_points', 'candidate_communication_feasible_points']}
                   for r in episode_records if r['visits_U'] and r['joint_position_los_candidate_points_after_U']],
               'source_support': {name: {'rows': int(mask.sum()),
                    'continuous_witness_rows': int((mask & witnessed).sum()),
                    'strict_context_exact_graph_reachable_rows': int((mask & strict).sum()),
                    'same_task_goal_exact_graph_reachable_rows': int((mask & shared).sum())}
                    for name, mask in [('U_interior', interior), ('U_west_exit', west_exit)]}}
    return {'summary': summary, 'episodes': selected, 'episode_records': episode_records,
            'anchor_records': anchor_records, 'valid_anchors': selected_anchors,
            'sources': sources, 'target_rows': target_rows, 'interior': interior,
            'west_exit': west_exit, 'witnessed': witnessed, 'tail_costs': tail_costs,
            'strict': strict, 'shared': shared, 'planner_rows': planner_rows}


def write_trajectory_plot(output, raw, support, scenario):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Wedge
    obs = raw.all_observations.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    for wall in scenario['obstacles']:
        ax.add_patch(Rectangle((wall['x_min'], wall['y_min']), wall['x_max']-wall['x_min'],
                               wall['y_max']-wall['y_min'], color='#555555'))
    target = next(d for d in scenario['device_catalog']['devices'] if d['id'] == support['summary']['device_id'])
    bearing = np.rad2deg(target['observation']['preferred_bearing_rad'])
    spread = np.rad2deg(target['observation']['bearing_tolerance_rad'])
    ax.add_patch(Wedge(target['position'], target['observation']['max_distance'], bearing-spread, bearing+spread,
                       width=target['observation']['max_distance']-target['observation']['min_distance'],
                       color='#6aaed6', alpha=.35))
    ax.scatter(*target['position'], marker='*', color='black', s=90, zorder=5)
    labels_used = set()
    for e in support['episodes']:
        r = e['record']
        if not r['visits_U'] and not r['continuous_goal_witness']:
            continue
        category = ('Recorded success (never visited U)' if r['continuous_goal_witness'] else
                    'U trajectory reaching observation position' if r['joint_position_los_candidate_points_after_U'] else
                    'Other U trajectories')
        color = {'Recorded success (never visited U)': '#269354',
                 'U trajectory reaching observation position': '#d55d32',
                 'Other U trajectories': '#9a9a9a'}[category]
        label = category if category not in labels_used else None
        labels_used.add(category)
        states = obs[e['obs_ids']]
        ax.plot(states[:, 0], states[:, 1], color=color, alpha=.9 if color != '#9a9a9a' else .18,
                linewidth=1.2 if color != '#9a9a9a' else .6, label=label)
    xmin, ymin, xmax, ymax = scenario['bounds']
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal', title='Recorded trajectories for u_trap_target', xlabel='x', ylabel='y')
    ax.legend(fontsize=7, loc='upper right')
    group = next(g for g in support['summary']['groups'] if g['group'] == 'visits_U')
    stage_labels = ['Visited U', 'West exit', 'Around walls', 'East of wall', 'Near target', 'Position + LOS', 'Observation', 'Goal + anchor']
    counts = [group['episodes']] + [group['stages'][k] for k in STAGES if k != 'task_feasible']
    axes[1].barh(stage_labels[::-1], counts[::-1], color='#2876b8')
    for y, n in enumerate(counts[::-1]):
        axes[1].text(n+1, y, str(n), va='center')
    axes[1].set(title='Ordered continuous trajectory funnel', xlabel='Episodes', xlim=(0, max(counts)*1.15))
    fig.savefig(output/'trajectory_funnel.png', dpi=160)
    plt.close(fig)


@torch.no_grad()
def goal_distances(critic, observations, goals, device, chunk_size):
    output = []
    for start in range(0, len(observations), chunk_size):
        output.append(critic(torch.from_numpy(observations[start:start+chunk_size]).to(device),
                             torch.from_numpy(goals[start:start+chunk_size]).to(device)).reshape(-1).cpu().numpy())
    values = np.concatenate(output) if output else np.empty(0)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite task goal distances")
    return values


def proximity_audit(raw, support, env, scenario, heading_degrees):
    obs = raw.all_observations.numpy()
    eps = support['episodes']
    ids = np.concatenate([e['obs_ids'] for e in eps])
    contexts = np.concatenate([np.full(len(e['obs_ids']), e['context']) for e in eps])
    successful = [e for e in eps if e['record']['continuous_goal_witness']]
    success_ids = np.concatenate([e['obs_ids'] for e in successful]) if successful else np.array([], dtype=int)
    success_contexts = np.concatenate([np.full(len(e['obs_ids']), e['context']) for e in successful]) if successful else np.array([], dtype=int)
    anchors = sorted(support['valid_anchors'])
    anchor_ids = support['sources'][anchors]
    expected_goal = env.abstract_goal_for_task(support['summary']['device_id'])
    if any(not np.array_equal(e['goal'], expected_goal) for e in eps):
        raise ValueError("Different task goals cannot share proximity candidates")
    probes = build_probe_records(scenario)
    rows = []
    for probe in probes:
        if probe['device_id'] != support['summary']['device_id']:
            continue
        query = env.observation_for_task(np.asarray(probe['state'], np.float32), probe['device_id'])
        row = {'probe_id': probe['probe_id'], 'x': float(query[0]), 'y': float(query[1]),
               'heading_label': probe['heading_label'], 'same_task_candidates_only': True}
        for name, candidates, contexts_for_candidates in [
                ('physical_replay', obs[ids], contexts),
                ('successful_suffix', obs[success_ids], success_contexts),
                ('legal_anchor', obs[anchor_ids], np.array([int(raw.transition_infos['context_id'][r]) for r in anchors]))]:
            near = nearest_state(query, candidates, heading_tolerance=np.deg2rad(heading_degrees))
            row[name+'_position_distance'] = near['position_distance'] if near else None
            row[name+'_heading_difference_rad'] = near['heading_difference_rad'] if near else None
            row[name+'_exact_match'] = near['exact_full_observation_match'] if near else False
            row[name+'_context_id'] = int(contexts_for_candidates[near['candidate_index']]) if near else None
        rows.append(row)
    attachments = []
    for e in eps:
        if not e['record']['visits_U']:
            continue
        candidates = obs[success_ids]
        best = None
        for t, query in enumerate(obs[e['obs_ids']][e['record']['first_U_timestep']:]):
            near = nearest_state(query, candidates, heading_tolerance=np.deg2rad(heading_degrees))
            if near and (best is None or near['position_distance'] < best['position_distance']):
                best = {**near, 'query_timestep': t + e['record']['first_U_timestep']}
        attachments.append({'episode_id': e['episode_id'], 'context_id': e['context'],
                            'outcome': e['record']['outcome'], 'heading_tolerance_deg': heading_degrees,
                            'minimum_xy_to_successful_suffix': best['position_distance'] if best else None,
                            'query_timestep': best['query_timestep'] if best else None,
                            'successful_context_id': int(success_contexts[best['candidate_index']]) if best else None,
                            'exact_match': best['exact_full_observation_match'] if best else False,
                            'is_recorded_connector': False})
    return rows, attachments


def evaluate_checkpoint(spec, raw, support, device, chunk_size, tolerance):
    agent, _, step, _, _ = load_models(spec, device)
    obs = raw.all_observations.numpy()
    rows = support['target_rows']
    goals = raw.transition_infos['task_goal_observations'].numpy()
    costs = (-raw.rewards.numpy()).clip(min=0)
    endpoint_ids = np.array([e['obs_ids'][-1] for e in support['episodes']])
    endpoint_goals = np.stack([e['goal'] for e in support['episodes']])
    results = []
    for i, critic in enumerate(agent.critics):
        print(f"Task-path model audit: {Path(spec['checkpoint']).parent.name} step={step} critic={i}", flush=True)
        distances, _, _ = evaluate_edges(critic, raw, chunk_size=chunk_size, device=device)
        task_values = np.full(raw.num_transitions, np.nan)
        task_values[rows] = goal_distances(critic, obs[support['sources'][rows]], goals[rows], device, chunk_size)
        endpoints = goal_distances(critic, obs[endpoint_ids], endpoint_goals, device, chunk_size)
        carry = np.full(raw.num_transitions, np.nan)
        for e, value in zip(support['episodes'], endpoints):
            carry[e['rows']] = value
        groups = {}
        for name, mask in [('U_interior', support['interior']), ('U_west_exit', support['west_exit']),
                           ('successful_suffix_sources', support['witnessed'])]:
            ids = np.flatnonzero(mask)
            has_path = support['witnessed'][ids]
            path_bound = support['tail_costs'][ids]  # only absolute when witnessed
            relative_bound = path_bound + carry[ids]
            groups[name] = {'rows': len(ids), 'task_distance': quantiles(task_values[ids]),
                            'remaining_recorded_segment_cost': quantiles(path_bound),
                            'unanchored_endpoint_goal_distance': quantiles(carry[ids[~has_path]]),
                            'absolute_witness_bound_rows': int(has_path.sum()),
                            'absolute_witness_bound_violation_count': int(np.sum(task_values[ids[has_path]] > path_bound[has_path] + tolerance)),
                            'relative_bound_violation_count': int(np.sum(task_values[ids] > relative_bound + tolerance)),
                            'one_step_constraints': summarize_edges(distances[ids], costs[ids], tolerance=tolerance)}
        success_eps = np.array([e['record']['continuous_goal_witness'] for e in support['episodes']], dtype=bool)
        results.append({'critic': i, 'groups': groups,
                        'witnessed_terminal_goal_distance': quantiles(endpoints[success_eps]),
                        'failed_terminal_goal_distance': quantiles(endpoints[~success_eps]),
                        'anchor_constraints': summarize_edges(distances[sorted(support['valid_anchors'])], costs[sorted(support['valid_anchors'])], tolerance=1e-6)})
    return {'checkpoint': spec['checkpoint'], 'optim_steps': step, 'critics': results}


def write_report(output, payload):
    summary = payload['support']
    group = next(g for g in summary['groups'] if g['group'] == 'visits_U')
    lines = ['# U 到 task anchor 的经验路径支持审计', '',
             f"目标：`{summary['device_id']}`；replay SHA256：`{payload['replay_sha256']}`。",
             '所有计数来自已验证的 replay；没有采集新数据或启动训练。', '',
             '## 连续轨迹的缺口', '',
             f"访问 U 的目标任务轨迹共 {group['episodes']} 条，完整成功到合法 anchor 的记录 {group['continuous_witnesses']} 条；",
             f"这些 context 中有合法 anchor 的轨迹 {group['valid_context_anchors']} 条。合法 anchor 本身不证明轨迹接到了它。", '',
             '| 按时间正序到达的阶段 | 轨迹数 |', '|---|---:|']
    lines += [f'| {k} | {n} |' for k, n in group['stages'].items()]
    lines += ['', '满足位置、扇区和 LOS 的候选点进一步检查：', '',
              '| context | 合法位置点数 | 朝向满足点数 | 通信满足点数 | 最好的 heading margin | 终止原因 |',
              '|---|---:|---:|---:|---:|---|']
    for r in summary['terminal_position_candidates']:
        lines.append(f"| {r['context_id']} | {r['joint_position_los_candidate_points_after_U']} | {r['candidate_heading_feasible_points']} | {r['candidate_communication_feasible_points']} | {r['candidate_best_heading_margin']:.4f} | {r['outcome']} |")
    lines += ['', '终止原因与失败前沿：', '', '```json', json.dumps(summary['failure_frontiers'], ensure_ascii=False, indent=2), '```', '',
              '## 严格见证与近邻诊断', '', '```json', json.dumps(summary['source_support'], ensure_ascii=False, indent=2), '```', '',
              '`strict_context` 不跨 episode context；`same_task_goal` 允许跨 context，但只连接相同设备、',
              '相同完整 goal 和逐字节相同的完整 observation。context_id 是采集编号，不是新的环境任务。',
              '两个图均没有近邻边、状态取整或最短路成本回归。近邻报告中的距离只表示可能缺口；',
              '没有记录转移的近邻不能提供严格见证。', '',
              f"合法 anchor 共 {summary['valid_anchors_all_tasks']}/{summary['abstract_edges_all_tasks']}，",
              f"物理动力学验证边数 {summary['physical_dynamics']['checked_edges']}，不匹配数 0。", '',
              '## 模型检查', '', '| 模型 | critic | U 状态数 | 有绝对路径上界的 U 状态数 | U task 距离中位数 | 未锚定末尾到 goal 距离中位数 | 普通边容差违反数 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for cp in payload['checkpoints']:
        for c in cp['critics']:
            g = c['groups']['U_interior']
            median = g['task_distance']['median'] if g['task_distance'] else None
            carry = g['unanchored_endpoint_goal_distance']['median'] if g['unanchored_endpoint_goal_distance'] else None
            lines.append(f"| {Path(cp['checkpoint']).parent.name} {cp['optim_steps']} | {c['critic']} | {g['rows']} | {g['absolute_witness_bound_rows']} | {median} | {carry} | {g['one_step_constraints']['violation_count']} |")
    lines += ['', '失败片段只给出相对界 `d(s,g) ≤ C(s→end) + d(end,g)`；末尾没有接到合法终止',
              '状态时，不能把片段成本单独当作到 goal 的绝对上界。成功路径成本只作为审计值，',
              '不作为新训练 target，不假定是最优路径。缺少记录链不证明环境不可达，有限网络也',
              '可能通过共享参数泛化；零局部违反不是全局最优性证明。', '',
              '![Recorded trajectory funnel](trajectory_funnel.png)', '',
              '文件：`episodes.csv`（逐轨迹前沿）、`u_sources.csv`（逐 U 状态支持）、',
              '`anchors.csv`（anchor 合法性）、`probe_support.csv`（固定探针覆盖）、',
              '`near_suffix_attachments.csv`（近邻敏感性，非见证）、`task_path_support_audit.json`（完整统计）。']
    lines += ['', '## 按数据来源区分路径支持', '',
              '新增规划器轨迹有见证不意味着原 replay 的 U 状态已经连通；模型泛化需单独评估。',
              '', '```json', json.dumps(summary['source_support_by_origin'], ensure_ascii=False, indent=2), '```']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--replay-cache')
    parser.add_argument('--task-device-id', default='u_trap_target')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--heading-tolerance-deg', type=float, default=15)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.chunk_size, args.cpu_threads) <= 0 or not 0 <= args.heading_tolerance_deg <= 180 or args.tolerance < 0:
        parser.error('Invalid audit counts/tolerances')
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = [load_run_spec(p) for p in args.checkpoints]
    if len({s['expected_sha256'] for s in specs}) != 1:
        raise ValueError('All checkpoints must use exactly the same replay')
    raw = load_replay(specs[0], args.replay_cache or output/'verified_replay.pt')
    for spec in specs:
        validate_replay(raw, spec['expected_sha256'])
        if spec['scenario'] != specs[0]['scenario']:
            raise ValueError('Scenario mismatch')
    env = CommInspectionDubinsUAV2D(**get_env_kwargs(specs[0]['args']))
    support = build_support(raw, env, specs[0]['scenario'], args.task_device_id)
    write_csv(output/'episodes.csv', support['episode_records'])
    write_csv(output/'anchors.csv', support['anchor_records'])
    records = []
    for e in support['episodes']:
        observation_rows = raw.all_observations.numpy()[e['obs_ids']]
        exits = inside(observation_rows, specs[0]['scenario']['metadata']['exploration_diagnostic_regions']['u_trap_west_exit'])
        future_exit = np.maximum.accumulate(exits[::-1])[::-1]
        for t, row in enumerate(e['rows']):
            if support['interior'][row] or support['west_exit'][row]:
                records.append({'transition_index': int(row), 'episode_id': e['episode_id'],
                                'context_id': e['context'], 'timestep': t,
                                'planner_path_support': bool(support['planner_rows'][row]),
                                'device_id': args.task_device_id,
                                'x': float(observation_rows[t, 0]), 'y': float(observation_rows[t, 1]),
                                'heading_rad': float(np.arctan2(observation_rows[t, 3], observation_rows[t, 2])),
                                'region': 'interior' if support['interior'][row] else 'west_exit',
                                'outcome': e['record']['outcome'],
                                'west_exit_recorded_at_or_after_this_state': bool(future_exit[t]),
                                'continuous_goal_witness': bool(support['witnessed'][row]),
                                'strict_context_exact_graph_reachable': bool(support['strict'][row]),
                                'same_task_goal_exact_graph_reachable': bool(support['shared'][row]),
                                'remaining_segment_cost': float(support['tail_costs'][row]),
                                'absolute_goal_path_bound': float(support['tail_costs'][row]) if support['witnessed'][row] else None})
    write_csv(output/'u_sources.csv', records)
    probes, attachments = proximity_audit(raw, support, env, specs[0]['scenario'], args.heading_tolerance_deg)
    write_csv(output/'probe_support.csv', probes)
    write_csv(output/'near_suffix_attachments.csv', attachments)
    write_trajectory_plot(output, raw, support, specs[0]['scenario'])
    models = [evaluate_checkpoint(spec, raw, support, auto_device(args.device), args.chunk_size, args.tolerance) for spec in specs]
    sensitivity = {str(t): sum(r['minimum_xy_to_successful_suffix'] is not None and r['minimum_xy_to_successful_suffix'] <= t for r in attachments)
                   for t in [0.05, 0.1, 0.25, 0.5, 1.0]}
    payload = {'schema_version': 1, 'replay_sha256': specs[0]['expected_sha256'],
               'no_training_or_path_targets_added': True, 'support': support['summary'],
               'near_suffix_heading_tolerance_deg': args.heading_tolerance_deg,
               'near_suffix_position_threshold_episode_counts': sensitivity,
               'proximity_is_not_a_recorded_connector': True, 'checkpoints': models}
    (output/'task_path_support_audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    write_report(output, payload)
    print(json.dumps(support['summary'], indent=2), flush=True)
    print(f'Task path support audit saved: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
