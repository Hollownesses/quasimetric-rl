"""Fixed replay pair banks for the Global Push support ablations.

Like the original environment pair bank, each replay row owns one fixed pair.
No costs or regression targets are added. B0 draws both endpoints independently;
B1 uses the row's source and a uniformly sampled strictly later physical state.
"""
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch


def replay_fingerprint(raw):
    """Hash the complete replay constraints and task contexts, excluding push pairs."""
    digest = hashlib.sha256()
    for tensor in (raw.episode_lengths, raw.all_observations, raw.actions,
                   raw.rewards, raw.terminals, raw.timeouts,
                   raw.transition_infos["context_id"],
                   raw.transition_infos["task_goal_observations"]):
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def configure_global_push_sampling(dataset, *, state_mode="environment",
                                  task_mode="all", seed=42):
    if state_mode not in ("environment", "replay", "trajectory"):
        raise ValueError(f"Unknown Global Push state sampling: {state_mode}")
    if task_mode not in ("all", "witnessed"):
        raise ValueError(f"Unknown Global Push task sampling: {task_mode}")
    raw = dataset.raw_data
    infos = raw.transition_infos
    if "task_goal_observations" not in infos:
        raise ValueError("Global Push support experiments require goal-set replay")
    n = len(dataset)
    obs = raw.all_observations.numpy()
    ctx = infos["context_id"].numpy()
    goals = infos["task_goal_observations"].numpy()
    abstract = infos["abstract_goal_edge"].numpy().astype(bool)
    terminal_source = infos["source_terminal_goal_state"].numpy().astype(bool)
    success = infos["task_success_episode"].numpy().astype(bool)
    terminals = raw.terminals.numpy()
    timeouts = raw.timeouts.numpy()
    rewards = raw.rewards.numpy()
    feature_indices = getattr(dataset.create_env(), "task_context_indices", {})
    source_obs = np.arange(n) + dataset.indices_to_episode_indices.numpy()
    end_obs = np.full(n, -1, dtype=np.int64)
    witnessed = np.zeros(n, dtype=bool)
    pools = defaultdict(list)
    # A witness must actually connect to a recorded zero-cost legal anchor.
    # A sampled anchor for a failed rollout cannot turn that rollout into a success.
    def context_key(i):
        return int(ctx[i]), goals[i].tobytes()

    anchors = {
        (context_key(i), obs[source_obs[i]].tobytes())
        for i in np.flatnonzero(abstract & terminal_source)
        if rewards[i] == 0 and np.array_equal(obs[source_obs[i] + 1], goals[i])
        and ("task_feasible" not in feature_indices
             or obs[source_obs[i], feature_indices["task_feasible"]] > 0.5)
    }
    transition_start = 0
    for length in raw.episode_lengths.tolist():
        stop = transition_start + length
        # Split at artificial edges, termination/timeout, or a context change.
        start = transition_start
        while start < stop:
            if abstract[start]:
                start += 1
                continue
            end = start + 1
            while (end < stop and not abstract[end]
                   and not bool(terminals[end - 1] or timeouts[end - 1])
                   and context_key(end) == context_key(start)):
                end += 1
            last_obs = source_obs[end - 1] + 1
            if end < stop:
                # MultiEpisodeData stores only the final independent next state.
                # At an internal reset/context/abstract boundary, the next row
                # may start somewhere else: omit that ambiguous endpoint.
                last_obs -= 1
            end_obs[start:end] = last_obs
            pools[context_key(start)].extend(range(int(source_obs[start]), int(last_obs) + 1))
            if (end == stop and bool(terminals[end - 1]) and not bool(timeouts[end - 1])
                    and success[end - 1]
                    and (context_key(start), obs[last_obs].tobytes()) in anchors
                    and ("task_feasible" not in feature_indices
                         or obs[last_obs, feature_indices["task_feasible"]] > 0.5)):
                witnessed[start:end] = True
            start = end
        transition_start = stop

    rng = np.random.default_rng(int(seed) + 63_127_019)
    pair_mask = ~abstract
    pair_source = np.full(n, -1, dtype=np.int64)
    pair_goal = np.full(n, -1, dtype=np.int64)
    spans = np.full(n, -1, dtype=np.int64)
    if state_mode == "replay":
        rows_by_context = defaultdict(list)
        for i in np.flatnonzero(pair_mask):
            rows_by_context[context_key(i)].append(i)
        for key, rows in rows_by_context.items():
            pool = np.unique(pools[key])
            pair_source[rows] = rng.choice(pool, size=len(rows))
            pair_goal[rows] = rng.choice(pool, size=len(rows))
    elif state_mode == "trajectory":
        pair_mask &= end_obs > source_obs
        rows = np.flatnonzero(pair_mask)
        pair_source[rows] = source_obs[rows]
        # Uniform k in [1, remaining segment length], including k=1 and s_last.
        spans[rows] = rng.integers(1, end_obs[rows] - source_obs[rows] + 1)
        pair_goal[rows] = source_obs[rows] + spans[rows]
    if state_mode != "environment":
        sources = np.zeros_like(infos["global_push_source_observations"].numpy())
        targets = np.zeros_like(sources)
        sources[pair_mask] = obs[pair_source[pair_mask]]
        targets[pair_mask] = obs[pair_goal[pair_mask]]
        infos["global_push_source_observations"] = torch.from_numpy(sources)
        infos["global_push_goal_observations"] = torch.from_numpy(targets)
        infos["global_push_pair_mask"] = torch.from_numpy(pair_mask)
    task_candidate = ~abstract & ~terminal_source
    infos["global_push_task_witness_mask"] = torch.from_numpy(witnessed & task_candidate)
    infos["global_push_task_mask"] = torch.from_numpy(
        task_candidate & witnessed if task_mode == "witnessed" else task_candidate
    )
    infos["global_push_pair_steps"] = torch.from_numpy(spans)
    infos["global_push_pair_source_obs_index"] = torch.from_numpy(pair_source)
    infos["global_push_pair_goal_obs_index"] = torch.from_numpy(pair_goal)
    return {"state_sampling": state_mode, "task_sampling": task_mode,
            "seed": int(seed), "pair_bank": "fixed_one_pair_per_replay_row",
            "trajectory_span_sampling": "uniform_positive_remaining_steps",
            "path_cost_target_added": False,
            "whole_push_has_recorded_witnesses": state_mode == "trajectory" and task_mode == "witnessed"}


def write_global_push_coverage(dataset, env, output_dir, *, sampling,
                               position_resolution=1.0, heading_bins=12,
                               success_weight=1.0, regions=None):
    """Report the actual fixed bank, replay-relative coverage, and sampler mass.

    CSV rows include zero-pair replay bins so missing signal stays visible.
    Coverage is empirical replay coverage, never coverage of unseen free space.
    """
    if not np.isfinite(position_resolution) or position_resolution <= 0 or heading_bins <= 0:
        raise ValueError("Global Push coverage bin sizes must be positive")
    raw = dataset.raw_data
    infos = raw.transition_infos
    n = len(dataset)
    abstract = infos["abstract_goal_edge"].numpy().astype(bool)
    terminal = infos["source_terminal_goal_state"].numpy().astype(bool)
    success = infos["task_success_episode"].numpy().astype(bool)
    pair_mask = infos["global_push_pair_mask"].numpy().astype(bool) & ~abstract
    task_candidate = ~abstract & ~terminal
    task_mask = infos["global_push_task_mask"].numpy().astype(bool) & task_candidate
    witness = infos["global_push_task_witness_mask"].numpy().astype(bool)
    contexts = infos["context_id"].numpy()
    devices = infos["device_index"].numpy()
    observations = raw.all_observations.numpy()
    source_ids = np.arange(n) + dataset.indices_to_episode_indices.numpy()
    counts = defaultdict(Counter)
    region_counts = defaultdict(Counter)

    def record(observation, row, field):
        state = env.observation_to_state(observation)
        x, y, theta = map(float, state)
        key = (int(devices[row]), int(contexts[row]),
               int(np.floor((x - env.x_min) / position_resolution)),
               int(np.floor((y - env.y_min) / position_resolution)),
               int(np.floor(((theta + np.pi) % (2 * np.pi)) / (2 * np.pi) * heading_bins)) % heading_bins)
        counts[key][field] += 1
        for name, (xmin, ymin, xmax, ymax) in (regions or {}).items():
            if xmin <= x <= xmax and ymin <= y <= ymax:
                region_counts[name][field] += 1

    for i in np.flatnonzero(~abstract):
        record(observations[source_ids[i]], i, "replay_sources")
        record(observations[source_ids[i] + 1], i, "replay_targets")
        if task_candidate[i]:
            record(observations[source_ids[i]], i, "task_candidates")
        if task_mask[i]:
            record(observations[source_ids[i]], i, "task_retained")
        if witness[i]:
            record(observations[source_ids[i]], i, "task_witnessed")
        if pair_mask[i]:
            record(infos["global_push_source_observations"][i].numpy(), i, "pair_sources")
            record(infos["global_push_goal_observations"][i].numpy(), i, "pair_targets")
    fields = ("replay_sources", "replay_targets", "pair_sources", "pair_targets",
              "task_candidates", "task_retained", "task_witnessed")
    output_dir = Path(output_dir)
    csv_path = output_dir / "global_push_pair_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("device_index", "context_id", "x_bin", "y_bin", "heading_bin") + fields
                        + ("pair_source_present", "pair_endpoint_present", "task_retained_fraction"))
        for key, values in sorted(counts.items()):
            writer.writerow(key + tuple(values[field] for field in fields)
                            + (int(bool(values["pair_sources"])),
                               int(bool(values["pair_sources"] or values["pair_targets"])),
                               values["task_retained"] / values["task_candidates"] if values["task_candidates"] else ""))

    def summarize(values):
        values = list(values)
        replay_bins = [v for v in values if v["replay_sources"] or v["replay_targets"]]
        candidates = sum(v["task_candidates"] for v in values)
        retained = sum(v["task_retained"] for v in values)
        return {**{field: sum(v[field] for v in values) for field in fields},
                "replay_bins": len(replay_bins),
                "pair_endpoint_bin_coverage": sum(bool(v["pair_sources"] or v["pair_targets"]) for v in replay_bins) / max(1, len(replay_bins)),
                "pair_source_bin_coverage": sum(bool(v["pair_sources"]) for v in replay_bins) / max(1, len(replay_bins)),
                "task_retained_fraction": retained / max(1, candidates)}

    weights = np.where(success, float(success_weight), 1.0)
    spans = infos["global_push_pair_steps"].numpy()[pair_mask]
    positive_spans = spans[spans > 0]
    per_context = defaultdict(list)
    per_device = defaultdict(list)
    per_heading = defaultdict(list)
    for (device, context, *_bins), values in counts.items():
        per_context[f"{device}:{context}"].append(values)
        per_device[str(device)].append(values)
        per_heading[str(_bins[-1])].append(values)
    def quantiles(values):
        return dict(zip(("min", "p25", "p50", "p75", "p95", "max"),
                        np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist())) if len(values) else None

    pair_rows = np.flatnonzero(pair_mask)
    physical_sources = np.asarray([env.observation_to_state(infos["global_push_source_observations"][i].numpy()) for i in pair_rows]).reshape(-1, 3)
    physical_targets = np.asarray([env.observation_to_state(infos["global_push_goal_observations"][i].numpy()) for i in pair_rows]).reshape(-1, 3)
    spatial_spans = np.linalg.norm(physical_targets[:, :2] - physical_sources[:, :2], axis=1)
    heading_spans = np.abs((physical_targets[:, 2] - physical_sources[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    return {
        **sampling, "coverage_reference": "physical_replay_position_heading_context_bins",
        "position_resolution": float(position_resolution), "heading_bins": int(heading_bins),
        "physical_replay_sha256": replay_fingerprint(raw),
        "coverage_csv": str(csv_path.resolve()),
        "overall": summarize(counts.values()),
        "per_device": {k: summarize(v) for k, v in per_device.items()},
        "per_context": {k: summarize(v) for k, v in per_context.items()},
        "per_heading": {k: summarize(v) for k, v in per_heading.items()},
        "diagnostic_regions": {k: {f: v[f] for f in fields} for k, v in region_counts.items()},
        "state_pairs_success_rows": int(np.sum(pair_mask & success)),
        "state_pairs_failed_rows": int(np.sum(pair_mask & ~success)),
        "state_pair_slot_retained_fraction": float(pair_mask.sum() / max(1, np.sum(~abstract))),
        "state_pair_success_sampler_fraction": float(weights[pair_mask & success].sum() / max(1.0, weights[pair_mask].sum())),
        "task_retained_sampler_fraction": float(weights[task_mask].sum() / max(1.0, weights[task_candidate].sum())),
        "task_witness_sampler_fraction": float(weights[witness].sum() / max(1.0, weights[task_candidate].sum())),
        "span_steps_histogram": {str(k): v for k, v in sorted(Counter(positive_spans.tolist()).items())},
        "span_steps_quantiles": quantiles(positive_spans),
        "spatial_span_quantiles": quantiles(spatial_spans),
        "heading_span_radians_quantiles": quantiles(heading_spans),
        "pairs_without_temporal_span": int(np.sum(spans < 0)),
    }
