import csv
import json
import os
from pathlib import Path
import subprocess
import uuid

import gym
import numpy as np
import pytest
import torch

from minimal_qrl.dataset import _goal_set_transition_infos, _slice_episode
from minimal_qrl.global_push_sampling import configure_global_push_sampling, write_global_push_coverage
from quasimetric_rl.data import Dataset, EpisodeData, register_offline_env
from quasimetric_rl.modules.quasimetric_critic.losses import CriticBatchInfo
from quasimetric_rl.modules.quasimetric_critic.losses.global_push import GlobalPushLoss


class TinyEnv(gym.Env):
    x_min = y_min = 0.0

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1000, 1000, (4,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def observation_to_state(self, obs):
        return np.array([obs[0], obs[1], np.arctan2(obs[3], obs[2])])


def episode(xs, *, context=0, success=False, anchor=False, goal=999):
    observations = np.array([[x, 0, 1, 0] for x in xs], dtype=np.float32)
    n = len(xs) - 1
    infos = _goal_set_transition_infos(
        n=n, task_goal_obs=np.array([goal, 0, 1, 0], np.float32),
        context_id=context, abstract_goal_edge=anchor,
        source_terminal_goal_state=anchor, task_success_episode=success,
    )
    return EpisodeData.from_simple_trajectory(
        observations=observations[:-1], next_observations=observations[1:],
        actions=np.zeros((n, 1), np.float32),
        rewards=np.full(n, 0 if anchor else -1, np.float32),
        terminals=np.array([False] * (n - 1) + [True]),
        timeouts=np.zeros(n, dtype=bool), transition_infos=infos,
    )


def dataset(*episodes):
    key = uuid.uuid4().hex
    register_offline_env(key, key, load_episodes_fn=lambda: iter(episodes), create_env_fn=TinyEnv)
    return Dataset(key, key, future_observation_discount=.99)


def test_b0_draws_both_endpoints_from_matching_replay_context_and_task():
    data = dataset(episode(range(80)), episode(range(100, 150)),
                   episode([40, 999], anchor=True),
                   episode(range(200, 250), context=1),
                   episode(range(300, 350), goal=998))
    configure_global_push_sampling(data, state_mode="replay", seed=3)
    infos = data.raw_data.transition_infos
    mask = infos['global_push_pair_mask']
    assert not mask[128]  # synthetic anchor row
    for source, target, ctx, goal in zip(
            infos['global_push_source_observations'][mask, 0],
            infos['global_push_goal_observations'][mask, 0],
            infos['context_id'][mask], infos['task_goal_observations'][mask, 0]):
        allowed = set(range(80)) | set(range(100, 150)) if ctx == 0 and goal == 999 else (
            set(range(200, 250)) if ctx == 1 else set(range(300, 350)))
        assert source.item() in allowed and target.item() in allowed
    # Independent draws allow reversed temporal order and endpoints in different episodes.
    x = infos['global_push_source_observations'][:128, 0]
    y = infos['global_push_goal_observations'][:128, 0]
    assert bool((x > y).any())
    assert bool(((x < 80) & (y >= 100)).any())


def test_b1_positive_time_same_segment_both_outcomes_and_no_extra_cost_target():
    first = episode(range(100))
    first.terminals[30] = True  # an internal boundary may not be crossed
    data = dataset(first, episode(range(200, 250), success=True),
                   episode([249, 999], anchor=True))
    original_rewards = data.raw_data.rewards.clone()
    original_keys = set(data.raw_data.transition_infos)
    configure_global_push_sampling(data, state_mode="trajectory", seed=7)
    infos = data.raw_data.transition_infos
    mask = infos['global_push_pair_mask']
    sources = infos['global_push_pair_source_obs_index'][mask]
    goals = infos['global_push_pair_goal_obs_index'][mask]
    assert torch.all(goals > sources)
    assert torch.all(goals[sources < 31] <= 30)
    assert not mask[30]  # ambiguous endpoint at an internal reset is omitted
    assert torch.all(goals[(sources >= 31) & (sources < 100)] <= 99)
    assert torch.all(goals[sources >= 100] <= 150)
    assert torch.equal(goals - sources, infos['global_push_pair_steps'][mask])
    assert bool((infos['global_push_pair_steps'][mask] == 1).any())
    assert int(infos['global_push_pair_steps'].max()) > 10
    assert bool(infos['task_success_episode'][mask].any())
    assert bool((~infos['task_success_episode'][mask]).any())
    assert torch.equal(original_rewards, data.raw_data.rewards)
    assert not any('cost' in k or 'target' in k for k in set(infos) - original_keys)
    assert bool(infos['global_push_task_mask'][~infos['abstract_goal_edge']].all())


def test_task_witness_requires_complete_success_and_matching_recorded_anchor():
    complete = episode([0, 1, 2, 3], success=True)
    truncated = _slice_episode(episode([10, 11, 12, 13], success=True), 2)
    data = dataset(complete, episode([3, 999], anchor=True),
                   truncated, episode([13, 999], anchor=True),
                   episode([20, 21], success=True),  # no matching anchor
                   episode([30, 31]), episode([31, 999], anchor=True),
                   episode([40, 41], success=True, context=2),
                   episode([41, 999], anchor=True, context=3))
    configure_global_push_sampling(data, state_mode="trajectory", task_mode="witnessed")
    infos = data.raw_data.transition_infos
    assert torch.equal(torch.nonzero(infos['global_push_task_mask']).flatten(), torch.tensor([0, 1, 2]))
    assert torch.equal(infos['global_push_task_mask'], infos['global_push_task_witness_mask'])
    # Main minibatch keeps failure states and one-step constraints.
    batch = data[torch.arange(len(data))]
    assert batch.num_transitions == len(data)
    assert torch.equal(batch.transition_infos['global_push_task_mask'], infos['global_push_task_mask'])


@pytest.mark.parametrize('has_success', [False, True])
def test_task_witness_mask_gates_loss_and_gradient_without_fallback(has_success):
    data = (dataset(episode([0, 1, 2], success=True), episode([2, 999], anchor=True),
                    episode([10, 11])) if has_success else dataset(episode([0, 1, 2])))
    configure_global_push_sampling(data, state_mode="trajectory", task_mode="witnessed")
    batch = data[torch.arange(len(data))]

    class L1(torch.nn.Module):
        def forward(self, x, y):
            return (x - y).abs().sum(-1)

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Identity()
            self.quasimetric_model = L1()

    zx = batch.observations.clone().requires_grad_()
    loss = GlobalPushLoss(objective='linear', softplus_beta=.1, softplus_offset=15,
                          abstract_goal_ratio=.6, state_goal_ratio=0)
    result = loss(batch, CriticBatchInfo(critic=Critic(), zx=zx, zy=zx))
    expected_loss = -.6 * 998.5 if has_success else 0
    assert result.loss.item() == pytest.approx(expected_loss)
    expected_retention = 2 / 3 if has_success else 0
    assert result.info['global_push_task_set/retained_fraction'].item() == pytest.approx(expected_retention)
    result.loss.backward()
    mask = batch.transition_infos['global_push_task_mask']
    assert torch.equal(zx.grad[~mask], torch.zeros_like(zx.grad[~mask]))
    if has_success:
        assert bool((zx.grad[mask].abs().sum(-1) > 0).all())


def test_coverage_reports_span_retention_and_zero_signal_bins(tmp_path):
    data = dataset(episode([0, 1, 2, 3], success=True),
                   episode([3, 999], anchor=True), episode([10, 11, 12]))
    sampling = configure_global_push_sampling(data, state_mode='trajectory', task_mode='witnessed')
    stats = write_global_push_coverage(data, TinyEnv(), tmp_path, sampling=sampling,
                                       success_weight=4, regions={'U': [9, -1, 13, 1]})
    assert stats['overall']['task_retained_fraction'] == 3 / 5
    assert stats['task_retained_sampler_fraction'] == 12 / 14
    assert stats['state_pairs_success_rows'] == 3
    assert stats['state_pairs_failed_rows'] == 2
    assert sum(stats['span_steps_histogram'].values()) == 5
    assert stats['diagnostic_regions']['U']['task_retained'] == 0
    with open(stats['coverage_csv']) as handle:
        rows = list(csv.DictReader(handle))
    assert any(row['x_bin'] == '10' and row['task_retained_fraction'] == '0.0' for row in rows)
    assert stats['overall']['pair_source_bin_coverage'] < 1  # final endpoints have no outgoing pair
    json.dumps(stats)  # serialization must not leak numpy types


def test_pair_bank_seed_reproducible_without_changing_training_rng():
    data = dataset(episode(range(20)))
    torch_state = torch.get_rng_state().clone()
    configure_global_push_sampling(data, state_mode='trajectory', seed=19)
    pairs = data.raw_data.transition_infos['global_push_goal_observations'].clone()
    configure_global_push_sampling(data, state_mode='trajectory', seed=19)
    assert torch.equal(pairs, data.raw_data.transition_infos['global_push_goal_observations'])
    assert torch.equal(torch_state, torch.get_rng_state())


def test_b1_does_not_use_a_state_from_the_next_context_as_endpoint():
    physical = episode(range(6))
    physical.transition_infos['context_id'][2:] = 1
    data = dataset(physical)
    configure_global_push_sampling(data, state_mode='trajectory')
    infos = data.raw_data.transition_infos
    assert infos['global_push_pair_goal_obs_index'][0] == 1
    assert not infos['global_push_pair_mask'][1]
    assert bool(infos['global_push_pair_mask'][2:].all())


def test_disabling_environment_pairs_keeps_real_rollouts_identical():
    from minimal_qrl.dataset import QRLExploreConfig, create_dataset
    from minimal_qrl.test_comm_inspection_dubins_uav import make_env
    config = QRLExploreConfig(attempted_env_steps=30, start_heading_bins=2)
    def collect(include):
        return list(create_dataset(make_env(), max_steps_per_episode=10, seed=13,
                                    qrl_explore_config=config,
                                    include_global_push_pairs=include))
    original, replay = collect(True), collect(False)
    assert len(original) == len(replay)
    for a, b in zip(original, replay):
        for field in ('all_observations', 'actions', 'rewards', 'terminals', 'timeouts'):
            assert torch.equal(getattr(a, field), getattr(b, field))
        assert not bool(b.transition_infos['global_push_pair_mask'].any())


def test_skipping_pair_generation_does_not_call_environment_state_sampler():
    class NoSampling:
        active_device_index = 0
        def sample_valid_state(self, **kwargs):
            raise AssertionError('off-replay pair generation should be disabled')
    infos = _goal_set_transition_infos(n=2, task_goal_obs=np.zeros(4, np.float32),
                                      context_id=0, env=NoSampling(),
                                      include_global_push_pairs=False)
    assert not infos['global_push_pair_mask'].any()


@pytest.mark.parametrize('variant,state_mode,task_mode', [
    ('control', 'environment', 'all'), ('b0', 'replay', 'all'),
    ('b1', 'trajectory', 'all'), ('witnessed', 'trajectory', 'witnessed'),
])
def test_diagnostic_phase_forwards_isolated_sampling_experiment(tmp_path, variant, state_mode, task_mode):
    import sys
    fake_python = tmp_path / 'python'
    fake_python.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                           'if len(sys.argv) > 1 and sys.argv[1] == "minimal_qrl/train.py":\n'
                           '    open(os.environ["CAPTURE_ARGS"], "w").write(json.dumps(sys.argv[2:]))\n')
    fake_python.chmod(0o755)
    capture = tmp_path / 'args.json'
    env = {**os.environ, 'PYTHON_BIN': str(fake_python), 'ORACLE_BANK_EVAL': '0',
           'PHASE': f'train_qrl_push_support_{variant}', 'OUTPUT_ROOT': str(tmp_path),
           'CAPTURE_ARGS': str(capture)}
    subprocess.run(['bash', str(Path(__file__).with_name('run_comm_inspection_diagnostic.sh'))],
                   env=env, check=True, capture_output=True, text=True)
    args = json.loads(capture.read_text())
    def arg(name):
        return args[args.index('--' + name) + 1]
    assert arg('global-push-state-sampling') == state_mode
    assert arg('global-push-task-sampling') == task_mode
    assert arg('global-push-objective') == 'linear'
    assert arg('global-push-abstract-goal-ratio') == '0.6'
    assert arg('global-push-state-goal-ratio') == '0.4'
    for name in ('qrl-temporal-constraint-weight', 'qrl-goal-return-constraint-weight',
                 'qrl-nstep-goal-constraint-weight', 'qrl-mqe-waypoint-consistency-weight'):
        assert float(arg(name)) == 0
    assert arg('output-dir').endswith(f'qrl_training_push_support_{variant}')
