import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from minimal_qrl import constraint_only_recovery as recovery
from minimal_qrl.test_global_push_sampling import TinyEnv, dataset, episode
from quasimetric_rl.modules.quasimetric_critic.losses import CriticBatchInfo
from quasimetric_rl.modules.quasimetric_critic.losses.latent_dynamics import LatentDynamicsLoss


def test_complete_pass_visits_each_edge_once_including_last_partial_batch():
    raw = dataset(episode(range(6)), episode([5, 999], anchor=True),
                  episode([20, 21, 22], success=True)).raw_data
    batches = list(recovery.replay_batches(raw, 3, seed=42, pass_index=1))
    indices = np.concatenate([ids.numpy() for ids, _ in batches])
    assert len(batches[-1][0]) == 2
    np.testing.assert_array_equal(np.sort(indices), np.arange(raw.num_transitions))
    # Source/target alignment survives shuffling and episode offsets.
    sources = recovery.edge_metadata(raw)[2]
    for ids, batch in batches:
        assert torch.equal(batch.observations, raw.all_observations[sources[ids.numpy()]])
        assert torch.equal(batch.next_observations, raw.all_observations[sources[ids.numpy()] + 1])
    again = list(recovery.replay_batches(raw, 3, seed=42, pass_index=1))
    assert all(torch.equal(a[0], b[0]) for a, b in zip(batches, again))
    different = np.concatenate([ids.numpy() for ids, _ in recovery.replay_batches(raw, 3, seed=42, pass_index=2)])
    assert not np.array_equal(indices, different)


def test_disabled_latent_loss_skips_model_and_has_no_dynamics_gradient():
    raw = dataset(episode([0, 1])).raw_data
    _, batch = next(recovery.replay_batches(raw, 1, seed=42, pass_index=1))
    class NoDynamics(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError('Disabled dynamics should not run')
    class Critic:
        latent_dynamics = NoDynamics()
    zx = batch.observations.clone().requires_grad_()
    result = LatentDynamicsLoss(weight=0)(batch, CriticBatchInfo(critic=Critic(), zx=zx, zy=zx))
    assert result.loss.item() == 0
    result.loss.backward()
    assert torch.equal(zx.grad, torch.zeros_like(zx))


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_restore_keeps_adam_dual_and_scheduler_state_but_disables_objectives(tmp_path):
    # Metadata is generated independently of local results so tests can run in CI.
    import argparse
    source_args = argparse.Namespace(
        seed=42, num_critics=1, qrl_local_constraint_mode='kkt_functional',
        global_push_objective='linear', abstract_goal_edge_loss_weight=1.0,
        qrl_kkt_init_lagrange_multiplier=.01, qrl_kkt_augmented_lagrangian_rho=1,
        qrl_kkt_dual_hidden_sizes=[8, 8], qrl_kkt_dual_max=100000,
        qrl_kkt_dual_steps=1, qrl_kkt_dual_start_violation_fraction=0,
        qrl_kkt_dual_projected_step_size=.1, qrl_kkt_dual_feature_scale=5,
        qrl_kkt_dual_raw_min=-10, qrl_kkt_dual_lr=1e-4,
    )
    env = TinyEnv()
    original, losses = recovery.recovery_conf(source_args).make(
        env_spec=recovery.EnvSpec.from_env(env), total_optim_steps=100)
    # Seed genuine Adam moments in both the critic and dynamics before disabling
    # the latter: zero gradients would still move it if the loss didn't skip it.
    losses.critic_losses[0].latent_dynamics.weight = .1
    raw = dataset(episode([0, 2, 3]), episode([3, 999], anchor=True)).raw_data
    _, batch = next(recovery.replay_batches(raw, 3, seed=42, pass_index=0))
    losses(original, batch, optimize=True)
    path = tmp_path / 'source.pth'
    checkpoint = {'optim_steps': 3, 'agent': copy.deepcopy(original.state_dict()),
                  'losses': copy.deepcopy(losses.state_dict()),
                  'experiment': {'local_constraint_mode': 'kkt_functional'}}
    torch.save(checkpoint, path)
    agent, restored, step, _ = recovery.load_recovery_models(
        {'checkpoint': str(path), 'args': source_args}, env, 'cpu')
    assert step == 3
    assert_nested_equal(agent.state_dict(), checkpoint['agent'])
    assert_nested_equal(restored.state_dict(), checkpoint['losses'])
    cl = restored.critic_losses[0]
    assert cl.global_push.abstract_goal_ratio == cl.global_push.state_goal_ratio == 0
    assert cl.latent_dynamics.weight == 0
    assert cl.abstract_goal_edge.weight == 1
    for name in ('temporal_path', 'goal_return', 'nstep_goal', 'mqe_waypoint'):
        assert getattr(cl, name).weight == 0
    initial_dynamics = copy.deepcopy(agent.critics[0].latent_dynamics.state_dict())
    critic_state = copy.deepcopy(cl.critic_optim.state_dict())
    result = restored(agent, batch, optimize=True)
    info = result.info['critic_00']
    assert info['global_push']['dist'].item() == 0
    assert info['latent_dynamics']['sq_dists'].item() == 0
    assert_nested_equal(agent.critics[0].latent_dynamics.state_dict(), initial_dynamics)
    assert all(p.grad is None for p in agent.critics[0].latent_dynamics.parameters())
    # The active critic actually performs an update; this is not a freeze-all test.
    old_steps = [state['step'].item() for state in critic_state['state'].values()]
    new_steps = [state['step'].item() for state in cl.critic_optim.state_dict()['state'].values()]
    assert max(new_steps) > max(old_steps)


def test_recovery_phase_uses_full_pass_budget_and_saved_replay(tmp_path):
    fake = tmp_path / 'python'
    capture = tmp_path / 'args.json'
    fake.write_text(f'#!{sys.executable}\nimport os,json,sys\n'
                    'open(os.environ["RECOVERY_CAPTURE"],"a").write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = {**os.environ, 'PYTHON_BIN': str(fake), 'PHASE': 'train_qrl_constraint_only_recovery',
           'OUTPUT_ROOT': str(tmp_path), 'QRL_INIT_CHECKPOINT': 'b1_3k.pth',
           'RECOVERY_PASSES': '2', 'TOTAL_STEPS': '120000', 'RECOVERY_CAPTURE': str(capture)}
    subprocess.run(['bash', str(Path(__file__).with_name('run_comm_inspection_diagnostic.sh'))],
                   env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1  # No regeneration of the training scenario.
    args = calls[0]
    assert args[:2] == ['-m', 'minimal_qrl.constraint_only_recovery']
    assert args[args.index('--recovery-passes') + 1] == '2'
    assert args[args.index('--init-checkpoint') + 1] == 'b1_3k.pth'
    assert '--total-steps' not in args and '--batch-size' not in args
