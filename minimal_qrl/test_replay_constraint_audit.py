import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from minimal_qrl.eval import replay_constraint_audit as audit
from minimal_qrl.global_push_sampling import replay_fingerprint
from minimal_qrl.test_global_push_sampling import TinyEnv, dataset, episode
from quasimetric_rl.data import EnvSpec
from quasimetric_rl.modules import QRLConf
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss


class ScaledL1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, source, target):
        return self.scale * (source - target).abs().sum(-1)


def test_audit_uses_real_next_state_in_each_episode_and_restores_eval_mode():
    raw = dataset(episode([0, 1, 3]), episode([100, 104]),
                  episode([3, 999], anchor=True)).raw_data
    critic = ScaledL1()
    a, _, _ = audit.evaluate_edges(critic, raw, chunk_size=1, device='cpu')
    b, _, _ = audit.evaluate_edges(critic, raw, chunk_size=100, device='cpu')
    np.testing.assert_array_equal(a, [2, 4, 8, 1992])
    np.testing.assert_array_equal(a, b)
    assert critic.training
    assert critic.scale.grad is None


def test_summary_separates_strict_tolerance_and_weighted_violations():
    stats = audit.summarize_edges(
        np.array([1, 2, 3, 4], np.float32), np.array([1, 1.99999, 2, 5], np.float32),
        tolerance=1e-4, weights=[1, 1, 4, 1])
    assert stats['edges'] == 4
    assert stats['strict_violation_count'] == 2
    assert stats['violation_count'] == 1
    assert stats['violation_fraction'] == .25
    assert stats['sampler_weighted_violation_fraction'] == 4 / 7
    assert stats['max_violation'] == 1
    assert stats['squared_positive_residual_mean'] == pytest.approx(.25)


def test_empty_and_nonfinite_groups_are_never_reported_as_feasible():
    empty = audit.summarize_edges([], [], tolerance=0)
    invalid = audit.summarize_edges([float('nan'), 2], [1, 1], tolerance=0)
    assert empty['edges'] == 0 and empty['violation_fraction'] is None
    assert not invalid['audit_valid']
    assert invalid['nonfinite_edges'] == 1
    assert invalid['violation_fraction'] is None
    json.dumps(invalid, allow_nan=False)


def test_dual_diagnostics_condition_on_violated_edges_and_use_configured_floor():
    stats = audit.summarize_edges([5, 3, 0], [1, 1, 1], tolerance=0,
                                  dual=[.0001, 10, 100], raw_dual=[-8, 10, 100],
                                  dual_raw_floor=-7)
    assert stats['violation_dual_below_1e_3_fraction'] == .5
    assert stats['violation_raw_dual_below_floor_fraction'] == .5
    assert stats['violation_dual_mean'] == pytest.approx(5.00005)


def test_fingerprint_rejects_a_different_constraint_replay():
    raw = dataset(episode([0, 1])).raw_data
    digest = replay_fingerprint(raw)
    assert audit.validate_replay(raw, digest) == digest
    raw.rewards[0] = -2
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        audit.validate_replay(raw, digest)


def test_training_log_is_parsed_as_literals_without_execution(tmp_path):
    path = tmp_path / 'train.log'
    path.write_text("INFO - 参数: Namespace(seed=41)\nINFO - 参数: Namespace(seed=42, x=[1, 2], _scenario_data={'bounds': [0, 1]})")
    assert audit.load_training_args(path).seed == 42
    path.write_text("INFO - 参数: Namespace(seed=__import__('os').system('false'))")
    with pytest.raises(ValueError):
        audit.load_training_args(path)


def test_replay_cache_is_verified_against_training_digest_on_every_read(tmp_path):
    raw = dataset(episode([0, 1])).raw_data
    cache = tmp_path / 'replay.pt'
    torch.save({'raw': {name: getattr(raw, name) for name in (
        'episode_lengths', 'all_observations', 'actions', 'rewards', 'terminals',
        'timeouts', 'transition_infos', 'observation_infos')}}, cache)
    assert audit.load_replay({'expected_sha256': replay_fingerprint(raw)}, cache).num_transitions == 1
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        audit.load_replay({'expected_sha256': 'wrong'}, cache)


def test_checkpoint_restores_nested_loss_module_and_each_critic_dual(tmp_path, monkeypatch):
    env = TinyEnv()
    agent, _ = QRLConf(actor=None, num_critics=2).make(env_spec=EnvSpec.from_env(env), total_optim_steps=1)
    duals = [LocalConstraintLoss.Conf(mode='kkt_functional', dual_hidden_sizes=(8, 8),
                                     dual_feature_scale=5, dual_max=100000,
                                     dual_raw_min=-10).make(c) for c in agent.critics]
    with torch.no_grad():
        duals[0].dual_raw_offset.fill_(2)
        duals[1].dual_raw_offset.fill_(4)
    state = {f'critic_losses.{i}.local_constraint.{k}': v
             for i, local in enumerate(duals) for k, v in local.state_dict().items()}
    path = tmp_path / 'checkpoint.pth'
    torch.save({'agent': agent.state_dict(), 'losses': {'module': state, 'optim_scheds': {}},
                'optim_steps': 3000}, path)
    args = argparse.Namespace(seed=42, num_critics=2, qrl_local_constraint_mode='kkt_functional',
                              qrl_kkt_dual_hidden_sizes=[8, 8], qrl_kkt_dual_feature_scale=5,
                              qrl_kkt_dual_raw_min=-10, qrl_kkt_dual_max=100000)
    monkeypatch.setattr(audit, 'get_env_kwargs', lambda args: {})
    monkeypatch.setattr(audit, 'CommInspectionDubinsUAV2D', lambda **kw: env)
    restored, loaded_duals, step, _, _ = audit.load_models({'checkpoint': str(path), 'args': args}, 'cpu')
    assert step == 3000
    for i in range(2):
        for key, value in agent.critics[i].state_dict().items():
            assert torch.equal(restored.critics[i].state_dict()[key], value)
        assert loaded_duals[i].dual_raw_offset == 2 + 2 * i


def test_groups_keep_anchors_successes_failures_and_regions_separate():
    raw = dataset(episode([0, 1], success=True), episode([1, 999], anchor=True),
                  episode([10, 11])).raw_data
    env = TinyEnv()
    env.device_ids = ('target',)
    masks = audit.group_masks(raw, env, {'metadata': {'exploration_diagnostic_regions': {'U': [9, -1, 12, 1]}}})
    np.testing.assert_array_equal(masks['real'], [True, False, True])
    np.testing.assert_array_equal(masks['anchor'], [False, True, False])
    np.testing.assert_array_equal(masks['real_success'], [True, False, False])
    np.testing.assert_array_equal(masks['real_terminal_success'], [True, False, False])
    np.testing.assert_array_equal(masks['region/U/real'], [False, False, True])


def test_top_violations_include_success_terminal_and_keep_anchor_family():
    raw = dataset(episode([0, 1], success=True), episode([1, 999], anchor=True),
                  episode([10, 11])).raw_data
    env = TinyEnv()
    env.device_ids = ('target',)
    masks = audit.group_masks(raw, env, {})
    rows = audit.top_edge_rows(raw, np.array([100, 2, 20]), np.array([1, 0, 1]),
                               'test', 0, masks, 1)
    assert len(rows) == 2
    assert rows[0]['transition_index'] == 0 and rows[0]['terminal']
    assert rows[0]['success_episode']
    assert rows[1]['family'] == 'anchor' and rows[1]['cost'] == 0


def test_checkpoint_training_metadata_mismatch_fails_before_evaluation(tmp_path):
    path = tmp_path / 'wrong_seed.pth'
    torch.save({'experiment': {'global_push_sampling': {'seed': 7}}}, path)
    args = argparse.Namespace(seed=42, qrl_local_constraint_mode='kkt_functional')
    with pytest.raises(ValueError, match='sampling seed'):
        audit.load_models({'checkpoint': str(path), 'args': args}, 'cpu')


def test_audit_phase_forwards_checkpoints_and_does_not_regenerate_training_config(tmp_path):
    fake = tmp_path / 'python'
    capture = tmp_path / 'args.json'
    fake.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                    'open(os.environ["AUDIT_CAPTURE"], "a").write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = {**os.environ, 'PYTHON_BIN': str(fake), 'PHASE': 'replay_constraint_audit',
           'QRL_CHECKPOINTS': 'two.pth\nthree.pth five.pth', 'OUTPUT_ROOT': str(tmp_path),
           'AUDIT_CAPTURE': str(capture), 'REPLAY_AUDIT_CHUNK_SIZE': '32'}
    subprocess.run(['bash', str(Path(__file__).with_name('run_comm_inspection_diagnostic.sh'))],
                   env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][:2] == ['-m', 'minimal_qrl.eval.replay_constraint_audit']
    assert calls[0][3:6] == ['two.pth', 'three.pth', 'five.pth']
    assert calls[0][calls[0].index('--chunk-size') + 1] == '32'
