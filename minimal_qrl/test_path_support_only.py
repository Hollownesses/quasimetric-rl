"""A complete witnessed subgraph, unchanged learner, read-only diagnostics."""
import csv
import json

import numpy as np
import pytest
import torch

from minimal_qrl import dataset as data_module
from minimal_qrl import path_support as ps
from minimal_qrl.eval import replay_constraint_audit as audit
from minimal_qrl.eval.path_support_diagnostic import PathSupportDiagnostic
from minimal_qrl.global_push_sampling import configure_global_push_sampling, replay_fingerprint
from minimal_qrl.test_barrier_push_support import capture_phase, arg
from minimal_qrl.test_path_support import bank_path, protocol_args
from minimal_qrl.test_comm_inspection_oracle_bank import make_env
from quasimetric_rl.data import MultiEpisodeData


def only_episodes(bank_path):
    return list(data_module.create_dataset(make_env(), path_support_bank=str(bank_path),
                path_support_only=True, include_global_push_pairs=False))


def test_only_mode_never_calls_exploration_or_adds_other_physical_data(bank_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('bank-only diagnostic called exploration collector')
    monkeypatch.setattr(data_module, 'create_qrl_explore_comm_dataset', forbidden)
    stats = {}
    actual = list(data_module.create_dataset(make_env(), path_support_bank=str(bank_path),
                  path_support_only=True, include_global_push_pairs=False, collection_stats=stats))
    expected = list(ps.bank_episodes(make_env(), bank_path, context_start=0, include_global_push_pairs=False))
    raw = MultiEpisodeData.cat(actual)
    assert replay_fingerprint(raw) == replay_fingerprint(MultiEpisodeData.cat(expected))
    assert len(actual) == 2
    assert raw.transition_infos['planner_path_support'].all()
    assert not raw.transition_infos['exploration'].any()
    assert not raw.transition_infos['goal_return_mask'].any()
    assert not raw.transition_infos['goal_return_cost'].any()
    assert stats['attempted_env_steps'] == 0
    assert stats['stored_real_transitions'] == actual[0].num_transitions
    assert stats['path_support']['base_episode_count'] == 0
    assert stats['path_support']['data_scope'] == 'bank_only'
    with pytest.raises(ValueError, match='bank and no exploration'):
        list(data_module.create_dataset(make_env(), path_support_bank=str(bank_path),
             path_support_only=True, qrl_explore_config=data_module.QRLExploreConfig()))


def test_all_task_sampling_has_full_witness_support_in_path_only_replay(bank_path):
    # Use the actual environment for terminal feature validation in the sampler.
    import uuid
    from quasimetric_rl.data import Dataset, register_offline_env
    key = uuid.uuid4().hex
    register_offline_env(key, key, create_env_fn=make_env, load_episodes_fn=lambda: iter(only_episodes(bank_path)))
    data = Dataset(key, key, future_observation_discount=.99)
    sampling = configure_global_push_sampling(data, state_mode='trajectory', task_mode='all')
    ps.validate_path_only_replay(data.raw_data, batch_size=1)
    assert sampling['whole_push_has_recorded_witnesses']
    real = ~data.raw_data.transition_infos['abstract_goal_edge']
    assert data.raw_data.transition_infos['global_push_task_mask'][real].all()
    assert (data.raw_data.transition_infos['global_push_pair_steps'][real] > 0).all()
    with pytest.raises(ValueError, match='batch size'):
        ps.validate_path_only_replay(data.raw_data, data.raw_data.num_transitions + 1)
    data.raw_data.transition_infos['goal_return_mask'][0] = True
    with pytest.raises(ValueError, match='without return targets'):
        ps.validate_path_only_replay(data.raw_data, batch_size=1)


@pytest.mark.parametrize('change,value', [('path_support_bank', None), ('init_checkpoint', 'old.pth'),
    ('global_push_state_sampling', 'environment'), ('path_support_probe_interval', 0)])
def test_only_protocol_rejects_missing_bank_resume_and_unsupported_push(change, value):
    args = protocol_args()
    args.comm_dataset_mode = 'path_support_only'
    ps.validate_training_protocol(args)
    setattr(args, change, value)
    with pytest.raises(ValueError):
        ps.validate_training_protocol(args)


def test_only_phase_preserves_learner_and_forces_B1_without_exploration_budget(tmp_path, bank_path):
    result, only = capture_phase(tmp_path, 'only', PHASE='train_qrl_path_support_only_b1',
        PATH_SUPPORT_BANK=str(bank_path), PATH_SUPPORT_PROBE_INTERVAL='17')
    assert result.returncode == 0, result.stderr
    assert arg(only, 'comm-dataset-mode') == 'path_support_only'
    assert arg(only, 'path-support-probe-interval') == '17'
    assert '--explore-attempted-env-steps' not in only
    assert '--target-env-transitions' not in only
    assert arg(only, 'total-steps') == '2000'
    result, full = capture_phase(tmp_path, 'full', PHASE='train_qrl_path_support_positive_b1', PATH_SUPPORT_BANK=str(bank_path))
    assert result.returncode == 0
    def learner_args(args):
        starts = [i for i, token in enumerate(args) if token.startswith('--')] + [len(args)]
        pairs = {args[i]: args[i+1:j] for i, j in zip(starts[:-1], starts[1:])}
        return {k: v for k, v in pairs.items() if k not in ('--output-dir', '--oracle-bank-dir',
                '--comm-dataset-mode', '--path-support-probe-interval') and not k.startswith('--explore-')}
    assert learner_args(only) == learner_args(full)
    result, _ = capture_phase(tmp_path, 'resume', PHASE='train_qrl_path_support_only_b1',
        PATH_SUPPORT_BANK=str(bank_path), QRL_INIT_CHECKPOINT='old.pth')
    assert result.returncode != 0 and 'fresh initialization' in result.stderr


def test_only_audit_reconstructs_from_relocatable_bank_without_exploration_starts(bank_path, tmp_path, monkeypatch):
    raw = MultiEpisodeData.cat(only_episodes(bank_path))
    bank = ps.load_bank(bank_path)
    ps.save_bank(tmp_path/'path_support_bank.json', bank)
    args = protocol_args()
    args.comm_dataset_mode = 'path_support_only'
    args.path_support_bank = '/original/location/removed.json'
    args.path_support_bank_sha256 = bank['content_digest']
    args.qrl_cost_source = 'negative_reward'
    args._scenario_data = {'diagnostic': True}
    args.seed, args.max_steps_per_episode = 42, 300
    (tmp_path/'train.log').write_text('INFO - 参数: '+repr(args))
    (tmp_path/'global_push_sampling_stats.json').write_text(json.dumps({'physical_replay_sha256': replay_fingerprint(raw)}))
    spec = audit.load_run_spec(tmp_path/'checkpoint.pth')
    assert spec['config'] is None
    assert not (tmp_path/'exploration_start_bank.json').exists()
    monkeypatch.setattr(audit, 'get_env_kwargs', lambda args: {})
    monkeypatch.setattr(audit, 'CommInspectionDubinsUAV2D', lambda **kw: make_env())
    rebuilt = audit.load_replay(spec, tmp_path/'cache.pt')
    assert replay_fingerprint(rebuilt) == replay_fingerprint(raw)
    assert rebuilt.transition_infos['planner_path_support'].all()
    bank['records'][0]['diagnostic_path_cost_only'] += 1
    ps.save_bank(tmp_path/'path_support_bank.json', bank)
    with pytest.raises(ValueError, match='snapshot disagrees'):
        audit.load_run_spec(tmp_path/'checkpoint.pth')


def test_path_probes_do_not_change_rng_parameters_modes_or_replay(bank_path, tmp_path):
    class Distance(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.))
        def forward(self, source, target):
            return self.scale * (source[:, 0] - target[:, 0]).abs()
    agent = torch.nn.Module()
    agent.critics = torch.nn.ModuleList([Distance()])
    agent.train()
    agent.critics[0].eval()
    raw = MultiEpisodeData.cat(only_episodes(bank_path))
    before = replay_fingerprint(raw)
    probe = PathSupportDiagnostic(raw, tmp_path)
    rng = torch.get_rng_state().clone()
    probe.snapshot(agent, 0)
    probe.snapshot(agent, 0)  # Initial/final coincidence must not duplicate rows.
    assert torch.equal(rng, torch.get_rng_state())
    assert agent.training and not agent.critics[0].training
    assert agent.critics[0].scale.item() == 2 and agent.critics[0].scale.grad is None
    assert replay_fingerprint(raw) == before
    assert not raw.transition_infos['goal_return_cost'].any()
    rows = list(csv.DictReader((tmp_path/'values.csv').open()))
    n = int(raw.episode_lengths[0])
    assert len(rows) == n+1
    assert float(rows[0]['recorded_suffix_cost_upper_bound']) == pytest.approx(float(-raw.rewards[:n].sum()))
    assert float(rows[-1]['recorded_suffix_cost_upper_bound']) == 0
    assert rows[-1]['local_residual'] == ''
    assert float(rows[0]['value_delta']) == pytest.approx(float(rows[0]['next_value'])-float(rows[0]['value']))
    with torch.no_grad():
        agent.critics[0].scale.fill_(1.)
    probe.snapshot(agent, 3)
    probe.write_report()
    assert (tmp_path/'report.md').exists() and (tmp_path/'values.png').exists()
    summaries = list(csv.DictReader((tmp_path/'summary.csv').open()))
    assert [int(row['step']) for row in summaries] == [0, 3]
    assert float(summaries[1]['initial_value']) == float(summaries[0]['initial_value'])/2
