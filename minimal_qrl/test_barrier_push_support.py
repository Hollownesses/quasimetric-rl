import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import minimal_qrl.train as train_module


def capture_phase(tmp_path, variant, **overrides):
    fake = tmp_path / 'python'
    capture = tmp_path / f'{variant}.json'
    fake.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                    'if sys.argv[1] == "minimal_qrl/train.py":\n'
                    '    open(os.environ["CAPTURE_ARGS"],"w").write(json.dumps(sys.argv[2:]))\n')
    fake.chmod(0o755)
    env = {**os.environ, 'PYTHON_BIN': str(fake), 'ORACLE_BANK_EVAL': '0',
           'PHASE': f'train_qrl_goal_set_barrier_push_support_{variant}',
           'OUTPUT_ROOT': str(tmp_path), 'CAPTURE_ARGS': str(capture),
           'GLOBAL_PUSH_STATE_SAMPLING': 'replay', 'GLOBAL_PUSH_TASK_SAMPLING': 'witnessed',
           'GLOBAL_PUSH_ABSTRACT_GOAL_RATIO': '0', 'GLOBAL_PUSH_STATE_GOAL_RATIO': '1',
           'QRL_LOCAL_CONSTRAINT_MODE': 'kkt_functional', 'QRL_DATASET_MODE': 'standard',
           **overrides}
    if 'TOTAL_STEPS' not in overrides:
        env.pop('TOTAL_STEPS', None)
    if 'QRL_INIT_CHECKPOINT' not in overrides:
        env.pop('QRL_INIT_CHECKPOINT', None)
    result = subprocess.run(['bash', str(Path(__file__).with_name('run_comm_inspection_diagnostic.sh'))],
                            env=env, capture_output=True, text=True)
    return result, json.loads(capture.read_text()) if capture.exists() else None


def arg(args, name):
    return args[args.index('--' + name) + 1]


@pytest.mark.parametrize('variant,sampling', [('control', 'environment'), ('b1', 'trajectory')])
def test_barrier_pair_defaults_and_isolation(tmp_path, variant, sampling):
    result, args = capture_phase(tmp_path, variant)
    assert result.returncode == 0, result.stderr
    assert arg(args, 'total-steps') == '2000'
    assert arg(args, 'comm-dataset-mode') == 'qrl_explore'
    assert arg(args, 'task-aware-teacher-ratio') == '0.0'
    assert arg(args, 'qrl-local-constraint-mode') == 'goal_set_barrier'
    assert arg(args, 'global-push-state-sampling') == sampling
    assert arg(args, 'global-push-task-sampling') == 'all'
    assert arg(args, 'global-push-objective') == 'linear'
    assert arg(args, 'global-push-abstract-goal-ratio') == '0.6'
    assert arg(args, 'global-push-state-goal-ratio') == '0.4'
    assert arg(args, 'qrl-barrier-weight') == '500.0'
    assert arg(args, 'qrl-barrier-lr') == '0.00002'
    for name in ('abstract-goal-edge-loss-weight', 'qrl-temporal-constraint-weight',
                 'qrl-goal-return-constraint-weight', 'qrl-nstep-goal-constraint-weight',
                 'qrl-mqe-waypoint-consistency-weight'):
        assert float(arg(args, name)) == 0
    assert '--init-checkpoint' not in args
    assert arg(args, 'output-dir').endswith(f'qrl_training_goal_set_barrier_push_support_{variant}')


def test_pair_differs_only_in_state_sampling_and_derived_output_paths(tmp_path):
    normalized = []
    for variant in ('control', 'b1'):
        result, args = capture_phase(tmp_path, variant, QRL_BARRIER_WEIGHT='700',
                                     QRL_BARRIER_AUDIT_INTERVAL='10', SEED='13')
        assert result.returncode == 0
        assert arg(args, 'qrl-barrier-weight') == '700'
        assert arg(args, 'qrl-barrier-audit-interval') == '10'
        for name in ('output-dir', 'oracle-bank-dir', 'global-push-state-sampling'):
            args[args.index('--' + name) + 1] = '<paired-difference>'
        normalized.append(args)
    assert normalized[0] == normalized[1]


def test_pair_rejects_ambient_checkpoint_resume(tmp_path):
    result, args = capture_phase(tmp_path, 'b1', QRL_INIT_CHECKPOINT='kkt_3k.pth')
    assert result.returncode != 0
    assert 'fresh initialization' in result.stderr
    assert args is None


@pytest.mark.parametrize('state_sampling', ['environment', 'trajectory'])
def test_pair_metadata_retains_barrier_optimizer_and_sampling_axes(monkeypatch, state_sampling):
    captured = {}
    monkeypatch.setattr(train_module, 'train', lambda args: captured.setdefault('args', args))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--qrl-local-constraint-mode', 'goal_set_barrier',
                                    '--global-push-state-sampling', state_sampling,
                                    '--global-push-task-sampling', 'all'])
    train_module.main()
    meta = train_module._build_topology_improvement_metadata(
        captured['args'], mqe_waypoint_weight=0, mqe_terminal_anchor_loss_weight=0,
        mqe_diagnostic_device_names=())
    assert meta['optimizer_variant'] == 'goal_set_quotient_replay_margin_barrier'
    assert meta['variant'].startswith('goal_set_quotient_replay_margin_barrier')
    assert meta['goal_set_barrier']['goal_set_quotient']
    assert meta['global_push_state_sampling'] == state_sampling
    assert meta['global_push_task_sampling'] == 'all'
    assert not meta['goal_set_barrier']['hard_feasibility_guarantee']
