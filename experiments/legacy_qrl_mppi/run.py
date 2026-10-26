"""Add one historical model arm without modifying the frozen R-series experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from experiments.r_series_mppi.common import (
    CONFIG, SEEDS, LoggedMPPI, COST_KEYS, load, file_digest, tensor_digest, read, write,
)
from experiments.r_series_mppi.run import verify as verify_baseline, store_rollout
from experiments.r_series_mppi.analyze import pair_details, detailed_group
from minimal_qrl.baselines.base import rollout_controller_episode
from minimal_qrl.baselines.hybrid_astar_value import _signature_payload, _digest
from minimal_qrl.baselines.hybrid_astar import HybridAStarConfig
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.gc_agents import QRLGoalValueAdapter
from quasimetric_rl.data import EnvSpec
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic


BASE = Path('results/r_series_mppi_20261026/main')
LEGACY_ROOT = Path('../quasimetric-rl-industrial-inspection/results/diagnostic_u_shadow_corridors_aligned_v2')
CHECKPOINT = LEGACY_ROOT / 'qrl_training/checkpoint_final.pth'
ARM = 'legacy_gp300'


def legacy_adapter(env):
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert saved['optim_steps'] == 100000
    agent = nn.Module()
    spec = EnvSpec.from_env(env)
    agent.critics = nn.ModuleList([QuasimetricCritic.Conf().make(env_spec=spec) for _ in range(2)])
    # Encoder's backward-compatible loader inserts only the absent False buffer.
    agent.load_state_dict(dict(saved['agent']), strict=True)
    agent.eval().requires_grad_(False)
    actual = agent.state_dict()
    for key, value in saved['agent'].items():
        assert torch.equal(actual[key], value), key
    added = set(actual) - set(saved['agent'])
    assert added == {f'critics.{i}.encoder.comm_goal_set_quotient' for i in range(2)}
    assert all(not bool(actual[key]) for key in added)
    return QRLGoalValueAdapter(agent, env, torch.device('cpu'), distance_scale=1.0), dict(
        checkpoint_steps=saved['optim_steps'], critics_loaded=2, critic_used=0,
        original_state_tensors_exact=True, added_false_compatibility_buffers=sorted(added),
        terminal_quotient_enabled=False, distance_scale=1.0,
    )


@torch.no_grad()
def validate_adapter(value, data):
    env = data.env
    env.reset(seed=0, options={'device_id':'u_trap_target', 'start':[4.,3.65,0.]})
    observations = data.raw.all_observations[:1024].numpy()
    goal = np.repeat(env.abstract_goal_observation()[None], len(observations), axis=0)
    critic = value.qrl_agent.critics[0]
    o, g = torch.from_numpy(observations), torch.from_numpy(goal)
    direct = critic.quasimetric_model(critic.encoder(o), critic.encoder(g)).reshape(-1).numpy()
    actual = value.batch_value(observations, goal)
    np.testing.assert_array_equal(actual, direct)
    chunks = np.concatenate([value.batch_value(observations[i:i+37], goal[i:i+37])
                             for i in range(0,len(observations),37)])
    np.testing.assert_allclose(actual, chunks, rtol=3e-6, atol=8e-5)
    terminal = np.stack([env.state_to_observation(env.sample_task_terminal_state(seed=8100+i)) for i in range(16)])
    terminal_goal = np.repeat(goal[:1], len(terminal), axis=0)
    terminal_values = value.batch_value(terminal, terminal_goal)
    # These may be positive: preserving the historical network forbids a new quotient.
    assert np.isfinite(terminal_values).all() and (terminal_values >= 0).all()
    goal_value = value.batch_value(goal[:1], goal[:1])
    assert float(goal_value[0]) == 0
    assert np.isfinite(actual).all() and (actual >= 0).all()
    return dict(direct_first_critic_max_error=float(abs(actual-direct).max()),
                chunk_max_error=float(abs(actual-chunks).max()),
                physical_terminal_values=terminal_values.tolist(), abstract_goal_self_distance=0.,
                note='Physical terminal predictions preserved; no forced zero or new canonicalization.')


def audit_episode(data, record, seed, result, controller, baseline):
    key = (record['id'],seed)
    env = data.env
    expected_bank = baseline[key]['first_candidate_sha256']
    assert controller.banks[0] == expected_bank
    # Replay every action through actual dynamics before storing the episode.
    env.reset(seed=seed,options={'device_id':'u_trap_target','start':np.asarray(record['pose'],np.float32)})
    np.testing.assert_array_equal(env.state,result['states'][0])
    for i,action in enumerate(result['actions']):
        _,reward,done,truncated,info = env.step(action)
        np.testing.assert_array_equal(env.state,result['states'][i+1])
        assert reward == result['rewards'][i]
        for field in COST_KEYS:
            assert info[field] == result['step_infos'][i][field]
        assert not(done or truncated) or i == len(result['actions'])-1
    assert done or truncated
    assert bool(info['success']) == result['success']
    assert bool(info['collision']) == result['collision']
    assert bool(info['out_of_bounds']) == result['out_of_bounds']
    assert bool(truncated and not info['success']) == bool(result['truncated'] and not result['success'])


def run(root):
    torch.set_num_threads(1)
    original = verify_baseline(BASE)
    assert asdict(CONFIG) == original['config']
    root.mkdir(parents=True,exist_ok=False)
    data = load()
    records = read(BASE/'states.json')['records']
    assert len(records) == 32 and original['planner_seeds'] == list(SEEDS)
    old_scenario = read(LEGACY_ROOT/'config/diagnostic_scenario.json')
    old_env = CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(old_scenario))
    old_env.set_task_by_device_id('u_trap_target')
    data.env.set_task_by_device_id('u_trap_target')
    old_signature = _signature_payload(old_env,HybridAStarConfig())
    new_signature = _signature_payload(data.env,HybridAStarConfig())
    assert old_signature == new_signature
    value, validation = legacy_adapter(data.env)
    model_digest = tensor_digest(*value.state_dict().values())
    validation.update(validate_adapter(value,data))
    # Separate development poses. Both complete rollouts must reproduce exactly.
    smoke = []
    for pose in ([4.,3.65,.31],[7.8,3.8,2.95]):
        first = None
        for repeat in range(2):
            control = LoggedMPPI(CONFIG,terminal_mode='qrl',qrl_agent=value)
            result = rollout_controller_episode(control,data.env,episode_seed=SEEDS[0],
                device_id='u_trap_target',start_state=np.asarray(pose,np.float32))
            if first is not None:
                for field in ('states','actions','rewards'):
                    np.testing.assert_array_equal(first[field],result[field])
            else:
                first = result
        smoke.append(dict(pose=pose,repeat_exact=True,success=result['success'],steps=len(result['actions'])))
    assert tensor_digest(*value.state_dict().values()) == model_digest
    write(root/'adapter_validation.json',validation)
    write(root/'smoke.json',dict(passed=True,episodes=4,formal_starts_used=False,cases=smoke))
    inputs = {str(CHECKPOINT.resolve()):file_digest(CHECKPOINT),
              str(LEGACY_ROOT/'config/diagnostic_scenario.json'):file_digest(LEGACY_ROOT/'config/diagnostic_scenario.json'),
              str(LEGACY_ROOT/'qrl_training/train.log'):file_digest(LEGACY_ROOT/'qrl_training/train.log')}
    for f in ('manifest.json','FROZEN','states.json','adapter_validation.json','comparison/report.md','comparison/summary.json'):
        inputs[str(BASE/f)] = file_digest(BASE/f)
    baseline_rows = {}
    for arm in ('none','R0','R1','oracle'):
        directory = BASE/'arms'/arm
        for f in ('COMPLETE','episodes.jsonl'):
            inputs[str(directory/f)] = file_digest(directory/f)
        baseline_rows[arm] = [json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
    baseline = {(r['state_id'],r['planner_seed']):r for r in baseline_rows['R0']}
    shutil.copy2(BASE/'states.json',root/'states.json')
    source = Path(__file__)
    sources = dict(original['sources'])
    sources[str(source)] = file_digest(source)
    external = dict(original['external_sources'])
    write(root/'manifest.json',dict(
        arm=ARM,checkpoint=str(CHECKPOINT.resolve()),checkpoint_sha256=file_digest(CHECKPOINT),
        baseline_root=str(BASE),baseline_manifest_sha256=file_digest(BASE/'manifest.json'),
        model_validation=validation,model_digest=model_digest,config=asdict(CONFIG),
        planner_seeds=SEEDS,episodes=64,immutable_inputs=inputs,sources=sources,external_sources=external,
        environment_signature_digest=_digest(new_signature),
        comparison_scope='Historical trained system; training data, mechanisms, budget and training seed differ.',
        evaluation_only=True,calibration=False,formal_starts_sha256=file_digest(root/'states.json')))
    shutil.copy2(source,root/'source_run.py')
    write(root/'FROZEN',dict(manifest_sha256=file_digest(root/'manifest.json')))
    out = root/'arms'/ARM
    out.mkdir(parents=True,exist_ok=False)
    rows = []
    started = time.perf_counter()
    with (out/'episodes.jsonl').open('x') as stream:
        for record in records:
            for seed in SEEDS:
                control = LoggedMPPI(CONFIG,terminal_mode='qrl',qrl_agent=value)
                result = rollout_controller_episode(control,data.env,episode_seed=seed,
                    device_id='u_trap_target',start_state=np.asarray(record['pose'],np.float32))
                audit_episode(data,record,seed,result,control,baseline)
                assert tensor_digest(*value.state_dict().values()) == model_digest
                row = store_rollout(out,ARM,record,seed,result,control)
                rows.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps(dict(completed=len(rows),episodes=64,state=record['id'],seed=seed,
                    success=row['success'],steps=row['steps'],elapsed_seconds=time.perf_counter()-started)),flush=True)
    assert len(rows) == 64
    write(out/'COMPLETE',dict(episodes=64,episodes_sha256=file_digest(out/'episodes.jsonl'),
        elapsed_seconds=time.perf_counter()-started,all_actions_replayed_exact=True,parameters_unchanged=True))
    baseline_rows[ARM] = rows
    analyze(root,baseline_rows)
    verify_baseline(BASE)
    for group in (inputs,sources,external):
        for f,h in group.items():
            assert file_digest(f) == h, ('input/source changed',f)
    assert tensor_digest(*value.state_dict().values()) == model_digest
    artifacts = {str(f):file_digest(f) for f in root.rglob('*') if f.is_file() and f.name not in ('COMPLETE','final_integrity.json')}
    write(root/'final_integrity.json',dict(passed=True,legacy_checkpoint_unchanged=True,
        original_experiment_unchanged=True,model_parameters_unchanged=True,formal_episodes=64,
        source_and_input_hashes_verified=True,first_candidate_banks_exact=True,
        replayed_steps=sum(r['steps'] for r in rows),actions_states_rewards_costs_termination_exact=True,
        artifacts=artifacts))
    write(root/'COMPLETE',dict(passed=True,episodes=64,final_integrity_sha256=file_digest(root/'final_integrity.json')))


def analyze(root,rows):
    out=root/'comparison';out.mkdir(exist_ok=False)
    groups={'all':lambda r:True,'inside':lambda r:r['stratum']=='inside','west':lambda r:r['stratum']=='west'}
    for h in range(8):
        groups[f'heading_{h}']=lambda r,h=h:r['heading_bin']==h
    grouped={g:{a:detailed_group([r for r in values if predicate(r)]) for a,values in rows.items()}
             for g,predicate in groups.items()}
    comparisons={g:{ARM+'_to_'+a:pair_details([r for r in rows[ARM] if predicate(r)],
                   [r for r in rows[a] if predicate(r)]) for a in ('none','R0','R1','oracle')}
                 for g,predicate in groups.items()}
    summary=dict(grouped=grouped,paired=comparisons,baseline_root=str(BASE),added_arm=ARM,
        episode_count=64,training_performed=False,validation=read(root/'adapter_validation.json'),
        scope='Single historical checkpoint; 32 shared initial poses x two planner seeds; descriptive paired comparison.')
    write(out/'summary.json',summary)
    lines=['# 历史GP300模型加入当前MPPI评价','',
        '复用 aligned_v2 的100k旧checkpoint，保留原网络与第一个critic，未训练、未校准、未启用后来新增的终态quotient。',
        '在当前冻结32起点×2 planner seeds上补跑64回合；其他四组复用已完成的同协议结果。',
        '这比较历史训练系统与当前系统的表现，训练数据、预算、机制和初始化不同，不能把全部差异归因于PHR或源均衡。','',
        '|区域|组别|成功|碰撞/越界/超时|成功成本均值|成功步数均值|',
        '|---|---|---:|---|---:|---:|']
    def fmt(x): return '—' if x is None else f'{x:.3f}'
    for g in ('all','inside','west'):
        for a in ('none',ARM,'R0','R1','oracle'):
            x=grouped[g][a]
            lines.append(f"|{g}|{a}|{x['successes']}/{x['episodes']}|{x['collision_count']}/{x['out_of_bounds_count']}/{x['timeout_count']}|{fmt(x['successful_cost_mean'])}|{fmt(x['successful_steps_mean'])}|")
    lines+=['','配对方向为旧模型→对比组；成本差=对比组−旧模型，只在同一实例双方成功时计算。','',
        '|区域|对比组|新增成功|失去成功|共同成功|共同成功成本差|共同成功步数差|',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in ('all','inside','west'):
        for a in ('none','R0','R1','oracle'):
            p=comparisons[g][ARM+'_to_'+a]
            lines.append(f"|{g}|{a}|{p['improved']}|{p['worsened']}|{p['both_success']}|{fmt(p['joint_success_cost_delta'])}|{fmt(p['joint_success_steps_delta'])}|")
    lines+=['','MPPI统一H10/N128、noise=.8、temperature=1、terminal_weight=1，真实环境上限300步。',
        '正式起点和首次候选bank与原评价一致；闭环分化后仅保证同样的随机扰动生成规则。',
        '保留全部失败和真实早停成本；成功集合不同，不能依据成功成本均值直接排名。',
        'Oracle为离散格点终值参照，其执行成本不代表连续最优成本或严格上界。',
        '新模型共用一个训练seed606，旧模型训练seed42，两个planner seeds共用32起点，不能视为64个独立地理样本。',
        '本轮未强制旧模型对物理合法终态预测为零；这属于保留历史网络语义。MPPI对已经成功的候选不再查询终值。',
        '验证见 ../adapter_validation.json、../smoke.json 和 ../final_integrity.json；完整分组/配对见 summary.json。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({g:{a:f"{v['successes']}/{v['episodes']}" for a,v in grouped[g].items()} for g in ('all','inside','west')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
