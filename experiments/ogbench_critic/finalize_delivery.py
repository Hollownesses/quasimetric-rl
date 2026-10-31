"""File-only final integrity audit and figure; never import JAX or run an episode."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / 'results/ogbench_critic_20261029'
B = ROOT / 'results/ogbench_official_qrl_20261026_seed0'
E = ROOT / 'experiments/ogbench_critic'
UPSTREAM = ROOT / 'experiments/ogbench_official_qrl/vendor/ogbench'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run():
    # Use the checked-in pin rather than a second independently maintained string.
    pin = (ROOT/'experiments/ogbench_official_qrl/OFFICIAL_COMMIT').read_text().strip()
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == pin
    assert not subprocess.check_output(['git', '-C', str(UPSTREAM), 'diff', '--stat'], text=True).strip()
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'exp/ogbench'
    baseline = read(B/'summary.json')
    frozen_path = B/'main/frozen_config.json'
    frozen = read(frozen_path)
    assert sha(frozen_path) == (B/'main/frozen_config.sha256').read_text().strip()
    assert frozen['official_commit'] == pin == baseline['official_commit']
    for name, expected in frozen['source_sha256'].items():
        assert sha(UPSTREAM/name) == expected, name
    checkpoint_dir = B/'main/OGBench/navigate_100k/sd000_20261026_115628'
    assert baseline['status'] == 'complete' and (B/'COMPLETE').exists()
    assert len(baseline['checkpoint_sha256']) == 10
    for name, expected in baseline['checkpoint_sha256'].items():
        assert sha(checkpoint_dir/name) == expected, name
    assert baseline['smoke']['passed'] and baseline['final_checkpoint_verification']['passed']
    assert baseline['final_checkpoint_verification']['optimizer_step'] == 100000
    flags = read(checkpoint_dir/'flags.json')
    assert flags['seed'] == 0 and flags['train_steps'] == 100000 and flags['restore_path'] is None
    for name in ['train', 'validation']:
        item = read(B/'data_audit.json')[name]
        assert sha(item['path']) == item['sha256']
    for filename, count in [('train.csv', 100), ('eval.csv', 11)]:
        with (checkpoint_dir/filename).open() as handle:
            assert len(list(csv.DictReader(handle))) == count
    hard = read(G/'hard_goal_1000/analysis.json')
    assert hard['complete'] and hard['training']['passed'] and hard['training']['steps'] == 1000
    assert sum(v['success'] for v in hard['neural']['cases'].values()) == 25
    assert sha(G/'hard_goal_1000/model/params_101000.pkl') == hard['training']['checkpoint_sha256']
    extraction = read(G/'extraction_1000/analysis.json')
    assert extraction['complete'] and extraction['training']['passed'] and extraction['paired_250_initials_and_goals']
    controls = [read(G/f'extraction_1000/{group}_evaluation/analysis.json') for group in ['original', 'improved']]
    counts = [[a['official'][str(t)]['successes'] for t in range(1, 6)] for a in controls]
    assert counts == [[50, 50, 0, 50, 50], [50, 50, 31, 50, 50]]
    gains = losses = 0
    for name, old in controls[0]['official_episodes'].items():
        new = controls[1]['official_episodes'][name]
        assert old['initial_xy'] == new['initial_xy'] and old['goal'] == new['goal']
        gains += new['success'] and not old['success']
        losses += old['success'] and not new['success']
    assert len(controls[0]['official_episodes']) == 250 and gains == 31 and losses == 0
    for name, item in extraction['training']['checkpoints'].items():
        assert sha(G/'extraction_1000'/name/f"params_{item['epoch']}.pkl") == item['sha256']
        assert item['frozen'] and item['full_state_roundtrip']
    counter = read(G/'official_counterfactual_completion/analysis.json')
    assert counter['complete'] and counter['no_teacher_modules'] and counter['amended_runtime_budget']
    assert (G/'official_counterfactual/failure.json').exists()
    assert counter['original_protocol']['matched_reset_inputs'] == 250
    assert counter['original_protocol']['first_steps_bitwise_verified'] == 100
    critic_counts = [sum(v['success'] for v in counter['initials'][group].values()) for group in ['original', 'improved']]
    rescues = sum(v['success'] for v in counter['actor_timeout_counterfactual'].values())
    assert critic_counts == [0, 42] and rescues == 9 and len(counter['actor_timeout_counterfactual']) == 19
    trace_count = 0
    for paths in counter['trace_paths'].values():
        for path in paths.values():
            with np.load(path) as arrays:
                assert all(np.isfinite(arrays[key]).all() for key in arrays.files if arrays[key].dtype.kind in 'fci')
            trace_count += 1
    assert trace_count == 119
    smoke = read(G/'standalone_smoke/analysis.json')
    assert smoke['complete'] and not smoke['formal_performance_claim'] and smoke['actual'] == dict(value=2, dynamics=2, actor=2)
    assert len(smoke['evaluation']['cases']) == 5 and smoke['evaluation']['no_teacher_imports']
    for name, expected in smoke['protocol']['source_sha256'].items():
        assert sha(E/name) == expected, name
    for stage, item in smoke['stage_proofs'].items():
        assert item['frozen'] and item['full_state_roundtrip']
        assert sha(G/f"standalone_smoke/model/params_{item['epoch']}.pkl") == item['checkpoint_sha256']
    for group in ['official_full_seed0', 'official_100k_seed1']:
        stopped = read(G/group/'USER_STOPPED.json')
        assert stopped['exit_code'] == -15 and not stopped['restart_authorized']
    tracked = subprocess.check_output(['git', 'ls-files', 'results/ogbench_critic_20261029'], cwd=ROOT, text=True).splitlines()
    assert all(Path(p).suffix in ['.md', '.png'] for p in tracked)
    out = G/'research_summary'; out.mkdir(exist_ok=True)
    audit = dict(passed=True,official_commit=pin,baseline_checkpoints_verified=10,official_sources_verified=len(frozen['source_sha256']),
                 data_hashes_verified=2,official_paired_episodes=250,actor_counts=counts,paired_gains=gains,paired_losses=losses,
                 task3_critic_counts=critic_counts,actor_timeout_rescues=rescues,counterfactual_traces=trace_count,
                 smoke_stage_updates=smoke['actual'],smoke_seconds=smoke['elapsed_seconds'],
                 no_training_or_episodes_executed_by_audit=True,standalone_full_recipe_rerun=False,
                 seed1_and_full_stopped_records_verified=True,cross_seed_or_full_paper_claim=False)
    (out/'delivery_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    fig, axes = plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    x=np.arange(5);width=.34
    for i,(values,label,color) in enumerate(zip(counts,['Original critic + matched extraction','Improved critic + matched extraction'],['#888888','#297aa3'])):
        axes[0].bar(x+(i-.5)*width,values,width,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=[f'Task {i}' for i in range(1,6)],ylim=(0,58),ylabel='Successes / 50',title='Official actor: 80.0% → 92.4%')
    axes[0].legend(fontsize=7,loc='upper center')
    bars=axes[1].bar(['Original\ncritic','Improved\ncritic','Actor-timeout\nrescue'],[0,42,9],color=['#888888','#297aa3','#dc9b37'])
    for bar,text in zip(bars,['0 / 50','42 / 50','9 / 19']):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,text,ha='center')
    axes[1].set(ylim=(0,52),ylabel='Successful diagnostic episodes',title='True-successor greedy: diagnosis only')
    fig.savefig(out/'overview.png',dpi=170);plt.close(fig)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    run()
