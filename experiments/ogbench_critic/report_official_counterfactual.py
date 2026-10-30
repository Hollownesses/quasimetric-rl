"""Verify both read-only phases and report exact matched-input counterfactuals."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from report_iqe_distill import read,sha


def run(root):
    root=Path(root).resolve();repo=Path(__file__).resolve().parents[2];source=root/'official_counterfactual';out=root/'official_counterfactual_completion'
    a=read(out/'analysis.json');p=a['protocol'];old=a['original_protocol'];prior=read(root/'extraction_1000/analysis.json')
    assert a['complete'] and a['amended_runtime_budget'] and a['no_teacher_modules'] and a['elapsed_seconds']<300
    assert a['new_complete_episodes']==79 and a['reused_complete_episodes']==40
    assert sha(source/'protocol.json')==p['original_protocol_sha256'] and sha(source/'failure.json')==p['original_failure_sha256']
    failure=read(source/'failure.json');assert not failure['complete'] and failure['completed']==dict(original=40)
    assert failure['elapsed_seconds']>=240 and a['combined_seconds']==failure['elapsed_seconds']+a['elapsed_seconds']
    assert old['matched_reset_inputs']==250 and old['first_steps_bitwise_verified']==100
    assert sha(source/'official_task3_bank.npz')==old['bank_sha256'] and sha(root/'extraction_1000/analysis.json')==old['prior_analysis_sha256']
    assert sha(root/'hard_goal_1000/model/params_101000.pkl')==old['improved_checkpoint_sha256']
    for manifest,snapshot in [(old,source/'source_snapshot'),(p,out/'source_snapshot')]:
        for name,digest in manifest['source_sha256'].items():
            path=repo/'experiments/ogbench_diagnostics'/name if name in ['core.py','fast_pointmaze.py'] else Path(__file__).with_name(name)
            if name=='official_evaluation.py':path=repo/'experiments/ogbench_official_qrl/vendor/ogbench/impls/utils/evaluation.py'
            assert sha(snapshot/name)==digest==sha(path)
    for name,digest in p['completed_trace_sha256'].items():assert sha(source/'original'/f'{name}.npz')==digest
    for name,digest in old['trace_sha256'].items():
        arm,case=name.split('/');assert sha(root/'extraction_1000'/f'{arm}_evaluation/official'/f'{case}.npz')==digest
    original=a['initials']['original'];improved=a['initials']['improved'];rescue=a['actor_timeout_counterfactual']
    actor_a={n:v for n,v in prior['evaluations']['original']['official_episodes'].items() if v['task']==3}
    actor_b={n:v for n,v in prior['evaluations']['improved']['official_episodes'].items() if v['task']==3}
    assert list(original)==list(improved)==list(actor_a)==list(actor_b) and len(original)==50
    assert set(rescue)=={n for n,v in actor_b.items() if not v['success']} and len(rescue)==19
    for label,infos in [('original',original),('improved',improved),('improved_from_actor_timeout',rescue)]:
        assert set(a['trace_paths'][label])==set(infos)
        for name,info in infos.items():
            with np.load(a['trace_paths'][label][name]) as f,np.load(root/'extraction_1000/improved_evaluation/official'/f'{name}.npz') as actor:
                assert f['observations'].shape==(info['steps']+1,2) and f['actions'].shape==(info['steps'],2) and f['state_values'].shape==(info['steps']+1,)
                assert f['distances'].shape==(info['steps'],) and f['final_integration'].ndim==1 and all(np.isfinite(f[k]).all() for k in f.files)
                np.testing.assert_array_equal(f['goal'],actor['goal']);np.testing.assert_array_equal(f['observations'][-1],info['final_xy'])
                np.testing.assert_array_equal(f['observations'][0],actor['observations'][-1 if label=='improved_from_actor_timeout' else 0])
    count=lambda d:sum(v['success'] for v in d.values())
    gains=[n for n in original if improved[n]['success'] and not original[n]['success']];losses=[n for n in original if original[n]['success'] and not improved[n]['success']]
    names=list(original);fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    axes[0].bar(range(4),[count(actor_a),count(actor_b),count(original),count(improved)],color=['C0','C1','.6','C2'])
    axes[0].set(title='Same 50 official task3 inputs',xticks=range(4),xticklabels=['Actor A','Actor B','Greedy A','Greedy B'],ylabel='Successes / 50',ylim=(0,55))
    for success,color in [(True,'C2'),(False,'C3')]:
        selected=[n for n in names if actor_b[n]['success']==success]
        axes[1].scatter([improved[n]['steps'] for n in selected],[actor_b[n]['steps'] for n in selected],s=22,color=color,label='Actor B '+('success' if success else 'timeout'))
    axes[1].set(title='B: neural distance control vs direct actor',xlabel='Greedy episode steps',ylabel='Actor episode steps');axes[1].legend(fontsize=8)
    axes[2].bar(range(len(rescue)),[v['steps'] for v in rescue.values()],color=['C2' if v['success'] else 'C3' for v in rescue.values()])
    axes[2].set(title='All 19 actor-timeout states (separate test)',xlabel='Index in preserved failure set',ylabel='Extra diagnostic steps');fig.savefig(out/'official_counterfactual.png',dpi=160);plt.close(fig)
    with np.load(root/'hard_goal_1000/neural_evaluation/field.npz') as f:walls=f['walls'].copy()
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    for i,ax in enumerate(axes):
        name=f'task3_episode{i:03d}'
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,path,color in [('Actor A',root/'extraction_1000/original_evaluation/official'/f'{name}.npz','C0'),
                ('Actor B',root/'extraction_1000/improved_evaluation/official'/f'{name}.npz','C1'),
                ('Greedy B',Path(a['trace_paths']['improved'][name]),'C2')]:
            with np.load(path) as f:ax.plot(*f['observations'].T,label=label,color=color,lw=1.2);goal=f['goal'];points.extend([f['observations'].copy(),goal[None]])
        ax.scatter(*goal,marker='*',s=70,color='green');ax.set(title=f'Episode {i:03d}; actor B '+('success' if actor_b[name]['success'] else 'timeout'),aspect='equal')
    points=np.concatenate(points);lo=points.min(0)-1;hi=points.max(0)+1
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=7);fig.savefig(out/'task3_counterfactual.png',dpi=160);plt.close(fig)
    rescue_table='\n'.join(f"| {n} | {v['success']} | {v['steps']} | {v['strict_decrease_fraction']:.4f} | {v['increase_fraction']:.4f} |" for n,v in rescue.items())
    initial_fails=[n for n,v in improved.items() if not v['success']];rescue_fails=[n for n,v in rescue.items() if not v['success']]
    extraction_only=[n for n in rescue if rescue[n]['success'] and improved[n]['success']]
    entered_trap=[n for n in rescue_fails if improved[n]['success']]
    shared_trap=[n for n in rescue_fails if not improved[n]['success']]
    report=f'''# M6：官方新目标流上的距离恢复与残余提取缺口

同一50个官方task3输入，原critic贪心{count(original)}/50、改进critic贪心{count(improved)}/50；对应E17直接actor为0/50和31/50。改进critic从全部19个actor失败末状态续行，{count(rescue)}/19成功。官方actor指标仍31/50，这些真模拟器候选控制和超时后额外诊断绝不加进正式策略成功率。

## 运行时预算偏离完整保留

原240秒协议实际在{failure['elapsed_seconds']:.2f}秒触发TimeoutError、exit1，只完成40个原critic案例，第41例中断；未执行改进组。原目录及失败源码/日志/轨迹只读保留，不能称原240秒协议通过。因为振荡轨迹每步输入不同、精确缓存miss多，部分单例约12–14秒，算时估计不足。

显式[预算修订](../../../experiments/ogbench_critic/OFFICIAL_COUNTERFACTUAL_COMPLETION.md)在新目录只补未执行的10+50+19个episode，新阶段{a['elapsed_seconds']:.2f}秒/300秒上限，合计{a['combined_seconds']:.2f}秒。40个完整旧案例经逐例文件/hash验证后复用，未重跑；原未完整第41例从同初态重新完成。合计119个完整反事实episode及一次中断尝试。修订只针对只读计算成本，未改变训练更新（始终0）、模型、控制器、样本、episode上限或基于成功率改预算；未换seed、缩数据、启动1M。不能写成“未扩计算预算”。

## 相同输入与物理状态证据

按固定官方evaluate的np rng0调用顺序复建五任务250 reset，每任务前同一次actor key随机数；所有250 observation/goal与E17 A/B逐位一致。50个task3保存完整reset integration后，使用实际core.load_case恢复，分别执行原两组已存首动作，100个第一后继完整观测逐位一致。测试另将复建序列与实际官方evaluate在确定mock上比较，并用step消耗随机数的mock证明不可泛化跳过step的假设，两项通过。

这里并未声称将原未保存的初始mjData逐字段读回比较；证据是固定无传送PointMaze源码、全部reset输入、100实际首步和后续每个候选所选真实后继核验。对原始随机环境不能无条件套用此复建方法。未重跑旧完整actorepisode，无新增评价seed。

## 全部50例配对距离控制

| 控制器 | task3成功 | 范围 |
|---|---:|---|
| 原critic + E17同预算官方actor | {count(actor_a)}/50 | 正式actor评价 |
| 改进critic + 同预算官方actor | {count(actor_b)}/50 | 正式actor评价 |
| 原100k critic真后继贪心 | {count(original)}/50 | 本只读诊断 |
| E16 critic真后继贪心 | {count(improved)}/50 | 本只读诊断 |

改进critic贪心配对新增{len(gains)}、失去{len(losses)}。改进初态贪心失败名单{initial_fails}，末状态续行失败名单{rescue_fails}。全部50新目标在E16/E17终点冻结后才用于评价，没有回填训练或修改模型/超参数；它们与已知25-case方法选择库不同。但本反事实用途为事后解释，不当独立训练seed/全域泛化证明。

两组同原121动作真实后继/1000步上限，绕过actor和潜在动力学，仅原神经forward IQE评分。T1物理/精确有效输入缓存源码身份未变，每个选中后继仍逐位同实际env.step。轨迹日志state_values以固定128行padding前向，下降/上坡容差1e−4；这只改变日志批形状，候选评分原121行不变，不把小FP32误差作上坡证据。没有教师/图模块导入或在线训练标签。

![相同50例四控制器、步数及19个失败末状态诊断](official_counterfactual.png)

![预定官方task3前五例，未挑成功；与E17原actor轨迹配对](task3_counterfactual.png)

## 19个actor超时末状态

| 原失败episode | 同critic续行成功 | 额外步数 | 下降比例 | 上坡比例 |
|---|---|---:|---:|---:|
{rescue_table}

完整保留全部19例。起点为原actor第1000步结束integration，重新给1000步TimeLimit只为反事实诊断，不是原episode继续被算成功。模型和goal未变，能够或不能够从这些状态离开，才是这里的测量。

本项将残留actor失败与是否存在可用critic引导分开；贪心同时绕过actor和潜在动力学，不能再进一步唯一归因BC权重、actor容量或动力学误差。候选仅121个离散动作，也不是连续最优性证明。所有结论限当前模型、任务/目标流和控制范围。

本轮结果不能概括为“剩余都是actor问题”：{len(shared_trap)}例从原初态及actor末状态贪心都失败；{len(entered_trap)}例从原初态贪心成功，但actor进入了另一个同critic仍无法脱困的状态；{len(extraction_only)}例从原初态及失败末状态贪心都成功。后一组支持提取链路仍有缺口，前两组直接保留了残余距离陷阱证据。新初态42/50相对原0/50、正式actor31/50相对0/50且其他四任务无损，说明恢复可扩展到既有小bank之外；不是对全部状态都可下降或task3完全解决的证明。

## 复现与只读证明

原100k/E16参数摘要及文件前后不变，原E17的100条task3轨迹文件哈希前后不变；50-bank及所有旧/新源码、40复用/79新轨迹的起点/goal/步数/结束integration/有限性已核验。原协议[OFFICIAL_COUNTERFACTUAL_PROTOCOL.md](../../../experiments/ogbench_critic/OFFICIAL_COUNTERFACTUAL_PROTOCOL.md)与修订均保留。入口run_official_counterfactual.py，修订finish_official_counterfactual.py仅在特定超时/完整前缀被验证后补齐；本报告report_official_counterfactual.py只读重建。

raw results/ogbench_critic_20261029/official_counterfactual及official_counterfactual_completion，日志同级.log；后者analysis.json记录每条轨迹来源。只提交源码、测试、协议、中文报告及两张引用图，bank/模型/JSON/NPZ/日志不提交。19个actor失败仍是最终方法的限制；本项不自动追加学习或调参数。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');run(p.parse_args().root)
