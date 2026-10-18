"""Post-run report; all failures retained and cohorts reported separately."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
import numpy as np
from experiments.four_path_execution.common import read,write,file_digest
from experiments.full_replay_four_path.evaluate import ARMS,OLD


def episodes(path):return [json.loads(x) for x in path.read_text().splitlines()]


def counts(rows):
    success=[r for r in rows if r['success']]
    return dict(episodes=len(rows),success=len(success),rate=len(success)/len(rows),
        **{k:sum(bool(r[k]) for r in rows) for k in ('collision','out_of_bounds','timeout')},
        success_cost_mean=float(np.mean([r['cost'] for r in success])) if success else None,
        exit_eligible=sum(r['exit_eligible'] for r in rows),exit_completed=sum(r['exit_completed'] is True for r in rows),
        wrap_eligible=sum(r['wrap_eligible'] for r in rows),wrap_completed=sum(r['wrap_completed'] is True for r in rows))


def paired(before,after):
    before={(r['state_id'],r['planner_seed']):r for r in before};after={(r['state_id'],r['planner_seed']):r for r in after}
    assert before.keys()==after.keys()
    gained=lost=0;costs=[]
    for key,a in before.items():
        b=after[key];gained+=int(not a['success'] and b['success']);lost+=int(a['success'] and not b['success'])
        if a['success'] and b['success']:costs.append(b['cost']-a['cost'])
    return dict(gained=gained,lost=lost,common_success=len(costs),common_success_cost_delta=float(np.mean(costs)) if costs else None)


def summarize_candidates(directory,independent=False):
    scores=read(directory/'scores.json');pairs=read(directory/'paired.json');summary={}
    for arm in ARMS:
        r=[x for x in scores if x['arm']==arm];available=[x for x in r if x['useful_count']]
        pr=[x for x in pairs if x.get('arm',x.get('after'))==arm]
        correlations=[x['safe_rank_spearman'] for x in pr if x['safe_rank_spearman'] is not None]
        summary[arm]=dict(batches=len(r),no_safe=sum(x['safe_count']==0 for x in r),
            useful_available=len(available),useful_weight_mean=float(np.mean([x['useful_weight'] for x in available])) if available else None,
            oracle_spearman_mean=float(np.mean(correlations)) if correlations else None,
            first_control_safe=sum(x['first_control_safe'] for x in r))
    return summary


def number(x,digits=3):return '—' if x is None else f'{x:.{digits}f}'


def run(root):
    out=root/'comparison';out.mkdir(exist_ok=False)
    assert (root/'train/COMPLETE').exists() and (root/'value_analysis/COMPLETE').exists()
    for group in ('original','independent'):
        assert (root/group/'replay_audit.json').exists()
    reuse=read(root/'baseline_reuse.json')
    original={};independent={};all_rows=[]
    for arm in ARMS:
        olddir=root/'original/arms'/arm if arm=='merged_606' else OLD/'arms'/arm
        if arm!='merged_606':assert file_digest(olddir/'episodes.jsonl')==reuse[arm]['episodes_sha256']
        assert (olddir/'COMPLETE').exists() and (root/'independent/arms'/arm/'COMPLETE').exists()
        original[arm]=episodes(olddir/'episodes.jsonl');independent[arm]=episodes(root/'independent/arms'/arm/'episodes.jsonl')
        assert len(original[arm])==64 and len(independent[arm])==48
        for cohort,rows in [('original',original[arm]),('independent',independent[arm])]:
            for r in rows:all_rows.append(dict(cohort=cohort,**r))
    cohorts=dict(original=original,independent=independent)
    summary={cohort:{arm:counts(rows) for arm,rows in arms.items()} for cohort,arms in cohorts.items()}
    strata={s:{a:counts([r for r in original[a] if r['stratum']==s]) for a in ARMS} for s in ('support','perturbed','holdout')}
    pairs={c:{a:paired(arms[a],arms['merged_606']) for a in ARMS if a!='merged_606'} for c,arms in cohorts.items()}
    candidates=dict(development=summarize_candidates(root/'independent/candidates'),independent=summarize_candidates(root/'independent_candidates',True))
    graph=read(root/'value_analysis/recorded_graph.json');four=read(root/'value_analysis/four_path.json')
    curve=episodes(root/'train/training.jsonl');scans=[r for r in curve if 'original' in r];last=scans[-1];assert last['step']==25000
    report=dict(cohorts=summary,strata=strata,paired=pairs,candidates=candidates,final_constraints=last,
        four_path=four,graph=graph,training_seed=606,checkpoint_sha256=file_digest(root/'train/final.pt'))
    write(out/'summary.json',report)
    fields=['cohort','arm','state_id','stratum','planner_seed','success','collision','out_of_bounds','timeout','cost','steps']
    with (out/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_rows)
    write(out/'all_failures.json',[r for r in all_rows if not r['success']])
    objective=read(root/'value_analysis/objective_diagnostic.json')
    text=['# 完整 20 万转移＋四路径：单 seed 训练与冻结 MPPI 实验','',
        '仅训练 seed 606 的合并数据组，从独立初始化开始。固定完成 20,000 步 PHR 与 5,000 步降步长；未新增采集数据、未训练原始 replay 单独对照、未按 Oracle/图参考选 checkpoint。', '',
        '**结论：本次合并数据训练没有改善泛化，且未能恢复四路径的长程价值结构。** 原状态集为 28/64（既有五个四路径模型合计 225/320）；原 holdout 为 0/16（既有五个均为 4/16）；新独立集为 2/48，No Terminal 为 3/48，既有四路径五模型合计 18/240，Oracle 为 37/48。小样本差异不作显著性宣称。记录边可行性和四路径估值也同时失败，因此这不是一次已恢复约束后检验纯泛化上限的实验。', '',
        '## 数据、训练和边界','',
        '合并 200,000＋335 条物理转移及 748＋4 条零成本 anchor，共 201,087 条约束、200,665 个规范节点。完整动力学、合法 anchor、一步成本与 B1 轨迹边界审计通过；一步成本重算最大绝对差 2.60e-5，训练使用未经改写的原始奖励。各任务条件全部保留，没有把其他设备任务重标为 U-trap。', '',
        '每步 256 条原始物理边（轮末较小 batch，无放回遍历）＋全部四路径边；两个来源的 push 与 PHR 各占 50%。每个来源 push 为 0.6 task(all)＋0.4 固定 B1。四路径 B1 与历史完全一致；原始 B1 单独按 seed 42 重建。均匀混合时路径物理边仅约占 0.167%，固定分层权重用于避免其长程信号被淹没。', '',
        '共享 production encoder/projector/IQE，rho=100、逐记录边独立 multiplier、访问时 eta=1 更新。原始 multiplier 的更新频次低于四路径；这与小图全边 PHR 不同。没有大规模 QP。原始零 anchor 在 quotient 后恒等，训练可省略其零项，所有 anchor 仍纳入全量检查。', '',
        f"原始边每条访问 {last['original_visits']['min']}–{last['original_visits']['max']} 次，平均 {last['original_visits']['mean']:.3f} 次；四路径每条参与 25,000 次。每 1,000 步全量检查。主阶段 lr=2e-5，后 5,000 步从 2e-6 指数降至 2e-7，Adam 历史连续保留。训练墙钟约 {last['seconds']/60:.1f} 分钟。", '',
        '## 全量约束与长程记录图','',
        '| 来源 | 物理边 | h>1e-5 | 最大正残差 | 平均正残差 |', '|---|---:|---:|---:|---:|']
    for name,n in [('original',200000),('paths',335)]:
        r=last[name];text.append(f"| {name} | {n} | {r['violation_count']} | {r['max_violation']:.6g} | {r['mean_positive']:.6g} |")
    local=read(root/'value_analysis/local_constraint_groups.json')['groups']
    text+=['','| 进入终态的边 | 边数 | h>1e-5 | 最大正残差 |','|---|---:|---:|---:|']
    for name in ('original_terminal_entry','path_terminal_entry'):
        r=local[name];text.append(f"| {name} | {r['count']} | {r['violations']} | {number(r['max_violation'],6)} |")
    text+=['',f"全部 anchor 最大绝对残差 {last['anchor_max']:.6g}。约束容差没有放宽；有违反就不能称作完成记录图近似恢复。",
        f"四路径 task RMSE={four['task_rmse']:.3f}，NRMSE={four['task_nrmse_percent']:.3f}%；push={four['push']:.3f}，图参考={four['push_reference']:.3f}，gap={four['push_gap_percent']:.3f}%。这些指标单独在四路径上计算，不把大 replay 的无终态链状态当作有限图标签。为与旧实验一致，此处 task RMSE 将四个零终态计入分母；下表物理 source RMSE 不含这四个零值，所以略有不同。",'',
        '| 图检查子集 | 状态数 | 有记录终态链 | 无记录终态链 | 有限参考 RMSE |','|---|---:|---:|---:|---:|']
    for name in ('original','paths','original_U','paths_U'):
        r=graph['groups'][name];text.append(f"| {name} | {r['total']} | {r['finite']} | {r['unreachable']} | {number(r['rmse'])} |")
    text+=['','严格图只连接逐字节相同的完整规范状态，没有增加近邻边或位置取整。无记录链表示没有数据内的绝对路径上界，不表示真实环境不可达。B1 的记录片段成本另存为上界，不默认它就是最短路。所有图计算发生在 final checkpoint 固定之后。','',
        '## 闭环成功与所有失败','',
        'MPPI 保持 H10/N128，噪声、温度、终值权重、碰撞判定、环境成本与合法任务终态定义均不变。两个 planner seed 为 314159、271828。原有 32 状态的七个基线按数据/代码/模型指纹复用；新模型重新运行 64 次。独立测试为预先冻结的 24 个 U 内状态，八个 heading 分层；八组均运行 48 次。两个集合分别报告。','']
    for cohort in ('original','independent'):
        text += [f'### {cohort}','', '| 组别 | 成功 | 碰撞 | 越界 | 超时 | 成功成本均值 |','|---|---:|---:|---:|---:|---:|']
        for arm,r in summary[cohort].items():text.append(f"| {arm} | {r['success']}/{r['episodes']} ({100*r['rate']:.1f}%) | {r['collision']} | {r['out_of_bounds']} | {r['timeout']} | {number(r['success_cost_mean'])} |")
        text.append('')
    text+=['| 组别 | 原 support /24 | 原 perturbed /24 | 原 holdout /16 |','|---|---:|---:|---:|']
    for a in ARMS:text.append(f"| {a} | {strata['support'][a]['success']} | {strata['perturbed'][a]['success']} | {strata['holdout'][a]['success']} |")
    text+=['','失败的提前截断成本不用于成本优劣排名。下表成本差只比较共同成功实例；新增/失去成功以参考组为起点。','',
        '| 集合 | 参考 → merged_606 | 新增成功 | 失去成功 | 共同成功 | 共同成功成本差 |','|---|---|---:|---:|---:|---:|']
    for cohort,pp in pairs.items():
        for a,r in pp.items():text.append(f"| {cohort} | {a} | {r['gained']} | {r['lost']} | {r['common_success']} | {number(r['common_success_cost_delta'])} |")
    text+=['','## 固定候选评分与未见状态估值','',
        '沿用 16 个开发候选批次，并对 24 个独立状态×2 RNG 生成 48 个独立候选批次。每批动作和模拟结果固定，由所有八组评分；保存全部成本、权重和轨迹。safe 指 horizon 内无碰撞/越界，useful 指完整安全 horizon、向西至少 0.25 且末态朝西，只是局部进展启发式。Spearman 比较同一批安全候选与 Oracle 的总成本排序。','',
        '| 候选集合 | 组别 | 无 safe 批次 | 有 useful 批次 | useful 权重均值 | 与 Oracle Spearman | 首动作安全批次 |','|---|---|---:|---:|---:|---:|---:|']
    for cohort,arms in candidates.items():
        for a,r in arms.items():text.append(f"| {cohort} | {a} | {r['no_safe']}/{r['batches']} | {r['useful_available']} | {number(r['useful_weight_mean'])} | {number(r['oracle_spearman_mean'])} | {r['first_control_safe']}/{r['batches']} |")
    text+=['','候选不足时终值无法选择不存在的轨迹；MPPI 加权平均控制也不保证对应某条安全候选。局部排序和 useful 权重不能替代完整闭环成功。Oracle 是具有完整模型的离散格点参考，不是连续最优距离，也不保证产生成功动作。','']
    ov=read(root/'value_analysis/oracle_state_comparison.json')
    text+=['| 状态分层 | merged 对 Oracle RMSE | qp_101 对 Oracle RMSE |','|---|---:|---:|']
    for s,r in ov.items():text.append(f"| {s} | {number(r['merged_606']['rmse'])} | {number(r['qp_101']['rmse'])} |")
    novelty=read(root/'value_analysis/state_support.json');new=[r for r in novelty if r['stratum']=='independent']
    text+=['',f"独立状态中与训练规范节点完全相同者 {sum(r['exact_replay_node'] for r in new)}/24；到原始同任务 replay 最近位置距离中位数 {np.median([r['nearest_original_position'] for r in new]):.4f}。位置接近不代表 heading 或后继路径支持充分；独立抽样也不等于远离全部训练数据。",'',
        '## 解释与结论范围','',
        '首先是优化没有达到本轮所需的基础条件。原始边有 43,845/200,000（21.92%）违反；14 条原始终态入口边中 12 条失败，最坏边成本只有 0.1、残差却为 171.253。原始每条边的 dual 只更新 31–32 次，而路径边更新 25,000 次，相差约 782 倍。这个按访问次数计时的随机 PHR 扩展没有保留小图全量 dual 的时间尺度，稀少的关键入口也未被专门保障；其影响不能由总 primal 步数掩盖。', '',
        '其次是无见证 task push 与多任务数值规模失衡。原始 200,000 个 source 中 197,414 个没有记录终态链；corridor_target 的全部 50,000 个 source 都无有限图参考，其平均 task 估值达到 16,869.64，平均 push 为 10,183.61。相比之下，四路径平均 push 仅 22.06，原 U 内 Oracle 格点可达的 6,236 个状态上，模型平均估值约 22.62，Oracle 参考平均约 240.00。给路径 50% 采样权重并没有保持其长程结构：四个起点只预测 16.64–28.79，记录路径参考却为 194.49–270.55。', '',
        '终点追加的只读目标分解（每个原始任务固定抽 512 行）测得 corridor push 梯度范数约 30,705，四路径约 1,180；原始混合 push 与路径 push 梯度余弦为 +0.335，含 PHR 的目标下降方向余弦约 −0.037。它们提示尺度与约束梯度问题，并不证明持续反向梯度冲突；这是单个终点的局部欧氏梯度，不是 Adam 实际更新，也没有用于重训或挑选模型。详见 objective_diagnostic.json。', '',
        '最后，连接与执行瓶颈仍然存在。合并后原始 U 内 6,407 个 source 仍没有严格记录链接到终态；独立候选中 8/48 批没有完整安全候选，但 Oracle 仍完成 37/48，而合并模型只有 2/48，故失败不能全部归因于候选覆盖。合并模型在独立安全候选上的 Oracle 排序相关为 −0.122；其 useful 权重 0.643 虽高于若干四路径模型，仍未形成闭环收益，验证了局部西向启发式不足以代表正确终值。', '',
        '后续若继续使用同一数据，应先验证有界/有见证的任务 push 支持、跨任务目标尺度，以及按边覆盖校准的 dual 更新和关键终态入口约束工作集，再讨论数据互补性。上述建议未在本轮实施。本轮只能判断当前固定训练方案失败，不能判断更充分、正确优化后的合并数据必然无效，也不能把失败归结为 encoder 容量上限。', '',
        '本轮的结果需要同时检查约束残差、四路径估值、严格图连通性、未见状态排序及闭环成功。扩大数据确实增加局部覆盖，但共享参数能否将没有记录链的分量连接起来，不能从零 anchor 残差或较大 push 推出。task(all) 对无终态链片段没有由记录路径给出的有限上界，可能推动相对关系之外的整体数值增长；若仍有局部违反，还同时存在约束优化误差。', '',
        '本实验只有一个新训练初始化，而且同时改变数据支持、采样权重和约束更新频次，也没有原始 20 万单独对照，不能把改善或退化单独归因于数据量。旧五 seed 汇总是同一实例集合上的描述性结果，不是独立样本数的增加。保留本次全部失败，不以最佳子集代替完整结果。', '',
        '## 验证、产物与复现','',
        '训练前完成全量数据审计、110 步独立 smoke、CPU 线程基准和 16 项测试（含真实恢复更新一致性及冻结 MPPI/Oracle 回归）。初次测试中的测试夹具 deepcopy 问题已修复为重新构造再加载 state_dict；没有据评估结果重训。训练 checkpoint 包含模型、Adam、所有逐边 multiplier、采样排列与游标、访问计数及 RNG。latest 原子写入；完整恢复方法见实验 README。', '',
        '新执行的原有状态 64 次和独立状态 384 次全部保存逐步 NPZ，并在不改动的真实环境中逐动作重放核验状态、奖励和终止分类。复用基线自身的历史 448 次重放审计也保留。所有失败汇总为 all_failures.json；完整分层与配对指标见 summary.json、episodes.csv。', '',
        f"最终模型 SHA256：`{report['checkpoint_sha256']}`。源数据与合并数据指纹在 train/data_audit.json；冻结训练/执行源码在 source_snapshot/；协议、状态与基线复用记录在 protocol.json、independent_states.json 和 baseline_reuse.json。分析脚本指纹另列 analysis_manifest.json。", '',
        '复现入口：`experiments/full_replay_four_path/README.md`。原始训练曲线和全部中间诊断位于 train/training.jsonl，最终全边残差位于 train/final_constraints.npz；固定候选批次位于 independent/candidates/ 与 independent_candidates/；记录图/Oracle/支持诊断位于 value_analysis/。','',
        '![训练和约束](training.png)','', '![闭环结果](outcomes.png)']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    paths=[*Path('experiments/full_replay_four_path').glob('*')]
    write(out/'analysis_manifest.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files={str(p):file_digest(p) for p in paths if p.is_file()}))
    import shutil
    for p in paths:
        if p.is_file():
            dest=out/'analysis_source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    plot(out,curve,summary)
    write(out/'COMPLETE',dict(new_episodes=448,reused_episodes=448,training_seed=606))


def plot(out,curve,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    scans=[r for r in curve if 'original' in r];updates=[r for r in curve if 'loss' in r]
    for key in ('original','paths'):
        axes[0].semilogy([r['step'] for r in scans],[max(r[key]['max_violation'],1e-8) for r in scans],label=key)
        axes[1].plot([r['step'] for r in scans],[r[key]['violation_count']/(200000 if key=='original' else 335) for r in scans],label=key)
    for ax,key in zip(axes[2:],('original_push','path_push')):ax.plot([r['step'] for r in updates],[r[key] for r in updates],label=key)
    for ax,title in zip(axes,('Full replay max violation','Fraction h > 1e-5','Original replay push','Four-path push')):
        ax.set_title(title);ax.set_xlabel('Update');ax.axvline(20000,color='gray',ls='--');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,(cohort,arms) in zip(axes,summary.items()):
        bottom=np.zeros(len(ARMS))
        for key,color in [('success','#258c63'),('collision','#c45959'),('out_of_bounds','#df9d42'),('timeout','#8c78ba')]:
            height=np.array([arms[a][key]/arms[a]['episodes'] for a in ARMS]);ax.bar(ARMS,height,bottom=bottom,label=key,color=color);bottom+=height
        ax.set_title(cohort);ax.tick_params(axis='x',rotation=55);ax.set_ylim(0,1);ax.set_ylabel('Episode fraction')
    axes[1].legend(loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(out/'outcomes.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
