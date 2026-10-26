"""Chinese post-budget report and scientific plots; never used to select models."""
import numpy as np

ARMS=('R0','Rhalf','R1','R2')
LABELS={'records':'42,536记录','unique_sources':'6,766去重源','longest_quartile':'最远四分位','unique_starts':'165独立起点','starts':'307原始起点'}


def assessment(s):
    a=s['results']['R1'];b=s['results']['R2'];x=s['extra']['R1'];y=s['extra']['R2']
    checks={
        'unique_edge_mean_not_worse':y['local']['unique_edges']['mean_positive']<=x['local']['unique_edges']['mean_positive'],
        'record_mean_not_worse':y['local']['records']['mean_positive']<=x['local']['records']['mean_positive'],
        'unique_edge_p99_not_worse':y['local']['unique_edges']['p99_positive']<=x['local']['unique_edges']['p99_positive'],
        'record_p99_not_worse':y['local']['records']['p99_positive']<=x['local']['records']['p99_positive'],
        'maximum_not_worse':y['local']['unique_edges']['max_positive']<=x['local']['unique_edges']['max_positive'],
        'fixed_R0_path_mean_not_worse':y['fixed_R0_paths165']['mean']<=x['fixed_R0_paths165']['mean'],
        'far_triangle_slack_not_worse':b['decomposition']['groups']['longest_quartile']['triangle_slack']<=a['decomposition']['groups']['longest_quartile']['triangle_slack'],
        'far_nrmse_not_worse':b['distance']['longest_quartile']['nrmse_percent']<=a['distance']['longest_quartile']['nrmse_percent'],
        'path_regret_not_worse':b['decisions']['unique_start_rollouts']['mean_successful_regret']<=a['decisions']['unique_start_rollouts']['mean_successful_regret'],
        'ranking_not_worse':b['decisions']['pairwise_order_accuracy']>=a['decisions']['pairwise_order_accuracy'],
    }
    left={p['start']:p['sum_positive'] for p in x['fixed_R0_paths165']['paths']}
    right={p['start']:p['sum_positive'] for p in y['fixed_R0_paths165']['paths']}
    delta=np.array([right[k]-left[k] for k in left])
    return dict(direction_checks=checks,all_reported_directions_nonworsening=all(checks.values()),
        fixed_paths_improved=int((delta < -1e-5).sum()),fixed_paths_worsened=int((delta>1e-5).sum()),
        fixed_paths_unchanged=int((abs(delta)<=1e-5).sum()),
        scope='Descriptive endpoint comparisons only, not a statistical significance test or a scalar selection objective')


def report(out,s):
    r=s['results'];extra=s['extra'];a=r['R1'];b=r['R2'];x=extra['R1'];y=extra['R2']
    lines=['# R2：PHR唯一物理边均衡（seed606，正式50k）','',
        f"R2相对R1：去重边平均正违反 {x['local']['unique_edges']['mean_positive']:.6f}→{y['local']['unique_edges']['mean_positive']:.6f}，"
        f"最大正违反 {x['local']['unique_edges']['max_positive']:.6f}→{y['local']['unique_edges']['max_positive']:.6f}；"
        f"最远四分位NRMSE {a['distance']['longest_quartile']['nrmse_percent']:.4f}%→{b['distance']['longest_quartile']['nrmse_percent']:.4f}%，"
        f"165独立起点平均路径regret {a['decisions']['unique_start_rollouts']['mean_successful_regret']:.4f}→{b['decisions']['unique_start_rollouts']['mean_successful_regret']:.4f}。"
        '局部违反、长程紧致性和路径决策分别判断，不能用最大违反下降或到达率代替决策改善。','',
        '## 冻结协议与必要验证','',
        '本轮仅训练seed606 R2，直接复用经配置/数据/完整初始化/预算/hash核验的R0/R1/Rhalf。'
        '保留R1 task源均衡 `-.6 mean(w*d_task)` 及原B1 `-.4 mean(d_B1)`。'
        '唯一变化是PHR按全局唯一物理边权重聚合：`mean(v_i*PHR_i)`，'
        '`v_i=N/(K*m(e_i))`，边身份为完整canonical source、target、action字节、原始cost。','',
        '逐记录PHR分子先乘v再取均值，最后除共同2ρ。所有副本的multiplier分别进入各自PHR；'
        '这等价于每条唯一边内部对副本PHR取平均后，再对边等权平均，不能用代表行PHR替代。'
        '不按minibatch权重和归一化，不改变采样或乘子数量。dual仍为逐访问、primal前残差的 `lambda_i=relu(lambda_i+h_i)`，不乘v。','',
        f"实测N={s['edge_weights']['N']}、K={s['edge_weights']['K']}；平均v={s['edge_weights']['mean']:.10f}，"
        f"范围{s['edge_weights']['min']:.6f}–{s['edge_weights']['max']:.6f}，最大副本数{s['edge_weights']['max_copies']}。"
        'task权重与R1指纹一致，M=6766。','',
        '共享encoder/projector/IQE；seed606，batch256，sampler seed2307，B1 seed42，task_mode=all，'
        'AdamW/weight_decay0、rho100、eta1、4线程；前45k学习率2e-5，后5k从2e-6几何衰减到2e-7。'
        '每1000步全量审计，保留epoch末尾40条小批次，固定50k停止。训练/权重/停止不读取图参考或Oracle。','',
        f"初始model/optimizer/zero dual/visits/sampler/RNG/counters与全部基线逐项一致。"
        f"完整有序采样序列独立重建通过；各行访问{s['pairing']['visits_min']}–{s['pairing']['visits_max']}次，"
        f"共{s['pairing']['visits_sum']:,}次访问，R2有{s['pairing']['full_audits']}次全量审计（含0步）。"
        '最终乘子正常独立演化，未要求最终参数/乘子一致。','',
        '7项训练测试通过：边权重均值/边总权重、完整边身份、不同副本乘子的PHR值与梯度（含gradcheck）、'
        '全1边权重与R1精确更新一致、task/B1/PHR及不加权dual、50k访问预算、真实4步保存/恢复及权重/模式/配置拒绝。'
        '另有3项后验统计测试通过。运行环境与旧实验一致；源码、数据、协议均已冻结，旧代码与产物未覆盖。','',
        '## 距离误差与偏差','',
        'bias=预测−修正参考，NRMSE=RMSE/RMS(参考)。修正参考仅在训练预算完成后读取。','',
        '| 分组 | 模型 | MAE | RMSE | NRMSE % | bias |','|---|---|---:|---:|---:|---:|']
    verdict=assessment(s)
    joint=verdict['all_reported_directions_nonworsening']
    conclusion=('本轮各项所列方向均未恶化，但仍须保留单seed和固定图限制。' if joint else
        '本轮未实现“降低局部违反的同时保留R1的长程紧致性和路径收益”。不同统计测度下的违反变化方向不同，不能把唯一边均值或最大值下降概括成约束问题已解决。')
    lines[2:2]=[conclusion,'']
    for group,label in LABELS.items():
        for arm in ARMS:
            d=r[arm]['distance'][group];lines.append(f"| {label} | {arm} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['nrmse_percent']:.4f} | {d['bias']:+.4f} |")
    lines+=['','## 长程累计边项与三角松弛','',
        '沿固定参考最短路径：`D*−J = Σ(cost−d_edge) + [Σd_edge−J]`。前项为有符号累计边项，后项为累计三角松弛。'
        '该恒等式分解有符号误差及其均值，不是RMSE可加归因；更负边项可能掩盖正三角松弛。','',
        '| 分组 | 模型 | D*−J | 累计边项 | 累计三角松弛 |','|---|---|---:|---:|---:|']
    for group in ('unique_sources','longest_quartile','unique_starts'):
        for arm in ARMS:
            d=r[arm]['decomposition']['groups'][group];lines.append(f"| {LABELS[group]} | {arm} | {d['gap']:.4f} | {d['edge_slack']:.4f} | {d['triangle_slack']:.4f} |")
    lines+=['','## 局部违反：均值、尾部和最大值','',
        '残差h=d(s,s′)−原始一步成本，正违反=relu(h)。违反数阈值1e-5；分位数对全部正违反值（含零）计算。'
        'top1%均值取最大的ceil(0.01n)项。完整JSON还包括p90、signed mean及仅违反项的p95。','',
        '| 分布 | 模型 | 违反数/总数 | 平均正违反 | p95 | p99 | top1%均值 | 最大值 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for distribution in ('records','unique_edges'):
        for arm in ARMS:
            d=extra[arm]['local'][distribution]
            lines.append(f"| {distribution} | {arm} | {d['violations']}/{d['count']} | {d['mean_positive']:.6f} | {d['p95_positive']:.6f} | {d['p99_positive']:.6f} | {d['top1pct_mean_positive']:.6f} | {d['max_positive']:.6f} |")
    lines+=['','### 按唯一边的副本数分组','',
        '分组边界在训练前固定，仅使用物理记录副本数。不是源重复次数分组。每组分别给唯一边等权与记录加权统计。','',
        '| m(e) | 唯一边数 | 物理记录数 | 原记录PHR权重占比 | R2边均衡PHR权重占比 |','|---|---:|---:|---:|---:|']
    groups=list(y['edge_copy_groups'])
    for group in groups:
        g=y['edge_copy_groups'][group];lines.append(f"| {group} | {g['unique_edges']} | {g['records']} | {100*g['record_mass']:.2f}% | {100*g['unique_edge_mass']:.2f}% |")
    for distribution,label in [('unique_edge_violations','唯一边等权'),('record_violations','记录加权')]:
        lines+=['',f'#### {label}','',
            '| 副本数 | 模型 | 违反数/总数 | 平均正违反 | p95 | p99 | top1%均值 | 最大值 |','|---|---|---:|---:|---:|---:|---:|---:|']
        for group in groups:
            for arm in ARMS:
                d=extra[arm]['edge_copy_groups'][group][distribution]
                lines.append(f"| {group} | {arm} | {d['violations']}/{d['count']} | {d['mean_positive']:.6f} | {d['p95_positive']:.6f} | {d['p99_positive']:.6f} | {d['top1pct_mean_positive']:.6f} | {d['max_positive']:.6f} |")
    lines+=['','## 固定R0路径的累计正违反','',
        '使用seed606 R0的50k冻结路径，所有模型的h都在同一套物理边路径上累加relu(h)。'
        '165唯一起点等权为主；307原始起点为辅助。以下累计值是模型残差诊断，不是额外环境成本，也不等价于regret。','',
        '| 路径集 | 模型 | 路径平均Σh+ | p95路径Σh+ | 最大路径Σh+ |','|---|---|---:|---:|---:|']
    for population in ('fixed_R0_paths165','fixed_R0_paths307','own_paths165'):
        for arm in ARMS:
            d=extra[arm][population];lines.append(f"| {population} | {arm} | {d['mean']:.6f} | {d['p95']:.6f} | {d['max']:.6f} |")
    lines+=['','固定R0路径的直接累加值与唯一边路径暴露次数加权值独立核对相等。各自路径统计仅辅助，不能通过路线更换掩盖违反。','',
        '## 分支、后继预测误差差分与路径regret','',
        '| 模型 | 最优分支/58 | 严格排序正确率 | 分支平均regret | 165起点路径regret | 165最优路径数 | 307路径regret |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        d=r[arm]['decisions'];u=d['unique_start_rollouts']
        lines.append(f"| {arm} | {d['optimal_branch_choices']} | {d['pairwise_order_accuracy']:.4%} | {d['mean_branch_regret']:.4f} | {u['mean_successful_regret']:.4f} | {u['optimal_rollouts']} | {d['mean_successful_regret']:.4f} |")
    lines+=['','所有模型使用同一真实候选边及cost+预测后继距离贪心规则。图为DAG、唯一汇点是目标，最长275步；'
        '307/307到达率主要由图结构保证，仅为完整性检查。','',
        '令ε(t)=J(t)−D*(t)，对模型所选边a和参考最优边b，核对'
        '`score(a)−score(b)=regret(a)+ε(t_a)−ε(t_b)`。'
        '错误选择通常需足够负的后继误差差分抵消真实regret。所选错误分支的集合因模型而变，其均值不能直接当作配对改善。','',
        '| 模型 | 非最优分支数 | 所选边误差差分均值（全部58） | 非最优所选边差分均值 | 固定严格候选对差分RMS | 排序正确对数 |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        d=extra[arm]['successor_error'];wrong=d['suboptimal_chosen_contrast']
        lines.append(f"| {arm} | {d['suboptimal_branches']} | {d['all_chosen_contrast']['mean']:.4f} | {wrong.get('mean',0):.4f} | {d['strict_pair_contrast']['rms']:.4f} | {d['correct_pairs']}/{d['strict_pairs']} |")
    pairs=s['branch_changes']['R1_to_R2_strict_pairs']
    lines+=['',f"对完全相同的严格参考候选对，R1→R2：错误变正确{pairs['wrong_to_correct']}对，正确变错误{pairs['correct_to_wrong']}对。"
        f"共有{s['branch_changes']['changed_branch_count']}个分支更换所选边。下表列出所有更换分支；完整58分支和全部候选对明细见JSON。",'',
        '| 源ID | R1选择→R2选择 | R1 regret | R2 regret | R1所选后继误差差分 | R2所选后继误差差分 |','|---|---|---:|---:|---:|---:|']
    for d in s['branch_changes']['changed_branches']:
        l=d['R1'];q=d['R2'];lines.append(f"| {d['source']} | {l['chosen_row']}→{q['chosen_row']} | {l['regret']:.4f} | {q['regret']:.4f} | {l['successor_error_contrast']:.4f} | {q['successor_error_contrast']:.4f} |")
    lines+=['','## R2相对R1：是否降低违反并保留收益','']
    for distribution in ('records','unique_edges'):
        l=x['local'][distribution];q=y['local'][distribution]
        lines.append(f"- {distribution}：平均正违反变化 {q['mean_positive']-l['mean_positive']:+.6f}，p99变化 {q['p99_positive']-l['p99_positive']:+.6f}，最大值变化 {q['max_positive']-l['max_positive']:+.6f}；违反数变化 {q['violations']-l['violations']:+d}。")
    lines.append(f"- 固定R0的165起点路径平均累计正违反：{x['fixed_R0_paths165']['mean']:.6f}→{y['fixed_R0_paths165']['mean']:.6f}。")
    for group in ('unique_sources','longest_quartile','unique_starts'):
        l=a['distance'][group];q=b['distance'][group]
        lines.append(f"- {LABELS[group]}：NRMSE {l['nrmse_percent']:.4f}%→{q['nrmse_percent']:.4f}%，bias {l['bias']:+.4f}→{q['bias']:+.4f}。")
    l=a['decomposition']['groups']['longest_quartile'];q=b['decomposition']['groups']['longest_quartile']
    lines.append(f"- 远端有符号gap变化 {q['gap']-l['gap']:+.4f} = 累计边项变化 {q['edge_slack']-l['edge_slack']:+.4f} + 累计三角松弛变化 {q['triangle_slack']-l['triangle_slack']:+.4f}。")
    l=a['decisions'];q=b['decisions']
    lines.append(f"- 最优分支 {l['optimal_branch_choices']}→{q['optimal_branch_choices']}，排序正确率 {l['pairwise_order_accuracy']:.4%}→{q['pairwise_order_accuracy']:.4%}，分支平均regret {l['mean_branch_regret']:.4f}→{q['mean_branch_regret']:.4f}，165路径regret {l['unique_start_rollouts']['mean_successful_regret']:.4f}→{q['unique_start_rollouts']['mean_successful_regret']:.4f}。")
    lines+=['','### 改善与代价出现在哪些边','']
    for group in groups:
        l=x['edge_copy_groups'][group]['unique_edge_violations'];q=y['edge_copy_groups'][group]['unique_edge_violations']
        lines.append(f"- m(e)={group}：唯一边平均正违反 {l['mean_positive']:.6f}→{q['mean_positive']:.6f}，p99 {l['p99_positive']:.6f}→{q['p99_positive']:.6f}，最大值 {l['max_positive']:.6f}→{q['max_positive']:.6f}。")
    lines+=['',f"固定R0的165条唯一起点路径中，{verdict['fixed_paths_worsened']}条累计正违反上升，{verdict['fixed_paths_improved']}条下降，{verdict['fixed_paths_unchanged']}条近似不变。"
        '这避免了把本模型路线更换误当成局部可行性改善。','',
        '远端累计边项是有符号量，负残差可抵消正残差；固定R0路径的累计正违反只累加正项，而且路径集不同。'
        '因此有符号累计边项变得更不负，与固定R0路径累计正违反增加并不矛盾。', '',
        '最优分支数量相同也不能保证路径收益相同：错误发生在哪些分支、真实advantage大小及哪些起点经过它们，都会改变完整路径regret。'
        '后继误差差分恒等式直接核验选择原因，但不证明某条局部违反是某次错误选择的因果来源。', '',
        '本轮不据此追加训练、调rho/dual或搜索其他权重，按固定50k如实保留结果。']
    lines+=['','## 45k仅作事后分析','',
        '| 模型/步数 | 去重源NRMSE % | 远端NRMSE % | 远端三角松弛 | 165路径regret | 最大违反 | 固定R0路径平均Σh+ |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for step,d,e in [(45000,s['stage45000'][arm],s['stage45000_extra'][arm]),(50000,r[arm],extra[arm])]:
            lines.append(f"| {arm}/{step} | {d['distance']['unique_sources']['nrmse_percent']:.4f} | {d['distance']['longest_quartile']['nrmse_percent']:.4f} | {d['decomposition']['groups']['longest_quartile']['triangle_slack']:.4f} | {d['decisions']['unique_start_rollouts']['mean_successful_regret']:.4f} | {e['local']['unique_edges']['max_positive']:.6f} | {e['fixed_R0_paths165']['mean']:.6f} |")
    lines+=['','50k固定为正式终点，不挑最好checkpoint，不搜索配置，不追加预算。单seed606、4线程、不承诺逐位复现；'
        '结论仅适用于该有限记录图，不证明多seed稳定性或连续控制泛化。共享副本与路径不是统计独立样本。','',
        '复现命令与实现：`experiments/phr_edge_balance/PROTOCOL.md`；冻结配置 `../protocol.json`，复用核验 `../reuse_verification.json`，'
        '测试 `../../tests.log` 与 `../../analysis_tests.log`，完整结果 `summary.json`，候选对/分支变更 `branch_differences.json`，'
        '节点预测、残差、分解、边副本ID `arrays.npz`。','', '![统一比较](comparison.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def plot(out,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=s['results'];e=s['extra'];x=np.arange(4);fig,axes=plt.subplots(2,3,figsize=(16,8))
    for j,g in enumerate(('unique_sources','longest_quartile','unique_starts')):
        axes[0,0].bar(x+(j-1)*.25,[r[a]['distance'][g]['nrmse_percent'] for a in ARMS],width=.25,label=g)
    axes[0,0].set(xticks=x,xticklabels=ARMS,ylabel='NRMSE (%)');axes[0,0].legend(fontsize=8)
    axes[0,1].bar(x,[r[a]['decisions']['unique_start_rollouts']['mean_successful_regret'] for a in ARMS]);axes[0,1].set(xticks=x,xticklabels=ARMS,ylabel='165-start mean path regret')
    axes[0,2].bar(x,[e[a]['fixed_R0_paths165']['mean'] for a in ARMS]);axes[0,2].set(xticks=x,xticklabels=ARMS,ylabel='Mean cumulative positive h on FIXED R0 paths')
    for j,k in enumerate(('edge_slack','triangle_slack','gap')):
        axes[1,0].bar(x+(j-1)*.25,[r[a]['decomposition']['groups']['longest_quartile'][k] for a in ARMS],width=.25,label=k)
    axes[1,0].axhline(0,color='k',lw=.6);axes[1,0].set(xticks=x,xticklabels=ARMS,ylabel='Farthest-quartile signed decomposition');axes[1,0].legend(fontsize=8)
    groups=list(e['R2']['edge_copy_groups']);xx=np.arange(len(groups))
    for a in ARMS:
        axes[1,1].plot(xx,[e[a]['edge_copy_groups'][g]['unique_edge_violations']['mean_positive'] for g in groups],'-o',label=a)
        axes[1,2].plot(xx,[e[a]['edge_copy_groups'][g]['unique_edge_violations']['max_positive'] for g in groups],'-o',label=a)
    axes[1,1].set(xticks=xx,xticklabels=groups,xlabel='Exact physical-edge copies',ylabel='Unique-edge mean positive violation');axes[1,1].legend()
    axes[1,2].set(xticks=xx,xticklabels=groups,xlabel='Exact physical-edge copies',ylabel='Maximum positive violation');axes[1,2].legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
