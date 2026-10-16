"""Build a reviewable Chinese comparison from completed, audited runs."""
from pathlib import Path
import json
import sys

NAMES={'encoder_barrier':'共享 encoder + barrier','encoder_phr':'共享 encoder + 全边 PHR',
       'latent_barrier':'自由节点 latent + barrier','latent_phr':'自由节点 latent + 全边 PHR'}
ORDER=list(NAMES)

def main(root):
    out=root/'comparison'
    summary=json.loads((out/'summary.json').read_text())
    audit=json.loads((out/'independent_audit.json').read_text())
    assert len(summary)==8 and all(r['complete'] for r in summary.values())
    assert len(audit['runs'])==8
    rows=[]
    for ci in (0,1):
        for key in ORDER:
            r=summary[f'{key}_c{ci}']; m=r['final']
            assert m['step']==(20000 if ci==0 else 5000)
            rows.append((ci,key,m))
    c={key:summary[f'{key}_c0']['final'] for key in ORDER}
    ref=c[ORDER[0]]['push_reference']
    passed=[NAMES[key] for key in ORDER if c[key]['max_violation']<=1e-5 and c[key]['push_gap_percent']<=1 and c[key]['task_nrmse_percent']<=2]
    s=['# 四路径参数化 × 约束优化对照实验','',
       '日期：2026-10-16。范围：冻结四条成功路径形成的记录图，不涉及连续环境最优性或未见状态泛化。', '',
       '## 核心结果','',
       f'主实验四格各完成 20,000 次更新；第二个独立 critic 初始化四格各完成 5,000 次更新。预先登记的联合通过门槛：全部真实边最大正残差 ≤1e-5、终态为零、同支持 push gap ≤1%、task NRMSE ≤2%。通过组：{("、".join(passed)) if passed else "无"}。', '',
       f'**判断：证据更接近约束优化方案与IQE几何优化的耦合，不支持将共享encoder容量不足视为主要单一原因。** 固定共享encoder参数化，切换到全量独立乘子PHR，使task RMSE从 {c["encoder_barrier"]["task_rmse"]:.3f} 降到 {c["encoder_phr"]["task_rmse"]:.3f}；深处路径2/3的三角余量从 {c["encoder_barrier"]["paths"][2]["triangle_slack"]:.3f}/{c["encoder_barrier"]["paths"][3]["triangle_slack"]:.3f} 降到 {c["encoder_phr"]["paths"][2]["triangle_slack"]:.3f}/{c["encoder_phr"]["paths"][3]["triangle_slack"]:.3f}。因此，同一共享网络能够展开接近记录图参考的长程几何。剩余的关键问题是让这套约束求解真正达到逐边可行，而非继续只追求更高的原始push。', '',
       '不能把无违反当作最大化成功，也不能把存在约束违反时的高push当作可行最优解。下面同时报告原始值和统一缩放到可行后的值。', '',
       '## 20k 主比较（初始化 critic 0）','',
       f'同一 task/B1 支持的记录图最优 push 为 **{ref:.6f}**。gap = (参考−预测)/参考；负 gap 表示超过该可行参考，不能解释为更优。', '',
       '| 参数化 × 方法 | push | 原始 gap % | 可行缩放后 gap % | task RMSE | task NRMSE % | 违反边 /335 | 最大正残差 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in ORDER:
        m=c[key]
        s.append(f'| {NAMES[key]} | {m["push"]:.4f} | {m["push_gap_percent"]:.2f} | {m["feasible_push_gap_percent"]:.2f} | {m["task_rmse"]:.3f} | {m["task_nrmse_percent"]:.2f} | {m["violation_count"]} | {m["max_violation"]:.6g} |')
    s+=['','![20k训练曲线](curves_c0.png)','',
        '残差对数图将精确零画在1e-7显示下限；原始数值保存在CSV/JSON中。', '',
        '统一可行缩放仅用于独立评估：对正成本物理边取 `a=min(1,min_e c_e/max(d_e,1e-30))`（anchor结构上恒为零），将全部 projected latent 乘以 a，IQE 的正齐次性使所有距离同乘 a。重新通过实际 IQE 头验证了缩放后的全部边；没有用该缩放训练或挑选模型。它是保守的可行解诊断，不是训练结束时原模型已可行的声明。', '',
        '## 四条路径的起点与低估分解','',
        '| 方法 | 路径0 | 路径1 | 路径2 | 路径3 |','|---|---:|---:|---:|---:|',
        '| 记录图距离 | '+' | '.join(f'{p["reference"]:.3f}' for p in c[ORDER[0]]['paths'])+' |']
    for key in ORDER:
        s.append('| '+NAMES[key]+' | '+' | '.join(f'{p["start"]:.3f}' for p in c[key]['paths'])+' |')
    s+=['','![四路径估值](paths_c0.png)','',
        '下表各格是“边成本余量之和 / 三角余量之和”。两项相加严格等于记录路径成本减起点估值；PHR 有违反时，第一项可为负，不应称作全部正 slack。', '',
        '| 方法 | 路径0 | 路径1 | 路径2 | 路径3 |','|---|---:|---:|---:|---:|']
    for key in ORDER:
        s.append('| '+NAMES[key]+' | '+' | '.join(f'{p["edge_slack"]:.2f} / {p["triangle_slack"]:.2f}' for p in c[key]['paths'])+' |')
    s+=['','沿每条完整路径的正残差总和如下。它衡量允许局部违反后路径上界最多被放宽多少；不抵消逐边容差失败，但能补充统一缩放受单条短边主导的保守性。', '',
        '| 方法 | 路径0 | 路径1 | 路径2 | 路径3 |','|---|---:|---:|---:|---:|']
    for key in ORDER:
        s.append('| '+NAMES[key]+' | '+' | '.join(f'{p["sum_positive_residual"]:.4f}' for p in audit['runs'][key+'_c0']['path_violation_budgets'])+' |')
    s+=['','## 2×2 的效应与解释边界','',
        '| 同参数化切换到 PHR | 原始 push 增量 | 可行缩放后 push 增量 |','|---|---:|---:|']
    for param in ('encoder','latent'):
        b,p=c[param+'_barrier'],c[param+'_phr']
        s.append(f'| {param} | {p["push"]-b["push"]:.4f} | {p["feasible_push"]-b["feasible_push"]:.4f} |')
    interaction=(c['latent_phr']['push']-c['latent_barrier']['push'])-(c['encoder_phr']['push']-c['encoder_barrier']['push'])
    s+=['',f'原始 push 的差中差（latent 的 solver 效应减 encoder 的 solver 效应）为 {interaction:.4f}；这是单初始化的描述量，不是显著性检验。若模型不可行，该量不能当作约束最优性增益。', '',
        '自由 latent 直接学习 IQE 的2048维输入，取消 encoder 和 projector 的共享参数；其函数类能容纳已验证的图距离见证，但从同一初始距离出发的优化不保证更快。参数化之间使用适合尺度的不同坐标学习率，因此不能把结果解释为仅函数类容量的纯效应。所有 IQE 头的学习率均为2e-5。', '',
        '## 第二初始化的 5k 复核','',
        '两个初始化来自原 seed=42 的双 critic checkpoint，不是两个独立实验 seed。此处将 critic 0 的5k也列出，避免拿20k与5k比较。', '',
        '| 初始化 | 方法 | push | gap % | 可行缩放 gap % | task RMSE | 违反数 | 最大正残差 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ci in (0,1):
        for key in ORDER:
            curves=[json.loads(x) for x in (root/'main'/f'{key}_c{ci}'/'curve.jsonl').read_text().splitlines()]
            m=next(x for x in curves if x['step']==5000)
            s.append(f'| {ci} | {NAMES[key]} | {m["push"]:.3f} | {m["push_gap_percent"]:.2f} | {m["feasible_push_gap_percent"]:.2f} | {m["task_rmse"]:.3f} | {m["violation_count"]} | {m["max_violation"]:.6g} |')
    s+=['','![第二初始化曲线](curves_c1.png)','',
        '## 预算与计算开销','',
        '| 初始化 | 方法 | 更新数 | 可学习参数 | 训练核心秒 | 总计秒 |', '|---|---|---:|---:|---:|---:|']
    for ci,key,m in rows:
        cfg=summary[f'{key}_c{ci}']['config']
        s.append(f'| {ci} | {NAMES[key]} | {m["step"]} | {cfg["trainable_parameters"]} | {m["train_seconds"]:.1f} | {m["seconds"]:.1f} |')
    extras=[]
    for config_file in sorted(root.rglob('config.json')):
        if 'main' in config_file.parts or not (config_file.parent/'curve.jsonl').exists(): continue
        cc=[json.loads(x) for x in (config_file.parent/'curve.jsonl').read_text().splitlines()]
        if cc:
            extras.append(dict(run=str(config_file.parent.relative_to(root)),recorded_steps=cc[-1]['step'],
                recorded_train_seconds=cc[-1]['train_seconds'],complete=(config_file.parent/'COMPLETE').exists()))
    (out/'extra_compute.json').write_text(json.dumps(extras,indent=2)+'\n')
    if extras:
        s+=['',f'正式预算共100,000次更新。此外，[试跑与控制修正开销](extra_compute.json)共记录{sum(x["recorded_steps"] for x in extras):,}次更新、{sum(x["recorded_train_seconds"] for x in extras):.1f}训练核心秒。两个中止运行只计到最后保存点，各自之后不足250步未计入，所以这是额外开销的下界。测试、导入、分析耗时不包含在训练核心秒中。']
    s+=['','![时间与残差](time_and_residuals.png)','',
        'CPU、每进程2线程，最多4个训练进程并行；秒数包含资源竞争，不能据此给出独占硬件加速比。每步统一计算339条边及335个task、335个B1距离；因此20k的基础IQE前向比较数为20,180,000（另加评估）。PHR反传全部约束；barrier仍按原工作集选top8反传。encoder额外计算共享MLP，latent进行表格索引。没有把步数等同FLOPs。', '',
        '## 数据与实现控制','',
        '- 335条真实边、4条零成本anchor；规范化后336节点。节点由生产终态映射后的完整42维float32状态精确确定，保留真实共享节点；只统一正负零，无取整或空间近邻合并。',
        '- 复用生产B1生成器的seed=42与固定每行一个正向pair。统一对全部335个物理源计算0.6 task + 0.4 B1的完整期望，无 minibatch 噪声；未增加数据、auxiliary loss或轨迹成本监督。',
        '- encoder结构保持默认42→512→512→128→512→2048；IQE为64×32、同一max/mean reduction。自由表用初始encoder的projected输出初始化。四格的初始距离逐项一致，同参数化两个solver的初始state_dict指纹一致。',
        '- AdamW，weight_decay=0；encoder lr=2e-5，自由表lr=.01，所有IQE头lr=2e-5。常数学习率，不用pilot权重续训。',
        '- barrier直接继承原实现：weight500、margin=temperature=.01、64条工作边top8、每25步刷新全量audit缓存。',
        '- PHR共享原公式：mean_e(([lambda_e+rho*h_e]_+)²−lambda_e²)/(2rho)，rho=100；339个独立lambda从0开始，每步以更新前全量残差作lambda←[lambda+h]_+。均值尺度下实际逐边求和系数是lambda/339。没有functional dual网络或精确LP乘子输入。',
        '- 300步试跑比较latent lr=.01/.05及PHR rho=10/100、eta=.1/1；依据push增长和全量残差选更保守设置。初版自由表把大lr也用于IQE头，已中止并保留为exploratory_head_lr，不混入正式表；统一头lr后重新试跑、测试、从初始checkpoint重启。',
        '- 训练函数不调用图距离；反向Dijkstra只在评估中使用。单元测试将graph_reference替换为抛异常函数后，四格训练仍能反传。图距离仅说明记录支持上的可达上界及最优性差距，不表示环境最优距离。', '',
        '### 指纹','', '| 对象 | SHA256 / 内容摘要 |','|---|---|']
    for k,v in summary['encoder_barrier_c0']['config']['fingerprints'].items(): s.append(f'| {k} | `{v}` |')
    s+=['','## 验证与历史 baseline','',
        '47项测试通过：PHR梯度/投影、原functional PHR回归、终态与完整节点身份、B1冻结、初始函数等价、barrier损失一致、训练不依赖图标签、头学习率一致，以及已有四路径和barrier支持测试。独立审计重新载入8个最终checkpoint，逐项重算push、全部339条边、anchors与终态，并与保存结果对齐。', '',
        '历史20k是随机batch=256的训练，新的主表使用完整push期望，所以不直接复用作配对一格。对旧checkpoint在同一支持上独立重算：', '',
        '| 旧critic | push | gap % | task RMSE | 违反数 |','|---|---:|---:|---:|---:|']
    for ci,m in enumerate(audit['historical_stochastic_baseline']):
        s.append(f'| {ci} | {m["push"]:.4f} | {m["push_gap_percent"]:.2f} | {m["task_rmse"]:.3f} | {m["violation_count"]} |')
    s+=['','## 机制判断与下一步建议','',
        f'在当前有限预算和学习率下，自由 latent + barrier 的 push 为 {c["latent_barrier"]["push"]:.3f}，共享 encoder + barrier 为 {c["encoder_barrier"]["push"]:.3f}。'+
        ('解除共享 encoder 没有改善该软目标的最大化；不支持“共享 encoder 的容量是唯一瓶颈”的解释。' if c['latent_barrier']['push']<c['encoder_barrier']['push'] else
         '解除共享 encoder 改善了该软目标的最大化，显示参数化及其优化尺度对结果有影响。'), '',
        '切换到全量独立乘子 PHR 的作用必须以同参数化的两行比较，并同时考虑可行缩放差距。'+
        ('两种参数化的原始push都改善，说明原先的约束惩罚/优化方式确实是重要因素。' if all(c[p+'_phr']['push']>c[p+'_barrier']['push'] for p in ('encoder','latent')) else
         '两种参数化的原始push变化并不一致，提示方法与参数化存在交互。')+
        ('但至少一组PHR尚未达到逐边容差，不能据此声称硬约束问题已求解。' if any(c[p+'_phr']['max_violation']>1e-5 for p in ('encoder','latent')) else
         'PHR最终点通过逐边容差后，仍须由push gap检验最大化程度。'), '',
        '本实验能区分“换成自由表是否自动改善”和“同一参数化换求解方法是否改变轨迹”，不能将参数化、Adam坐标尺度、IQE分段梯度和有限预算完全解耦。单个20k初始化也不足以证明全局收敛或统计显著。', '',
        '方法轴同时改变固定软惩罚、独立乘子与实际反传的约束集合（工作集top8对比全量339边），所以结论针对这两套约束优化方案，不能单独归因于softplus函数形状。统一缩放可能被一条低成本边主导；它是保守可行诊断，不代表最好的局部修复，也不能独自推翻原始任务值显著改善的证据。', '',
        '1. 先在同一339边上改进可行性收敛：以完整检查控制outer PHR的rho/dual更新频率与primal学习率，或复用已有可行方向/回溯工具作全量验收；继续同时跟踪push gap，防止只收缩成低值可行解。当前常数学习率、每步同步primal-dual不是收敛证书。',
        '2. 若自由IQE仍难展开长程距离，下一组诊断应使用任务potential参数化 `d(x,y)=[F(x)-F(y)]_+`，仍只训练push和一步约束。它直接检验IQE区间/分量的优化问题，不添加suffix、最短路或LP乘子标签，也不宣称一般全状态对表示能力。',
        '3. 对候选可行解补完整20k多初始化/多seed与学习率敏感性，而不是仅将当前第二初始化5k视为最终稳健性证据。四路径联合门槛通过后，再加入真实分支动作并检验连续环境泛化；此实验没有连续最优性结论。', '',
        '## 复现与产物','',
        '在仓库根目录执行以下命令，输出目录必须是新目录：','',
        '```sh', '../quasimetric-rl/.venv/bin/python experiments/four_path_factorial/run_suite.py \\',
        '  --output results/NEW_FACTORIAL --jobs 4',
        '```','',
        '分析：`summarize.py OUTPUT_ROOT`、`audit.py OUTPUT_ROOT`、`write_report.py OUTPUT_ROOT`。', '',
        '- [汇总JSON](summary.json)、[汇总CSV](summary.csv)、[独立checkpoint/梯度/dual审计](independent_audit.json)。',
        '- [IQE存在性复核](representation_witness.json)：以记录图距离构造重复坐标，只作独立评估，验证同一IQE配置可同时达到全部边/task/B1参考；没有进入训练或初始化。',
        '- 每组目录包含config.json、初始权重、完整curve.jsonl、逐路径values.csv、support.npz、最终边残差及模型/optimizer/dual状态，以及2k/5k/10k/20k里程碑（在预算内的项）。',
        '- [主协议](../../../experiments/four_path_factorial/PROTOCOL.md)、[47项验证记录](../validation_head_matched.txt)、[初版来源指纹](../provenance.json)、[统一头lr来源指纹](../provenance_head_matched.json)。旧实验产物均保留。',
        '- source_snapshot保留encoder启动时/初版探索代码；source_snapshot_head_matched保留统一IQE头lr后的版本。config.json记录各run实际脚本指纹，允许重建此控制修正。', '']
    (out/'report.md').write_text('\n'.join(s))
    print(out/'report.md')

if __name__=='__main__': main(Path(sys.argv[1]))
