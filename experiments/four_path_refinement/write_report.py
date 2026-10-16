"""Build the report from saved measurements, including unsuccessful attempts."""
import argparse
import json
from pathlib import Path
import numpy as np


def readlines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def table(headers, rows):
    escape=lambda value:str(value).replace('|',r'\|')
    return '| '+' | '.join(map(escape,headers))+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(escape,row))+' |\n' for row in rows)


def write(root):
    out=root/'comparison'
    s=json.loads((out/'summary.json').read_text())
    cfg=json.loads((root/'main/A/config.json').read_text())
    initial=readlines(root/'main/A/curve.jsonl')[0]
    finals={a:s[a]['final'] for a in 'ABC'}
    repairs={name:readlines(root/name/'curve.jsonl')[-1] for name in ('repair','repair_push')}
    # This scientific narrative is specific to the recorded run. A rerun must
    # be interpreted again rather than silently reusing its factual claims.
    for a,expected in [('A',81.06521606445312),('B',81.00922393798828),('C',81.01052856445312)]:
        if abs(finals[a]['push']-expected)>1e-7:
            raise ValueError('This report narrative describes the recorded 20261016 run; interpret the new summary before rewriting it.')
    for label,name in [('R0','repair'),('R1','repair_push')]: finals[label]=repairs[name]
    comparison=[]
    for name,r in [('20k 起点',initial)]+[(f'{a} +{step}',s[a][key]) for step,key in [(2000,'at_2000'),(5000,'final')] for a in 'ABC']+[(k,v) for k,v in finals.items() if k.startswith('R')]:
        comparison.append([name,f"{r['push']:.6f}",f"{r['push_gap_percent']:.4f}",f"{r['task_nrmse_percent']:.4f}",f"{r['max_violation']:.8f}",r['violation_count'],'通过' if r['joint_pass'] else '未通过'])
    tail=[]
    for a in 'ABC':
        u=readlines(root/'main'/a/'updates.jsonl')[-1000:]
        h=[x['pre_primal_max_violation'] for x in u]
        tail.append([a,f'{min(h):.6f}–{max(h):.6f}',f'{np.std(h):.6g}',
                     f"{np.mean([x['parameter_update_l2'] for x in u]):.6g}",
                     f"{finals[a]['complementarity']:.6f}",f"{finals[a]['complementarity_max']:.6f}",
                     f"{finals[a]['dual_stationarity_eta1_max']:.6f}"])
    terminals=[];path_rows=[];starts=[]
    for p in range(4):
        entry=initial['terminal_entry_edges'][p]
        terminals.append([p,entry['row']]+[f"{r['terminal_entry_edges'][p]['residual']:+.8f}" for r in [initial]+list(finals.values())])
        starts.append([p,f"{initial['paths'][p]['reference']:.4f}"]+[f"{r['paths'][p]['start']:.4f}" for r in finals.values()])
        path_rows.append([p]+[f"{r['paths'][p]['positive_residual_sum']:.6f} / {r['paths'][p]['triangle_slack']:.6f}" for r in finals.values()])
    repair_rows=[]
    for label,name in [('R0','repair'),('R1','repair_push')]:
        q=readlines(root/name/'qp.jsonl');r=repairs[name];rc=json.loads((root/name/'config.json').read_text())
        attempts=[c for x in q for c in x['attempts']]
        repair_rows.append([label,len(q),sum(x['accepted'] for x in q),len(attempts),
            sum(c['push']<rc['fixed_push_floor'] for c in attempts),f"{r['seconds']:.2f}",f"{r['push_loss_from_start']:.8f}"])
    cost_rows=[[a,5000,250 if a=='C' else 5000,f"{finals[a]['train_seconds']:.2f}",f"{finals[a]['seconds']:.2f}"] for a in 'ABC']
    fingerprints=table(['冻结对象','SHA-256'],[[k,f'`{v}`'] for k,v in cfg['fingerprints'].items()])
    text=f'''# 从 encoder_phr_c0 20k 出发的可行性精修

日期：2026-10-16。承接 [四路径 2×2 报告](../../four_path_factorial_20261016/comparison/report.md)。范围仅为冻结四路径记录图；没有声称恢复连续环境最优距离，也没有增加数据、辅助 loss、图距离标签或 LP multiplier 标签。

## 结论

**降低 primal 学习率显著压低了更新波动，但没有解决精确可行性；本次内外层 PHR 与低学习率同步更新结果几乎一致。** A/B/C 各完成额外 5000 次 primal 更新，所有联合门槛均保持原值，没有一组通过。

B 相对 A 只改变学习率，保留 Adam 历史；最后 1000 步实际参数更新 L2 均值从 0.001877 降到 0.00005657（约 33 倍），逐步最大违反的标准差从 0.08180 降到 0.0004487（约 182 倍）。这比仅观察旧曲线更直接地支持“原步长导致较大边界波动”。但是 B/C 已稳定在约 0.011 的违反水平，仍是 1e-5 门槛的约千倍，不能把全部残差归结为随机波动或宣称已经收敛。

随后进行了带固定价值下限的全边局部 QP 修复。它将最大违反进一步降至约 0.00233，且 push 损失不超过预设 0.1；task NRMSE 仍约 1.07%。但仍有 31–32 条边超过 1e-5，两个修复分支也均未通过。**这说明能以较小价值代价修掉大部分残差；尚未证明能在该价值预算内消除全部残差，更不能从局部失败推断不存在这样的解。**

## 固定条件与三个分支

共同起点是 `results/four_path_factorial_20261016/main/encoder_phr_c0/latest.pt`，step=20000，SHA-256：`{cfg['checkpoint_sha256']}`。使用带完整优化器状态的 latest checkpoint；旧 milestone 文件没有 Adam 状态，因此没有拿它代替。

四条成功路径、完整 42D 规范状态身份、336 个节点、335 条物理边、4 条零 anchor、终态 quotient、固定 task/B1 配对与一步成本完全不变。训练 push 为 335 个物理 source 上的完整期望 `0.6 mean(task) + 0.4 mean(B1)`。共享 encoder、projector、IQE 64×32 和可学习 maxmean reduction 均沿用原模型；A/B/C 中全部原 primal 参数继续训练。

三组恢复同一个模型、339 个逐边乘子、单参数组 AdamW（包括 exp_avg、exp_avg_sq、step=20000）和 RNG。初始张量与 Adam 哈希逐项相同。ρ 始终为 100；不重置动量，不增大惩罚。

| 分支 | primal 学习率（包括 IQE reduction） | 乘子更新 | 额外 primal / dual 次数 |
|---|---|---|---|
| A 原配置 | 常数 2e-5 | 每次 primal 后用更新前 h，λ←[λ+h]+ | 5000 / 5000 |
| B 降步长 | 第 j 步为 2e-6×0.1^((j−1)/4999)，末步 2e-7 | 与 A 完全相同 | 5000 / 5000 |
| C 内外层 | 与 B 同一衰减 | 固定 λ 做 20 次 primal，重新计算全部 339 边当前 h，再 λ←[λ+20h]+ | 5000 / 250 |

C 的外层步长20配合每20步更新，使名义步长/每 primal 为1；这不意味着动态等价。C 同时改变分块、残差时刻和乘子更新频率，是组合求解策略，不能称为单因素试验。固定20次内层没有证明每个子问题已充分收敛；本结果也不能否定更充分或自适应内层求解。

先执行了三个100步试跑，使用同一个5000步衰减计划；试跑后正式实验全部重新加载原20k起点。正式完整评估每25步一次，逐步记录实际参数更新幅度、更新前全边最大违反和 push。2000/5000 checkpoint 均含完整模型、乘子与 Adam。

## 同预算结果与不变的联合门槛

联合通过要求：335条物理边最大正残差≤1e-5，4条anchor与规范终态距离≤1e-6（本次实际均严格为0），同支持 push gap≤1%，task NRMSE≤2%。违反数量按 h>1e-5 计数。表中对齐检查点失败的原因都是逐边可行性，价值侧门槛没有放宽；A的部分中间检查点也会因push波动而超过1%的gap门槛。

记录图参考 push 为 **{initial['push_reference']:.9f}**；它只用于独立评估。gap 为 `(参考−模型)/参考`。

{table(['分支 / 额外预算','push','gap %','task NRMSE %','最大正残差','违反 /335','联合门槛'],comparison)}

R0 为最小参数改动的价值保持修复；R1 为保留原 push 上升方向的探索性修复对照，均从 B+5000 重新开始，详情见后文。

![训练曲线](curves.png)

图中每25步一个全量评估；对数轴的零值仅为显示截到1e-9。统一缩放系数只是保守诊断，未用来训练、验收或替代逐边可行性。所有检查点的原始指标见各组 `curve.jsonl`。

## 波动、参数更新与乘子

以下残差范围/标准差来自最后1000次更新**每一步的更新前全量残差**，对应额外step4000–4999，不只取25步稀疏评估。参数幅度使用 step4001–5000 实际前后参数差，包含全部 primal 参数。

{table(['分支','逐步 max h+ 范围','标准差','平均 ‖Δθ‖₂','末点 mean|λh|','末点 max|λh|','末点 ‖[λ+h]+−λ‖∞'],tail)}

互补性是339边上的 mean/max |λh|，没有把它当成完整 KKT 证书；最后一列统一用 η=1 做可比较的投影 dual 变化诊断。B/C 的互补性和 dual 变化也仍非零。A 的逐步最大违反甚至到约0.491，单独挑末点0.081会低估其波动。B/C 的结果差距很小，单初始化下没有证据认定本次 C 优于 B。

## 四条终态入口边与路径估值

“到规范终态的自距离为0”与“进入终态那条物理边满足成本”是两个检查。以下四条入口边成本均为0.1，表内是**有符号 h=d−c**，负数表示有余量；row 是稳定 replay 行号。

{table(['路径','row','20k起点','A+5k','B+5k','C+5k','R0','R1'],terminals)}

A 末点最坏边正是路径2终态入口 row240；B/C 和两个修复末点最坏边都转为路径2的内部 row203，其一步成本为2.781761。只修终态入口或只看 terminal zero 都会漏掉这个瓶颈。

{table(['路径','记录图参考起点','A+5k','B+5k','C+5k','R0','R1'],starts)}

下表每格为“沿路径累计正残差 / 累计三角余量”。累计正残差不做正负抵消；三角余量是 Σ[d(sₜ,sₜ₊₁)+d(sₜ₊₁,G)−d(sₜ,G)]，不是累计成本训练标签。

{table(['路径','A+5k','B+5k','C+5k','R0','R1'],path_rows)}

![终态入口、累计正残差和三角余量](path_diagnostics.png)

## 带价值保持条件的局部修复

预先选 B+5000 作为修复起点；模型 push=81.009223938，固定全程下限 **80.909223938**，δ=0.1（起点 push 的约0.123%）。该右侧只来自起点模型，未读取记录图参考。每个候选步均用真实非线性网络重新计算339边和同一 push，实际 push 低于此下限的候选一律拒绝；没有逐步下调下限。

复用仓库 `feasible_direction.solve_direction`，每个局部 QP 包含339条边加1条 push guard。优化既有最后 projector 的全部 **1,050,624** 个权重/偏置，较早的 encoder/projector 层和 IQE reduction 固定。这仍是原共享网络中的局部修复，不是自由节点 latent；但它也不是全网络参数空间的 SQP。用端点梯度和隐藏特征内积构造精确 Jacobian Gram，分解为340×340因子送入现有 QP，再把数值乘子拉回真实权重方向，避免保存339×1050624稠密 Jacobian。Gram 与直接参数 autograd 已对照验证。这里的 QP multiplier 只是当前线性子问题的临时变量，不是图 LP 标签，也没有覆盖原 PHR multiplier。

物理边线性约束附加5e-5内部余量，anchor保持零；参数信赖半径0.1，QP最多2000轮，最多13级回溯，必要时使用现有elastic投影（penalty100）。从不可行点允许恢复：要求正残差平方和下降，最大违反最多增加5%加1e-7，同时严格保住实际 push 下限；不要求每一个微小步马上全部可行。最终验收仍是原联合门槛。

R0 投影零方向，求小幅恢复；当它在价值下限附近连续三轮无可接受步后，保留失败结果，再执行探索性 R1：从相同 B+5000 起点投影范数0.1的原 push 上升方向。R1改变的是局部 QP 目标，不改变δ、数据、全边约束或联合门槛。两者预算上限40轮，均因连续3轮停滞提前结束。

{table(['修复','尝试轮数','接受步数','真实候选前向次数','低于push下限候选数','墙钟秒','实际push损失'],repair_rows)}

![价值保持修复](repair.png)

R0 的148个被拒候选中，146个低于价值下限；R1的116个被拒候选中，78个低于下限（拒绝条件可重叠）。价值 guard 已活跃，而非放任全局收缩。最终 QP 的线性 slack 也非严格零，且 IQE 活跃分段改变会使线性预测与实际前向有差异。因此停步同时受到局部线性化、有限 QP 精度、回溯准则和最后层子空间的限制，不能据此宣称高价值与精确可行性必然冲突。

修复只保留旧 PHR 乘子以便对照记录；它们没有更新，修复后的互补性不是新优化器的 KKT 证书。保存的 Adam 状态明确标记为 stale，不能当作无缝 Adam 续训状态。

## 判断与下一步

1. **更新步长确实是原大幅波动的重要因素。** A/B 的干预控制了 Adam、乘子与目标，只改学习率；幅度和残差波动同步下降。结论限于本 checkpoint 和预算，不需要把旧曲线猜测当作证明。
2. **残余瓶颈仍更接近约束求解与当前 IQE 几何的局部优化耦合。** B/C保持已有长程价值，三角余量也没有退回 barrier 的大幅塌缩，但仍不能精确可行。C的20步分块没有提供明显额外收益；这不是“所有内外层PHR都无效”的结论。
3. **下一步优先改善有价值保护的局部求解可靠性，而非提高ρ或constraint-only修复。** 从稳定的B状态开始，针对row203及其全边耦合检查线性预测误差、QP精度和活跃分段；采用随实际/预测恢复比调整的信赖域，并逐步开放较早网络层。保持同一全量约束与固定push下限，验证是否能通过更可靠的非线性恢复达到原门槛。当前结果不能把失败归因于不存在可行解。
4. 若继续检验PHR外层机制，应把内层终止条件设为可测的子问题优化进展，而非直接把20改成某个“更快dual”。清楚报告实际primal、残差重算和dual次数；在四路径联合达标前，不扩大数据或把此结果推广到连续环境最优性。

## 计算开销、验证与复现

{table(['组','primal次数','dual次数','训练段秒','含评估保存的墙钟秒'],cost_rows)}

三组CPU并发，每组torch线程2；表中训练时间包含参数幅度测量，C还含250次更新后的残差前向。三组训练段合计{sum(finals[a]['train_seconds'] for a in 'ABC'):.2f}秒，运行区间最长{max(finals[a]['seconds'] for a in 'ABC'):.2f}秒（不含启动导入）。另有300次pilot primal更新、25轮局部修复尝试、31次QP求解与283次候选前向，均单独保留。不同方法计算开销不同，没有把一次外层当作一次primal混计。

验证结果：16项相关测试通过；包括原PHR/冻结支持测试、精确checkpoint恢复、保留Adam、原更新公式、C的新残差时序、全边/入口诊断、不可行起点恢复与价值guard、真实权重映射、Gram和投影方向对照。串行CPU做公式等价测试；并行float32归约可能产生舍入差异，复现不承诺逐bit训练轨迹一致。初始化与原始产物SHA-256已核验。

独立重新加载A/B/C各自0、2000、5000 checkpoint及两份修复终点，共11份checkpoint，重新检查339边、终态、push、NRMSE与保存的日志；也核对Adam实际步数和C的250次dual更新。所有匹配，见 [独立审计](independent_audit.json)、[汇总JSON](summary.json)、[汇总CSV](summary.csv)。原实验产物未覆盖。源文件快照与SHA-256在上级目录 `source_snapshot`、`source_snapshot_final`、`final_provenance.json`。

{fingerprints}

复现环境沿用 `../quasimetric-rl/.venv/bin/python`，Python3.9 / torch2.8，CPU。为保护结果，输出目录必须不存在；可改用新的run名称：

```sh
export MPLCONFIGDIR=/tmp/qrl-factorial-mpl
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=../quasimetric-rl/.venv/bin/python
$PY -m pytest experiments/four_path_refinement/test_refinement.py experiments/four_path_factorial/test_factorial.py -q
$PY -m experiments.four_path_refinement.suite --steps 100 --output results/NEW_RUN/pilot
$PY -m experiments.four_path_refinement.suite --steps 5000 --output results/NEW_RUN/main
$PY -m experiments.four_path_refinement.repair --checkpoint results/NEW_RUN/main/B/checkpoint_05000.pt --output results/NEW_RUN/repair
$PY -m experiments.four_path_refinement.repair --checkpoint results/NEW_RUN/main/B/checkpoint_05000.pt --output results/NEW_RUN/repair_push --push-step 0.1
$PY -m experiments.four_path_refinement.summarize --root results/NEW_RUN
```

`write_report.py` 是本轮已保存结果的叙述生成器，并核对本轮测量值；复跑训练可能因float32归约产生差异，应根据新的summary重新解读，不直接复制本轮结论。

每个main目录的 `config.json` 记录初始模型/dual/Adam摘要、数据指纹、学习率计划与代码摘要；`updates.jsonl`逐步记录参数幅度；`curve.jsonl`包含全部入口边、逐路径残差、互补性与门槛；`values.csv`提供路径逐节点估值。修复另存 `qp.jsonl`，完整保留被拒候选和实际验收。训练与修复函数都不接收评估图距离；图参考只由独立evaluate/diagnose计算。
'''
    (out/'report.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/four_path_refinement_20261016')
    write(Path(p.parse_args().root))
