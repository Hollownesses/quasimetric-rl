"""Report all prespecified batches; distinguish gradient projections from Adam."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def stats(values):
    a=np.asarray(values)
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)),positive=int((a>0).sum()),negative=int((a<0).sum()))


def run(output):
    output=Path(output)
    data=json.loads((output/'analysis.json').read_text())
    protocol=data['protocol'];rows=data['rows'];s=data['statistics']
    assert protocol['batches']==32 and protocol['seed']==4400 and len(rows)==32
    assert [r['batch'] for r in rows]==list(range(32))
    assert all(np.isfinite(v) for r in rows for v in r.values())
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    parts=['push','pair','goal']
    axes[0].bar(np.arange(3),[s[f'sgd/{p}/shape']['mean'] for p in parts],
                yerr=[s[f'sgd/{p}/shape']['std'] for p in parts],capsize=5,
                color=['#c65a4f','#3478b5','#8d69a5'],alpha=.9)
    axes[0].set(xticks=range(3),xticklabels=['Global push','Pair penalty','Goal penalty'],
        title='SGD first-order projection (mean ± batch SD)',ylabel='Projected change of relative barrier')
    for ax,key,label in zip(axes[1:],['gap','shape'],['Distance barrier','Relative barrier']):
        a=np.array([r[f'adam/pair/{key}'] for r in rows]);b=np.array([r[f'adam/goal/{key}'] for r in rows])
        for av,bv in zip(a,b):ax.plot([0,1],[av,bv],c='#bfc5cb',lw=.7,zorder=0)
        jitter=np.linspace(-.065,.065,len(a))
        ax.scatter(jitter,a,s=16,c='#3478b5',alpha=.8)
        ax.scatter(1+jitter,b,s=16,c='#c65a4f',alpha=.8)
        ax.scatter([0,1],[a.mean(),b.mean()],c='black',s=40,marker='D',label='Batch mean')
        ax.axhline(data['zero_new_gradient_adam_probe_delta'][['gap','shape'].index(key)],c='#6c7075',ls='--',label='Zero new gradient / old Adam moments')
        ax.set(xticks=[0,1],xticklabels=['A / pair','B / goal'],
               title=f'Actual one-step Adam: {label.lower()}',ylabel=f'Change of {label.lower()}')
        ax.legend(fontsize=7,loc='best')
    for ax in axes:
        ax.axhline(0,c='black',lw=.7);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'gradient_conflict.png',dpi=180);plt.close(fig)
    projected_rows=['| 分量 | gap一阶变化均值 ± SD | 相对上坡一阶变化均值 ± SD | 相对上坡增大批次 |','|---|---:|---:|---:|']
    for part,label in [('push','随机目标距离最大化'),('pair','lambda × pair约束'),('goal','lambda × goal约束')]:
        a=s[f'sgd/{part}/gap'];b=s[f'sgd/{part}/shape']
        projected_rows.append(f'| {label} | {a["mean"]:+.6f} ± {a["std"]:.6f} | {b["mean"]:+.6f} ± {b["std"]:.6f} | {b["positive"]}/32 |')
    for arm in ['pair','goal']:
        a=stats([r['sgd/push/gap']+r[f'sgd/{arm}/gap'] for r in rows])
        b=stats([r['sgd/push/shape']+r[f'sgd/{arm}/shape'] for r in rows])
        projected_rows.append(f'| push + {arm}（投影相加） | {a["mean"]:+.6f} ± {a["std"]:.6f} | {b["mean"]:+.6f} ± {b["std"]:.6f} | {b["positive"]}/32 |')
    actual_rows=['| 实际Adam一次更新 | gap变化均值 ± SD | 相对上坡变化均值 ± SD | gap增大 / 相对上坡增大 |','|---|---:|---:|---:|']
    for arm in ['pair','goal']:
        a=s[f'adam/{arm}/gap'];b=s[f'adam/{arm}/shape']
        actual_rows.append(f'| {arm} | {a["mean"]:+.6f} ± {a["std"]:.6f} | {b["mean"]:+.6f} ± {b["std"]:.6f} | {a["positive"]}/32；{b["positive"]}/32 |')
    z=data['zero_new_gradient_adam_probe_delta']
    actual_rows.append(f'| 新梯度为零、保留原Adam矩 | {z[0]:+.6f} | {z[1]:+.6f} | 单次参照 |')
    paired=stats([r['adam/goal/shape']-r['adam/pair/shape'] for r in rows])
    text=f'''# M1：原始100k模型的距离上坡训练压力

日期：2026-10-29。本项分析严格使用原始100k模型，不查看E1中间模型，不串联续训，也不保存新checkpoint。预先固定seed4400、32个官方随机训练批次、每批1024。各次反事实从相同参数与Adam状态开始。

**局部证据：随机目标距离最大化项在当前失败排序上提供了上坡压力，原pair约束提供相反压力，二者在平均一阶投影上接近抵消。直接替换成goal差约束并不自动加强纠正作用。** 这还不是对整个训练历史或完整预算模型的根因证明。

## 固定probe与问题边界

目标为{protocol['goal']}；锚点与走廊点为{protocol['points']}，来自E0已固定参考路线的第0与第15个节点。网络距离差初值为{protocol['initial_probe'][0]:.6f}。相对上坡定义为 `gap / ((d_peak+d_anchor)/2+1e-6)`，初值{protocol['initial_probe'][1]:.6f}；它移除统一正缩放的影响，不声称对任意变换不敏感。

这些点仅用于评价梯度的投影，未作为训练样本或标签。批次仍由官方GCDataset产生，value目标仍为官方随机目标。actor/dynamics参数与Adam矩逐位冻结。

## 损失分量的普通SGD一阶投影

下表计算 `-lr × <grad(probe), grad(loss_part)>`，lr为固定3e-4。正数表示朝增大上坡方向，负数表示朝减小方向。**这些是局部导数投影，不是执行大幅SGD参数步后的实测结果，也不是实际Adam步的贡献分解。** SD为批次标准差，不是多初始化误差。

{chr(10).join(projected_rows)}

最大化项对相对上坡的投影32/32为正，原pair项32/32为负，说明纠正方向并非原pair梯度完全缺失。goal项的平均纠正投影更小；它改变梯度与可行集合，不能从“更直接涉及目标”推出更好的距离恢复。目标相关残差由准度量三角关系弱于pair残差，但本项只量化了当前probe上的投影，不能推断所有参数方向的梯度强弱。

## 从同一checkpoint出发的一次实际Adam反事实

{chr(10).join(actual_rows)}

B相对A的“相对上坡变化”配对差均值为{paired['mean']:+.6f}，SD={paired['std']:.6f}，{paired['positive']}/32批次中B增大得更多或减小得更少。原pair实际一步平均减小上坡，goal替换实际一步平均增大上坡；因此本项不支持直接声称B会更有效修正当前局部排序。它仍不能预测固定10k续训后的结果。

零新梯度参照说明旧Adam动量本身就可产生参数变化；它只是一项反事实参照。由于Adam的预条件与动量更新有非线性，不能把该参照与各损失一步结果相减后声称得到唯一的因果贡献，也不能将SGD分量投影当作Adam分量贡献。

实际Adam步与probe一阶预测的gap平均绝对误差：A={s['linearization_error/pair/gap']['mean_absolute']:.6g}、B={s['linearization_error/goal/gap']['mean_absolute']:.6g}；最大绝对误差分别为{s['linearization_error/pair/gap']['max_absolute']:.6g}和{s['linearization_error/goal/gap']['max_absolute']:.6g}。这支持这些一次局部变化的线性解释，不能外推到多步训练轨迹。

![分量方向冲突与一次Adam的配对反事实](gradient_conflict.png)

## 对“为什么学出这个场”的推进与限制

现在有训练信号层面的证据：在这个模型与这两个固定走廊点附近，距离最大化会扩大错误相对排序，局部约束则提供相反作用；当前平均局部统计接近容差，并不意味着目标距离场已经处处获得下降方向。该场可以处在存在相互抵消压力的优化位置，而不是简单缺少所有纠正梯度。

这些结果不证明该竞争是训练全程唯一原因，不排除有限参数化、优化路径、随机采样或额外预算的影响。32批次都复用同一checkpoint，不能当成32个模型复现；单次probe的即时变化也不能当成五任务闭环表现。参考路线、模拟器或最短路未进入训练。

E1仍按原固定10k预算执行，不因这项检查中止、延长、调权重或换终点；结论仍以终点下降通道、五任务真实后继贪心及两类约束为准。官方1M与独立初始化检查仍需完成，用户要求的“稳定改善”总目标保持未完成。

## 验证与复现

3项新测试通过：梯度分解与实际A/B价值梯度一致；相对上坡移除正距离缩放；零新梯度下旧Adam动量仍能移动value且冻结actor/dynamics。实际32批次反事实也逐一验证冻结证明与有限性。

运行方式见 experiments/ogbench_critic/MECHANISM_PROTOCOL.md 和 README.md。原始protocol.json、analysis.json、progress.json与日志留在本地，不提交；本报告、引用PNG、源码、测试和协议进入本项独立commit。
'''
    (output/'report.md').write_text(text)
    print(output/'report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(a.output)
