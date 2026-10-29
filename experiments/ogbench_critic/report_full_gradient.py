"""Report original-model complete-gradient evidence without long-run claims."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def run(output):
    output=Path(output);data=json.loads((output/'analysis.json').read_text());s=data['statistics'];rows=data['rows']
    assert len(rows)==data['protocol']['batches']==32
    assert data['previous_stream_equal'] and data['previous_value_only_counterfactuals_equal'] and data['original_state_unchanged']
    assert all(r['actor_to_value_gradient_norm']==0 for r in rows)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    names=['value','dynamics','total'];colors=['#3478b5','#d58d34','#bd4e50']
    for i,(name,color) in enumerate(zip(names,colors)):
        y=[r[f'sgd/{name}/shape'] for r in rows]
        axes[0,0].scatter(np.full(32,i)+np.linspace(-.12,.12,32),y,color=color,s=16,alpha=.7)
        axes[0,0].plot([i-.2,i+.2],[np.mean(y)]*2,color='black',lw=2)
    axes[0,0].axhline(0,color='gray',lw=1)
    axes[0,0].set(title='Relative barrier: first-order SGD projection',xticks=range(3),xticklabels=names,ylabel='Predicted relative-gap change')
    x=np.arange(32)
    for name,color in [('value_only','#3478b5'),('joint','#bd4e50')]:
        axes[0,1].plot(x,[r[f'adam/{name}/shape'] for r in rows],label=name,color=color,marker='.',lw=1)
    axes[0,1].axhline(data['zero_new_gradient_adam_probe_delta'][1],color='gray',ls='--',label='Zero new gradient / old Adam')
    axes[0,1].axhline(0,color='black',lw=.7);axes[0,1].set(title='Actual independent one-step Adam',xlabel='Fixed batch ID',ylabel='Relative-gap change');axes[0,1].legend(fontsize=8)
    axes[1,0].hist([r['gradient_cosine/value_dynamics'] for r in rows],bins=10,color='#d58d34',alpha=.8)
    axes[1,0].set(title='V-parameter gradient cosine',xlabel='cos(value loss, dynamics loss)',ylabel='Batches')
    a=np.array([r['adam/value_only/shape'] for r in rows]);b=np.array([r['adam/joint/shape'] for r in rows])
    axes[1,1].scatter(a,b,c='#6d599a',s=24)
    lo=min(a.min(),b.min());hi=max(a.max(),b.max());axes[1,1].plot([lo,hi],[lo,hi],c='gray',ls='--')
    axes[1,1].set(title='Paired actual Adam (same original state)',xlabel='Value-only relative-gap change',ylabel='Full joint relative-gap change')
    axes[1,1].xaxis.set_major_locator(MaxNLocator(5));axes[1,1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1,1].ticklabel_format(axis='both',style='sci',scilimits=(-3,-3),useMathText=True)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(output/'full_gradient.png',dpi=180);plt.close(fig)
    table=['| SGD一阶相对上坡投影 | 均值 | 标准差 | 正 / 负 / 零 |','|---|---:|---:|---:|']
    for part in ['value','dynamics','actor','total']:
        item=s[f'sgd/{part}/shape'];table.append(f'| {part} | {item["mean"]:.8g} | {item["std"]:.8g} | {item["positive"]} / {item["negative"]} / {item["zero"]} |')
    adam=['| 实际一次Adam相对上坡变化 | 均值 | 标准差 | 正 / 负 / 零 |','|---|---:|---:|---:|']
    for part in ['value_only','joint','joint_minus_value']:
        item=s[f'adam/{part}/shape'];adam.append(f'| {part} | {item["mean"]:.8g} | {item["std"]:.8g} | {item["positive"]} / {item["negative"]} / {item["zero"]} |')
    dyn=s['sgd/dynamics/shape'];total=s['sgd/total/shape'];paired=s['adam/joint_minus_value/shape']
    direction='增大' if dyn['mean']>0 else '减小' if dyn['mean']<0 else '不改变'
    actual='更不利' if paired['mean']>0 else '更有利' if paired['mean']<0 else '平均相同'
    report=f'''# M2：补齐官方动力学损失对value距离场的梯度作用

在严格固定的原始100k checkpoint附近，动力学损失对当前probe的一步更新影响较小：完整联合Adam相对value-only的平均相对上坡变化差为{paired['mean']:.6g}，32批次实际变化符号不同的次数为{int(np.sum(np.sign(a)!=np.sign(b)))}。动力学分量的一阶SGD投影平均{direction}上坡，实际联合一步相对value-only平均{actual}，但两组实际平均仍为{s['adam/value_only/shape']['mean']:.6g}、{s['adam/joint/shape']['mean']:.6g}。actor_loss对value的梯度在本配置下为零。这补齐M1的范围限制，不支持将动力学在此处的一步作用认定为主导原因，也不证明全训练历史的唯一原因。

## 控制与实测核对

与M1使用同一seed4400、32个官方batch1024、同原始模型及Adam状态；probe仍为预先固定参考路线节点0/15及task3目标。每个反事实独立从原模型出发，不串联训练，不保存权重。M1的整条批次流hash一致，32次value-only实际反事实逐项相等，原始状态摘要前后相同。运行约{data['elapsed_seconds']:.2f}秒。

value+dynamics+actor对value的分量梯度之和与直接total梯度核验通过，各批次最大叶元素误差的均值为{s['gradient_sum_max_error']['mean']:.4g}，全体最大为{max(r['gradient_sum_max_error'] for r in rows):.4g}；actor对value的梯度范数32/32为零。三项独立测试另外检查原生全参数total梯度切片与分解相等、动力学确实到达value而actor不达，以及数值比较拒绝实质梯度改变。没有把同批次32次测量称为32个初始化。

首次运行只按最大绝对误差<5e-5验证，在第二个批次（batch=1）5.3406e-5处中止，失败日志保留在full_gradient.log。此处重新运行相同32批次，使用逐元素rtol2e-5、atol5e-5以处理独立FP32融合及梯度尺度；没有改变公式、样本、模型或方向判据。最终最大误差和分布仍保留，不将误差截为零。运行时协议的失败批次曾误写“第一批”，此为文字勘误；原运行协议按哈希保留在source_snapshot中。

## 一阶梯度证据

{chr(10).join(table)}

正值表示该损失分量的普通SGD方向使相对上坡增大，负值表示减小。相对量为gap / 两点距离均值，去除统一正缩放的主要影响；不能代表全图所有目标的空间排序。动力学投影正{dyn['positive']}/32、负{dyn['negative']}/32；完整total投影正{total['positive']}/32、负{total['negative']}/32。

value/dynamics在value参数上的梯度夹角cos均值{s['gradient_cosine/value_dynamics']['mean']:.6g}；各自梯度范数均值{s['gradient_norm/value']['mean']:.6g}、{s['gradient_norm/dynamics']['mean']:.6g}。梯度整体夹角、范数与特定probe投影是不同测量，不能相互替代。

## 实际官方Adam反事实

{chr(10).join(adam)}

joint直接调用原生agent.update，actor/dynamics也更新；本表只读取更新后的value场，不将joint称为冻结actor实验。zero-new-gradient、仅旧Adam矩产生的相对上坡变化为{data['zero_new_gradient_adam_probe_delta'][1]:.8g}。完整联合相对value-only的配对差为正{paired['positive']}/32、负{paired['negative']}/32。Adam预条件与动量是非线性的，不能把此差值当作可加的“动力学Adam贡献”。

两组实际变化符号不一致的批次数为{int(np.sum(np.sign(a)!=np.sign(b)))}/32。相较value-only每批绝对变化均值{float(np.abs(a).mean()):.6g}，联合与value-only的平均差仅{paired['mean']:.6g}。因此在当前probe的一步更新上，动力学影响较小；这既不支持其在此处占主导，也不能排除其在整个训练历史中长期累积的作用。

![完整梯度、实际Adam与配对变化](full_gradient.png)

## 对当前研究的约束

M1只测value_loss的表述应继续保留；M2补测了官方联合梯度，但仍是原模型附近的单步局部证据。特定损失在这个probe上朝不利方向，并不足以证明该损失应被移除；动力学也承担策略提取所需的建模功能。不会据此修改正在进行的E2、MQE或官方基线配置。

距离场为何导致失败的执行机制已有独立图和真实后继控制支持；训练为何形成此场的历史解释仍需结合固定预算终点、完整官方预算及独立seed核验。即使某次SGD投影向好，M1/E1已经表明也不能外推长期通道恢复。稳定改善、其他任务不退化以及actor利用新critic仍待完成。

源码及固定协议见[full_gradient.py](../../../experiments/ogbench_critic/full_gradient.py)与[FULL_GRADIENT_PROTOCOL.md](../../../experiments/ogbench_critic/FULL_GRADIENT_PROTOCOL.md)。原始逐批数据位于本目录analysis.json，成功运行日志位于同级full_gradient_fp32.log；仅源码、测试、协议、报告及引用PNG进入Git。
'''
    (output/'report.md').write_text(report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(a.output)
