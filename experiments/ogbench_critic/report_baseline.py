"""Create the reproducible baseline recheck report and its only committed figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def run(root):
    root=Path(root)
    actor=json.loads((root/'baseline_actor/summary.json').read_text())
    greedy=json.loads((root/'baseline_greedy/summary.json').read_text())
    field=json.loads((root/'baseline_field/summary.json').read_text())
    assert actor['bank_sha256']==greedy['bank_sha256']
    data=np.load(root/'baseline_field/field.npz')
    output=root/'baseline_recheck'
    output.mkdir(exist_ok=True)
    nodes=data['nodes'];walls=data['walls'];path=data['path']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for axis,values,title in zip(axes[:2],[data['reference'],data['values']],['Graph steps to success region','Learned IQE distance to goal']):
        plot=axis.scatter(nodes[:,0],nodes[:,1],c=values,s=5,cmap='viridis',rasterized=True)
        fig.colorbar(plot,ax=axis,shrink=.8)
        for x,y,hx,hy in walls:
            axis.add_patch(Rectangle((x-hx,y-hy),2*hx,2*hy,color='#39414c'))
        axis.plot(path[:,0],path[:,1],color='white',lw=1.4,label='Shortest graph route')
        axis.scatter(*field['anchor'],c='#f0525a',s=35,zorder=5)
        axis.scatter(*field['goal'],c='cyan',marker='*',s=80,zorder=5)
        axis.set(title=title,xlim=(-2,22),ylim=(-2,22),aspect='equal',xlabel='x',ylabel='y')
    ax=axes[2]
    ax.plot(np.arange(len(path)),data['path_values'],label='IQE distance',color='#bf3843')
    ax.set(xlabel='Steps along reference route',ylabel='IQE distance',title='Feasible progress can require an uphill field')
    secondary=ax.twinx()
    secondary.plot(np.arange(len(path)),data['path_costs'],label='Graph steps remaining',color='#286da8',linestyle='--')
    secondary.set_ylabel('Graph steps remaining')
    handles,labels=ax.get_legend_handles_labels();h2,l2=secondary.get_legend_handles_labels()
    ax.legend(handles+h2,labels+l2,fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'distance_field.png',dpi=180)
    plt.close(fig)
    table=['| 任务 | 冻结官方 actor | 真实后继贪心 |','|---|---:|---:|']
    for task in actor['tasks']:
        a=actor['tasks'][task];g=greedy['tasks'][task]
        table.append(f'| {task} | {a["successes"]}/{a["episodes"]} | {g["successes"]}/{g["episodes"]} |')
    pathrows=['| 路线步数 | 图参考剩余步数 | 网络距离 |','|---:|---:|---:|']
    for i in [0,10,15,20,30,40,50,60]:
        if i<len(path):pathrows.append(f'| {i} | {data["path_costs"][i]:.0f} | {data["path_values"][i]:.3f} |')
    text=f'''# E0：task3 距离陷阱复核与 critic 对照入口

日期：2026-10-29。分支：exp/ogbench。固定官方100k checkpoint的SHA256为397e919f2c453cee02ec840c615dce3532f31532fd639fbea6757c970fedbf0f。本项重新建立可复用的评价与训练工具，并核验基线；尚不包含A/B处理组终点结论。

## 已验证的基线结果

采用全部五任务、每任务5个固定扰动，保存完整初始integration state与目标。每个模型使用相同的库（SHA256 `{actor['bank_sha256']}`）。贪心控制每步枚举固定11×11动作，在完整MuJoCo状态恢复后实际执行候选并用critic评分，最终执行后继要求与枚举结果逐位一致。均使用官方成功条件和1000步TimeLimit。

{chr(10).join(table)}

这复现了task3的actor失败。其他任务中若贪心控制失败，也应视作诊断工具和距离场的额外限制；不能把这组结果混同于官方actor成功率。5个扰动仅供机制检查，不足以宣称统计稳定。

## 距离场与真实可达性

在用户给出的固定锚点(10.1156,14.6827)、目标(4.2055,12.0898)上，以环境墙体和pointbody半径{field['radius']}构建0.2间距的独立图。边检查采用线段到矩形的精确最小距离，不只检查端点；没有训练使用的oracle标签。

图含{field['nodes']}个可行节点、{field['edges']}条有向边；锚点吸附至{field['anchor_node']}，偏移{field['snap_distance']:.4f}。图参考剩余成本为{field['reference_steps']:.0f}步，参考路线在真实MuJoCo中{field['reference_replay_steps']}步成功，最大节点跟踪误差{field['reference_replay_max_tracking_error']:.3g}。这是离散图近似，不能称为连续环境严格最优成本。

网络距离起值{field['anchor_value']:.3f}；搜索全部图路线后的最低峰值为{field['minimum_peak']:.3f}，仍须上升{field['minimum_peak_gap']:.3f}（整图距离IQR的{field['minimum_peak_gap_over_iqr']:.3f}倍）。锚点没有严格下降通道。最短路线中的最低峰值也为{field['shortest_route_peak']:.3f}，不是挑选任意成功路线后判断距离错误。

{chr(10).join(pathrows)}

![独立参考图、网络距离与参考路线上的距离变化](distance_field.png)

全图参考步数与网络距离Spearman={field['all_graph_spearman']:.3f}；出口走廊区域={field['corridor_spearman']:.3f}。全图下降可达比例={field['strict_descent_reachable_fraction']:.3%}。这些整体统计不能掩盖当前锚点的局部排序障碍，也不直接揭示训练为什么产生障碍。网络距离未严格校准到环境步数，主要检查方向排序、下降可达性和执行。

## 控制实现与测试

- A逐字调用官方value_loss；损失及全参数梯度与官方实现相等测试通过。
- B只替换平均平方正部约束的残差；实测其梯度到达d(s,g)和d(s_next,g)两个端点，且公式依旧允许目标距离上坡。
- 使用非零预训练动量测试：两组更新value，actor/dynamics参数和Adam一二阶矩均逐位不变，避免零梯度但旧动量仍移动参数。
- 测试了准度量三角关系下goal残差不大于pair残差；扫掠边不能穿墙角；全部路线minimax与最短路线minimax区分正确。
- 在墙角接触态，交替候选后恢复完整mjData和TimeLimit，actor动作后继与原后继精确一致；候选枚举被选动作的真实执行后继亦精确一致。
- 共7项测试通过；另以真实100k模型完成配对两次更新、冻结检查和checkpoint保存烟雾。没有修改官方源码和原始数据。

## 已预注册的下一项实验

E1采用相同100k模型和Adam状态，pair/goal两组各额外10k，训练采样seed2001，同一批次和目标。actor/dynamics冻结，固定终点；每500步记录两类held-out残差，5k和10k保存。B是更弱的约束集合和不同梯度，不能声称原目标不变。

E1已启动，未完成。协议见仓库experiments/ogbench_critic/PROTOCOL.md。完成后用同一评价库和参考图评价110k终点，先比较下降通道与真实后继贪心执行，再决定是否进入多步。没有根据中途约束统计修改预算、学习率、容差或权重。后续若进入多步，需要纳入[MQE官方工作](https://mqe-paper.github.io/)，不将多步本身作为新颖贡献。

本项只确认原始陷阱可复核、实验控制可信；“为何学出该距离场”和“怎样稳定改善”的总目标仍未完成。原始数据、检查点、NPZ轨迹、JSON/CSV、日志及其他产物均不提交，仅提交源码、测试、协议、说明、本报告和引用PNG。
'''
    (output/'report.md').write_text(text)
    print(output/'report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();run(args.root)
