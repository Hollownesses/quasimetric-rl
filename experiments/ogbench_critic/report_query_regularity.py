"""File-only diagonal-limit proof checks and fixed off-diagonal responses."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from report_iqe_distill import read,sha


def run(root):
    root=Path(root).resolve();out=root/'query_regularity';a=read(out/'analysis.json')
    assert a['complete'] and a['elapsed_seconds']<a['time_limit_seconds']==60 and a['training_updates']==a['new_episodes']==0
    assert not (out/'failure.json').exists()
    assert sha(root/'local_shape_1000/model/params_101000.pkl')==a['student_sha256']
    assert sha(root/'roadmap_metric/graph.npz')==a['teacher_graph_sha256'] and sha(root/'roadmap_metric/distances.npy')==a['teacher_distance_sha256']
    assert sha(root/'evaluation_bank.npz')==a['bank_sha256']
    for name,digest in a['source_sha256'].items():
        directory=Path(__file__).parents[1]/'ogbench_diagnostics' if name=='core.py' else Path(__file__).parent
        assert sha(out/'source_snapshot'/name)==digest==sha(directory/name)
    with np.load(out/'diagonal.npz') as f:arrays={k:f[k].copy() for k in f.files}
    e=arrays['epsilon'];limit=arrays['expected_limit'];stable=arrays['stable_neighbors'];alias=arrays['float32_alias']
    assert arrays['points'].shape==(76,2) and len(a['point_labels'])==76 and len(a['responses'])==11
    assert np.isfinite(limit).all() and (limit>0).sum()==a['positive_limit_points'] and stable.sum()==a['stable_points']
    np.testing.assert_allclose(arrays['teacher'][stable,-1],limit[stable],rtol=0,atol=1e-4)
    flat=[v for pair in a['responses'].values() for v in pair.values()]
    response={k:np.max([v[k] for v in flat],axis=0) for k in ['teacher_max_abs_change','original_max_abs_change','student_max_abs_change']}
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for key,label in [('teacher','Abstract graph teacher'),('original','Original QRL'),('student','E13 neural')]:
        data=arrays[key];axes[0,0].plot(e,np.median(data,axis=0),marker='o',label=label)
    axes[0,0].fill_between(e,arrays['teacher'].min(axis=0),arrays['teacher'].max(axis=0),alpha=.15,color='C0',label='Teacher min--max')
    axes[0,0].set(title='76 fixed points: d(x, x + epsilon u)',xscale='log',xlabel='epsilon',ylabel='Distance');axes[0,0].invert_xaxis();axes[0,0].legend(fontsize=8)
    axes[0,1].scatter(limit,arrays['teacher'][:,-1],s=15);m=float(limit.max());axes[0,1].plot([0,m],[0,m],color='gray',ls=':')
    axes[0,1].set(title='All 76 neighbor sets stable at 1e-7',xlabel='Predicted limit: 2 min attachment cost',ylabel='Actual teacher distance at 1e-7')
    for key,label in [('teacher_max_abs_change','Teacher'),('original_max_abs_change','Original QRL'),('student_max_abs_change','E13')]:
        axes[1,0].plot(e,np.maximum(response[key],1e-8),marker='o',label=label)
    axes[1,0].set(title='11 fixed non-diagonal queries, all four directions',xscale='log',yscale='log',xlabel='epsilon',ylabel='Max |d_perturbed - d_current|');axes[1,0].invert_xaxis();axes[1,0].legend(fontsize=8)
    axes[1,1].plot(e,alias.sum(axis=0),marker='o');axes[1,1].set(title='Effective FP32 inputs can alias',xscale='log',xlabel='epsilon',ylabel='Identical float32 input pairs / 76',ylim=(-1,77));axes[1,1].invert_xaxis()
    fig.savefig(out/'query_regularity.png',dpi=160);plt.close(fig)
    table='\n'.join(f"| {name} | {v['source']['teacher_current']-v['source']['student_current']:.4f} | {v['source']['teacher_max_abs_change'][-1]:.2e} | {v['goal']['teacher_max_abs_change'][-1]:.2e} | {v['source']['changed_neighbors'][-1]} / {v['goal']['changed_neighbors'][-1]} |" for name,v in a['responses'].items())
    report=f'''# M5：新增教师存在对角跳变，固定远距离查询未发现相同跳变

只读检查用时{a['elapsed_seconds']:.2f}秒、训练/episode均0。76个预定点全部有正对角极限（0.11504–10.63807，中位{np.median(limit):.5f}），且最小epsilon处8邻居集合全部稳定，预测极限与实际误差最大{np.max(np.abs(arrays['teacher'][stable,-1]-limit[stable])):.2e}。因此E10抽象查询教师不是欧氏坐标下连续的距离标签，不能要求连续神经距离在全域一致拟合它。

但另11个实际非对角查询，在四个坐标方向epsilon=1e−7处教师变化最大仅{response['teacher_max_abs_change'][-1]:.2e}、邻居集合未变。两个失败终点的教师−学生差仍约17–18个距离单位，不能将这些远距离误差归因于已证明的对角跳变。此检查修正教师可拟合性边界，未解决或证明原QRL的唯一训练根因。

## 构造与解析证据

E10对查询点x连接原图的8个代价为b_i=a_i+delta。不同查询必须经过原图，故在邻居集合稳定、y→x且y≠x时：lim t(x,y)=min_ij[b_i+Dij+b_j]=2 min_i b_i。Dij≥|b_i−b_j|给出至少2 min b的下界，i=j选最小b取到等号。实现却在相同完整观测时直接取t(x,x)=0。

正极限表示每个连续坐标查询被当成带正连接代价的独立辅助图顶点；这是抽象图度量的合法语义，不能当作连续物理最优距离。连续且自距离为0的神经距离无法在该邻域将一致误差压到小于min b；这不排除随机对分布上的期望MSE很小，不否定已观察到的25/25教师执行。

两项合成测试通过：稳定连接的正对角极限与同点零，及不同float64查询映射到相同float32输入却有不同教师标签。76点来自既有固定验证批次前32 observations/前32 value_goals、canonical锚点/目标、五个既有task3起点/目标，无按结果挑点。验证seed9001只是复现原批次，SHA {a['validation_sha256']}同E13；没有新训练seed或重做覆盖审计。

![对角极限、解析核验、非对角响应和FP32别名](query_regularity.png)

epsilon固定1e−2..1e−7，方向固定(1,1/sqrt(2))。原100k/E13的近对角距离随epsilon下降；最小epsilon处E13最大约{arrays['student'][:,-1].max():.2e}，其中{alias[:,-1].sum()}/76对有效float32输入已经相同。数值平台/零值有FP32别名及归约影响，不能当解析梯度证明；图中非对角响应仅作有限扰动检查。

## 与实际失败分开解释

固定canonical锚点、五个task3起点、五个E13结束观测，共11对(s,g)，分别扰动source/goal沿±x/±y四方向。所有查询和所有epsilon保留，不找最大跳变另加点；无新episode或参考墙体/路线输入。

| 固定查询 | 当前教师−E13 | 最小epsilon source响应最大值 | goal响应最大值 | 邻居变化方向数 source/goal |
|---|---:|---:|---:|---:|
{table}

这些非对角输入在最小扰动下稳定，不能从“教师对角不连续”跳到“两个task3停滞完全由教师跳变造成”。也不能据四方向有限检查证明全域连续或排除别处连接边界。之前报告称教师仅抽象先验、非连续物理真值的限制现在有直接反例；三角性、全图相关及随机拟合误差仍不是局部动作排序的充分证据。

## 下一步与产物

停止把完整教师当可全域一致逼近的真值，不扩大原MSE/形状预算或地标数搜索。现有随机对低RMSE与固定远距离大误差的缺口仍需处理；下一项可另冻结对非对角全局监督的目标分布均衡检查，而不是反复修改局部约束或只追一个锚点。任何教师修改/训练须另立协议，不能用本检查给新标签预先背书。

原100k与E13完整参数摘要/文件只读不变，模型SHA {a['baseline_sha256']} / {a['student_sha256']}；图/全距离/源码/bank身份已核对。协议[QUERY_REGULARITY_PROTOCOL.md](../../../experiments/ogbench_critic/QUERY_REGULARITY_PROTOCOL.md)，测试test_query_regularity.py，入口query_regularity.py <根目录> --output <全新目录>，本报告report_query_regularity.py只读重建。raw results/ogbench_critic_20261029/query_regularity，日志同级.log；只提交代码/测试/协议/报告与引用图，数组/JSON/日志不提交。不换seed、不运行1M，总目标仍未完成。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
