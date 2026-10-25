"""Render the committed narrative from the local, read-only diagnostic summary."""
import argparse
import hashlib
import json
from pathlib import Path


def render(s):
    runs=s['runs'];comparisons=s['comparisons'];sizes=s['group_definitions']
    lines=['# 源均衡新增违反：位置、路径暴露与乘子窗口诊断','',
           '本项基于已有R0/R1三个初始化及seed606 Rhalf，只读重算45k/50k模型，不新增训练或环境交互。'
           '新增违反与正残差增量主要落在低重复源/低副本边；没有发现分支三跳邻域或远端源的额外富集。'
           'R1降低路径regret的同时，固定R0路径上的累计正违反也增加。乘子确实在更新，仍未实现可行性；'
           '两个时间快照不足以将根因认定为dual追不上残差。','',
           '## 范围、定义和核验','',
           '冻结协议见 `experiments/constraint_exposure_diagnostic/PROTOCOL.md`。使用原42536物理记录、6958条'
           '完整source/target/action/cost去重边、6767节点和58个分支；参考是已修正的'
           '`connected_reference_fix_20261024`。远端为1692个去重源的最高距离四分位，参考仅用于事后分组。',
           '', '源重复数n(s)与边副本数m(e)分别统计。前者决定task权重，后者决定某条独特约束拥有几份记录及总访问。'
           '高重复源也可能存在只有一份记录的候选动作，二者不可混用。PHR仍保留逐记录乘子，未用代表行替代重复记录的dual。',
           '', '重新计算七组模型两个时间点的输出，50k预测与原评价数组逐位相同；与训练保存的残差最大差不超过'
           f"{max(v['scan_residual_max_difference'] for v in s['verification'].values()):.3g}。"
           '重复物理边的重算残差一致。全部165独立起点及307原始起点路径通过连通、真实成本、目标及regret恒等式核验。'
           '输入模型、replay、参考及决策文件运行前后SHA256不变，分析源码在运行中未变。','',
           '## 1. 低重复定位：贡献与富集分开看','',
           '|比较|新增违反边|消失违反边|全图新增比例|n(s)=1新增比例|m(e)=1新增比例|n(s)=1正残差净增贡献|m(e)=1正残差净增贡献|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,c in comparisons.items():
        g=c['groups'];total=c['sum_positive_delta']
        lines.append(f"|{name}|{c['new_edges']}|{c['vanished_edges']}|{100*g['all']['new_fraction']:.2f}%|"
                     f"{100*g['multiplicity_1']['new_fraction']:.2f}%|{100*g['edge_copies_1']['new_fraction']:.2f}%|"
                     f"{100*g['multiplicity_1']['sum_positive_delta']/total:.2f}%|"
                     f"{100*g['edge_copies_1']['sum_positive_delta']/total:.2f}%|")
    lines += ['',f"n(s)=1包含{sizes['multiplicity_1']}条边，占全部边的{100*sizes['multiplicity_1']/sizes['all']:.2f}%；"
              f"m(e)=1包含{sizes['edge_copies_1']}条边，占{100*sizes['edge_copies_1']/sizes['all']:.2f}%。"
              '所以不能只依据贡献大就宣称富集；本表同时给出新增比例。贡献使用各组正残差变化的有符号净和，'
              '某组可以为负，比例不是概率。新增违反阈值1e-5，使用1e-3的敏感性检查只变化少量边。',
              '',f"三个R1共有{s['shared_new_edges']}条新增违反边，三者并集{s['union_new_edges']}条。"
              '跨初始化稳定的是群体趋势，并非同一份逐边错误清单。',
              '', '每条物理记录最终访问299–300次，45k至50k追加29–31次。低副本边并非从未访问；'
              '它们的总约束访问少于拥有大量副本的边。只改task权重并未增加这些独特约束的总采样。',
              '具体而言，单副本边最终299–300次访问；副本数65+的边合计19463–50605次。'
              '这些访问并非同等数量独立数据，且每份副本的dual独立演化。',
              '', '## 2. 分支和远端：没有观察到额外集中','',
              '|分组|唯一边数|606新增比例|707新增比例|808新增比例|',
              '|---|---:|---:|---:|---:|']
    labels=[('all','全图'),('branch_outgoing','分支源出边'),('near_branch3','分支三跳邻域'),
            ('away_branch3','邻域外'),('far_source','远端源出边'),('far_shortest_paths','远端最短路径使用边')]
    for group,label in labels:
        values='|'.join(f"{100*comparisons[str(seed)]['groups'][group]['new_fraction']:.2f}%" for seed in (606,707,808))
        lines.append(f'|{label}|{sizes[group]}|{values}|')
    lines += ['', '三跳邻域采用无向记录图距离，边任意端落入邻域即计入；不是欧氏空间距离。'
              '远端最短路径使用边包含共享近目标后缀，不能与远端源出边等同。分组互相重叠，不作比例相加。'
              '三组R1的分支邻域、远端源和远端最短路径新增比例都低于各自全图；分支源出边的方向则不统一。'
              '因此当前证据不支持仅按“分支附近”或“远端”重采约束。',
              '', '![正残差变化的空间位置](localization.png)',
              '', '图中全部唯一边保留，以转移中点表示；红色为正残差增加，颜色固定截断到±1，'
              '更大的真实幅度保留在数值结果中。空间图只用于定位，不证明障碍或某区域导致优化失败。',
              '', '## 3. 实际路径暴露：约束退化没有被路线变化解释掉','',
              '累计正违反定义为每条路径沿边sum(max(h,0))，再对165个独立起点等权平均；它不是路径额外成本。'
              '以下固定R0路径只替换残差模型，另列平衡模型自身路径，控制路线变化的混淆。',
              '', '|比较|R0路径/R0残差|R0路径/平衡残差|平衡路径/平衡残差|R0路径regret|平衡路径regret|',
              '|---|---:|---:|---:|---:|---:|']
    for name,c in comparisons.items():
        a,b=c['base'],c['arm']
        lines.append(f"|{name}|{c['R0_paths_R0_h']:.4f}|{c['R0_paths_arm_h']:.4f}|{c['arm_paths_arm_h']:.4f}|"
                     f"{runs[a]['unique_start_regret']:.4f}|{runs[b]['unique_start_regret']:.4f}|")
    lines += ['', '四个比较的固定R0路径累计正违反全部增加。因此约束退化并非只是选择了不同路线造成的统计变化。'
              '同时三个R1路径regret下降，说明局部违反总量与实际成本损失不是同一个量。',
              '', '高暴露示例（按本模型路径暴露×正违反排名，表中含原本已违反的边，不限新增违反）：',
              '', '|模型|记录行/source→target|位置(x,y)|源重复/边副本|R0→本模型路径经过次数|h45→h50|lambda45→lambda50（副本均值）|',
              '|---|---|---|---|---|---|---|']
    for label,c in comparisons.items():
        for e in c['top_edges'][:2]:
            lines.append(f"|{c['arm']}|{e['row']}/{e['source']}→{e['target']}|({e['xy'][0]:.3f},{e['xy'][1]:.3f})|"
                         f"{e['source_count']}/{e['copies']}|{e['R0_exposure']}→{e['arm_exposure']}|"
                         f"{e['h45']:.4f}→{e['h50']:.4f}|{e['lambda45_mean']:.2f}→{e['lambda50_mean']:.2f}|")
    lines += ['', '例如source484虽有63份源记录，但所选出边row22365只有1份副本。'
              '不能由“该源重复多”推断它的全部动作约束都获得了大量访问。',
              '', '每组最大违反的位置及实际暴露（与高暴露排名分开）：',
              '', '|模型|记录行/source→target|位置(x,y)|h50|本模型路径经过次数/165|近分支/远端源|lambda50副本均值|',
              '|---|---|---|---:|---:|---|---:|']
    for c in comparisons.values():
        e=max(c['top_edges'],key=lambda item:item['h50'])
        lines.append(f"|{c['arm']}|{e['row']}/{e['source']}→{e['target']}|({e['xy'][0]:.3f},{e['xy'][1]:.3f})|"
                     f"{e['h50']:.4f}|{e['arm_exposure']}|{e['near_branch']}/{e['far_source']}|{e['lambda50_mean']:.2f}|")
    lines += ['',
              '', '## 4. 成本损失的直接来源：高暴露错误分支及后继误差差分','',
              '逐路径核验：sum[c+J*(next)−J*(state)] = 路径成本−J*(start)。'
              '因此以下贡献是可加的成本分解；与局部违反的相关性不能替代它。',
              '', '|模型|分支源|经过次数/165|单次regret|平均路径贡献|所选边h|所选动作相对参考最优动作的预测分数差|后继预测误差差分|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('606_R1','707_R1','808_R1','606_Rhalf'):
        for b in runs[name]['branches'][:3]:
            lines.append(f"|{name}|{b['source']}|{b['exposure']}|{b['regret']:.4f}|{b['contribution']:.4f}|"
                         f"{b['chosen_h']:.4f}|{b['predicted_choice_minus_reference_best']:.4f}|{b['successor_error_contrast']:.4f}|")
    lines += ['', '分数差恒等式为：预测分数差 = 真实动作regret + 两后继的预测误差差分。'
              '错误后继的误差优势可以抵消真实额外成本，让高regret动作以很小预测分数优势胜出。'
              '这是冻结输出上的精确代数分解，不证明某条边约束违反是这些后继误差的因果来源。',
              '', 'Rhalf虽降低总体误差和平均违反，其高频分支24、468、436仍产生显著成本。'
              '只修复最大违反或继续调alpha，都没有现成证据保证修复这些动作排序。',
              '', '## 5. 乘子窗口：确实在更新，但未得到可行解','',
              '以下群体仅指45k、50k两个端点均h>1e-5的物理记录；不是这5000步连续违反。'
              '每一组是在同一模型的相同记录子集上比较两时刻，跨模型子集不同。',
              '', '|模型|两端均违反记录数|其中lambda增加|其中lambda增加且h也增加|平均h45→h50|平均lambda45→lambda50|',
              '|---|---:|---:|---:|---|---|']
    for name,r in runs.items():
        w=r['dual45to50']['both_endpoints_positive'];p=w['persistent']
        lines.append(f"|{name}|{w['persistent_positive']}|{w['persistent_growing_dual']}|"
                     f"{w['persistent_growing_dual_worsening']}|{p['h45_mean']:.4f}→{p['h50_mean']:.4f}|"
                     f"{p['lambda45_mean']:.2f}→{p['lambda50_mean']:.2f}|")
    lines += ['', 'R1两端均违反群体的平均残差在三个seed中都下降，平均乘子都增加；'
              '但其中仍有数千条记录同时出现乘子增加、端点残差增加。它们的乘子和有效PHR系数并非为零。'
              '这排除了“这些约束根本没有访问/没有乘子响应”的简单解释；不区分dual步长、primal响应、'
              '共享参数冲突或非凸局部解。不能据此单独宣布乘子太小，也不能以乘子很大宣布它已经足够。',
              '', '完整结果另含effective coefficient=max(lambda+100*h,0)、mean(abs(lambda*h))和eta1投影dual残差。'
              '这些是标量约束层面的诊断，不等于参数梯度平衡或完整KKT证书。lambda净增量/访问数受投影影响，'
              '不是期间平均残差。1000步日志只有聚合量，不能恢复逐边时间序列。',
              '', '![路径暴露和乘子端点](exposure_dual.png)',
              '', '## 6. 下一轮该检验什么','',
              '当前定位支持优先检查唯一边副本数造成的约束经验测度失衡。若做训练干预，建议只引入一个明确的'
              '约束测度对照：固定R1 task/B1和记录采样，通过全局逆边副本权重调整PHR聚合，并显式规定逐记录dual时钟。'
              '它是新的增广目标/有限预算优化实验，不是数学上必须与task分布匹配；本次未实现或运行。'
              '不能同时切换采样、dual步长和rho后仍将收益归因于边均衡。',
              '', '评价至少包括全图/低副本边可行性、远端三角松弛、固定路径累计正违反及高暴露分支成本。'
              '如果可行性改善却破坏长程收益，应作为明确取舍报告；如果约束与距离改善但分支仍错，'
              '继续分析后继误差差分与未见状态执行。当前证据不支持先按分支地理邻域追加采样，'
              '也不支持简单加大dual或先叠加QP/Oracle标签。',
              '', '## 复现、测试与提交范围','',
              '本目录只读计算源码为 `experiments/constraint_exposure_diagnostic/{core,analyze,report}.py`。'
              '核心测试8项覆盖完整边身份、分支图距离、远端后缀暴露、独立起点去重、路径成本/advantage恒等式及错误拒绝、'
              '逐记录乘子与投影、分支误差差分、残差容差；参考及分组回归5项通过。'
              '参考/预测/残差/路径恒等式与输入不变性还在七组真实数据上核验。',
              '', '分析命令及报告生成命令见README。输出目录必须为新目录；源码和输入SHA见本地integrity.json，'
              '报告渲染SHA另见report_manifest.json。仅提交源码、测试、协议、README、本报告以及两张引用图。'
              '逐边数组、完整分支明细、JSON、checkpoint、日志、快照和首轮输出均保留本地。'
              '本项不改变训练代码、模型、数据或历史正式结果。']
    return '\n'.join(lines)+'\n'


def main(root):
    root=Path(root)
    if not json.loads((root/'integrity.json').read_text())['passed']:
        raise ValueError('diagnostic integrity check failed')
    source=root/'summary.json';raw=source.read_bytes()
    (root/'report.md').write_text(render(json.loads(raw)))
    (root/'report_manifest.json').write_text(json.dumps(dict(
        summary_sha256=hashlib.sha256(raw).hexdigest(),
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256((root/'report.md').read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    main(parser.parse_args().root)
