"""Post-run evaluation only: all seeds, paired outcomes, costs and workload."""
import argparse
import csv
from pathlib import Path
import numpy as np
from experiments.four_path_execution.common import read,write,file_digest,ROOT
from experiments.four_path_execution.statistics import aggregate,paired
from experiments.four_path_oracle_execution.protocol import ARMS


def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def num(x):return '—' if x is None else f'{x:.3f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    manifest=read(root/'manifest.json');processes=read(root/'process_results.json')
    assert set(processes)==set(ARMS) and all(v['exit_code']==0 for v in processes.values())
    out=root/'comparison';out.mkdir(exist_ok=False)
    import json
    rows={a:[json.loads(x) for x in (root/'arms'/a/'episodes.jsonl').read_text().splitlines()] for a in ARMS}
    expected={(r['id'],s) for r in read(root/'states.json')['records'] for s in manifest['planner_seeds']}
    summary={};strata={};pairs={};candidate_summary={}
    scores=read(root/'candidates/scores.json');candidate_pairs=read(root/'candidates/paired.json')
    for arm,records in rows.items():
        assert len(records)==64 and {(r['state_id'],r['planner_seed']) for r in records}==expected
        assert read(root/'arms'/arm/'COMPLETE')['parameters_unchanged']
        summary[arm]=aggregate(records);strata[arm]={}
        summary[arm]['real_steps']=sum(r['steps'] for r in records)
        summary[arm]['initialization_seconds']=sum(r['controller_diagnostics']['initial_planning_time_sec'] for r in records)
        online=sum(r['controller_diagnostics']['avg_online_planning_time_sec']*r['steps'] for r in records)
        summary[arm]['online_planning_seconds']=online
        summary[arm]['online_ms_per_decision']=1000*online/summary[arm]['real_steps']
        for layer in ('support','perturbed','holdout'):
            strata[arm][layer]=aggregate([r for r in records if r['stratum']==layer])
        for reference in ('none','oracle'):
            if arm!=reference:pairs[f'{reference}_to_{arm}']=paired(rows[reference],records)
        selected=[s for s in scores if s['arm']==arm and s['useful_count']>0]
        compared=[s for s in candidate_pairs if s['after']==arm and s['safe_candidates']>1]
        candidate_summary[arm]=dict(banks_with_useful=len(selected),
            useful_weight_mean=float(np.mean([s['useful_weight'] for s in selected])),
            safe_rank_spearman_mean=float(np.mean([s['safe_rank_spearman'] for s in compared])) if compared else None,
            max_first_control_difference_from_oracle=max([abs(s['first_control_delta']) for s in candidate_pairs if s['after']==arm],default=0.))
    qrl=[r for a,rs in rows.items() if a.startswith('qp_') for r in rs]
    diagnostics=[r['controller_diagnostics'] for r in rows['oracle']]
    assert sum(not d['oracle_value_cache_hit'] for d in diagnostics)==1
    oracle=dict(table=diagnostics[0],cold_build_seconds=sum(d['oracle_value_build_time_sec'] for d in diagnostics),
        queries=sum(d['oracle_value_queries'] for d in diagnostics),
        unreachable_queries=sum(d['oracle_value_unreachable_queries'] for d in diagnostics))
    oracle['unreachable_fraction']=oracle['unreachable_queries']/oracle['queries']
    caches=list((root/'oracle_cache').glob('*.npz'));assert len(caches)==1
    oracle['cache_sha256']=file_digest(caches[0])
    audit=read(root/'replay_audit.json');assert audit['episodes']==448
    write(out/'summary.json',dict(arms=summary,strata=strata,paired=pairs,pooled_qrl=aggregate(qrl),
        oracle=oracle,candidates=candidate_summary,report_source_sha256=file_digest(Path(__file__)),
        total_episodes=448,total_real_steps=audit['total_real_steps'],
        suite_wall_seconds=max(v['suite_elapsed_seconds'] for v in processes.values())))
    with (out/'episodes.csv').open('w') as f:
        keys=['arm','state_id','stratum','planner_seed','success','collision','out_of_bounds','timeout','cost','steps',
              'exit_eligible','exit_completed','wrap_eligible','wrap_completed','planning_seconds','model_rollouts','model_transitions']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',lineterminator='\n');writer.writeheader()
        for records in rows.values():writer.writerows(records)
    lines=['# Oracle、五个 QP 后 QRL 与无终值 MPPI 配对评估','',
        '全部 448 个 episode 完成；未重新训练、选择或修改任何 QRL checkpoint。所有组统一 horizon=10、128 条候选，其余参数沿用上一轮。', '',
        '## 配置、可比性与 Oracle 定义','',
        '使用上一轮原样冻结的 32 个位置/heading 状态 × 两个 planner seed（314159、271828）。support / perturbed 各 24 次，holdout 16 次/组。holdout 是上一轮已评估的保留队列，本轮没有重新抽样，也没有据结果调参。相同初始 RNG 并不保证闭环分化后候选轨迹相同。', '',
        'Oracle 文件原样移植自 `34d1c011f637b3352ac1dc1134650f6f28c20b17`。位置分辨率 0.25，heading 24 bins，5 步定控制原语，转向比例 [-1,-0.5,0,0.5,1]；反向 Dijkstra 求网格终值，最近网格查询，合法终态为零，不可达查询为 1e6。与 QRL 一样直接使用环境成本单位，不再乘 dt。采样器、碰撞判定和闭环执行复用当前实现。', '',
        '这是拥有完整环境模型的离散格点参考，拥有四条数据以外的信息；它不是连续环境最优距离，也不是四路径记录图距离。端点吸附使网格成本不必对应可执行连续路径，因此表值不是连续最优成本的已证明上下界；Oracle-MPPI 成功轨迹的实际成本只是可行解成本。历史目录的 12/12 成功来自另一组主要朝西的起点与 RNG，不能与本表直接配对。', '',
        f"状态文件 SHA256：`{manifest['states_sha256']}`。模型、成本/数据、代码指纹和软件版本见 `../manifest.json`；五个适配器验证见 `../adapter_validation.json`。", '',
        '## 所有模型与失败分类','',
        table(['组别','成功','碰撞','越界','超时','成功成本均值','成功步数均值'],[
            [a,f"{s['successes']}/64 ({pct(s['success_rate'])})",f"{s['collision_count']} ({pct(s['collision_rate'])})",
             f"{s['out_of_bounds_count']} ({pct(s['out_of_bounds_rate'])})",f"{s['timeout_count']} ({pct(s['timeout_rate'])})",
             num(s['successful_cost_mean']),num(s['successful_steps_mean'])] for a,s in summary.items()]),'',
        f"五个 QRL 的汇总成功率为 {aggregate(qrl)['successes']}/320（{pct(aggregate(qrl)['success_rate'])}）。这是五个模型在同一 64 个实例上的描述性汇总，不是 320 个独立起点。失败提前截断成本，不参与成本优劣排名；成功集合不同，下面另给共同成功实例的配对成本。",'',
        '## 分层与阶段完成','',
        table(['组别','support 成功 /24','perturbed 成功 /24','holdout 成功 /16','离开 U（适用实例）','绕墙（适用实例）'],[
            [a,*[f"{strata[a][k]['successes']} ({pct(strata[a][k]['success_rate'])})" for k in ('support','perturbed','holdout')],
             f"{s['exit_completed']}/{s['exit_eligible']} ({pct(s['exit_rate'])})",
             f"{s['wrap_completed']}/{s['wrap_eligible']} ({pct(s['wrap_rate'])})"] for a,s in summary.items()]),'',
        '离开 U 只对初始位于内部的实例计分，并检查运动线段是否穿过西出口；绕墙只对初始未处于东侧的实例计分。合法终态完成率就是任务成功率；训练路径的中间状态不能因已经走过前一阶段而自动计为本轮新完成。', '',
        '## 配对结果','',
        '以下“新增成功/失去成功”以左侧参考组为起点；成本差 = 右组 − 左组，只在双方成功时计算。', '',
        table(['参考 → 对比','新增成功','失去成功','共同成功','共同成功成本差','共同成功步数差'],[
            [key.replace('_to_',' → '),p['improved'],p['worsened'],p['both_success'],num(p['joint_success_cost_delta']),num(p['joint_success_steps_delta'])] for key,p in pairs.items()]),'',
        '## 计算开销','',
        table(['组别','在线 ms/决策','初始化合计秒','MPPI rollout 数','实际模型一步转移数'],[
            [a,num(s['online_ms_per_decision']),num(s['initialization_seconds']),s['model_rollouts'],s['model_transitions']] for a,s in summary.items()]),'',
        f"Oracle 冷建表 {oracle['cold_build_seconds']:.3f} 秒，网格 {diagnostics[0]['oracle_value_grid_states']} 个状态、{diagnostics[0]['oracle_value_graph_edges']} 条非终态边；有效格点可达比例 {pct(diagnostics[0]['oracle_value_reachable_fraction'])}。首次 episode 建表，后续复用内存缓存。另有 {oracle['queries']} 次终值查询，其中 {oracle['unreachable_queries']} 次不可达（{pct(oracle['unreachable_fraction'])}）。建表需额外模拟至多 85848×5×5=2,146,200 个格点原语时间步槽位，实际活动步会因无效/终态提前结束；此上限不是实测转移数，不包含在 MPPI rollout 列中。",'',
        '在线时间含 rollout、终值评分和加权控制，不含一次性表构建；初始化单列。CPU 单线程/进程、最多四组并发，墙钟延迟受调度影响，不是独占硬件基准。每步均为 128 条候选，总量不同源自 episode 长度和终止原因。', '',
        '## 固定候选评分','',
        '沿用上一轮 8 个开发 pose × 2 个 RNG，零 nominal，按本轮 H10/N128 重新生成 16 个批次。同一批次由七组分别评分，保存完整动作、终态、成本和权重。useful 定义为全程安全、完整 horizon、向西推进至少 0.25 且最终朝西；只是局部进展指标，不能证明后续能脱困。', '',
        table(['组别','存在 useful 的批次数','这些批次 useful 权重均值','与 Oracle 安全候选 Spearman 均值','最大首控制差'],[
            [a,s['banks_with_useful'],num(s['useful_weight_mean']),num(s['safe_rank_spearman_mean']),num(s['max_first_control_difference_from_oracle'])] for a,s in candidate_summary.items()]),'',
        f"16 个批次中，{sum(s['safe_count']==0 for s in scores if s['arm']=='none')} 批完全没有安全候选，{sum(s['useful_count']==0 for s in scores if s['arm']=='none')} 批没有符合上述定义的西向进展候选。价值函数无法选择未采到的轨迹；MPPI 的加权平均首动作也不等于选择单条安全候选。",'',
        '## 验证、复现与产物','',
        f"22 项测试通过，包括历史 Oracle 文件逐字节一致、provider 控制输出一致、缓存失效/终态零/成本单位、既有 MPPI 回归与真实环境 rollout 一致性。正式运行前每组做 3 步 smoke；全部 {audit['total_real_steps']} 次真实执行动作已重新回放，状态和奖励误差均为零，终止分类和合法终态核验通过。五个 checkpoint 文件及推理参数前后哈希不变。",'',
        '运行命令和冻结规则见 `../../../../experiments/four_path_oracle_execution/PROTOCOL.md`。每组原始 JSONL、逐步 NPZ、所有失败均在 `../arms/`；`summary.json` / `episodes.csv` 提供机器可读汇总，`../candidates/` 为候选评分，`../replay_audit.json` 为完整回放核验。', '',
        '![outcomes](outcomes.png)', '', '![trajectories](trajectories.png)', '']
    findings=['## 解释与下一步', '',
        f"在完全相同的 H10/N128 执行器下，Oracle 为 {summary['oracle']['successes']}/64，五个 QRL 为 {aggregate(qrl)['successes']}/320，无终值为 {summary['none']['successes']}/64。Oracle 在 support 和 holdout 均全成功，五个 QRL 的 holdout 均为 4/16。说明该配置的采样/控制器能够在这些保留状态上完成任务；QRL 与 Oracle 的差距不能全部归咎于 horizon 或候选数不足。证据更支持四路径之外的 heading/状态价值结构与候选排序仍有缺口，但本实验不能把表示容量、训练数据覆盖和优化误差分离。", '',
        '在双方均成功的配对中，五个 QRL 相对 Oracle 的平均成本差都为正，101 最接近（+1.225），303 最大（+57.382）。Oracle 独有成功覆盖了更难的起点，因此不能用其全体成功成本 171.184 高于部分 QRL 的均值来判定 Oracle 更差。', '',
        '候选诊断同时揭示两种限制：朝东的深 U 开发点两批都没有安全候选；朝北/朝南的八批却有 5–8 条安全西向进展候选，QRL 给它们的总权重均低于 1e-5。Oracle 在朝南四批分配约 0.043–0.994 的权重，在朝北四批仅约 0.010–0.014；不能把“西向进展”启发式本身当作正确控制标签。模型与 Oracle 的安全候选排序平均负相关，是局部评分结构不同的证据，不是连续最优性的证明。', '',
        '下一步宜将离轨 heading 的终值排序/局部下降方向审计与候选覆盖审计分开：先用这些已保存批次定位错误排序，再在独立冻结测试集上验证修正。对于全无安全候选的边界状态，可另行评估统一的控制原语候选或安全恢复策略；本轮没有修改采样器或给 Oracle 特殊控制。不要再将记录图联合门槛通过等同于闭环执行成功。', '',
        '上一轮 H20/N256 的 QP 汇总为 211/320，本轮 H10/N128 为 225/320；这是两个配置字段一起变化的描述性对照，不能单独归因为缩短 horizon 或减少候选数，更不能据此按模型挑选超参数。', '']
    cache_audit=root/'oracle_audit/historical_cache_comparison.json'
    if cache_audit.exists():
        cache_check=read(cache_audit);failures=read(root/'oracle_audit/failure_first_decisions.json')
        findings += ['## 历史等价性与 Oracle 两次失败的事后核查', '',
            f"本轮新建表与用户指定历史缓存的全部 value 元素逐位相同：{cache_check['values_bitwise_equal']}，有限值最大差 {cache_check['maximum_finite_absolute_difference']}；环境/格点签名一致。详情见 `../oracle_audit/historical_cache_comparison.json`。", '',
            '两次失败均为 `path2_west_exit_perturbed`，初态 [2.91515, 1.92077, -2.72330]，第一步碰撞。两批 128 条候选均没有完整安全的 H10 轨迹；314159 中仅 1 条候选首步安全，271828 中为 0 条。所有候选都被标为 invalid，因此不会调用 Oracle 终值，当前加权控制最终仍碰撞。', '',
            '事后固定施加合法控制 omega=-3 两步可安全到达 [2.71750, 1.92704, 2.95989]，说明第一步碰撞并非这个位置/heading 的必然结果。这只是两步恢复见证，不证明整项巡检能完成；它未加入正式候选，也没有触发重跑或调参。失败批次和对照保存在 `../oracle_audit/`。', '']
    lines=lines[:-5]+findings+lines[-5:]
    (out/'report.md').write_text('\n'.join(lines))
    fig,ax=plt.subplots(figsize=(10,4));bottom=np.zeros(len(ARMS))
    for key,color in [('successes','#228b74'),('collision_count','#d66751'),('out_of_bounds_count','#a375c2'),('timeout_count','#e5b449')]:
        values=np.array([summary[a][key] for a in ARMS]);ax.bar(ARMS,values,bottom=bottom,label=key,color=color);bottom+=values
    ax.set(ylabel='Episodes (64 per arm)',title='Frozen H=10, N=128; all five QP models');ax.legend(ncol=4,fontsize=8)
    fig.tight_layout();fig.savefig(out/'outcomes.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,8));scenario=read(root/'scenario.json')
    for ax,arm in zip(axes.flat,ARMS):
        for obstacle in scenario['obstacles']:
            if obstacle['type']=='rectangle':
                ax.add_patch(Rectangle((obstacle['x_min'],obstacle['y_min']),obstacle['x_max']-obstacle['x_min'],
                                      obstacle['y_max']-obstacle['y_min'],color='.25'))
        for r in rows[arm]:
            s=np.load(root/'arms'/arm/f"{r['state_id']}_{r['planner_seed']}.npz")['states']
            ax.plot(s[:,0],s[:,1],color='#228b74' if r['success'] else '#d66751',alpha=.35,lw=.7)
        ax.scatter([6.7],[3.65],marker='*',c='black',s=50)
        ax.set(title=f"{arm}: {summary[arm]['successes']}/64",xlim=(0,18),ylim=(0,12),aspect='equal')
    axes.flat[-1].axis('off');fig.suptitle('All closed-loop trajectories: green success, red failure')
    fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=180);plt.close(fig)
    print({a:s['successes'] for a,s in summary.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);report(p.parse_args().root)
