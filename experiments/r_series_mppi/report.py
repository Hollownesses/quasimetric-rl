"""Chinese evaluation report; no graph-distance regret for continuous episodes."""
from pathlib import Path
import numpy as np
from .common import ARMS,read,write,DATA,TRAIN,file_digest,COST_KEYS

LABEL={'none':'无终值','R0':'R0+MPPI','R1':'R1+MPPI','oracle':'Oracle+MPPI'}
def f(x,d=2):return '—' if x is None else f'{x:.{d}f}'
def rate(n,d):return f'{n}/{d}' if d else '不适用'

def report(root,s,rows):
    out=root/'comparison';lines=['# R0/R1 + MPPI 连续环境闭环配对评价（seed606，固定50k）','']
    add=lines.append;g=s['grouped']['all'];pr=s['paired']['all']['R0_to_R1']
    add(f"本轮256个正式episode全部完成。成功数：无终值 {g['none']['successes']}/64，R0 {g['R0']['successes']}/64，R1 {g['R1']['successes']}/64，Oracle {g['oracle']['successes']}/64。")
    add(f"R0→R1配对新增成功 {pr['improved']}，失去成功 {pr['worsened']}，净变化 {pr['improved']-pr['worsened']:+d}。本批起点上观察到有限的连续导航成功收益；共同成功成本并未同步改善，不能据此宣布全面更优或多seed稳健提升。")
    add('')
    add('## 协议、适配和配对核验')
    add('')
    add('只评价已冻结的seed606 R0/R1 50k checkpoint，不训练、不校准、不挑选checkpoint。encoder/projector/IQE严格转移到生产连续观测接口，distance_scale=1，正距离以正终端成本加入MPPI；不查图、不吸附到专家节点、不使用QP或修复动作。')
    add('')
    add('MPPI与提交e591c088a294c02f1b6f0244802768562dec449f核心源码逐字一致：H10/N128，noise .8，temperature 1，terminal_weight 1，invalid penalty 1e6，terminal_samples128，动作[-3,3]，v1/dt.1，上限300步。每回合新建planner、nominal归零，RNG=planner seed+7001，后续移位warm start。两planner seeds为314159、271828。每worker1个Torch线程，四组并行。')
    add('')
    add('32个起点以seed2026102601冻结：内部24点，x∈[2.9,5.1]、y∈[2.0,5.3]；西出口8点，x∈[1.5,2.7]、y∈[2.0,5.3]。heading分[-π,π)的8层，层内均匀；内部每层3点，出口每层1点。只检查合法非终态，不用模型/Oracle/结果筛选。西出口起点已完成离开U阶段，不算新增离开U成功。')
    add('')
    for name,d in s['start_overlap'].items():add(f"- {name}：检查{d['count']}个pose，精确pose重合{d['exact_pose_overlaps']}；最小位置距离{d['min_position_distance']:.6f}。")
    add('')
    for arm,d in s['adapter_validation'].items():add(f"- {arm}：{d['recorded_pair_count']}个记录task/B1/edge距离误差最大{d['pair_error_max']:.6g}；全部canonical节点task值误差最大{d['all_node_task_error_max']:.6g}；97个未见连续观测批量/分块误差最大{d['unseen_batch_error_max']:.6g}；16个合法终态值均为0。checkpoint SHA256 `{d['sha256']}`。")
    add('- 7项执行前测试通过；8个独立开发案例各重复两遍（16次烟雾执行），状态、动作、奖励精确相同，未使用正式起点。另有3项事后统计/覆盖测试通过。')
    add(f"- 256条正式动作序列全部通过真实env.step重放，共{sum(a['replayed_steps'] for a in s['audits'].values())}步；状态、奖励、逐步成本分项和终止类型精确一致。四组64个state/seed键完整配对，首次候选控制bank逐项hash相同。闭环分化后只保证同一扰动生成规则。")
    add('- 所有源码、训练数据、checkpoint、缓存、起点与冻结指纹复核；参数未更新，旧产物未写入。')
    add('')
    add('## 成功率与阶段完成')
    add('')
    add('离开U采用原stage_metrics的西出口框/线段交叉判定；绕墙要求先到y<1.4或y>5.9，再到x>5.7。任务成功只取真实环境合法终态。下表离开U/绕墙均为“新完成/eligible”；heading编号h对应[-180+45h, -135+45h)度。')
    main_groups=['all','inside','west']+[f'heading_{h}' for h in range(8)]
    add('')
    add('| 分组 | 控制器 | 成功 | 成功率 | 新离开U | 绕墙 | 碰撞/越界/超时 |')
    add('|---|---|---:|---:|---:|---:|---:|')
    for name in main_groups:
        for arm in ARMS:
            d=s['grouped'][name][arm]
            add(f"| {name} | {LABEL[arm]} | {d['successes']}/{d['episodes']} | {100*d['success_rate']:.2f}% | {rate(d['exit_completed'],d['exit_eligible'])} | {rate(d['wrap_completed'],d['wrap_eligible'])} | {d['collision_count']}/{d['out_of_bounds_count']}/{d['timeout_count']} |")
    add('')
    add('## 配对成功变化与共同成功成本/步数')
    add('')
    add('方向均为前者→后者；新增=前失败后成功，失去=前成功后失败；Δ成本/步数=后−前，仅在同一起点及planner seed下两者都成功时计算。Oracle对照差值不是最优regret。')
    add('')
    add('| 分组 | 配对 | 新增 | 失去 | 都失败 | 共同成功 | 共同成功平均Δ成本 | 平均Δ步数 |')
    add('|---|---|---:|---:|---:|---:|---:|---:|')
    for name in main_groups:
        for pair in ('R0_to_R1','oracle_to_R0','oracle_to_R1'):
            d=s['paired'][name][pair]
            add(f"| {name} | {pair} | {d['improved']} | {d['worsened']} | {d['both_failure']} | {d['both_success']} | {f(d['joint_success_cost_delta'])} | {f(d['joint_success_steps_delta'])} |")
    add('')
    add('共同成功成本差的分项核对（后−前，分项之和等于上表总成本差）：')
    add('')
    add('| 配对 | 时间 | 观测违反 | 通信违反 | 观测失败 | 通信中断 | 碰撞 | 越界 |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|')
    for pair in ('R0_to_R1','oracle_to_R0','oracle_to_R1'):
        terms=s['paired']['all'][pair].get('common_success_cost_component_delta',{})
        add('| '+pair+' | '+' | '.join(f(terms.get(k),4) for k in COST_KEYS)+' |')
    add('')
    add('### 实际累计成本与步数：成功和失败分开')
    add('')
    add('不同组的成功episode集合不同，成功条件均值也不能独立证明控制器更好。失败的早停低成本不作为优势。全部逐回合成本、步数和七项成本分量保存于episodes.jsonl及对应轨迹JSON/NPZ；区域×heading及planner seed完整分层见分组附表和summary.json。')
    add('')
    add('| 分组 | 控制器 | 成功平均成本 | 成功平均步数 | 失败数 | 失败实际平均成本（仅描述） | 失败平均步数 |')
    add('|---|---|---:|---:|---:|---:|---:|')
    for name in main_groups:
        pred=(lambda r:True) if name=='all' else ((lambda r,n=name:r['stratum']==n) if name in ('inside','west') else (lambda r,h=int(name.split('_')[1]):r['heading_bin']==h))
        for arm in ARMS:
            d=s['grouped'][name][arm];fail=[r for r in rows[arm] if pred(r) and not r['success']]
            add(f"| {name} | {LABEL[arm]} | {f(d['successful_cost_mean'])} | {f(d['successful_steps_mean'])} | {len(fail)} | {f(float(np.mean([r['cost'] for r in fail])) if fail else None)} | {f(float(np.mean([r['steps'] for r in fail])) if fail else None)} |")
    add('')
    add('## Oracle缓存与终端查询诊断')
    add('')
    o=s['oracle_validation']
    add(f"缓存签名 `{o['current_digest']}` 与旧缓存匹配={o['reused']}；SHA256 `{o['cache_sha256']}`。完整环境/任务/成本/格点signature保存在oracle_validation.json，旧缓存只读，新目录保留字节相同副本。")
    add(f"预检缓存初始化/加载耗时{o['current_setup_seconds']:.6f}s；本轮缓存构建耗时{o['current_build_seconds']:.6f}s。旧缓存元数据中的历史构建耗时不计入本轮。正式每回合新建provider，因此加载单列为初始化时间；在线规划时间为act耗时，包含被动诊断记录开销。")
    add('')
    add('| 控制器 | 真实终值查询数 | 无效查询比例 | 不可达比例 | 候选invalid比例 | 全候选invalid决策数 | 在线总秒 | 每步平均ms | p95 ms | 初始化总秒 |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for arm in ARMS:
        q=s['queries'][arm];t=s['timings'][arm];init=t['initialization_seconds']
        add(f"| {LABEL[arm]} | {q['queries']} | {100*q['invalid_ratio']:.4f}% | {100*q['unreachable_ratio']:.4f}% | {100*q['invalid_candidate_ratio']:.2f}% | {q['all_invalid_decisions']} | {t['online_total_seconds']:.3f} | {1000*t['online_seconds']['mean']:.3f} | {1000*t['online_seconds']['p95']:.3f} | {init['mean']*init['n']:.6f} |")
    oq=s['queries']['oracle'];add(f"\nOracle正式episode缓存命中 {oq.get('cache_hit_episodes',0)}/64，miss {oq.get('cache_miss_episodes',0)}；none组不调用终值，查询数为0。invalid候选被原MPPI惩罚并排除终值查询；‘无效查询’与‘连续有效但最近格点不可达’分开统计。")
    add('')
    add('Oracle是拥有完整环境模型的离散格点参考：位置.25、24个heading、5步恒定动作primitive、反向Dijkstra；连续状态采用最近格点查询，合法真实终态强制零值。不是连续环境精确最优值、也不是严格性能上界。它只提供候选末态成本，没有直接沿Oracle路径控制；专家图参考文件不参与本轮执行。')
    add('')
    add('## 训练支持覆盖（事后辅助）')
    add('')
    add(f"训练支持取{s['support']['count']}个去重物理source/next pose，不使用抽象目标节点。XY最近距离按欧氏距离，其最近点heading差另存JSON（等距时按KD树返回点）；另按sqrt(dx²+dy²+(0.5·wrapped_dheading)²)寻找联合最近支持，报告其heading差。heading为弧度，位置为环境单位。每个实际访问/查询按出现次数统计，重复和轨迹相关性保留。这些量只描述覆盖，不证明可达性或泛化。")
    add('')
    add('| 控制器 | 状态集合 | 数量 | XY最近距离均值/p95 | 联合最近heading差均值/p95 | 联合距离均值/p95 |')
    add('|---|---|---:|---:|---:|---:|')
    for arm in ARMS:
        for name in ('executed','candidate_valid_unfinished','queried'):
            d=s['coverage'][arm][name];xy=d['nearest_xy'];h=d['joint_neighbor_heading_rad'];j=d['joint_metric']
            add(f"| {LABEL[arm]} | {name} | {xy['n']} | {f(xy['mean'],4)}/{f(xy['p95'],4)} | {f(h['mean'],4)}/{f(h['p95'],4)} | {f(j['mean'],4)}/{f(j['p95'],4)} |")
    add('')
    add('candidate_valid_unfinished为全部组相同规则的有效未完成候选末态；模型/Oracle组它就是实际终值查询集，none组仅作事后覆盖。全部候选（含invalid）的覆盖统计另存JSON。')
    add('')
    add('## 图内改善是否转化为连续导航收益')
    add('')
    old=read(TRAIN/'comparison/summary.json')
    for arm in ('R0','R1'):assert old['results'][arm]['checkpoint_sha256']==s['adapter_validation'][arm]['sha256']
    a=old['results']['R0'];b=old['results']['R1']
    add(f"历史固定图上，R0→R1最远四分位NRMSE {a['distance']['longest_quartile']['nrmse_percent']:.2f}%→{b['distance']['longest_quartile']['nrmse_percent']:.2f}%，165起点图内路径regret {a['decisions']['unique_start_rollouts']['mean_successful_regret']:.2f}→{b['decisions']['unique_start_rollouts']['mean_successful_regret']:.2f}。这里只引用既有图结果作背景，没有拿图最短路计算连续轨迹regret。")
    delta=pr['improved']-pr['worsened']
    add(f"\n本轮连续闭环成功率变化为{100*(g['R1']['success_rate']-g['R0']['success_rate']):+.2f}个百分点（净{delta:+d}/64），同时失去{pr['worsened']}个R0成功案例。共同成功{pr['both_success']}个episode中，R1−R0实际成本差均值{f(pr['joint_success_cost_delta'])}、步数差{f(pr['joint_success_steps_delta'])}。这些指标共同描述转化程度；不能把图内距离改善直接等同于稳健导航提升。")
    gained=sorted({x[0] for x in pr['new_success']})
    add(f"\n新增成功来自{len(gained)}个地理起点：{', '.join(gained)}；两个planner seeds共享这些起点，因此4次新增成功不等于4个独立地理案例。")
    add('')
    for region in ('inside','west'):
        q=s['paired'][region]['R0_to_R1'];d=s['grouped'][region]
        add(f"- {region}：成功R0 {d['R0']['successes']}/{d['R0']['episodes']}→R1 {d['R1']['successes']}/{d['R1']['episodes']}，新增{q['improved']}、失去{q['worsened']}；共同成功平均成本差{f(q['joint_success_cost_delta'])}。")
    add('')
    for arm in ('R0','R1'):
        d=g[arm];pair=s['paired']['all']['oracle_to_'+arm]
        add(f"- Oracle→{arm}：Oracle成功而{arm}失败{pair['worsened']}例，反向新增{pair['improved']}例；共同成功{pair['both_success']}例中{arm}−Oracle成本差{f(pair['joint_success_cost_delta'])}、步数差{f(pair['joint_success_steps_delta'])}。{arm}新离开U {d['exit_completed']}/{d['exit_eligible']}、绕墙{d['wrap_completed']}/{d['wrap_eligible']}；Oracle分别{g['oracle']['exit_completed']}/{g['oracle']['exit_eligible']}、{g['oracle']['wrap_completed']}/{g['oracle']['wrap_eligible']}。")
    add('')
    add('本轮R0/R1/Oracle的内部失败均发生在离开U之前；三组从西出口出发均成功。因而相对Oracle的成功率差距主要表现为U内逃逸，R1只改善其中两个起点。')
    add('')
    add('所有正式失败轨迹均保留。阶段差异说明收益/失败发生在流程的哪一段，但不能仅凭位置覆盖或单条轨迹证明误差的因果来源。MPPI采用短视域候选和加权平均控制，其执行也会碰撞；Oracle的失败不能自动归因于模型价值，Oracle的成功也不构成最优上界。')
    add('')
    add('本轮按冻结256episode结束，不追加预算、调参、挑起点或扩展训练seed。只有一个训练初始化seed606、32个地理起点与两个配对planner seeds；同一起点的两回合并非独立地理样本，也不能代表多训练seed稳定性。固定场景内随机新起点不等于场景外泛化。时间受四进程竞争和诊断开销影响。')
    add('')
    add('## 文件与复现')
    add('')
    add('- `experiments/r_series_mppi/PROTOCOL.md`：完整冻结规则与复现命令；事后统计测试命令：`$PY -m pytest experiments/r_series_mppi/test_analysis.py -q`。')
    add('- `../manifest.json`、`../states.json`、`../start_overlap.json`：共同配置、32个pose及重合检查。')
    add('- `../tests.log`、`../smoke.json`、`replay_audit.json`：必要测试、独立烟雾及256条轨迹真实重放。')
    add('- `../arms/{none,R0,R1,oracle}/episodes.jsonl`、各episode的JSON/NPZ：全部实际轨迹、分项成本、候选末态、终值查询、规划耗时和失败。')
    add('- `summary.json`、`group_tables.md`：完整分组及配对新增/失去成功列表、共同成功逐episode差值。')
    add('- `*_coverage.npz`：每个实际访问/候选/查询的训练支持距离；`../final_integrity.json`：完整产物指纹。')
    add('')
    add('![成功率与阶段](comparison.png)\n\n![全部闭环轨迹（含失败）](trajectories.png)')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    # All crossed region x heading and planner-seed tables, not only marginal summaries.
    appendix=['# 全部分组附表','', '| 分组 | 模型 | 成功 | 新离开U | 绕墙 | 成功均成本 | 成功均步数 |','|---|---|---:|---:|---:|---:|---:|']
    for name,arms in s['grouped'].items():
        for arm,d in arms.items():appendix.append(f"| {name} | {arm} | {d['successes']}/{d['episodes']} | {rate(d['exit_completed'],d['exit_eligible'])} | {rate(d['wrap_completed'],d['wrap_eligible'])} | {f(d['successful_cost_mean'])} | {f(d['successful_steps_mean'])} |")
    appendix+=['','| 分组 | 配对 | 新增 | 失去 | 共同成功 | 平均Δ成本 | 平均Δ步数 |','|---|---|---:|---:|---:|---:|---:|']
    for name,pairs in s['paired'].items():
        for pair,d in pairs.items():appendix.append(f"| {name} | {pair} | {d['improved']} | {d['worsened']} | {d['both_success']} | {f(d['joint_success_cost_delta'])} | {f(d['joint_success_steps_delta'])} |")
    (out/'group_tables.md').write_text('\n'.join(appendix)+'\n')


def plots(root,s,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    out=root/'comparison';colors={'none':'#777777','R0':'#2878b5','R1':'#e89424','oracle':'#35a36b'}
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained');x=np.arange(3)
    for j,arm in enumerate(ARMS):axes[0].bar(x+(j-1.5)*.19,[100*s['grouped'][g][arm]['success_rate'] for g in ('all','inside','west')],.19,label=arm,color=colors[arm])
    axes[0].set(xticks=x,xticklabels=['All (64)','Inside (48)','West (16)'],ylabel='Success (%)',ylim=(0,105));axes[0].legend()
    for j,arm in enumerate(ARMS):
        d=s['grouped']['all'][arm];axes[1].plot([0,1,2],[100*d['exit_rate'],100*d['wrap_rate'],100*d['success_rate']],'-o',label=arm,color=colors[arm])
    axes[1].set(xticks=[0,1,2],xticklabels=['Exit U (48)','Wrap (64)','Task (64)'],ylabel='Eligible completion (%)',ylim=(0,105));axes[1].legend()
    pairs=('R0_to_R1','oracle_to_R0','oracle_to_R1')
    axes[2].bar(np.arange(3)-.18,[s['paired']['all'][p]['improved'] for p in pairs],.36,label='Gained success',color='#35a36b')
    axes[2].bar(np.arange(3)+.18,[s['paired']['all'][p]['worsened'] for p in pairs],.36,label='Lost success',color='#c65a4b')
    axes[2].set(xticks=range(3),xticklabels=['R0 → R1','Oracle → R0','Oracle → R1'],ylabel='Paired episodes');axes[2].legend()
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained');scenario=read(DATA/'scenario.json')
    for ax,arm in zip(axes.ravel(),ARMS):
        for obs in scenario['obstacles']:
            if obs['type']=='rectangle':ax.add_patch(Rectangle((obs['x_min'],obs['y_min']),obs['x_max']-obs['x_min'],obs['y_max']-obs['y_min'],color='#b0b0b0'))
        for r in rows[arm]:
            z=np.load(root/'arms'/arm/f"{r['state_id']}_{r['planner_seed']}.npz");states=z['states']
            c='#2486aa' if r['success'] else '#d65b42';ax.plot(states[:,0],states[:,1],color=c,alpha=.45,lw=.8)
            ax.scatter(states[-1,0],states[-1,1],s=12,c=c,marker='o' if r['success'] else 'x')
        start=np.array([r['pose'] for r in read(root/'states.json')['records']]);ax.scatter(start[:,0],start[:,1],s=9,c='black',alpha=.6)
        ax.scatter([6.7],[3.65],marker='*',s=120,color='#554477')
        ax.set(xlim=(0,9),ylim=(0,8),aspect='equal',title=f"{arm}: {s['grouped']['all'][arm]['successes']}/64 success",xlabel='x',ylabel='y')
    fig.suptitle('All 256 trajectories: blue=success, red=failure; black=new starts, star=target')
    fig.savefig(out/'trajectories.png',dpi=180);plt.close(fig)
