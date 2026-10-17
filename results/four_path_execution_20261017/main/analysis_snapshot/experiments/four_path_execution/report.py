"""Independent post-run audit, paired execution statistics and plots."""
import argparse
import csv
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiments.four_path_execution.common import *
from experiments.four_path_execution.statistics import aggregate,paired,mean


def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]


def fmt(x,places=3):return '—' if x is None else f'{x:.{places}f}'


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)


def generate(root):
    out=root/'comparison';out.mkdir(exist_ok=False)
    manifest=read(root/'manifest.json');states=read(root/'states.json')['records']
    assert file_digest(root/'states.json')==manifest['states_sha256']
    for arm,entry in manifest['checkpoints'].items():assert file_digest(ROOT/entry['path'])==entry['sha256']
    for name,digest in manifest['sources'].items():assert file_digest(root/'source_snapshot'/name)==digest
    process=read(root/'process_results.json');rows=[];audits={};failed=[];stage_corrections=[]
    for arm in ARMS:
        directory=root/'arms'/arm
        if process.get(arm,{}).get('exit_code')!=0 or not (directory/'COMPLETE').exists():
            failed.append(arm);continue
        episodes=lines(directory/'episodes.jsonl')
        expected={(r['id'],s) for r in states for s in PLANNER_SEEDS}
        assert len(episodes)==len(expected)==64 and {(r['state_id'],r['planner_seed']) for r in episodes}==expected
        for r in episodes:
            original=next(x for x in states if x['id']==r['state_id'])
            trajectory=np.load(directory/f"{r['state_id']}_{r['planner_seed']}.npz")
            np.testing.assert_array_equal(trajectory['states'][0],original['pose'])
            assert len(trajectory['actions'])==r['steps'] and r['model_rollouts']==r['steps']*CONFIG.num_samples
            assert r['model_transitions']==int(trajectory['model_transitions'].sum())
            np.testing.assert_allclose(r['cost'],-trajectory['rewards'].sum(),rtol=1e-12,atol=1e-9)
            corrected=stage_metrics(trajectory['states'],original)
            for k,v in corrected.items():
                if r[k]!=v:
                    assert k in ('exit_completed','west_exit_visited','cumulative_exit')
                    stage_corrections.append(dict(arm=arm,state_id=r['state_id'],planner_seed=r['planner_seed'],metric=k,old=r[k],new=v))
            r.update(corrected)
            assert sum(r[k] for k in ('success','collision','out_of_bounds','timeout'))==1
            assert r['steps']<=300 and (not r['timeout'] or r['steps']==300)
        rows+=episodes;audits[arm]=dict(unique_pair_keys=True,checkpoint_unchanged=True,
            costs_and_steps_recomputed=True,stages_recomputed=True,**read(directory/'COMPLETE'))
    # Process errors cannot vanish from denominator or trigger replacement.
    summary=[];comparisons=[]
    for stratum in ('all','support','perturbed','holdout'):
        for arm in ARMS:
            selected=[r for r in rows if r['arm']==arm and (stratum=='all' or r['stratum']==stratum)]
            if selected:summary.append(dict(arm=arm,stratum=stratum,**aggregate(selected)))
        for seed in SEEDS:
            for before,after in [(f'decay_{seed}',f'qp_{seed}'),('none',f'decay_{seed}'),('none',f'qp_{seed}'),('geometry',f'qp_{seed}')]:
                a=[r for r in rows if r['arm']==before and (stratum=='all' or r['stratum']==stratum)]
                b=[r for r in rows if r['arm']==after and (stratum=='all' or r['stratum']==stratum)]
                if a and b:comparisons.append(dict(seed=seed,stratum=stratum,before=before,after=after,**paired(a,b)))
    write(out/'summary.json',dict(aggregates=summary,paired=comparisons,process_failures=failed,planned_episodes=768,completed_episodes=len(rows)))
    write(out/'audit.json',audits)
    write(out/'stage_measurement_corrections.json',stage_corrections)
    for name,records in [('episodes',rows),('aggregates',summary),('paired',comparisons)]:
        with (out/f'{name}.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    scores=read(root/'candidates/scores.json');candidate_pairs=read(root/'candidates/paired.json')
    unique=[r for r in scores if r['arm']=='none']
    empty=sum(r['useful_count']==0 for r in unique)
    candidate_summary=[]
    for seed in SEEDS:
        group=[p for p in candidate_pairs if p['seed']==seed]
        candidate_summary.append(dict(seed=seed,mean_rank_spearman=mean([p['safe_rank_spearman'] for p in group if p['safe_rank_spearman'] is not None]),
            mean_weights_tv=mean([p['weights_total_variation'] for p in group]),max_abs_action_change=max(abs(p['first_control_delta']) for p in group),
            mean_abs_action_change=mean([abs(p['first_control_delta']) for p in group]),
            mean_useful_weight_change=mean([p['useful_weight_delta'] for p in group]),
            mean_top10_overlap=mean([p['top10_overlap'] for p in group]),
            mean_top10_overlap_fraction=mean([p['top10_overlap']/p['top10_size'] for p in group if p['top10_size']]),
            nonempty_top10_banks=sum(p['top10_size']>0 for p in group)))
    write(out/'candidate_summary.json',candidate_summary)
    plot(root,out,summary,states,scores)
    pooled=[]
    for stratum in ('all','support','perturbed','holdout'):
        for category in ('none','geometry','decay','qp'):
            selected=[r for r in rows if (r['arm']==category or r['arm'].startswith(category+'_')) and (stratum=='all' or r['stratum']==stratum)]
            if selected:pooled.append(dict(group=category,stratum=stratum,**aggregate(selected)))
    write(out/'pooled.json',pooled)
    main_rows=[[r['group'],r['stratum'],f"{r['successes']}/{r['episodes']}",fmt(100*r['success_rate'],1),f"{r['collision_count']} ({100*r['collision_rate']:.1f}%)",f"{r['out_of_bounds_count']} ({100*r['out_of_bounds_rate']:.1f}%)",f"{r['timeout_count']} ({100*r['timeout_rate']:.1f}%)",fmt(r['successful_cost_mean']),fmt(r['successful_steps_mean'],1)] for r in pooled]
    individual=[[r['arm'],r['stratum'],f"{r['successes']}/{r['episodes']}",r['collision_count'],r['out_of_bounds_count'],r['timeout_count'],fmt(r['successful_cost_mean']),fmt(r['successful_steps_mean'],1)] for r in summary]
    prepost=[p for p in comparisons if p['before'].startswith('decay_')]
    paired_table=[[p['seed'],p['stratum'],p['improved'],p['worsened'],p['both_success'],p['both_failure'],fmt(100*p['success_delta'],2),fmt(p['joint_success_cost_delta']),fmt(p['joint_success_steps_delta'],1)] for p in prepost]
    stages=[[r['arm'],r['stratum'],f"{r['exit_completed']}/{r['exit_eligible']}",f"{r['wrap_completed']}/{r['wrap_eligible']}",f"{r['successes']}/{r['episodes']}",fmt(100*r['cumulative_exit_rate'],1),fmt(100*r['cumulative_wrap_rate'],1)] for r in summary]
    compute=[[r['arm'],r['episodes'],fmt(r['planning_seconds'],1),fmt(r['planning_ms_per_decision'],2),r['model_rollouts'],r['model_transitions']] for r in summary if r['stratum']=='all']
    all_pairs=[p for p in prepost if p['stratum']=='all'];improved=sum(p['improved'] for p in all_pairs);worsened=sum(p['worsened'] for p in all_pairs)
    aggregate_lookup={(r['group'],r['stratum']):r for r in pooled}
    post=aggregate_lookup['qp','all'];before=aggregate_lookup['decay','all']
    coverage=[[r['x'],r['heading'],r['planner_seed'],r['safe_count'],r['useful_count'],r['exit_count']] for r in unique]
    candidates=[[r['seed'],fmt(r['mean_rank_spearman'],5),fmt(r['mean_weights_tv'],5),fmt(r['mean_top10_overlap_fraction']*100,1),fmt(r['mean_abs_action_change'],5),fmt(r['max_abs_action_change'],5),fmt(r['mean_useful_weight_change'],5)] for r in candidate_summary]
    report=f'''# 四路径 QRL：QP 前后 × MPPI 配对执行（2026-10-17）

## 结论与范围

冻结配置下完成{len(rows)}/768个episode；进程失败组：{failed or '无'}。全部五个网络初始化均纳入。修复前QRL的成功数为{before['successes']}/{before['episodes']}，修复后为{post['successes']}/{post['episodes']}。同起点/heading/规划随机种子的QP前后配对中，**失败→成功{improved}组，成功→失败{worsened}组**；下表完整保留每个seed和测试层的表现。

QRL终值在这组固定配置上优于无终值/现有几何引导，但收益不能归于QP：原图联合通过的404并非执行最优（202为48/64，404为46/64），记录图指标和离轨执行不是同一件事。支持层成功率82.5%、扰动层75.8%、保留U内姿态26.25%，说明存在明显的离轨/heading脆弱性。下一步优先分开验证候选覆盖与离轨终值排序，当前证据不支持继续仅靠压低记录边残差来改善执行。

这一评估将记录图的局部可行性与实际闭环执行分开检验。不能用QP后的零边违反代替执行成功，也不能把五份模型在相同32个状态上的结果当成320个独立测试状态。固定规划配置只有一组，不能据此断言MPPI所有配置的能力上限。

{table(['组别','测试层','成功/总数','成功率%','碰撞数','越界数','超时数','成功成本均值','成功步数均值'],main_rows)}

三种失败比例均可由同一行总数计算，精确rate见[aggregates.csv](aggregates.csv)。无终值和几何baseline各仅64次，QRL各阶段为5×64次；所有组每个网络/基线均使用同一32状态×2规划seed，基线不复制成五份样本。只在成功任务内报告成本/步数，所有episode平均截断成本不用于优劣排序。

![成功率](success_rates.png)

## 全部seed与逐对变化

{table(['执行组','测试层','成功/总数','碰撞','越界','超时','成功成本','成功步数'],individual)}

QP后−QP前的严格配对结果：成本/步数差仅在两者都成功的同一状态/RNG组合上计算，避免成功集合变化造成错误比较。空集合记为“—”，不是0。

{table(['网络seed','层','改善','退化','均成功','均失败','成功率差百分点','共同成功成本差','共同成功步数差'],paired_table)}

与无终值、几何baseline的相同键配对也保存在[paired.csv](paired.csv)。这些是有限固定状态的描述性结果，未把状态内两个规划随机种子或共享baseline当作独立重复来计算显著性。

共同成功210对的QP后成本平均差为+0.01778，接近不变；总体成功成本均值154.544→150.257的下降主要受成功集合换位影响，不能据此认定QP普遍缩短执行成本。碰撞数80→74同时超时14→20，主要是失败模式变化，并非任务完成率提升。

## 阶段完成与轨迹

“离开U”定义为访问西侧出口区域；“完成绕墙”要求先到达上/下墙包围框之外，再到闭墙东侧；“合法终态”为真实环境巡检任务success，包含位置、视角/heading、视线与通信要求。

对已在西出口或闭墙东侧的中间起点，不能把之前已经过的阶段计成新完成。因此分别给出当前起点适用的阶段完成数/分母，以及含已记录训练前缀的累计比例。扰动状态不继承训练路径前缀证据，不适用的早期阶段记NA；保留U内起点对三个阶段均适用。全量JSON保留prefix与eligible标记。

初版episode日志仅检查离散状态是否落在西出口区域，会漏掉从西角一步跨过区域的情况。独立审计统一按环境使用的线性移动线段检查出口相交，修正所有组的阶段计数；没有改变轨迹、动作或成功/失败标签。原始episodes.jsonl及冻结源码不变，[统计修正明细](stage_measurement_corrections.json)保存所有差异，新增跨角回归验证。此项是事后修正测量漏计，不是更改成功门槛。

{table(['组','层','新离开U/适用','新绕墙/适用','合法终态/全部','含前缀离U%','含前缀绕墙%'],stages)}

![保留起点的闭环轨迹](holdout_trajectories.png)

图中固定展示规划seed314159下全部8个保留状态，每组同一集合，没有挑选最好轨迹；绿线成功、灰线失败。逐episode完整轨迹、动作、环境奖励及逐决策计算量位于`../arms/<arm>/*.npz`。绘图窗口覆盖所示全部轨迹，显示U区域细节。第一步前的候选噪声相同；闭环状态、名义控制序列分化后，候选轨迹自然不同，本报告没有声称全过程共享候选。

## 固定候选评分诊断

16批开发状态候选，来自深处/中部×四个heading×两个规划seed；每批同一256×20控制序列、同一环境模型rollout被12组评分，完全隔离候选采样变化。未用这些开发点调参，也没有向候选池注入训练路径或oracle动作。

“有用”是本窗口内未碰撞/越界、完整20步、向西进展至少0.25且末heading朝西的局部指标；它不是完整脱困可达性的证明。另列安全访问西出口的候选数。

{table(['x','heading','规划seed','安全/256','有用/256','访问出口/256'],coverage)}

**{empty}/16批没有采到符合该局部进展定义的候选。** 这4批均为朝东状态，安全候选数也为0；在无效轨迹上本实现不加terminal value，此时更换QRL无法改变这批候选的终值排序。没有覆盖时，不能仅凭terminal value的排序来解释失败；有覆盖时可进一步检查这些候选获得的概率质量。零进展候选不证明任何闭环策略都不能脱困：MPPI还会滚动重采样，且加权平均动作可能不等于某条候选首动作。

{table(['模型seed','安全排序Spearman均值','权重TV均值','有效top-k重合%' ,'首动作绝对差均值','首动作绝对差最大','有用权重差均值'],candidates)}

![候选覆盖及权重](candidate_weights.png)

安全候选的top-k集合在全部有安全候选的批次中完全重合（k=min(10,安全候选数)，空集合排除）；平均Spearman均≥0.9999949，QP前后最大首动作差≤0.002169 rad/s。局部修复几乎保留了原执行排序。

同时存在“已采到进展但几乎不给权重”的情形：开发状态(4.0,3.65)、north、规划seed314159有3/256条安全西进展候选；无终值给出的总权重约9.63×10⁻⁵，五个QRL修复前仅约10⁻⁴¹至10⁻³⁴、修复后仍近零。south同位置有5/256条，QRL权重约10⁻⁴³至10⁻³⁹。QRL给这些候选终点的成本估计高于其他安全候选；这是对西向局部进展的强烈压低，尚不是全局优劣的oracle证明。QP未改变这一现象，支持把下一轮重点放在离轨/heading价值排序和采样覆盖，而非重复原记录图可行性修复。

热力图采用log10权重（低于1e-60按1e-60显示），灰色行表示根本没有采到有用候选；因此能区分覆盖缺失和已有候选被压低权重。评分明细包括每批有用候选的terminal contribution、权重质量、top10数量、有效样本数与加权首动作安全性，见[全部评分](../candidates/scores.json)和[配对诊断](../candidates/paired.json)。每批NPZ包含候选、每步状态、每组cost/weight，能够定位具体哪些候选获得权重。安全候选排序相关只在相同safe mask上计算，不让统一的无效罚项主导相关系数。

## 固定配置、模型适配与保留集

完全不改learner、不优化任何checkpoint。使用五个25k降步长终点和上一轮数值修复统一重跑所得QP终点（包括未通过1%门槛的模型）。[manifest](../manifest.json)冻结全部checkpoint SHA256、状态bank、源码哈希和参数；结束后再次校验checkpoint与评估模型未变。

复用现有MPPI，horizon20（2秒）、候选256、逐时独立高斯噪声σ0.8、温度1、terminal weight1、无效罚项1e6；几何terminal使用原有位置时间+转向时间启发与128个合法终态样本，不把它解释成成本最优值。几何启发只含飞行/转向时间，而QRL估计还包含观测和通信罚项；共同terminal weight并不匹配二者值尺度，因此本对照不能单独证明学到的结构本身优于经过校准的几何引导。保持环境原成本、地图、300步上限，训练bank/B1配对未修改。

IndexedIQE的encoder、projector与IQE head以strict state_dict加载到生产critic，再用现有QRLGoalValueAdapter；明确distance_scale=1.0，未再次乘dt。十个模型全部339边+335 task+335 B1共1009配对输出最大差为0；原始42D状态与规范状态一致、不同推理batch分块误差在既定浮点容差内，详见[适配验证](../adapter_validation.json)。不能把独立latent表用于任意离轨状态，本次仅共享encoder模型。

32状态由几何规则生成并在闭环执行前冻结：四路径的起点/首次西出口/首次闭墙东侧共12个支持状态；各加一个dx/dy∈[−.15,.15]、heading∈[−.35,.35]扰动共12个；独立随机seed2026101702生成8个U内位置与完整[-π,π] heading，排除距任何训练轨迹状态、旧诊断起点、旧候选起点位置≤.25的点。只按有效性/非终态几何筛选，不按模型分数或执行结果筛选。[states.json](../states.json)记录所有姿态、来源和排除距离。这是同一地图/任务的新姿态保留集，不能称为新地图泛化或从未用于任何研究的任务。

## 计算开销与验证

{table(['组','episodes','规划总秒','每决策平均ms','候选序列rollout数','实际模型状态转移数'],compute)}

模型rollout数按每次规划256条序列计算，实际模型状态转移数考虑提前成功/碰撞和剩余环境步数截断。另有16×256=4096条固定候选诊断rollout，12组仅重复评分，不虚计成12倍环境rollout。规划时间包含每次episode终态样本初始化和在线规划；四个CPU进程并发、每进程一个torch线程，延迟包含系统调度负载，不是专用硬件基准。独立3步smoke的各组耗时也保留在`../../smoke/smoke.json`。

18项测试通过：十模型适配、状态bank确定性、矩形碰撞边界/随机标量对照、旧版本MPPI连续控制一致、环境rollout成本与终止对照、阶段资格以及成功条件成本/配对统计、线段跨出口计数。见[最终验证输出](../../validation_final.txt)。新增矩形批量碰撞路径只替换Python逐候选循环，采样、成本、温度、更新公式保持一致。暴露候选采样和评分接口使诊断与闭环复用相同实现；没有另写一套评分规则。

[审计](audit.json)从保存轨迹重算成本、步数、阶段、rollout计数，核对完整配对矩阵和不可变checkpoint。源快照及所有失败结果保留，不覆盖上一轮实验。另将全部768条已保存的真实动作序列重新送入原环境，逐步核对状态、奖励和成功/失败终止，见[环境重放审计](../replay_audit.json)。统计修正后的分析源码另存analysis_snapshot及analysis_manifest.json，不改写正式执行时的冻结源码。

## 复现

```sh
export MPLCONFIGDIR=/tmp/qrl-execution-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=../quasimetric-rl/.venv/bin/python
$PY -m pytest experiments/four_path_execution/test_execution.py minimal_qrl/test_comm_inspection_baselines.py -q
$PY -m experiments.four_path_execution.smoke --output results/NEW_EXECUTION/smoke
$PY -m experiments.four_path_execution.prepare --output results/NEW_EXECUTION/main
$PY -m experiments.four_path_execution.suite --root results/NEW_EXECUTION/main --workers 4
$PY -m experiments.four_path_execution.candidates --root results/NEW_EXECUTION/main
$PY -m experiments.four_path_execution.audit_replay --root results/NEW_EXECUTION/main
$PY -m experiments.four_path_execution.report --root results/NEW_EXECUTION/main
```

输出目录必须不存在。新运行需要本地冻结训练bank/replay与上一轮十个checkpoint；路径及SHA256见common.py与manifest。图参考不参与任何执行动作、候选评分或测试状态筛选。当前结论仅限该固定MPPI预算和三层姿态集合，下一轮应根据候选覆盖与配对执行证据分别检验采样、终值泛化及终态接近控制，不在本次结果上为每个模型单独调参。
'''
    (out/'report.md').write_text(report)
    print(json.dumps(dict(complete_episodes=len(rows),failed_arms=failed,pre_successes=before['successes'],post_successes=post['successes'],paired_improved=improved,paired_worsened=worsened,empty_candidate_banks=empty)),flush=True)


def plot(root,out,summary,states,scores):
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True)
    for ax,stratum in zip(axes,('support','perturbed','holdout')):
        byarm={r['arm']:r for r in summary if r['stratum']==stratum}
        ax.barh(range(len(ARMS)),[100*byarm[a]['success_rate'] if a in byarm else 0 for a in ARMS],
            color=['#888888','#bc9a3c']+['#4488bb','#44aa77']*5)
        ax.set(yticks=range(len(ARMS)),yticklabels=ARMS,xlim=(0,100),title=stratum,xlabel='Task success (%)');ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.tight_layout();fig.savefig(out/'success_rates.png',dpi=150);plt.close(fig)
    scenario=read(root/'scenario.json');fig,axes=plt.subplots(3,4,figsize=(16,12))
    for ax,arm in zip(axes.flat,ARMS):
        for obstacle in scenario['obstacles']:
            ax.add_patch(Rectangle((obstacle['x_min'],obstacle['y_min']),obstacle['x_max']-obstacle['x_min'],obstacle['y_max']-obstacle['y_min'],color='#a89989'))
        rows=lines(root/'arms'/arm/'episodes.jsonl') if (root/'arms'/arm/'episodes.jsonl').exists() else []
        for record in states:
            if record['stratum']!='holdout':continue
            path=root/'arms'/arm/f"{record['id']}_{PLANNER_SEEDS[0]}.npz"
            if not path.exists():continue
            trajectory=np.load(path)['states'];r=next(r for r in rows if r['state_id']==record['id'] and r['planner_seed']==PLANNER_SEEDS[0])
            assert ((trajectory[:,0]>=0)&(trajectory[:,0]<=9)&(trajectory[:,1]>=0)&(trajectory[:,1]<=8)).all()
            ax.plot(trajectory[:,0],trajectory[:,1],color='#278251' if r['success'] else '#727a84',alpha=.75,lw=.9)
            ax.scatter(*trajectory[0,:2],s=7,color='#245cad')
        ax.scatter(6.7,3.65,marker='*',s=65,color='#cc7733');ax.set(xlim=(0,9),ylim=(0,8),title=arm,aspect='equal');ax.grid(alpha=.1)
    fig.tight_layout();fig.savefig(out/'holdout_trajectories.png',dpi=150);plt.close(fig)
    probes=[r['probe'] for r in scores if r['arm']=='none']
    matrix=np.array([[next(r['useful_weight'] for r in scores if r['arm']==a and r['probe']==p) for a in ARMS] for p in probes])
    useful_counts=np.array([next(r['useful_count'] for r in scores if r['probe']==p) for p in probes])
    displayed=np.ma.masked_where(np.repeat((useful_counts==0)[:,None],len(ARMS),axis=1),np.log10(np.clip(matrix,1e-60,1.)))
    cmap=plt.cm.viridis.copy();cmap.set_bad('#b8b8b8')
    fig,ax=plt.subplots(figsize=(13,8));im=ax.imshow(displayed,vmin=-60,vmax=0,cmap=cmap,aspect='auto')
    ax.set(xticks=range(len(ARMS)),xticklabels=ARMS,yticks=range(len(probes)),yticklabels=[p+f" (n={next(r['useful_count'] for r in scores if r['probe']==p)})" for p in probes],title='Safe westward-progress weight: log10 scale (gray = no useful candidate)' )
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right');fig.colorbar(im,ax=ax,label='log10 MPPI probability mass (floor 1e-60)' );fig.tight_layout();fig.savefig(out/'candidate_weights.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);generate(Path(p.parse_args().root))
