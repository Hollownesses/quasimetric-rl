"""Read-only endpoint comparison and all existing task3-goal field checks."""
import argparse
import csv
import json
from pathlib import Path
import time

from diagnostics import REPO,core,load_checkpoint,value_components,aggregate,sha,UNIT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree


def read(path):return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as f:return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]


def paired(a,b):
    assert list(a)==list(b)
    return dict(wins=[k for k in a if b[k]['success'] and not a[k]['success']],
                losses=[k for k in a if a[k]['success'] and not b[k]['success']])


def run(root,output,resume_analysis=False):
    start=time.time();root=Path(root).resolve();output=Path(output).resolve()
    if resume_analysis:assert output.is_dir()
    else:output.mkdir(parents=True,exist_ok=False)
    training=root/'mqe_100k_seed0';diagnostics=root/'mqe_100k_diagnostics'
    verified=read(training/'checkpoint_verification.json');assert verified['passed'] and verified['adam_count']==100000
    assert verified['full_state_roundtrip'] and read(training/'runtime.json')['exit_code']==0
    assert read(diagnostics/'verification.json')['passed']
    result=read(diagnostics/'summary.json');meta=result['protocol'];assert meta['epoch']==100000
    checkpoint=Path(meta['checkpoint']);assert sha(checkpoint/'params_100000.pkl')==verified['checkpoint_sha256']==meta['checkpoint_sha256']
    bank=root/'evaluation_bank.npz';assert sha(bank)==meta['bank_sha256'];cases=core.read_bank(bank);assert len(cases)==25
    for v in result['rollouts'].values():assert list(v['cases'])==list(cases) and v['bank_sha256']==sha(bank)
    original=REPO/'results/ogbench_official_qrl_20261026_seed0/main'
    old_flags=list(original.glob('OGBench/*/*/flags.json'));assert len(old_flags)==1
    old_checkpoint=old_flags[0].parent
    official={'QRL':csv_rows(old_checkpoint/'eval.csv'),'MQE':csv_rows(checkpoint/'eval.csv')}
    assert all(v[-1]['step']==100000 for v in official.values())
    base_greedy=read(root/'baseline_greedy/summary.json');base_actor=read(root/'baseline_actor/summary.json')
    base_field=read(root/'baseline_field/summary.json');base_data=np.load(root/'baseline_field/field.npz')
    pairs={mode:paired(base_greedy['cases'],result['rollouts'][mode]['cases']) for mode in ['ensemble0','ensemble1','max']}
    pairs['actor']=paired(base_actor['cases'],result['rollouts']['actor']['cases'])
    agent,_,loaded=load_checkpoint(checkpoint,100000);assert loaded['network_digest']==meta['network_digest']
    # Reuse the exact existing graph; no new simulation episode or model update.
    from utils.env_utils import make_env_and_datasets
    env,_,_=make_env_and_datasets('pointmaze-medium-navigate-v0');walls,radius=core.geometry(env)
    nodes,graph=core.build_graph(walls,radius,.2);env.close();assert np.array_equal(nodes,base_data['nodes'])
    names=[k for k in cases if k.startswith('task3_')];assert len(names)==5
    fields={};goal_checks={};first_values=None
    for name in names:
        case=cases[name];traj=np.load(diagnostics/f'max_rollout/{name}.npz');goal=case['goal']
        assert np.array_equal(goal,traj['goal']) and np.array_equal(case['observation'],traj['observations'][0])
        cached=output/f'{name}_field.npz'
        if resume_analysis:
            with np.load(cached) as old:
                assert np.array_equal(old['nodes'],nodes) and np.array_equal(old['walls'],walls) and np.array_equal(old['goal'],goal)
                components=old['component_values'].copy()
                assert np.array_equal(old['values'],aggregate(components,'max'))
            check=np.array([0,len(nodes)//2,len(nodes)-1])
            np.testing.assert_allclose(components[:,check],np.asarray(value_components(agent,nodes[check],np.broadcast_to(goal,(3,2)))),rtol=1e-5,atol=1e-7)
        else:
            components=np.concatenate([np.asarray(value_components(agent,x,np.broadcast_to(goal,x.shape)))
                for x in np.array_split(nodes,int(np.ceil(len(nodes)/1024)))],axis=1)
        values=aggregate(components,'max');assert np.isfinite(values).all()
        if first_values is None:first_values=values
        anchors={'old_anchor':core.ANCHOR,'start':case['observation'],'greedy_final':traj['observations'][-1]}
        checks={}
        for label,point in anchors.items():
            stats,_=core.graph_metrics(nodes,graph,values,goal=goal,anchor=point,tol=1e-6*UNIT)
            checks[label]={k:stats[k] for k in ['anchor_node','snap_distance','anchor_value','reference_steps',
                'strict_descent_from_anchor','minimum_peak_gap_over_iqr','minimum_peak_gap','distance_iqr']}
        checks.update(goal=goal.tolist(),start_xy=case['observation'].tolist(),greedy_final_xy=traj['observations'][-1].tolist(),
            success=result['rollouts']['max']['cases'][name]['success'],steps=len(traj['actions']))
        goal_checks[name]=checks
        if not resume_analysis:np.savez_compressed(cached,nodes=nodes,walls=walls,goal=goal,component_values=components,values=values)
        print(name,json.dumps(checks),flush=True)
    for mode in ['ensemble0','ensemble1','max']:fields[mode]=result['fields'][mode]
    # Common canonical reference route, not separately chosen favorable paths.
    idx=cKDTree(nodes).query(base_data['path'])[1];colors={'QRL':'#636d79','MQE':'#c84c45'}
    fig,axes=plt.subplots(2,3,figsize=(15,8.5));x=np.arange(5)
    for i,(name,logs) in enumerate(official.items()):
        axes[0,0].bar(x+(i-.5)*.33,[logs[-1][f'evaluation/task{t}_success'] for t in range(1,6)],width=.33,label=name,color=colors[name])
    axes[0,0].set(title='Official actor: fixed 100k; 50 episodes / task',xticks=x,xticklabels=[f'T{t}' for t in range(1,6)],ylim=(0,1.05));axes[0,0].legend(fontsize=8)
    for i,(name,stats) in enumerate([('QRL',base_greedy),('MQE',result['rollouts']['max'])]):
        axes[0,1].bar(x+(i-.5)*.33,[stats['tasks'][f'task{t}']['successes']/5 for t in range(1,6)],width=.33,label=name,color=colors[name])
    axes[0,1].set(title='Same-bank true-successor V greedy; 5 / task',xticks=x,xticklabels=[f'T{t}' for t in range(1,6)],ylim=(0,1.05));axes[0,1].legend(fontsize=8)
    for t in range(1,6):axes[0,2].plot([r['step'] for r in official['MQE']],[r[f'evaluation/task{t}_success'] for r in official['MQE']],label=f'T{t}')
    axes[0,2].set(title='MQE actor learning curve (no selection)',xlabel='Updates',ylim=(-.02,1.05));axes[0,2].legend(fontsize=7,ncol=3)
    for name,values,iqr in [('QRL',base_data['values'],base_field['distance_iqr']),
                          ('MQE',np.load(diagnostics/'max_field/field.npz')['values'],fields['max']['distance_iqr'])]:
        v=values[idx];axes[1,0].plot((v-v[0])/iqr,label=name,color=colors[name])
    axes[1,0].set(title='Canonical goal: same old reference route',xlabel='Graph steps',ylabel='(d - d_start) / IQR');axes[1,0].legend(fontsize=8)
    first=names[0];goal=cases[first]['goal'];norm=(first_values-np.quantile(first_values,.25))/np.subtract(*np.quantile(first_values,[.75,.25]))
    sc=axes[1,1].scatter(nodes[:,0],nodes[:,1],c=norm,s=3,cmap='viridis',vmin=-1,vmax=3)
    fig.colorbar(sc,ax=axes[1,1],shrink=.65,label='(V - Q25) / IQR')
    for xx,yy,hx,hy in walls:axes[1,1].add_patch(Rectangle((xx-hx,yy-hy),2*hx,2*hy,color='#39414c'))
    for mode,color in [('max','#f3e36d'),('actor','#f09a85')]:
        a=np.load(diagnostics/f'{mode}_rollout/{first}.npz');p=a['observations']
        axes[1,1].plot(p[:,0],p[:,1],c=color,lw=1.3,label=mode);axes[1,1].scatter(*p[-1],c=color,s=18)
    axes[1,1].scatter(*goal,c='cyan',marker='*',s=60);axes[1,1].set(title='Actual goal: task3_seed3100',aspect='equal',xlim=(-2,22),ylim=(-2,22));axes[1,1].legend(fontsize=7)
    for mode,color in [('max','#c84c45'),('actor','#3478b5')]:
        a=np.load(diagnostics/f'{mode}_rollout/{first}.npz');obs=a['observations']
        v=aggregate(value_components(agent,obs,np.broadcast_to(goal,obs.shape)),'max')
        axes[1,2].plot(v,c=color,label=mode)
    axes[1,2].set(title='Actual goal: V along saved failed trajectories',xlabel='Environment steps',ylabel='Max V distance (log-discount)');axes[1,2].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(output/'endpoint.png',dpi=180);plt.close(fig)
    official_rows=['| 任务 | QRL官方actor | MQE官方actor |','|---|---:|---:|']
    task_rows=['| 任务 | QRL V贪心 | MQE max V贪心 | MQE ensemble0 | MQE ensemble1 | MQE actor |','|---|---:|---:|---:|---:|---:|']
    for t in range(1,6):
        official_rows.append(f'| task{t} | '+' | '.join(f"{round(official[n][-1][f'evaluation/task{t}_success']*50)}/50" for n in ['QRL','MQE'])+' |')
        stats=[base_greedy,*[result['rollouts'][m] for m in ['max','ensemble0','ensemble1','actor']]]
        task_rows.append(f'| task{t} | '+' | '.join(f"{s['tasks'][f'task{t}']['successes']}/5" for s in stats)+' |')
    goal_rows=['| 既有task3目标 | 原锚点下降通道 | 起点下降通道 | 贪心终点附近下降通道 | 终点最低峰差/IQR |','|---|---|---|---|---:|']
    for name,c in goal_checks.items():goal_rows.append(f'| {name} | {c["old_anchor"]["strict_descent_from_anchor"]} | {c["start"]["strict_descent_from_anchor"]} | {c["greedy_final"]["strict_descent_from_anchor"]} | {c["greedy_final"]["minimum_peak_gap_over_iqr"]:.6g} |')
    qrl_rate=official['QRL'][-1]['evaluation/overall_success'];mqe_rate=official['MQE'][-1]['evaluation/overall_success']
    p=pairs['max'];a_pair=pairs['actor'];runtime=read(training/'runtime.json')
    report=f'''# 原生 MQE 100k：局部通道恢复未转化为task3执行改善

固定原生MQE 100k已完成，实际Adam count100000、全状态有限性/保存加载/原生采样与同库诊断核验均通过。主要max V贪心在task3仍0/5，官方actor为2/50；不能据canonical锚点有下降通道宣称问题已解决。与原QRL相比整体官方actor均值为{qrl_rate:.1%}/{mqe_rate:.1%}，没有整体收益。

## 正式终点与方法范围

{chr(10).join(official_rows)}

原生MQE commit04c2697405233e4313fab43e05bd4d95f1b3e111，外部OGBench与QRL同commit a977253ce5b88434f4e4f531c50047cc95993a51、同原始数据。alpha0.03取论文pointmaze配置、diag_backup0.5为显式代码默认；环境选择与macOS适配沿用接入报告。完整模型SHA256为{meta['checkpoint_sha256']}。

仅取预定100k，不挑中间最好。MQE双MRN、原生B×B、future目标和actor联合学习等均与QRL不同，是方法参照；不是单变量多步消融。两者更新数相同、batch为256/1024，源转移25.6M/102.4M，不能称同计算预算。MQE主进程耗时{runtime['elapsed_seconds']/3600:.3f}小时，运行时存在其他CPU训练，不将耗时差简单当算法速度收益。官方50episode随机流不同，不作配对episode统计。

同一次MQE训练的task3官方曲线在60k记录为{next(r['evaluation/task3_success'] for r in official['MQE'] if r['step']==60000):.0%}，到100k为{official['MQE'][-1]['evaluation/task3_success']:.0%}，表现明显非单调。这里只报告已存在曲线，不将60k改选为正式结果；尚未据此区分critic变化、actor提取或评价样本的作用，不能仅凭曲线命名为某种遗忘机制。

## 同库25例执行与canonical距离场

{chr(10).join(task_rows)}

MQE max V相对QRL V贪心新增{len(p['wins'])}、失去{len(p['losses'])}；新增{p['wins']}，失去{p['losses']}。actor同库相对原QRL新增{len(a_pair['wins'])}、失去{len(a_pair['losses'])}，两者task3仍0/5。主聚合max在查看结果前固定，两个ensemble全部列出，不择优。

canonical目标沿用原诊断目标[4.2055,12.0898]。MQE两个ensemble和max均在原锚点有严格下降通道，max最低峰差/IQR为{fields['max']['minimum_peak_gap_over_iqr']:.6g}；原QRL无通道、峰差/IQR为{base_field['minimum_peak_gap_over_iqr']:.6g}。MQE max的整图/走廊Spearman为{fields['max']['all_graph_spearman']:.6g}/{fields['max']['corridor_spearman']:.6g}，原QRL为{base_field['all_graph_spearman']:.6g}/{base_field['corridor_spearman']:.6g}。这是特定区域的距离结构改善，不是全目标/全轨迹的成功保证。

## 既有五例task3实际目标的只读检查

{chr(10).join(goal_rows)}

本项只重新前向计算原模型在五个既有目标上的场，没有新增目标、训练seed、训练更新或episode。各位置吸附到同0.2网格，不能把网格结论当连续环境最优性证明。完整吸附位置/误差和场数组留在analysis.json与对应NPZ中。原锚点、起点与贪心停滞位置是不同状态；canonical目标与实际bank目标也不同，必须逐项区分。

五例原锚点均有下降通道，但五个起点及贪心终点附近均无下降通道；对应的是另一处未恢复的距离排序，不是原角点修复后执行便必然成功。起点吸附误差最大{max(c['start']['snap_distance'] for c in goal_checks.values()):.4g}，终点附近最大{max(c['greedy_final']['snap_distance'] for c in goal_checks.values()):.4g}，尤其接触状态与无碰撞图节点存在偏差，因此将后者明确称为“附近”检查；真实1000步失败本身来自已保存的模拟器轨迹。

第一例MQE max V贪心最终停在{goal_checks[first]['greedy_final_xy']}，此前QRL诊断锚点为{core.ANCHOR.tolist()}。下图使用这一例的实际目标，而非用canonical场冒充；轨迹来自已完成的正式诊断。存在某条下降路线也不保证当前贪心选出的分支能到达目标，不能把通道可达性单独当作执行证据。

![官方终点、同库执行、学习曲线与实际目标轨迹](endpoint.png)

## 结论与后续约束

本次MQE说明既有多步准度量方法能改变原失败区域的排序，但当前100k主要执行指标没有解决task3，也没有严格扩展原成功集。E2的IQE备份失败与原生MQE的局部通道改善不能混称同一实现。后续优先检查既有失败轨迹对应的局部距离与动作选择，避免只修一个锚点或只看约束统计；本报告不启动新训练。

2026-10-30按用户指令停止QRL seed1与1M完整预算，禁止自动恢复或换seed证明非偶然性。它们不再作为完成门槛，也不能声称已作跨seed/完整预算核验。范围以CURRENT_LIMITS.md为准；当前尚未达到稳定闭环改善，总目标继续。

协议见[ENDPOINT_ANALYSIS_PROTOCOL.md](../../../experiments/ogbench_mqe/ENDPOINT_ANALYSIS_PROTOCOL.md)，复现入口为[analyze_endpoint.py](../../../experiments/ogbench_mqe/analyze_endpoint.py)。仅源码、协议、报告及引用PNG提交，原始模型、CSV/JSON/NPZ与日志保留本地。

分析脚本首次在报告格式化阶段因位置与统计字段重名而退出，原日志/源码副本/已计算场均保留。修复后以resume-analysis复用同目标、同网格的场数组，并用原模型的固定节点前向校验缓存；未重复训练或增加episode。
'''
    summary=dict(checkpoint_sha256=meta['checkpoint_sha256'],bank_sha256=sha(bank),official_endpoint={k:v[-1] for k,v in official.items()},
        paired=pairs,canonical_fields=fields,actual_task3_goal_checks=goal_checks,read_only=True,elapsed_seconds=time.time()-start,
        protocol_sha256=sha(Path(__file__).parent/'ENDPOINT_ANALYSIS_PROTOCOL.md'),source_sha256=sha(__file__),reused_validated_cache=resume_analysis)
    (output/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');(output/'report.md').write_text(report)
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True)
    p.add_argument('--resume-analysis',action='store_true');a=p.parse_args();run(a.root,a.output,a.resume_analysis)
