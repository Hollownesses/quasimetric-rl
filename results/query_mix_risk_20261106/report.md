# OGBench PointMaze 查询分布与局部平方预算的 2×2 对照

2026-11-06 · pointmaze-medium-navigate-v0 · 官方 QRL seed 0 的 100k checkpoint

**观测方向符合机制预期，预定预算对照无法判别。** 紧档两组的实测 $R$ 均较对应松档低约 80%。B 重紧档的 $R=0.010186$，超出预先冻结的 0.01 上限约 0.000186，触发了预定的无法判别规则。 [预定协议](../../experiments/query_mix_risk/PROTOCOL.md)

| 查询分布 | 名义 $R$ 上限 | 实测 $R$ | A 后缀 $J_A$ | B 后缀 $J_B$ | $\widehat Q_A$ | $\widehat Q_B$ | $m=\widehat Q_B-\widehat Q_A$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 重 | 0.05 | 0.044643 | 119.9808 | 108.9161 | 143.0428 | 141.7337 | −1.3091 |
| B 重 | 0.05 | 0.049911 | 105.1756 | 141.0473 | 149.7048 | 151.2015 | +1.4967 |
| A 重 | 0.01 | 0.007936 | 84.4923 | 79.7866 | 106.5595 | 105.4644 | −1.0951 |
| B 重 | 0.01 | **0.010186** | 75.1398 | 102.1772 | 113.7588 | 114.9233 | +1.1646 |

$J_i=d(z_i,g)$ 是固定下游锚点的估价。$\widehat Q_i=1+d(s'_i,g)$ 是固定动作的成本评分，$s'_i$ 取同一 MuJoCo 初态实际执行该动作后的观测。$R=\operatorname{mean}([d(x,x')-1]_+^2)$ 在全部 100 万有效训练边上计算。 [全精度表](results.csv)

把同一份查询权重从 B 移到 A，$J_A$ 在松档上升 14.8053，$m$ 下降 2.8058。紧档对应变化为 +9.3524 和 −2.2597。换权在两档都翻转了动作排序。紧档的原始分差响应幅度小 19.5%，除以固定 2048 个验证查询的平均距离后仍小 10.0%。两档换权后，真实后继的两个动作成本评分都下降，B 的降幅更大。 [配对差分与尺度读数](results.json)

## 固定探针与匹配依据

- $s=(4,8)$，速度为零，$g=(8,20)$。A 动作 $(1,0)$ 的真实后继为 $(4.2,8)$，B 动作 $(0,1)$ 的真实后继为 $(4,8.2)$。沿环路两侧的格心续行分别在 116 和 115 步进入同一目标圆。后缀锚点为 $z_A=(12,8)$ 和 $z_B=(0,12)$。
- 背景起点取自训练轨迹。记录必须从对应侧第一格的格心 1.5 单位内出发，按该侧格序穿过后缀锚点的 1.5 单位邻域，再于同一 episode 进入半径 1 的目标圆。A、B 分别筛得 9 和 30 条，用剩余步数一对一匹配，各保留 9 条。两池从所选起点到目标圆的平均步数分别为 97.11 和 97.78，所有查询共享目标点 $g$。 [格序、逐条行号与物理初态](frozen.json)

## 复现

每组 512 次 critic 更新，batch 256。四组共享父模型和原 E6 seed 2001 的离线边与随机查询流。推远项为 50% 官方随机查询、5% 固定评估查询 $(s,g)$、45% 对应侧背景池均值。保留 value Adam 和学习率 0.0003，冻结 actor 与潜在动力学。两个预算共用快速对偶更新，从原 $\lambda$ 开始，以 0.85 倍预算作为控制设定点。端点训练及全量扫描共 524.7 秒。重新加载后核验了共享采样哈希、真实后继、两动作评分、全量 $R$ 汇总及各组 9216 条抽查边的前向输出。结论范围是这对动作和这次固定采样流的短续训。 [核验记录](verification.json)

从本仓库根目录运行，指定持有固定源码、数据和 checkpoint 的历史 worktree，以及新输出目录。

~~~bash
og_raw='/Users/z./.codex/worktrees/5670/quasimetric-rl-topology-improvements'
og_python="$og_raw/experiments/ogbench_official_qrl/.venv/bin/python"
query_out=results/query_mix_risk_reproduction
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$og_python" experiments/query_mix_risk/experiment.py freeze "$og_raw" "$query_out"
"$og_python" experiments/query_mix_risk/experiment.py run "$og_raw" "$query_out"
~~~

输入与协议哈希在冻结配置中。 [运行环境版本](run_started.json)
