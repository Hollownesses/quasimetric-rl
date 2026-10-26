# R0/R1 与 MPPI：seed606、固定50k、仅连续闭环评价

本轮四组 none、R0、R1、oracle，各32起点×planner seeds 314159/271828，共256回合。
不重新训练、不校准distance_scale、不挑选终点、不按执行结果更改起点/规划参数。
R0/R1使用 `results/task_source_balance_20261025/main/{R0,R1}/final.pt`。
严格转移encoder、projector、IQE至生产连续观测接口，distance_scale=1；没有图查表或吸附。
模型正距离作为正terminal cost加入MPPI；合法任务终态经原encoder canonicalization映射为同一目标，距离零。

MPPI核心源码逐字核对 e591c088a294c02f1b6f0244802768562dec449f。
H=10，N=128，noise_sigma=.8，temperature=1，terminal_weight=1，invalid_penalty=1e6，terminal_samples=128。
角速度限制[-3,3]，v=1，dt=.1，环境上限300步。每回合新建planner，nominal全0，NumPy RNG seed=planner_seed+7001。
正态噪声加当前nominal后裁剪，第0候选为nominal；指数权重加权平均整条控制序列，执行第一项，再移位warm start并尾部补0。
四组全部调用原MPPI并由真实env.step闭环，不限专家记录动作。相同起点/seed的首次候选完全一致，之后只保证扰动生成规则一致。
每worker Torch 1线程，四组并行；在线时间包含被动记录开销，不是纯内核benchmark。

起点采样seed=2026102601。U内部24点，x∈[2.9,5.1],y∈[2.0,5.3]；西出口8点，x∈[1.5,2.7],y∈[2.0,5.3]。
x/y均匀独立采样。heading在[-π,π)等分8层内均匀，内部每层3点，西出口每层1点。
仅验证有效且非终态，不使用模型/Oracle/图参考或结果筛选。保存完整pose与重合距离；新抽样不代表已证明独立泛化。
与全部训练起点、旧32探针、旧24新增探针及场景16局部探针检查精确pose重合；不拒绝靠近训练支持的点。
烟雾测试使用独立开发pose，不计入正式矩阵；所有失败轨迹保留。

Oracle复用原HybridAStarValueOracle，经环境、任务、动力学、成本、格点配置及源码逐项核验后复制旧缓存至新目录。
格点位置分辨率.25，24heading，5步恒定动作primitive，scales[-1,-.5,0,.5,1]，不可达1e6。
签名不匹配则在新目录构建，保留差异和计时；旧缓存永不写入。
Oracle仅通过terminal_value_provider查询候选末态，由相同MPPI控制。模型组绝不调用Oracle。
Oracle为拥有完整模型的离散格点参考；连续查询取最近位置/heading格点，不是连续精确最优值或严格性能上界。
与Oracle执行成本的差值仅是配对成本差，绝不称最优regret。

每episode保存实际状态/动作/逐步成本分项、终止类型、阶段完成、规划时间和全部候选末态、真正终值查询mask/值与诊断。
none组的末态只供覆盖统计，不查询终值。记录所有候选invalid/success标志，真正终值查询只发生在未完成且有效候选。
阶段沿用已有stage_metrics：离开U为到达/线段穿过西出口框；绕墙为先到y<1.4或y>5.9，再到x>5.7；终态由真实环境合法判定。
西出口起点的离开U阶段已完成，不纳入新增离开U分母；报告累计阶段比例与eligible分母。
失败实际成本保留并按终止类型单列，不能拿早停低成本当优势。主要成本/步数比较限共同成功episode。
R0→R1、oracle→R0/R1均按(state_id,planner_seed)配对，并按区域、heading、区域×heading、planner seed汇总。
事后重放全部执行动作验证状态/成本/终止；冻结源码、数据、模型、缓存和起点hash复核。
覆盖诊断仅使用物理训练观测支持：最近XY距离及该最近点heading差；另用半径.5的周期heading联合pose距离，保存联合最近点各分量。
覆盖不证明可达性。两个planner seeds共享32起点；单训练seed，不把64episode当作64独立地理样本。

从仓库根目录复现，使用全新RESULT路径（已完成目录不得覆盖）：
```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/qrl-r-mppi
PY=../quasimetric-rl/.venv/bin/python
RESULT=results/r_series_mppi_20261026/reproduction
mkdir -p "$RESULT/.."
$PY -m experiments.r_series_mppi.run prepare --root "$RESULT"
$PY -m pytest experiments/r_series_mppi/test_evaluation.py -q > "$RESULT/tests.log" 2>&1
$PY -m experiments.r_series_mppi.run smoke --root "$RESULT" > "$RESULT/smoke.log" 2>&1
$PY -m experiments.r_series_mppi.run freeze --root "$RESULT" --test-log "$RESULT/tests.log"
$PY -m experiments.r_series_mppi.run suite --root "$RESULT"
$PY -m experiments.r_series_mppi.analyze --root "$RESULT"
```
