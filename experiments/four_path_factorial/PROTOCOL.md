# 四路径参数化 × 约束优化对照协议

2026-10-16，在主实验运行前登记。300-step pilots 全部保留，不用于最终比较。

- 冻结原 four-path bank 与 verified replay：335 真实边、4 anchors；采用生产 encoder 的完整 42 维状态 canonicalization。精确状态共用节点，禁止位置近邻拼接。
- 生产 `configure_global_push_sampling(seed=42)` 重建固定 B1；task/B1 权重 0.6/0.4。四组统一对 335 个真实 source 求完整期望，不增加数据或辅助 loss。旧 stochastic 20k 只作背景，不能混入配对主表。
- encoder 保留 42→512→512→128→512→2048；自由 latent 是每节点 2048 维直接 IQE 输入，移除 encoder 和 projector 的共享耦合。IQE 均为 64×32，包含原可学习 max/mean reduction。
- 同一 critic 初始 checkpoint 的 projected latent 初始化自由表，两种参数化初始函数相同；同参数化两种 solver 的 state_dict 逐字节相同。未使用图距离初始化。
- barrier 原权重 500、margin/temperature .01、64 工作边 top8、每25步更新 audit cache。实现直接继承生产 updater。为了统一审计，四组每一步均计算全部339条边；barrier反传仍仅选中top8。
- PHR 复用已有 mean-form 公式，339个独立乘子从0开始，不拟合网络；每步使用完整 pre-primal-update 残差作 `lambda=relu(lambda+eta*h)`。记录其非硬可行性，不暗示自动收敛。
- AdamW weight_decay=0。encoder lr=2e-5；latent lr=.01。300步试跑中 latent .05 在 barrier 下仍有大量违反，.01消除了初始违反，因此用更保守的 .01。两种 solver 在同参数化下 lr 相同；参数化之间 lr 不同，因而该因素代表“参数化及适合其尺度的优化”，不是纯函数类隔离。
- PHR rho=100, eta=1：与 rho=10, eta=.1 比较的300步pilot中，较大值控制残差更好。两种参数化共用此设置。不根据图距离标签选择参数。
- 主实验：critic 0 的四格，各20,000次 primal update，常数lr。复核：独立初始化的 critic 1，四格各5,000步。二者来自同一个 seed=42 的初始双critic checkpoint；不称为多个随机seed或20k统计显著性。
- 每250步保存全支持push、335边残差、anchors、路径预测和余量分解；2k/5k/10k/20k留模型。339个乘子以固定replay顺序保存。
- 通过标准：所有真实边 max positive residual ≤1e-5、terminal=0、同支持push gap≤1%、task NRMSE≤2%。同时给出保守统一缩放到可行后的push，避免以违反约束的高push判断成功；该缩放只用于评估。
- 参考距离仅在评估中用反向Dijkstra计算，是记录图距离，绝不是连续环境最优距离。训练入口只收到节点、pair索引和一步成本，不接收参考距离、suffix、LP dual。
- 记录模型/数据/节点/pair/成本/代码指纹、线程数、实际训练秒数和参数量。主表按更新步数对齐，图同时显示实际时间。相同更新数并不等同计算预算。

## 运行

仓库根目录：

```sh
MPLCONFIGDIR=/tmp/qrl-factorial-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 ../quasimetric-rl/.venv/bin/python experiments/four_path_factorial/run.py \
 --output results/NEW_RUN --parameterization latent --method phr \
 --critic 0 --steps 20000 --lr .01 --rho 100 --dual-step 1 --threads 2
```

另三格修改 `parameterization` / `method`，encoder 使用 `--lr .00002`。critic 1复核设 `--critic 1 --steps 5000`。输出目录必须不存在，防止覆盖旧产物。所有pilot和正式实验从同一初始checkpoint重新开始，绝不续用pilot权重。

一次运行完整正式套件（4个CPU进程，每个2线程，非多agent）：

```sh
../quasimetric-rl/.venv/bin/python experiments/four_path_factorial/run_suite.py \
 --output results/NEW_FACTORIAL --jobs 4
```

独立分析：`summarize.py OUTPUT_ROOT` 与 `audit.py OUTPUT_ROOT`。`audit.py` 会重新加载完成的 checkpoint 并核验全部边、保存的曲线值与实际 IQE 可行缩放，以及重算历史20k对照。

## 正式自由 latent 重跑前的控制修正

初次实现把 latent 的 .01 lr 也用于 IQE max/mean 混合系数，形成不必要的头部优化混杂。发现后中止两个自由 latent 运行，完整保留在 `exploratory_head_lr/`；它们不是正式表格数据。正式四格的 IQE 头 lr 统一为2e-5，仅 latent 表用 .01。encoder 原实现所有参数均为2e-5，故无需重跑。新配置再次做300步试跑与验证，然后自由 latent 从相同初始 checkpoint 重启。保留新旧脚本指纹与此修正记录，不把早期探索曲线混入正式比较。
