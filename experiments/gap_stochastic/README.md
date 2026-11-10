**第一阶段：自由到达流与准度量的小批量联合训练**

从仓库目录复现到一个尚不存在的目录：

```sh
env PYTHONDONTWRITEBYTECODE=1 ../quasimetric-rl/.venv/bin/python experiments/gap_stochastic/run_all.py /private/tmp/qrl-gap-stage1-reproduction
```

运行所需：Python、NumPy、SciPy、Matplotlib；本次 Python 3.9、NumPy 2.0.2、SciPy 1.13.1。完整版本与平台在结果 `manifest.json` 中。只用 CPU FP64，不读取真实环境 checkpoint，不下载数据。

`PROTOCOL.md` 固定了主组、初始化、seed、预算、优化器和成功判据。`core.py` 只有图数据、损失与准度量结构投影；`train.py` 不导入任何评估/最短路/最优流函数。六个到达流坐标各自独立，只有非负投影，守恒完全由损失学习。终点的出流为零是目标吸收语义。

`audit.py` 检查采样与总体梯度、枚举 batch=2 的精确估计期望，并核对对角度量投影。`evaluate.py` 在训练完成后计算真距离、独立 LP 最优流、吸收链策略成本，以及四种期望目标的完整 36 变量参考解。`diagnose.py` 从有限分布精确计算 U-statistic 路线方向梯度方差，检查零出流失败和流成本校正恒等式。`plot.py` 导出图、PDF 和 CSV。

所有主组使用同一个 seed 2001 的边序列与查询均匀数，两个查询混合仅改变查询映射。每组固定 20000 次更新、batch 256。期望梯度对照也固定 20000 步。没有 warm-start、动作排序标签、已知路径混合、守恒修复、挑 checkpoint 或多 seed。

主结果目录：`results/gap_stochastic_20261108`。`evaluation.json` 的 `policy.query_cost=null` 且 `proper=false` 表示不能从全部正质量查询到达目标，按协议到达成本为无穷；不是缺失评估。CSV 对此直接写 `inf`。`flow_cost` 是未必守恒的网络流成本，不得冒充执行成本。

原型使用完整准度量表，隔离表示容量的影响；不等同于 IQE 网络或 OGBench 性能验证。图上的原 QRL 使用线性推远与平均平方风险约束，也不等同于官方完整神经实现。

数值说明：SciPy 1.13 的 NNLS 在退化活跃集处可能异常，当前实现以 SLSQP 求解**同一个对角度量投影**作为后备。每次后备及其步数均记录，既不改学习目标，也不修复流守恒。已有主实验的异常恢复记录完整保留在结果目录中。
