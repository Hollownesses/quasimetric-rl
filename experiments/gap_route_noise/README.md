**守恒原始随机梯度的分流投影消融**

从仓库目录执行到一个尚不存在的输出目录：

```sh
env PYTHONDONTWRITEBYTECODE=1 ../quasimetric-rl/.venv/bin/python experiments/gap_route_noise/run_all.py /private/tmp/qrl-route-noise-reproduction
```

只新增两种既有查询混合下的投影组。旧无偏基线读取 `results/gap_stochastic_20261108`，不重训。运行配置、优化器、采样流、初始化和策略评估均复用第一阶段代码。`gap_stochastic.train.run` 仅增加一个默认关闭的采样梯度回调，未改变默认训练路径。

在新检出中，先生成第一阶段的基线输出。运行本目录的 `run.py <消融输出目录> --baseline <基线输出目录>`，然后依次把同一个消融输出目录传给 `evaluate.py` 和 `finish.py`。三个命令都用第一阶段的 Python 环境，并设置 `OPENBLAS_NUM_THREADS=1` 和 `OMP_NUM_THREADS=1`。基线生成命令见 [第一阶段复现说明](../gap_stochastic/README.md)。

`estimator.py` 单独提取原 U-statistic 守恒梯度，在进入 Adam 前从完整梯度中减去 `z * (z @ g_cons)`；距离梯度、缺口项和局部风险项不变。方向由六个自由流坐标的关联矩阵零空间确定，不使用成本排序。六个流仍独立训练，未投影到守恒集合。

`audit.py` 在训练前精确枚举 batch=2 的全部独立样本对，验证投影后分流分量为零、分量隔离正确、期望总梯度仍是第一阶段总体目标梯度。训练时对全部 40000 个批次重复零分量与改动范围审计，保存于两个 `.npz` 的 `gradient_audit` 数组。列名在 `manifest.json` 中。

这是梯度估计器消融。仍记录原随机标量损失，但投影后的梯度不必是该标量对当前批次的精确导数；正确性条件是其期望对应原总体目标。投影不施加于 Adam 的二阶矩、最终更新或策略提取；相应非线性作用只作诊断。

`evaluate.py` 沿用第一阶段相同的真实守恒、吸收链策略成本、全部查询可达性及联合判据，固定第 20000 步。`finish.py` 在同一参数点比较原/投影估计器分流方差并绘制完整曲线，不增加训练。

主产物位于 `results/gap_route_noise_20261109`：`report.md`、`summary.csv`、`evaluation.json`、`preflight.json`、`noise_diagnostics.json`、`comparison.png` 和每组完整快照/逐批审计。未修改第一阶段原始结果，不更换 seed，不增加其它消融。
