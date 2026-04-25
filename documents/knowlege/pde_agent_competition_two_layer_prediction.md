# 神经算子 PDE Agent 赛题的两层预测结构

## 核心结论

这个赛题不能被理解为“只在公开 PDEBench 文件上训练并验证一个短窗模型”。更准确的结构是：

1. 本地公开数据只支持短窗监督训练与验证。
2. 赛方最终评测要求模型或提交程序在隐藏长窗输入上生成 190 步预测。
3. 因此本地优化目标必须同时包含短窗精度和长时稳定性的代理诊断。

换句话说，本地能直接看到的是 `10 -> 31`，最终要服务的是 `10 -> 190`。

## 数据尺度

公开 Task-1 固定物理环境训练数据是：

```text
1D_Burgers_Sols_Nu0.001.hdf5
```

真实检查结果：

```text
tensor raw shape: [10000, 201, 1024]
available keys: t-coordinate, tensor, x-coordinate
dtype: float32
```

官方 checkpoint 训练/验证使用：

```text
reduced_resolution_t = 5
reduced_resolution = 4
```

所以公开训练文件在模型尺度下约为：

```text
time: 201 -> 41
space: 1024 -> 256
reduced shape: [N, 41, 256]
```

如果使用前 10 个 reduced time steps 作为输入，本地最多有：

```text
context_steps = 10
forecast_steps = 31
total_steps = 41
```

这就是本地监督验证上限。

## 赛方长窗评测

规程描述了新的长时验证数据：

```text
raw: 1000 time steps, 1024 spatial points
reduced: 200 time steps, 256 spatial points
```

最终预测文件要求：

```text
task{N}_pred.hdf5 shape: (N, 200, 256)
```

其中：

```text
steps 0-9: 必须与输入初始条件完全一致
steps 10-199: 模型预测的 190 个未来 reduced steps
```

因此最终推理问题是：

```text
10 reduced steps -> 190 reduced steps
```

这不是公开训练文件可以直接监督验证的问题。

## 两层预测

### 第一层：本地短窗训练与验证

目的：

- 训练或微调 FNO/U-Net/自定义神经算子。
- 比较架构、损失函数、归一化、rollout 方式。
- 给 Agent 提供可重复、低成本、可量化的实验反馈。

本地可做：

```text
input:  [N, 10, 256]
target: [N, 31, 256]
```

评价指标：

- 逐时间步 MSE / Rel-MSE。
- 分段误差，模拟最终评分的早中晚段权重。
- rollout 误差随时间增长曲线。
- 频谱误差。
- 梯度/总变差误差。
- shock 位置与振幅误差。

局限：

- 只能看到 31 个未来 reduced steps。
- 不能直接验证 190 步后是否稳定。
- 短窗误差最优的模型可能长窗发散或过度耗散。

### 第二层：长窗代理诊断与最终推理

目的：

- 在没有 190 步 ground truth 的本地条件下，判断模型是否可能在最终评测中稳定。
- 让 Agent 不只盲目优化短窗 MSE，而是围绕 Burgers 方程的物理结构做策略选择。

本地代理做法：

```text
input: [N, 10, 256]
rollout: 递推到 [N, 200, 256]
```

即使没有 190 步 GT，也可以检查：

- NaN/Inf 是否出现。
- 解场幅值是否爆炸。
- 总能量或 L2 范数是否出现非物理增长。
- 高频能量是否异常堆积。
- 高频能量是否过度衰减导致 shock 被抹平。
- 总变差是否异常。
- 空间梯度统计是否偏离训练集分布。
- Burgers 黏性项对应的耗散趋势是否合理。
- 预测序列是否坍缩到常数或均值场。

最终当赛方提供 `task1_test.hdf5` / `task2_test.hdf5` 或平台环境中存在这些文件时，提交推理流程应当：

```text
1. 读取测试输入前 10 个 reduced steps。
2. 初始化输出 pred shape = (N, 200, 256)。
3. 复制 pred[:, :10, :] = input[:, :10, :]。
4. 用模型递推生成 pred[:, 10:200, :]。
5. 写入 task{N}_pred.hdf5。
6. 记录 inference_time。
```

## 对 Agent 设计的影响

Agent 不能只执行“短窗训练 -> 看 validation MSE -> 选最优”的普通 AutoML 流程。它需要明确区分三类信号：

1. 短窗监督信号：本地 `10 -> 31` 的真实误差。
2. 长窗代理信号：无 GT 情况下的稳定性、频谱、物理一致性。
3. 提交格式信号：最终必须输出 `(N, 200, 256)`，且前 10 步严格保真。

推荐 Agent skills：

- `inspect_dataset_shape`: 读取 HDF5 key、shape、dtype、降采样后长度。
- `build_local_windows`: 从公开训练文件构造 `10 -> 31` 数据。
- `train_short_window_model`: 训练或微调模型。
- `evaluate_short_window`: 计算短窗误差与分段误差。
- `rollout_long_proxy`: 自回归推到 200 步。
- `diagnose_long_stability`: 检查幅值、频谱、总变差、梯度、NaN/Inf。
- `compare_baselines`: 与 FNO/U-Net checkpoint 或简单物理 baseline 比较。
- `build_submission_prediction`: 生成 `task{N}_pred.hdf5`。
- `validate_submission_shape`: 检查 shape、前 10 步保真、dtype、文件结构。

## 当前项目状态

已经具备：

```text
data/pdebench/burgers/1D_Burgers_Sols_Nu0.001.hdf5
data/pdebench/checkpoints/burgers_FNO.tar
data/pdebench/checkpoints/burgers_Unet.tar
```

已验证：

```text
MD5 OK
tensor shape = [10000, 201, 1024]
reduced shape with sample_limit=1 = [1, 41, 256]
short smoke with forecast_steps=31 passes DATA_INSPECTED
```

当前缺口：

- 本地还没有赛方隐藏长窗测试输入。
- 还没有本地 scorer 实现赛方分段评分。
- 还没有用于 200 步 rollout 的模型训练/推理 harness。
- 还没有从官方 checkpoint 解包并加载 FNO/U-Net 的 baseline 复现路径。

## 下一步建议

优先级顺序：

1. 实现本地 `10 -> 31` window builder。
2. 实现赛方评分近似版 local scorer。
3. 解包并尝试加载官方 FNO checkpoint。
4. 建立 baseline rollout 脚本。
5. 增加长窗代理诊断指标。
6. 最后再接 Agent skill，让 Hermes 自动驱动这些工具。

这条路径比继续寻找可能不存在的公开测试附件更可靠。
