# [Feature]: 新增FusedAdamW 优化器

> 来源: gitee_issue/mindspore/mindspore | 可信度: gitee_issue_labeled (0.75) | 标签: feature, test
> URL: https://gitee.com/mindspore/mindspore/issues/IDKYI0
> 爬取时间: 2026-08-12T12:18:20.874064

### 🚀 背景描述
需求来源：ADS中SparseBEV和GOD网络，由于存在parameter较多且相对shape较小，优化器更新参数时有大量的H2D操作，从而降低了网络性能。
优化器更新参数时需反复把计算从 Host 推向 Device，每一次下发都带来一次 H2D 开销，训练速度随之下降。这些底层运算逻辑一致，区别仅在于目标数据，因此先在 Host 端一次性完成全部计算，再集中下发，可大幅减少 H2D 次数，提升性能。但是，合并后的批量下发需要更大的内存，无法再利用 Device 上的零散小块内存，因此融合优化器会占用更多内存空间。

SparseBEV和GOD网络由于存在parameter较多且相对shape较小，有大量的H2D操作，影响网络性能。为了减少网络中h2d的下发操作，新增优化器 FusedAdamW，提升网络性能。

### 设计思路

**主要方案**
对数据重排，然后对入参进行分组拼接，拼接后一次性下发，来减少 h2d 操作。

**具体实现如下**：
基于已有的优化器 optimizer 实现一个新的融合优化器 FusedAdamW，具体实现如下：
1. `contiguous_params` 里把本组参数展平后连续存放，让 `AdamW` 底层拿到 一块连续地址；
2. `mint.cat(..., dim=0)` 把 N 个小梯度拼成 **1 个大梯度**，原来要下发 N 次，现在只下发 **1 次**；
3. 所有 `continuous_*` 列表里每个元素对应 **一组** 参数，优化器只需要按照 **组数** 循环；
4. 最后只调 1 次 ·self.adamw_opt(...)· 完成整组更新，底层只需1次下发。

**作用**：
减少h2d的下发操作，提升性能。


### 涉及到的对外API

新增对外接口 FusedAdamW
【用法】
1. FusedAdamW(params, *, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, maximize=False)
2. FusedAdamW(params=group_params, lr=1e-3)，其中`group_params`为自定义参数组

参数：
- params (Union[tuple, list])：待优化的参数列表或自定义参数组。
- lr (float, 可选)：学习率。默认值为 1e-3。
- betas (Tuple[float, float], 可选)：用于计算梯度及其平方的运行平均值的指数衰减率。默认值为 (0.9, 0.999)。
- eps (float, 可选)：为提升数值稳定性而添加到分母中的项。必须大于 0。默认值为 1e-8。
- weight_decay (float, 可选)：权重衰减（L2 惩罚）。默认值为 1e-2。
- amsgrad (bool, 可选)：是否使用 AMSGrad 算法。默认值为 False。
- maximize (bool, 可选)：若为 True，则最大化目标函数对应的参数；若为 False，则为最小化。默认值为 False。

输入：gradients (tuple[Tensor])：params 对应的梯度张量元组。

返回：无。该操作用于原地更新参数。

约束：
- lr 必须为浮点数且不小于 0。
- eps 必须大于 0。
- betas 中的每个值必须在区间 [0, 1) 内。
- weight_decay 必须不小于 0。

异常：
- ValueError：若学习率不是浮点数。
- ValueError：若学习率小于 0。
- ValueError：若 eps 小于 0。
- ValueError：若 betas 不在区间 [0, 1) 内。
- ValueError：若 weight_decay 小于 0。

支持平台：Ascend

### 与其他模块的相关性描述
**测试计划**：
1. 明确优化器验收规格，由测试验收
2. 开发自验证保障优化器的功能与精度

**测试设计**：
1. 测试验收
2. 开发自验证


**1. 测试验收规格**：

1. 功能：原使用AdamW (mindspore.mint.optim.AdamW ) 的用例，改成 FusedAdamW 后用例无异常。
2. 精度：原使用AdamW (mindspore.mint.optim.AdamW ) 的用例，改成 FusedAdamW 后用例零偏差对齐。
3. 性能：不同网络的收益取决于网络中parameter的规格，其中SparseBEV和GOD网络整网收益5%，parameter规格如下：

| 网络| Parameter数量 | 中小shape个数（shape ≤ 192）|
| ---| ---| ---|
| SparseBEV   | 762  | 384|
|GOD | 570 |479|

SparseBEV 网络
| 维度 | 个数 | Shape（中小shape，shape ≤ 192） |
|---|---|---|
| **一维** | **309** | (32,), (64,), (128,), (96,), (192,), (100,), (10,), (5,), (3,), (2,), (1,) |
| **二维** | **5** | (64, 10), (1, 64), (32, 64), (1, 32), (240, 3) |
| **四维** | **70** | (32,3,3,3), (128,64,3,3), (128,128,3,3), (64,64,1,1), (128,64,1,1), (128,128,1,1),  (32,128,3,3)等 |

GOD 网络
| 维度 | 个数 | Shape（中小shape，shape ≤ 192） |
|---|---|---|
| **一维** | **328** | (32,), (64,), (128,), (96,), (192,), (101,), (33,), (16,), (10,), (9,), (5,), (3,), (2,), (1,) |
| **二维** | **1** | (64, 10) |
| **四维** | **150** | (32,3,3,3), (64,32,3,3), (64,64,3,3), (128,64,3,3), (128,128,3,3), (64,64,1,1), (128,64,1,1), (128,128,1,1)等 |

**2. 开发自验证场景**：
对FusedAdamW优化器进行功能、精度以及边界场景测试，新增测试用例（tests/st/mint/optim/test_fused_adamw.py）

1. FusedAdamW(params, *, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, maximize=False)
2. FusedAdamW(params=group_params, lr=1e-3)，其中`group_params`为自定义参数组


| 序号 | 测试函数名 | 主要测试场景 | 子场景/特殊点 |
|------|-----------|-------------|-------------|
| 1 | `test_mint_optim_fused_adamw_graph_compile` | **基础功能、图模式编译** | 默认参数、GRAPH_MODE编译 |
| 2 | `test_mint_optim_fused_adamw_group1_graph_compile` | **参数分组、不同超参数** | 分组设置不同lr/weight_decay |
| 3 | `test_mint_optim_fused_adamw_group2_graph_compile` | **参数分组、默认全局lr** | 分组覆盖全局默认值 |
| 4 | `test_mint_optim_fused_adamw_pynative` | **PyNative模式执行** | 动态图模式单步训练 |
| 5 | `test_mint_optim_fused_adamw_basic` | **基础参数组合、双模式** | 默认参数、自定义betas/weight_decay、GRAPH/PYNATIVE |
| 6 | `test_mint_optim_fused_adamw_basic_amsgrad` | **AMSGrad功能、双模式** | 开启AMSGrad、不同参数组合 |
| 7 | `test_mint_optim_fused_adamw_basic_maximize` | **最大化优化模式、双模式** | maximize=True、与AMSGrad组合 |
| 8 | `test_mint_optim_fused_adamw_basic_group` | **分组参数基础、双模式** | 分组参数默认配置 |
| 9 | `test_mint_optim_fused_adamw_basic_lr_dynamic` | **动态学习率、双模式** | 学习率调度器 |
| 10 | `test_mint_optim_fused_adamw_group_lr_dynamic` | **分组+动态学习率、双模式** | 两种功能组合 |
| 11 | `test_mint_optim_fused_adamw_group_lr_dynamic_change_param` | **参数动态更新、双模式** | 训练中改变优化器参数 |
| 12 | `test_mint_optim_fused_adamw_dtype_base` | **边界值、极端参数** | betas=(0.0, 1.0) |
| 13 | `test_mint_optim_fused_adamw_dtype_1000_3000` | **高维张量、数值精度** | 1000×1000大矩阵、严格loss容差 |
| 14 | `test_mint_optim_fused_adamw_dtype_float32` | **3D张量、分组参数** | 25×25×25三维张量 |
| 15 | `test_mint_optim_fused_adamw_dtype_20_30` | **复杂功能组合** | 分组+动态lr+AMSGrad+参数更新 |
| 16 | `test_mint_optim_fused_adamw_dtype_200_300` | **最大化+自定义参数** | maximize=True+自定义eps/weight_decay |
| 17 | `test_mint_optim_fused_adamw_bfloat16` | **bfloat16精度** | 低精度浮点数、AMSGrad |
| 18 | `test_mint_optim_fused_adamw_float32_4d` | **4D张量** | 四维张量形状兼容性 |
| 19 | `test_mint_optim_fused_adamw_float32_5d` | **5D张量** | 五维张量形状兼容性 |
| 20 | `test_mint_optim_fused_adamw_float32_6d` | **6D张量、零学习率** | 六维张量、lr=0.0边界 |
| 21 | `test_mint_optim_fused_adamw_float32_7d` | **7D张量** | 七维张量形状兼容性 |
| 22 | `test_mint_optim_fused_adamw_float32_8d` | **8D张量** | 八维张量形状兼容性 |
| 23 | `test_mint_optim_fused_adamw_dtype_base_discontinuous_tensor` | **非连续张量** | 内存不连续的数据处理 |
| 24 | `test_mint_optim_fused_adamw_dtype_float16_group_true_5d` | **float16+5D+AMSGrad+maximize** | 多特性组合压力测试 |
| 25 | `test_mint_optim_fused_adamw_dtype_float64` | **数据类型错误处理** | float64不支持→TypeError |
| 26 | `test_mint_optim_fused_adamw_withoutparam` | **空参数列表错误** | 无训练参数→ValueError |
| 27 | `test_mint_optim_fused_adamw_lr_neg` | **负学习率错误** | lr=-0.1→ValueError |
| 28 | `test_mint_optim_fused_adamw_dtype_20_30_lr_int` | **学习率类型错误** | lr为整数→TypeError |
| 29 | `test_mint_optim_fused_adamw_dtype_betas_neg` | **负betas值错误** | betas为负数→ValueError |
| 30 | `test_mint_optim_fused_adamw_dtype_weight_neg` | **负weight_decay错误** | weight_decay为负→ValueError |


### 其他信息



