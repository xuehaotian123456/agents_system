# [RFC]: HyperOffload动态图场景自动优化

> 来源: gitee_issue/mindspore/mindspore | 可信度: gitee_issue_labeled (0.75) | 标签: RFC
> URL: https://gitee.com/mindspore/mindspore/issues/IDLRGL
> 爬取时间: 2026-08-12T12:18:20.863372

### 背景与目标描述.

在大模型的训练与推理过程中，随着网络规模和节点数量的持续增长，显存占用逐渐成为系统性能与可扩展性的主要瓶颈。单纯依赖扩充设备侧显存来缓解该问题，不仅成本高昂，也难以满足长期演进需求。为此，我们引入了 HyperOffload 功能，通过对计算图执行序列进行深度分析，结合节点间的数据依赖关系，对部分权重参数及中间计算结果进行智能卸载与预加载调度，从而有效降低网络运行过程中的显存峰值占用。

与此同时，HyperOffload 通过对数据迁移时机进行精细化调度，最大程度减少主机与设备之间数据传输对计算性能带来的影响。该特性的核心目标是在尽可能保持计算性能无损的前提下，显著降低模型的显存需求，从而提升模型运行的整体性价比与部署灵活性。

当前，HyperOffload 功能仅支持在具备全局视角的静态图场景下启用；对于动态图场景，由于缺乏完整的全局执行与数据依赖信息，尚无法充分发挥 HyperOffload 的优化能力。本次特性旨在通过对动态图执行过程进行 trace，获取节点使用关系的详细信息，从而构建近似的全局视角，并在此基础上使能 HyperOffload 的全局显存优化能力。

### 建议的方案.

整体方案分为两个阶段：

WarmUp 阶段：
通常对应网络运行的第一个 Step。在该阶段需要首先保证网络能够稳定运行而不发生 OOM，同时通过 trace 机制采集动态图执行过程中的关键信息，包括生成对应的 ANF 图以及完整的执行序列，为后续分析提供基础数据。

Stable 阶段：
在获得执行序列和 ANF 图后，对其进行深入分析，自动识别需要进行卸载优化的节点，并生成相应的卸载与预加载策略。该策略将反馈给动态图基础框架，由基础框架据此对执行流程进行重新规划，动态插入卸载、加载以及异步流相关的节点，从而实现显存占用的全局优化。

整体流程如下：
![Image description](https://foruda.gitee.com/images/1769156295391189541/1659beb1_8204759.png "HyperOffloadPynative.png")

针对这个流程，需要将目前的HyperOffloadOptimizer的代码解耦出来（目前的HyperOffloadOptimizer是绑定在静态图KernelGraph上面的），整体解耦后的流程如下：
![Image description](https://foruda.gitee.com/images/1769156445239551337/fd0a169e_8204759.png "decouple.png")


### 涉及到的对外API

该特性需要添加新的对外API，目前暂定的API规格如下（后续需要进一步评审）


```python
from mindspore._c_expression import OffloadManager

net = Net()
net.auto_offload()

...

manager.analyse() # Trace code end and enable hyperoffload
```


### 测试验证

可以在动态图任意网络或者用例上使能该功能，结果正确，显存降低即可通过验证。

### 期望的反馈时间.

2026/3/30

### CC List.

 @zh_qh  @liangzhibo  @luochao60 @huangyi

### 其他补充信息.

该特性已穿刺流程，验证效果为主。接口相关的设计需要后续进一步讨论。

### Before submitting a new issue...

- [x] Make sure you already searched for previous [RFCs](https://gitee.com/mindspore/mindspore/issues?q=is%3Aall+label%3ARFC+sort%3Arecently-updated).


