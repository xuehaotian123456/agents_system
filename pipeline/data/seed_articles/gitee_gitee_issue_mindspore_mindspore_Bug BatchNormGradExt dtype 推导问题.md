# [Bug]: BatchNormGradExt dtype 推导问题

> 来源: gitee_issue/mindspore/mindspore | 可信度: gitee_issue_labeled (0.8) | 标签: bug
> URL: https://gitee.com/mindspore/mindspore/issues/IDLSBG
> 爬取时间: 2026-08-12T12:18:20.864077

### Checklist

- [x] 1. 我已经搜索过相关问题，但没有得到预期的帮助. (https://gitee.com/mindspore/mindspore/issues)
- [x] 2. 最新版本中该错误尚未修复.
- [x] 3. 请注意，如果您提交的Bug描述缺少相应的环境信息和最小可复现的demo，我们将很难复现和解决该问题，从而降低收到反馈的可能性，甚至该问题将被关闭.


### 🐞 问题详细描述

# BatchNormGradExt dtype 推导问题

## 背景
`BatchNormGradExt` 是 `batch_norm` 反向的扩展算子。该算子在 MindSpore 中通过类型推导确定
输出 `dx/dweight/dbias` 的数据类型。

## 现象
- 当输入为 `float64` 时，CPU 后端反向结果出现细微误差。
- 通过日志发现 `dweight` 和 `dbias` 变为 `float32`，与期望不一致。

## 根因
类型推导逻辑将 `dweight/dbias` 固定为 `float32`，忽略了 `weight` 与 `input` 的真实类型。
这会导致 `dweight/dbias` 输出张量类型被强制为 `float32`，从而在 `float64` 场景中产生误差。

## 修复
将 `BatchNormGradExt` 的类型推导规则改为：
- `dx` 的 dtype 与 `input` 一致。
- `weight` 不为 `None` 时，`dweight/dbias` 与 `weight` 一致。
- `weight` 为 `None` 时，`dweight/dbias` 与 `input` 一致。

## 覆盖测试
新增 C++ UT 覆盖以下场景：
- `input=float64`，`weight=float32` -> `dweight/dbias=float32`
- `input=float64`，`weight=None` -> `dweight/dbias=float64`

对应测试文件：`tests/ut/cpp/ops/test_ops_batch_norm_grad_ext.cc`。


### 详细的环境信息描述

CPU plugin后端

### 其他辅助信息



### 版本信息

master

