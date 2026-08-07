# 🏆 CC-Harness vs LangGraph — Agent 框架对比报告

> 生成时间: 2026-08-06 10:04:13
> 评测问题数: 5
> 模型: qwen-plus

---

## 📊 核心指标对比

| 指标 | CC-Harness (自研) | LangGraph | 差异 | 胜出 |
|:---|---:|---:|---|:---:|
| **成功率** | 100.0% | 100.0% | ➡️ → 0.0% | 🤝 平 |
| **平均延迟** | 9.5s | 26.5s | 📉 ↓ 64.0% | 🏆 CC |
| **关键词命中率** | 38.0% | 22.0% | 🚀 ↑ 72.7% | 🏆 CC |
| **平均推理轮次** | 2.2 | 0.0 | N/A | — |

---

## 🔍 架构对比

| 维度 | CC-Harness (自研) | LangGraph |
|:---|:---|:---|
| **核心引擎** | AgentLoop 异步循环 | StateGraph 状态图 |
| **流程定义** | 运行时 LLM 动态决策 | 编译时静态定义 |
| **依赖体量** | openai + chromadb + pydantic (~6 包) | LangChain 生态 (~50+ 包) |
| **RAG 定位** | 普通 Tool，按需调用 | 图节点，框架级耦合 |
| **多 Agent** | Debate / Map-Reduce / Hierarchy (3 模式) | 多 StateGraph 共享 State |
| **上下文管理** | 内置 LLM 分层摘要 + Token 预算 | 开发者自行实现 |
| **可观测性** | AgentTracer 内置追踪 + 回放 | 依赖 LangFuse 外部集成 |
| **MCP 支持** | ✅ 原生 MCP Client + Server Registry | ❌ 需额外集成 |
| **评测体系** | ✅ 内置 EvalRunner + LLMJudge | 依赖外部 LangSmith |

---

## 📈 各题详情

| ID | 问题 | CC-Harness | LangGraph |
|:---|:---|:---|:---|
| q001 | Agentic RAG 和普通 RAG 有什么区别？... | 延迟7.75s 命中20.0% | 延迟39.55s 命中0.0% |
| q002 | Claude Code 的 AgentLoop 与 LangGraph 的 St... | 延迟6.79s 命中0.0% | 延迟22.88s 命中0.0% |
| q003 | LangGraph 使用什么机制实现 Agent 工作流？... | 延迟5.64s 命中50.0% | 延迟3.08s 命中0.0% |
| q004 | GraphRAG 和 Agentic RAG 各有什么优缺点？... | 延迟20.44s 命中20.0% | 延迟40.43s 命中60.0% |
| q005 | 在 Agent 框架中，什么是上下文压缩？为什么需要它？... | 延迟7.06s 命中100.0% | 延迟26.62s 命中50.0% |

---

## 💡 关键发现

1. **架构灵活性**: CC-Harness 的 AgentLoop 运行时动态决策机制使 Agent 能够根据实际情况自主调整策略，而 LangGraph 的预定义图结构在面对意外场景时可能束手束脚。

2. **部署复杂度**: CC-Harness 仅依赖 6 个核心包，Docker 镜像 < 200MB；LangGraph 项目依赖 50+ 包，镜像 > 1GB。

3. **多 Agent 能力**: CC-Harness 实现了 3 种多 Agent 协作模式（辩论/分发归并/层级委托），每种模式有独立的编排器和上下文隔离策略。

4. **MCP 生态**: CC-Harness 原生支持 MCP 协议，可通过标准协议对接任意外部工具/数据源——这在 2026 年的 Agent 工程中是关键技术能力。

5. **评测体系**: CC-Harness 内置了完整的评测框架（Benchmark + LLMJudge + 指标计算），支持与 LangGraph 项目的 A/B 对比。

---

## ⚠️ 数据说明

- CC-Harness 数据来自本轮实际运行
- LangGraph 数据来自实际运行
- 关键词命中率作为 LLM Judge 的轻量替代指标
- 同一套 5 道 Benchmark 问题在两个框架上分别运行

---

*🤖 由 CC-Harness Agent Eval Framework v2.0 生成*
