# Agent 双引擎系统 — 技术规格书

> **项目名**：Agent 双引擎系统 (Agent Dual-Engine System)
> **根目录**：`E:\agent-system\`
> **最后更新**：2026-08-06
> **开发环境**：Windows 11 · Python 3.11 · Conda (rag_env)

---

## 一、项目概述

### 1.1 一句话

**双引擎 Agent 系统**：LangGraph StateGraph 负责 Pipeline 层（爬取→入库→推送），CC-Harness AgentLoop 负责 Interaction 层（对话→工具调用→报告），A2A 协议互通。

### 1.2 目录结构

```
E:\agent-system\
├── TECHNICAL_SPEC.md      ← 本文档
├── interaction\            ← Interaction 层 (CC-Harness 自研框架)
│   ├── harness/            # 核心引擎
│   ├── server/             # FastAPI + Web UI
│   ├── profiles/           # Agent 配置
│   ├── demos/              # Demo 脚本
│   ├── benchmarks/         # 对比评测
│   ├── tools/              # 工具系统
│   └── infrastructure/     # 基础设施
│
└── pipeline\               ← Pipeline 层 (LangGraph DevPilot)
    ├── agent/              # LangGraph 四节点
    ├── crawlers/           # 爬虫
    ├── rag/                # 向量检索 + KG
    ├── services/           # 趋势分析 + 图谱 + 定时 + 邮件
    ├── data/               # 文章 + 图谱文件
    ├── app.py              # Streamlit UI
    └── a2a_server.py       # A2A Server
```

### 1.3 为什么分两层

| | Pipeline 层 (pipeline/) | Interaction 层 (interaction/) |
|:---|:---|:---|
| **引擎** | LangGraph StateGraph | CC-Harness AgentLoop |
| **决策方式** | 编译时预设 DAG | 运行时 LLM 动态决策 |
| **触发** | 定时任务 / 手动触发 | 用户随时对话 |
| **核心动作** | 爬虫 → 清洗 → 分块 → embedding → 入库 → 邮件 | LLM推理 → A2A调工具 → 报告/图表/词云 |
| **资源** | CPU/网络密集 | GPU/API 密集 |
| **失败影响** | 重试即可 | 必须即时响应 |

分开的价值：各自用最合适的架构、独立演化、互不拖垮。A2A 是唯一接触面。

### 1.4 对标参考

| 参考项目 | 借鉴点 |
|:---|:---|
| [字节 DeerFlow](https://github.com/bytedance/deer-flow) | Harness vs App 分层思想 |
| [a2a-openai-agent](https://github.com/MuhammadAbdullah95/a2a-openai-agent) | 三种框架 A2A 混用 |
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | 技术调研 Agent Demo |
| [a2aproject/A2A](https://github.com/a2aproject/A2A) | 官方 A2A 协议 (24k+ stars) |
| [Agently](https://pypi.org/project/agently/) | 结构化流式输出 + TriggerFlow |

---

## 二、开发环境

### 2.1 统一环境

```bash
conda activate rag_env
python --version   # Python 3.11
```

### 2.2 环境变量

两个子目录各自有 `.env`（内容相同，或在根目录放一个 `.env` 两个都读）：

```bash
# LLM API
DASHSCOPE_API_KEY=sk-your-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEEPSEEK_API_KEY=sk-your-key         # 可选

# A2A 互连
DEVPILOT_A2A_URL=http://localhost:8010
HARNESS_API_URL=http://localhost:8020

# Pipeline 配置
CRAWL_INTERVAL_HOURS=6
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_FROM=devpilot@example.com
SMTP_PASSWORD=your-password
```

### 2.3 启动命令

```bash
# === Pipeline 层 ===
cd E:\agent-system\pipeline
streamlit run app.py                     # UI (8501)
python a2a_server.py                     # A2A Server (8010)

# === Interaction 层 ===
cd E:\agent-system\interaction
uvicorn server.app:app --port 8020       # API + Web UI (8020)
python demos/research_demo.py            # 技术调研 Demo
python benchmarks/compare_frameworks.py  # 框架对比
```

---

## 三、系统架构

### 3.1 整体架构

```
                       用户
                         │
          ┌──────────────┴──────────────┐
          │                             │
    Streamlit UI                   Web UI (Pico.css)
    (pipeline/app.py)              (interaction/server/static/)
          │                             │
          ▼                             ▼
┌─────────────────────┐     A2A      ┌──────────────────────────┐
│  Pipeline 层         │◄───────────►│  Interaction 层           │
│  pipeline/           │             │  interaction/             │
│  (LangGraph)         │  工具调用    │  (CC-Harness AgentLoop)   │
│                      │ ◄───────── │                           │
│  ┌────────────────┐  │            │  ┌───────────────────────┐│
│  │ 爬虫引擎        │  │  流式结果  │  │ AgentLoop 主循环       ││
│  │ 掘金/博客园     │  │ ─────────► │  │ Think→Act→Observe     ││
│  │ /GitHub (TODO)  │  │            │  └───────────────────────┘│
│  └────────────────┘  │            │                           │
│                      │            │  ┌───────────────────────┐│
│  ┌────────────────┐  │            │  │ 工具系统               ││
│  │ 数据处理         │  │            │  │ rag_search (本地)      ││
│  │ 分块→embed→入库 │  │            │  │ A2A Tools (9个远程)    ││
│  └────────────────┘  │            │  │ MCP Tools (外部)       ││
│                      │            │  └───────────────────────┘│
│  ┌────────────────┐  │            │                           │
│  │ 知识图谱        │  │            │  ┌───────────────────────┐│
│  │ 3180+ 实体      │  │            │  │ 安全层                 ││
│  │ pyvis 交互图    │  │            │  │ Guardrails → Policy    ││
│  └────────────────┘  │            │  │ → HITL                 ││
│                      │            │  └───────────────────────┘│
│  ┌────────────────┐  │            │                           │
│  │ 定时任务 (TODO) │  │            │  ┌───────────────────────┐│
│  │ 邮件推送 (TODO) │  │            │  │ 报告引擎               ││
│  └────────────────┘  │            │  │ 调研→对比→图表→词云    ││
│                      │            │  │ Map-Reduce 协作        ││
│  端口: 8010 (A2A)    │            │  └───────────────────────┘│
│                      │            │  端口: 8020 (API+WebUI)   │
└─────────────────────┘            └──────────────────────────┘
```

### 3.2 A2A 暴露工具 (Pipeline → Interaction)

| 工具 | 入参 | 说明 |
|:---|:---|:---|
| `rag_search` | query | 知识库搜索 |
| `trending_list` | source | 实时热榜 |
| `kg_lookup` | entity_name | 知识图谱查询 |
| `fetch_article` | url | 爬取文章 |
| `search_web` | query | 网络搜索 |
| `code_example` | keyword | 代码示例 |
| `compare_tech` | tech_a, tech_b | 技术对比 |
| `daily_digest` | — | 今日摘要 |
| `trend_report` | — | 趋势报告 |

---

## 四、当前完成状态

### 4.1 Interaction 层 (interaction/)

| 模块 | 文件 | 状态 |
|:---|:---|:---:|
| AgentLoop 引擎 | harness/agent_loop.py | ✅ |
| LLM 适配层（多Provider+流式） | harness/llm_adapter.py | ✅ |
| 上下文压缩引擎 | harness/context_engine.py | ✅ |
| 结构化追踪器 | harness/tracer.py | ✅ |
| 三层安全护栏 | harness/guardrails/ | ✅ |
| 确定性策略引擎 | harness/policy_engine.py | ✅ |
| Human-in-the-Loop | harness/hitl.py | ✅ |
| 断点恢复 (CheckpointSaver) | harness/checkpoint.py | ✅ |
| 评测框架 + pass^k | harness/eval/ | ✅ |
| MCP 协议 | harness/mcp/ | ✅ |
| A2A 协议 | harness/a2a/ | ✅ |
| 三种多Agent协作 | harness/multi_agent.py | ✅ |
| 双层记忆系统 | harness/memory/ | ✅ |
| Prompt 版本管理 + A/B | harness/prompt_manager.py | ✅ |
| Agent Profile 配置中心 | harness/agent_profile.py | ✅ |
| FastAPI + SSE + WebSocket | server/app.py | ✅ |
| Pico.css Web UI | server/static/index.html | ✅ |
| 4 个 Agent Profile | profiles/*.yaml | ✅ |
| 端到端 Demo | demos/research_demo.py | ✅ |
| 框架对比 Benchmark | benchmarks/compare_frameworks.py | ✅ |

### 4.2 Pipeline 层 (pipeline/)

| 模块 | 文件 | 状态 |
|:---|:---|:---:|
| LangGraph 四节点 | agent/graph.py, nodes.py, state.py | ✅ |
| 12 个工具 | agent/tools/ | ✅ |
| 爬虫（掘金+博客园） | crawlers/ | ✅ |
| 混合检索 | rag/hybrid_retriever.py | ✅ |
| 知识图谱 | rag/knowledge_graph.py | ✅ |
| 趋势分析 | services/trends.py | ✅ |
| 图谱可视化 | services/graph_viz.py | ✅ |
| Streamlit UI | app.py | ✅ |
| A2A Server (9工具) | a2a_server.py | ✅ |
| LLM 工厂 | model/factory.py | ✅ |
| **GitHub Trending 爬虫** | crawlers/github_trending.py | 🔲 |
| **定时爬取调度** | services/scheduler.py | 🔲 |
| **邮件推送** | services/mailer.py | 🔲 |

---

## 五、开发计划

### 5.1 Pipeline 层待做

| # | 任务 | 文件 | 说明 |
|:---|:---|:---|:---|
| **P1** | GitHub Trending 爬虫 | crawlers/github_trending.py | 解析 GitHub Trending 页面 |
| **P2** | 定时爬取调度 | services/scheduler.py | APScheduler，每 N 小时全量爬取 |
| **P3** | 邮件推送 | services/mailer.py | SMTP 发送技术日报 |
| **P4** | 增量更新去重 | crawlers/ 扩展 | 已爬文章不重复入向量库 |

### 5.2 Interaction 层待做

| # | 任务 | 说明 |
|:---|:---|:---|
| **H1** | A2A 工具自动注册到 ToolRegistry | 启动时自动从 DevPilot 发现工具 |
| **H2** | 调研报告生成链路打通 | 多源→KG→对比→Markdown报告 |
| **H3** | Web UI 完善 | 历史记录、错误优化 |

### 5.3 改进空间（对标 2026 夏季前沿）

| # | 改进点 | 当前状态 | 前沿标准 | 建议 |
|:---|:---|:---|:---|:---|
| **I1** | 结构化输出自动修复 | 有 `_extract_json` + `_repair_action`，单次尝试 | 所有主流框架标配 2 次自动修复循环（Schema错误→反馈给模型→重试） | 在 `llm_adapter.generate_structured()` 中加入 repair loop |
| **I2** | 可观测性分层 | AgentTracer 一层 | 行业标配 3-5 层：事件回调 + 指标导出 + 结构化追踪 + deep-debug + 实时面板 | Tracer 增加 Prometheus metrics 导出 |
| **I3** | 成本追踪 | Token 计数在 LLMAdapter 里 | 所有框架都在做 per-run cost rollup + per-model pricing table | LLMAdapter 已有基础，加一个全局 CostTracker |
| **I4** | 流式事件类型完善 | 6 种事件（session/thinking/tool/answer/done/error） | 行业标准 ~15 种细粒度事件类型 | 增加 step_start, guardrail_triggered, repair_attempted 等 |
| **I5** | Guardrails 双模式 | block/rewrite 两种 action | 行业标配 strict（抛异常）vs soft（优雅终止）+ 工具结果校验 | ToolGuard 增加 strict/soft mode |
| **I6** | Agent→Agent 统一 Trace | Tracer 每个 Session 独立 | 行业标配：handoff 链中所有 Agent 写入同一个 Trace | 多Agent模式中传递共享 Tracer |

这些改进不是"必须做"，而是**面试时的加分项**——你知道前沿在做什么，框架里有对应设计（哪怕简化版），面试官会觉得你眼界够。

---

## 六、执行路线图

### Phase 1: Pipeline 层补完

```
□ P1: GitHub Trending 爬虫
□ P2: 定时爬取调度  
□ P3: 邮件推送
□ P4: 增量去重
□ 验证: 手动触发→3源爬取→入库→KG更新→邮件收到
```

### Phase 2: Interaction 层补完

```
□ H1: A2A 工具自动注册
□ H2: 调研报告链路
□ H3: Web UI 完善
```

### Phase 3 (可选加分): 前沿改进

```
□ I1: 结构化输出自动修复循环
□ I2: Prometheus metrics 导出
□ I3: 全局 CostTracker
```

### Phase 4: 收尾

```
□ 两项目 README 重写
□ 录 Demo 视频
□ Gitee 推送
```

---

## 七、对话操作约定

**同一个对话中操作两个子目录**，说清楚当前在操作哪个：

```
cd E:\agent-system\interaction   # Interaction 层
cd E:\agent-system\pipeline      # Pipeline 层
```

每次开始新任务时报：`[Phase X] [任务编号] — interaction / pipeline`

---

## 八、简历呈现

### 项目名

**Agent 双引擎系统 — Pipeline + Interaction 分层架构**

### 一句话

基于 LangGraph 和自研 CC-Harness 的双层 Agent 系统，通过 A2A 协议实现跨框架互操作。

### 技术亮点

```
- 分层 Agent 架构: LangGraph StateGraph（管道层）+ 自研 CC-Harness AgentLoop（交互层）
- 跨框架互操作: Google A2A 协议连接异构 Agent，9 个工具自动发现与注册
- 自研 CC-Harness 框架: AgentLoop 动态决策、三层护栏、pass^k 评测、断点恢复
- 知识图谱: jieba + 共现矩阵 + pyvis 交互式，3180+ 实体
- 混合检索: 向量 + BM25 + BGE-Reranker，MRR@5 85%
- 技术调研 Demo: 多源搜索 → KG 关联 → 对比分析 → 结构化报告
```

### 规模

```
- CC-Harness: ~7000 行 Python，45 模块，零 LangChain 依赖
- Pipeline: 493 文档块，3180+ KG 实体，9 个 A2A 工具
```
