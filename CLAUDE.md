# Agent 双引擎系统 — 开发文档

> 给新 Claude 会话看的速览文档。读完就能上手开发。

---

## 一、一句话概览

**双引擎 Agent 系统**：LangGraph StateGraph（Pipeline 层，爬取→入库→推送）+ 自研 CC-Harness AgentLoop（Interaction 层，对话→工具调用→报告），A2A 协议互通。

---

## 二、目录结构

```
E:\agent-system\
├── CLAUDE.md                 ← 本文档
├── TECHNICAL_SPEC.md         ← 统一技术规格书（架构图、执行计划）
├── .gitignore
│
├── pipeline/                 ← Pipeline 层 (LangGraph DevPilot)
│   ├── a2a_server.py         # A2A Server (15工具, 端口8010)
│   ├── app.py                # Streamlit UI (端口8501)
│   ├── agent/                # LangGraph 四节点 + 14工具
│   │   ├── graph.py          # 图组装: Planner→Retriever→Reflector→Summarizer
│   │   ├── nodes.py          # 四节点实现
│   │   ├── state.py          # AgentState TypedDict
│   │   └── tools/            # 14个 @tool 函数
│   ├── crawlers/             # 爬虫: 掘金API / 博客园RSS / GitHub Trending / HackerNews / OSChina
│   ├── rag/                  # 混合检索: VectorStore + BM25 + BGE-Reranker + KG
│   │   ├── vector_store.py   # ChromaDB + chunking
│   │   ├── hybrid_retriever.py
│   │   └── knowledge_graph.py # jieba.posseg + 共现矩阵 + IDF, 5379实体
│   ├── services/             # 调度器/邮件/趋势/图谱可视化/状态去重
│   │   ├── scheduler.py      # APScheduler: 每6h增量爬取 + 每日摘要邮件
│   │   ├── mailer.py         # SMTP邮件 + 12种邮箱SMTP自动识别
│   │   ├── crawl_state.py    # URL去重 + 爬取状态 + SMTP配置持久化
│   │   ├── trends.py         # 热词趋势分析
│   │   ├── graph_viz.py      # pyvis 交互式知识图谱
│   │   └── digest_mail.py    # 旧版邮件（保留）
│   ├── model/                # LLM 工厂 (qwen-plus via DashScope)
│   ├── utils/                # 日志 + 文件处理 (UTF-8自适应)
│   ├── prompts/              # Prompt 模板
│   ├── config/               # 配置
│   ├── data/                 # 运行时数据 (gitignored)
│   │   ├── articles/         # 爬取的 .md 文章
│   │   ├── graphs/           # 生成的 pyvis HTML 图谱
│   │   ├── crawl_state.json   # 去重状态
│   │   └── kg_state.json     # KG 持久化
│   └── requirements.txt
│
└── interaction/              ← Interaction 层 (CC-Harness 自研框架)
    ├── server/
    │   ├── app.py            # FastAPI + SSE + WebSocket (端口8020)
    │   └── static/
    │       └── index.html    # Pico.css Web UI (会话历史/图谱嵌入)
    ├── harness/              # ★ 核心引擎 (45模块, ~7000行, 零LangChain)
    │   ├── agent_loop.py     # AgentLoop 主循环 (Think→Act→Observe)
    │   ├── llm_adapter.py    # LLM适配层 (多Provider + 结构化输出 + 自动修复)
    │   ├── session.py        # 会话管理 + 上下文压缩
    │   ├── prompt_engine.py  # 提示词引擎 (动态组装)
    │   ├── prompt_manager.py # Prompt 版本管理 + A/B
    │   ├── context_engine.py # 上下文压缩引擎
    │   ├── tracer.py         # 结构化追踪器
    │   ├── subagent.py       # 子Agent调度器
    │   ├── multi_agent.py    # 三种多Agent协作模式
    │   ├── types.py          # Pydantic v2 数据模型
    │   ├── checkpoint.py     # 断点恢复
    │   ├── hitl.py           # Human-in-the-Loop
    │   ├── agent_profile.py  # Agent Profile 配置中心
    │   ├── guardrails/       # 三层安全护栏
    │   ├── eval/             # 评测框架 + pass^k
    │   ├── mcp/              # MCP 协议
    │   ├── a2a/              # A2A 协议 (client + server)
    │   └── memory/           # 双层记忆系统
    ├── tools/                # 工具系统
    │   ├── base.py           # BaseTool 抽象基类
    │   ├── registry.py       # ToolRegistry (动态注册/筛选)
    │   └── rag_tool.py       # RAG 检索工具 (内部Agentic RAG循环)
    ├── profiles/             # 4个 Agent Profile (YAML)
    ├── infrastructure/       # 基础设施 (Retry/Cache/RateLimit)
    ├── demos/                # 端到端 Demo
    ├── benchmarks/           # 框架对比 Benchmark
    └── data/
        └── sessions/         # 会话持久化 (gitignored)
```

---

## 三、开发环境

### 统一环境

```bash
conda activate rag_env
python --version   # Python 3.11
```

### 环境变量 (.env)

两个子目录各需要 `.env`（或项目根目录放一个）：

```bash
# LLM API (必填)
DASHSCOPE_API_KEY=sk-your-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# A2A 互连
DEVPILOT_A2A_URL=http://localhost:8010

# Pipeline 配置
CRAWL_INTERVAL_HOURS=6
SMTP_HOST=smtp.example.com
SMTP_PORT=465
SMTP_USER=your@email.com
SMTP_PASSWORD=your-password
SMTP_FROM=devpilot@example.com
```

### 依赖安装

```bash
# Pipeline 层依赖
cd E:\agent-system\pipeline
pip install -r requirements.txt

# Interaction 层依赖
cd E:\agent-system\interaction
pip install -r requirements.txt

# 额外依赖 (如果缺失)
pip install apscheduler pyvis jieba chromadb feedparser httpx beautifulsoup4 streamlit
```

### 首次运行（新电脑）

```bash
# 1. Pipeline A2A 服务 (先启动，KG 首次需构建 ~30秒)
cd E:\agent-system\pipeline
python -m uvicorn a2a_server:app --host 0.0.0.0 --port 8010

# 2. Interaction 服务 (自动发现 Pipeline 工具 + 加载RAG)
cd E:\agent-system\interaction
python -m uvicorn server.app:app --host 0.0.0.0 --port 8020

# 3. (可选) Streamlit UI
cd E:\agent-system\pipeline
streamlit run app.py
```

### 启动后的端口

| 服务 | 端口 | URL |
|:---|:---|:---|
| Pipeline A2A | 8010 | http://localhost:8010/docs |
| Interaction Web UI | 8020 | http://localhost:8020 |
| Streamlit | 8501 | http://localhost:8501 |

---

## 四、核心架构

### 两层分工

```
用户 → Web UI (8020)
         │
         ▼
   CC-Harness AgentLoop ←── A2A ──→ LangGraph DevPilot (8010)
   (动态决策, 对话驱动)              (DAG编排, 定时爬取)
         │                                    │
    15 个工具                          爬虫 → 入库 → KG → 邮件
    (本地RAG + 14个远程)               (每6h增量 + 每日08:00推送)
```

### A2A 暴露的 15 个工具

| 工具 | 说明 |
|:---|:---|
| `rag_search` | 知识库混合检索 (向量+BM25+Reranker) |
| `trending_list` | 多源实时热榜 (掘金/博客园/GitHub/HN/OSChina) |
| `kg_lookup` | 知识图谱查询 + 交互式图谱生成 |
| `fetch_article` | 爬取指定 URL 文章 |
| `search_web` | 网络搜索 |
| `code_example` | 代码示例搜索 |
| `compare_tech` | 技术对比 |
| `daily_digest` | 多源技术日报生成 |
| `trend_report` | 技术热词趋势分析 |
| `send_digest_email` | 发送邮件日报 |
| `get_smtp_help` | 获取邮箱SMTP帮助链接 |
| `configure_smtp` | 配置SMTP (自动识别服务器) |
| `configure_daily_digest` | 配置每日定时推送 |
| `get_pipeline_status` | 查看爬取/调度状态 |
| `force_update` | 强制增量爬取 |

### 数据流（用户查询"掘金热榜"）

```
用户输入 → AgentLoop.think() → action_type=tool_call → trending_list
  → A2A → Pipeline trending_list 工具 → 掘金API → 标题+摘要+链接
  → AgentLoop.observe() → session.append_tool_result()
  → AgentLoop.think() → action_type=final_answer → 流式输出到前端
```

---

## 五、当前完成状态

### ✅ 已完成

| 功能 | 位置 | 说明 |
|:---|:---|:---|
| AgentLoop 引擎 | interaction/harness/agent_loop.py | Think→Act→Observe 循环 |
| LLM 适配层 | interaction/harness/llm_adapter.py | 结构化输出 + 自动修复 + 流式 |
| 上下文压缩 | interaction/harness/context_engine.py | LLM 驱动分层摘要 |
| 三层护栏 | interaction/harness/guardrails/ | block/rewrite + HITL |
| Prompt 管理 | interaction/harness/prompt_manager.py | 版本管理 + A/B |
| 会话持久化 | interaction/server/app.py | JSON文件存储 + 重启恢复 |
| A2A 自动发现 | interaction/server/app.py | 启动时注册14个远程工具 |
| RAG 本地检索 | interaction/tools/rag_tool.py | 338 chunks (来自94篇文章) |
| Web UI | interaction/server/static/ | 会话历史 + 图谱嵌入 + SSE流式 |
| LangGraph 四节点 | pipeline/agent/ | Planner→Retriever→Reflector→Summarizer |
| 多源爬虫 | pipeline/crawlers/ | 掘金/博客园/GitHub/HN/OSChina |
| 混合检索 | pipeline/rag/ | 向量 + BM25 + BGE-Reranker |
| 知识图谱 | pipeline/rag/knowledge_graph.py | 5379实体 + 持久化 |
| 定时调度 | pipeline/services/scheduler.py | 每6h爬取 + 每日邮件 |
| 邮件服务 | pipeline/services/mailer.py | 12种邮箱SMTP自动识别 |
| URL去重 | pipeline/services/crawl_state.py | 增量爬取不去重入 |
| SMTP对话配置 | pipeline/a2a_server.py | 用户Web端直接配置 |
| KG交互图谱 | pipeline/services/graph_viz.py | pyvis HTML + UTF-8修复 |

### ⚠️ 已知问题

1. **Windows GBK 编码**：pyvis 生成中文图谱时可能写 GBK → 已加 `_ensure_utf8()` 修复
2. **Python 循环闭包陷阱**：A2A 适配器创建时 `tool_def` 需用工厂函数捕获 → 已修复
3. **LLM 结构化输出**：qwen-plus 偶尔把工具名填到 `action_type` → `_repair_action()` 自动纠正
4. **LLM 丢失 GRAPH 标记**：回答时漏掉 `[GRAPH:...]` → `_append_graph_markers()` 自动追加
5. **首次 KG 构建慢**：需 ~30秒 → 已持久化到 `kg_state.json`，重启秒加载

---

## 六、面试话术

**这个项目的亮点**：

1. **双引擎分层架构**：LangGraph 处理有向无环的 Pipeline（爬取→入库→推送），自研 AgentLoop 处理开放式对话。各用最合适的架构，A2A 是唯一接触面。

2. **零 LangChain 依赖的 Agent 框架**：CC-Harness 从零实现 AgentLoop、Session、Tracer、Guardrails、PromptEngine，约 7000 行纯 Python。对标 Claude Code 的 Harness 架构思想。

3. **跨框架互操作**：Google A2A 协议连接 LangGraph Agent 和 CC-Harness Agent，15 个工具自动发现与注册。

4. **完整的工程实践**：会话持久化、断点恢复、pass^k 评测、Prompt A/B 测试、上下文压缩、三层安全护栏、结构化输出自动修复。

5. **实时多源数据管道**：5 源爬虫 → URL 去重 → 向量+BM25+Reranker 混合检索 → jieba 知识图谱(5379实体) → 邮件推送。

### 规模

- CC-Harness: ~7000 行 Python，45 模块，零 LangChain 依赖
- Pipeline: LangGraph 四节点 + 14 工具，5 源爬虫，5379 KG 实体，15 个 A2A 工具

---

## 七、快捷命令

```bash
# === 启动全部服务 ===
# 终端1: Pipeline A2A
cd E:\agent-system\pipeline && python -m uvicorn a2a_server:app --host 0.0.0.0 --port 8010

# 终端2: Interaction Web UI
cd E:\agent-system\interaction && python -m uvicorn server.app:app --host 0.0.0.0 --port 8020

# 终端3 (可选): Streamlit
cd E:\agent-system\pipeline && streamlit run app.py

# === 强制刷新数据 ===
curl -X POST http://localhost:8010/tools/force_update -d '{}'

# === 查看系统状态 ===
curl -X POST http://localhost:8010/tools/get_pipeline_status -d '{}'

# === 测试 KG 查询 ===
curl -X POST http://localhost:8010/tools/kg_lookup -H "Content-Type: application/json" -d '{"entity_name":"Python"}'

# === 查看会话历史 ===
curl http://localhost:8020/api/sessions
```
