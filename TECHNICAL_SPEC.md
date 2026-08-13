# Agent 双引擎系统 — 技术规格书

> **GraphRAG 增强的技术知识 Agent 系统**
>
> 版本: 2.0 | 更新: 2026-08-12

---

## 一、系统架构

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
│  (LangGraph)         │  15 工具    │  (CC-Harness AgentLoop)   │
│                      │             │                           │
│  Planner→Retriever   │             │  Think→Act→Observe        │
│  →Reflector→         │             │  动态决策引擎              │
│  Summarizer          │             │                           │
│                      │             │  Tracer 决策链可视化       │
│  GraphRAG 三路融合   │             │  ContextEngine 分层压缩    │
│  Vector+BM25+KG      │             │                           │
│                      │             │                           │
│  KG 多跳推理         │             │                           │
│  11,942 实体         │             │                           │
└─────────────────────┘             └──────────────────────────┘
```

### 为什么分两层

| | Pipeline 层 | Interaction 层 |
|:---|:---|:---|
| **引擎** | LangGraph StateGraph | CC-Harness AgentLoop |
| **决策方式** | 编译时预设 DAG 节点 | 运行时 LLM 动态决策 |
| **适合场景** | 爬取→入库→推送 (固定流程) | 对话→工具调用→反思 (开放式) |
| **状态管理** | TypedDict + MemorySaver | Pydantic Session 自包含 |
| **依赖** | LangGraph, LangChain | 零 LangChain，纯 openai 库 |

---

## 二、核心技术组件

### 2.1 AgentLoop 引擎 (`interaction/harness/agent_loop.py`)

**ReAct 循环** (Reasoning + Acting):

```
while session.can_continue():
    1. 压缩检查 → 超窗口 75% 触发分层摘要
    2. LLM 结构化输出 → AgentAction {action_type, thought, tool_call, answer}
    3. 动作执行:
       - final_answer  → 流式输出，退出
       - tool_call     → 执行工具，结果注入上下文
       - spawn_subagent → 独立 AgentLoop，摘要返回
    4. 结果写入 Session，回到步骤 1
```

**关键设计**:
- **一次一个工具**: 不用 OpenAI parallel tool calls，每个结果立即注入影响下一轮决策
- **结构化输出双重保障**: 原生 JSON mode + Prompt fallback + `_repair_action()` 自动修复
- **多 Provider 自动路由**: 模型名 → Provider → base_url → API key 链式推断
- **模型降级链**: 主模型失败 → 依次尝试 fallback_models

### 2.2 上下文引擎 (`interaction/harness/context_engine.py`)

**三层摘要金字塔**:

```
L0: 最近 4 条 — 保留原文
L1: 中期 6 条 — LLM 压缩 150 字
L2: 早期余下 — LLM 压缩 200 字
```

Token 预算 (qwen-plus 32k): System 800 + Tools 1000 + History 20000 + Answer 10200

### 2.3 决策链追踪器 (`interaction/harness/tracer.py`)

12 种事件类型，输出:
- `to_markdown()` — 面试级人类可读报告
- `to_chain()` — 结构化决策链
- `to_json()` — 持久化 / LangFuse

---

## 三、GraphRAG 检索系统

### 3.1 三路混合检索 (`pipeline/rag/hybrid_retriever.py`)

```
用户查询
  ├─ Vector 路: ChromaDB 语义相似度, top_k=10
  ├─ BM25 路: jieba 分词 + 词频匹配, top_k=10
  └─ Graph 路: KG 多跳扩散 → 关联 chunk, top_k=5
         ↓
    RRF 分数融合: Σ 1/(60+rank), alpha 加权 (向量×α, BM25×(1-α), 图×1.0)
         ↓
    可信度加权: score × (0.7 + 0.3 × credibility)
         ↓
    BGE-Reranker 精排 → top_k (默认 3, 评测统一 5)
```

### 3.2 知识图谱 (`pipeline/rag/knowledge_graph.py`)

**构建**: jieba.posseg 词性标注 → 实体归一化 → 共现矩阵 → IDF 过滤 → JSON 持久化

**规模**: 11,942 实体 | 11,234 共现边 | 4,331 chunks

**核心 API**:
| 方法 | 功能 | 
|:---|:---|
| `multi_hop_expand(query, max_hops=3)` | BFS 多层实体扩散，返回完整推理链 |
| `find_path(entity_a, entity_b)` | 两实体间最短关联路径 |
| `graph_retrieve(query, top_k=5)` | 图检索转为 Document 列表 |

### 3.3 分块策略

RecursiveCharacterTextSplitter: chunk_size=300, overlap=30, 分隔符 `["\n\n","\n","。",".","！","？","，",","]`

---

## 四、LangGraph Pipeline

### 4.1 四节点图

```
PLANNER ─→ RETRIEVER ─→ REFLECTOR ─→ SUMMARIZER → END
   │           │              │
   │     KG多跳扩散      置信度<0.5→重试
   │     全失败→降级     LLM失败→跳过
   └── LLM失败→规则兜底
```

### 4.2 三级异常降级

```
L1: 单工具失败 → 其他工具继续
L2: 所有工具失败 → degradation_triggered → 跳过 Reflector
L3: LLM 失败 → 规则兜底 (Planner) / 跳过反思 (Reflector)
```

---

## 五、数据质量治理

### 5.1 可信度分级

| 来源 | 可信度 |
|:---|:---|
| official_doc (GitHub/Gitee 官方文档) | 1.00 |
| gitee_repo_doc | 0.95 |
| github_issue_labeled | 0.85 |
| gitee_issue_labeled | 0.75 |
| tech_blog_quality | 0.50 |
| rss_headline | 0.30 |

### 5.2 冲突消解

同知识点多源冲突 → 高权威覆盖低权威 → 低权威文档降级标记 `suppressed=True`

### 5.3 脏数据过滤

字数<100 / 含广告词 / 纯外链(>30% URL) / 仅标题RSS → 丢弃

### 5.4 离线容错

GitHub: enable_github=false → REST API → 失败 3 次 → 离线缓存

---

## 六、评测体系

### 评测方法

| 方法 | 指标 | 查询数 |
|:---|:---|:---|
| 关键词匹配 | Recall@5, Hit Rate, MRR | 50 条 (6 类 × 3 难度) |
| LLM-as-Judge | Answerability (0-1) | 50 条 |
| 负样本测试 | 诚实拒答率 | 15 条 |

### 当前结果

**关键词匹配 (50 条, 同 k 口径)**:

| 指标 | 纯向量 | GraphRAG | 提升 |
|:---|:---|:---|:---|
| Recall@5 | 56% | 76% | **+19%** |
| 实体关联类 | 60% | 85% | +25% |
| 报错溯源类 | 86% | 94% | +8% |

**负样本测试 (15 条)**:

| 类型 | 拒答率 |
|:---|---:|
| hallucination_trap (编造术语) | 100% |
| vague (模糊问题) | 100% |
| irrelevant (无关问题) | 100% |
| **总体** | **80%** |

### 评测局限

- 关键词匹配是弱 proxy，非人工相关性别定
- 50 条不足统计显著 (建议 200+)
- LLM-as-Judge 自身有 bias

---

## 七、系统规模

| 组件 | 代码量 | 文件数 |
|:---|:---|:---|
| Interaction 层 | ~7,500 行 | 45 模块 |
| Pipeline 层 | ~3,500 行 | 20+ 模块 |
| 评测体系 | ~1,000 行 | 7 脚本 |
| **总计** | **~12,000 行** | **70+ 文件** |

| 数据 | 规模 |
|:---|:---|
| 文章 | 227 篇 |
| Chunks | 4,331 |
| KG 实体 | 11,942 |
| 爬虫源 | 6 个 (Gitee+GitHub+掘金+博客园+HN+OSChina) |
| A2A 工具 | 15 个 |

---

## 八、运行方式

### 启动

```bash
# Pipeline A2A (先启动)
cd pipeline && uvicorn a2a_server:app --host 0.0.0.0 --port 8010
# Interaction Web UI
cd interaction && uvicorn server.app:app --host 0.0.0.0 --port 8020
```

### 评测

```bash
cd pipeline
python eval/e2e_demo.py          # 端到端演示 (面试跑这个)
python eval/run_eval.py          # 对比评测
python eval/negative_test.py     # 负样本测试
python eval/llm_judge_eval.py    # LLM-as-Judge
python eval/generate_report.py   # 生成报告
```

### 数据维护

```bash
curl -X POST http://localhost:8010/tools/force_update -d '{}'     # 强制爬取
curl -X POST http://localhost:8010/tools/get_pipeline_status -d '{}'  # 状态
```
