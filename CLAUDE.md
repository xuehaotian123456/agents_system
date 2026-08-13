# Agent 双引擎系统 — 开发文档

> 给新 Claude 会话看的速览文档。读完就能上手开发。

---

## 一、一句话概览

**GraphRAG 增强的 Agent 双引擎系统**：LangGraph Pipeline（爬取→入库→GraphRAG 检索）+ 自研 CC-Harness AgentLoop（动态决策→多跳推理→决策链可视化），A2A 协议互通。

核心亮点：**知识图谱多跳推理 + LLM-as-Judge 评测 + Agent 决策链可追溯**。

---

## 二、目录结构

```
E:\agent-system\
├── CLAUDE.md                 ← 本文档
├── .gitignore
├── Dockerfile                # Docker 镜像
├── docker-compose.yml        # 一键部署
│
├── pipeline/                 ← Pipeline 层 (LangGraph DevPilot)
│   ├── a2a_server.py         # A2A Server (15工具, 端口8010)
│   ├── app.py                # Streamlit UI (端口8501)
│   ├── agent/                # LangGraph 四节点 + 14工具
│   │   ├── graph.py          # 图组装: Planner→Retriever→Reflector→Summarizer
│   │   ├── nodes.py          # 四节点实现 (含 KG 多跳扩散增强)
│   │   ├── state.py          # AgentState (含 errors + degradation)
│   │   └── tools/            # 14个 @tool 函数
│   ├── crawlers/             # 爬虫体系
│   │   ├── github_issues.py  # GitHub REST API (三层离线容错)
│   │   ├── gitee_adapter.py  # Gitee API v5 (国内直连, 全局节流)
│   │   ├── juejin.py         # 掘金 API
│   │   ├── cnblogs.py        # 博客园 RSS
│   │   ├── multi_source.py   # HackerNews / GitHub Trending / OSChina
│   │   └── source_credibility.py  # 可信度打分 + 冲突消解 + 脏数据过滤
│   ├── rag/                  # GraphRAG 三路混合检索
│   │   ├── vector_store.py   # ChromaDB + chunking + force_rebuild
│   │   ├── hybrid_retriever.py    # 三路融合 (Vector+BM25+Graph)
│   │   ├── knowledge_graph.py     # KG: 多跳扩散 + 路径查找 + 共现矩阵
│   │   └── graph_retriever.py     # KG→Document 包装器
│   ├── eval/                 # ★ 评测体系
│   │   ├── run_eval.py       # 对比评测 (纯向量 vs GraphRAG)
│   │   ├── llm_judge_eval.py # LLM-as-Judge 评测 (替代关键词匹配)
│   │   ├── negative_test.py  # 负样本测试 (测诚实拒答)
│   │   ├── e2e_demo.py       # 端到端演示脚本
│   │   ├── test_queries.json # 50条查询 (含难度分级)
│   │   └── negative_queries.json # 15条负样本
│   ├── services/             # 调度器/邮件/趋势/图谱/状态去重
│   ├── model/                # LLM 工厂 (qwen-plus via DashScope)
│   ├── config/               # data_source.yaml 等配置
│   ├── data/                 # 运行时数据 (gitignored)
│   └── requirements.txt
│
└── interaction/              ← Interaction 层 (CC-Harness 自研框架)
    ├── server/               # FastAPI + SSE + WebSocket (8020)
    ├── harness/              # ★ 核心引擎 (~7000行, 零LangChain)
    │   ├── agent_loop.py     # AgentLoop (Think→Act→Observe)
    │   ├── llm_adapter.py    # LLM适配 (多Provider + 结构化输出)
    │   ├── session.py        # 会话管理 + 上下文压缩
    │   ├── context_engine.py # 三层摘要金字塔
    │   ├── tracer.py         # ★ 决策链追踪 (to_markdown / to_chain)
    │   ├── types.py          # Pydantic v2 数据模型
    │   ├── subagent.py       # 子Agent调度
    │   ├── guardrails/       # 输入护栏(注入/越狱硬拦截, 已接入主链路)
    │   └── ...
    └── tools/                # 工具系统
```

---

## 三、开发环境

```bash
conda activate rag_env
python --version   # Python 3.11
```

### 快速启动

```bash
# 终端1: Pipeline A2A 服务
cd E:\agent-system\pipeline
python -m uvicorn a2a_server:app --host 0.0.0.0 --port 8010

# 终端2: Interaction Web UI
cd E:\agent-system\interaction
python -m uvicorn server.app:app --host 0.0.0.0 --port 8020
```

| 服务 | 端口 | URL |
|:---|:---|:---|
| Pipeline A2A | 8010 | http://localhost:8010/docs |
| Interaction | 8020 | http://localhost:8020 |

---

## 四、核心架构

### GraphRAG 三路混合检索

```
用户查询
  ├─ 第一路: Vector (ChromaDB 语义相似度)       → top_k=10
  ├─ 第二路: BM25 (jieba 分词 + 词频匹配)       → top_k=10
  └─ 第三路: Graph (KG 多跳扩散 → 关联 chunk)   → top_k=5
         ↓
    三路去重合并 → 可信度加权 → BGE-Reranker 精排 → top_k=3
```

### KG 多跳推理

```
query="MindSpore FusedAdamW"
  → Hop 1: FusedAdamW → [AdamW, 优化器, 融合]
  → Hop 2: 优化器 → [超参调优, 性能优化, RFC]
  → 扩散实体数: 15+
  → 注入 RAG 检索词 + 上下文
```

### Agent 四节点 Pipeline

```
PLANNER ─→ RETRIEVER ─→ REFLECTOR ─→ SUMMARIZER → END
   │           │              │              │
   │     KG多跳扩散      置信度判定     透传GRAPH标记
   │     工具≤3个       <0.5重试       可信度标注
   └── LLM失败→规则兜底──→ 全失败→降级直达──→ 诚实拒答
```

### 数据质量治理

```
多源数据 (Gitee Issues + Docs + 掘金 + 博客园)
  → 来源可信度打分 (official_doc=1.0 → rss=0.3)
  → 冲突消解 (高权威覆盖低权威)
  → 脏数据过滤 (广告/纯外链/过短)
  → 三级离线容错 (API→重试→缓存)
```

---

## 五、评测体系

### 运行评测

```bash
cd E:\agent-system\pipeline

# 对比评测 (纯向量 vs GraphRAG, 50条查询)
python eval/run_eval.py

# LLM-as-Judge 评测 (替代关键词匹配)
python eval/llm_judge_eval.py

# 负样本测试 (测诚实拒答)
python eval/negative_test.py

# 端到端演示 (面试直接跑)
python eval/e2e_demo.py
```

### 当前结果 (50 条查询, 4206 chunks, 11814 KG 实体, 干净库实测)

> 口径: 纯向量 = 仅 ChromaDB top-5；GraphRAG = 三路 RRF 融合 + 查询自适应权重 + Reranker top-5。**两边同 k 对比**。
> 数据源: 干净向量库 (4,206 embeddings 1:1, 评测前后 count 一致; 内容级去重 227→220 篇, 英文停用词清洗 KG)。

| 指标 | 纯向量 RAG | GraphRAG | 提升 |
|:---|:---|:---|:---|
| Recall@5 | 77% | 78.3% | **+1.3%** |
| 实体关联类 | 85.8% | 91.7% | +5.9% |
| 时效查询类 | 84.4% | 91.7% | +7.3% |
| 多跳推理类 | 75% | 79% | +4% |
| 报错溯源类 | 94.2% | 94.2% | 持平 |
| 事实查找类 | 64.2% | 59.2% | -5% |

**诚实结论**: 在 4.2k chunks 规模上，三路融合对关键词 Recall@5 的提升有限（+1.3%），优势集中在实体关联（+5.9%）与时效查询（+7.3%）。这符合预期——小语料下向量检索已能命中关键词，且 BGE-Reranker 主导最终排序；GraphRAG 的真实价值在语义相关性（LLM-as-Judge 评估）和 KG 多跳推理链（作为 Agent 工具提供推理路径），而非关键词召回。曾尝试分数级融合调 λ，实测提升 <0.5% 且有测试集过拟合风险，已撤回。

### 评测局限性 (面试时需说明)

- expected_entities 关键词匹配是弱标注 proxy，不是人工相关性别定
- LLM-as-Judge 补充了更准确的 answerability 评估（基线同为仅 ChromaDB）
- 50 条查询仍不够统计显著，建议 200+
- 负样本测试验证了诚实拒答能力（80% 拒答率, 幻觉陷阱 100% 拦截）
- 全量评测数据需运行 force_update 拉取后复现（种子数据仅保证 demo 可跑）

### 重大教训: 数据污染追查 (面试故事素材)

早期评测数字 (+13%~+19%) 曾被两类数据污染：① md5.txt 写入用 Windows 默认 GBK、读取用 UTF-8，中文文件名导致缓存静默失效，每次 load_articles 全量重复 add_documents（ChromaDB 膨胀 15 倍根因）；② `_recover_from_store` 的 metadata 过滤用子串匹配，'juejin_Cursor.md' 会串档匹配 'sched_juejin_Cursor.md' 等文件。重复文档压低纯向量基线 top-5 质量，制造了虚假提升。修复后重跑得到上表真实数字。**这个排查故事比任何漂亮数字都更能证明你的工程能力。**

---

## 六、当前数据规模

| 指标 | 数值 |
|:---|:---|
| 文章数 | 220 篇 (内容去重后) |
| Chunks | 4,206 |
| KG 实体 | 11,686 (英文停用词清洗后) |
| ChromaDB | 4,206 embeddings (1:1) |
| URL 去重 | 251 条 |
| 数据源 | Gitee (MindSpore + Paddle) + 掘金 + 博客园 + HN + OSChina |

---

## 七、面试话术

**这个项目的亮点**：

1. **GraphRAG 三路融合 + 多跳推理**：Vector + BM25 + KG 三路召回，RRF (Reciprocal Rank Fusion) 分数级融合（查询自适应权重）+ 可信度加权 + BGE-Reranker 精排。KG 支持 BFS 多跳实体扩散（2-3 hops）与实体最短路径查找。评测（同 k 口径, 干净库）验证实体关联类 Recall +5.9%。

2. **双引擎分层架构**：LangGraph StateGraph 处理 Pipeline，自研 CC-Harness AgentLoop 处理对话。各用最合适的范式。

3. **Agent 决策链完全可见**：Tracer 记录每一步思考→工具调用→结果→反思，`to_markdown()` 输出人类可读的决策报告。面试时跑 `e2e_demo.py` 直接展示。

4. **完整的评测体系**：对比评测（纯向量 vs GraphRAG）+ LLM-as-Judge 评测 + 负样本测试（诚实拒答），不是"看起来效果不错"而是有数据支撑。

5. **数据质量治理**：可信度打分（1.0→0.3）+ 冲突消解 + 脏数据过滤 + 三级离线容错。

6. **工程实践**：三级异常降级、结构化输出自动修复、ChromaDB 膨胀治理（440MB→88MB）、会话持久化。

### 规模

- Pipeline: ~3500 行 Python，4 节点 + 14 工具，6 源爬虫，11,686 KG 实体
- Interaction: ~7500 行 Python，45 模块，零 LangChain
- 评测: 50 条查询 + 15 条负样本 + LLM-as-Judge

---

## 八、快捷命令

```bash
# 强制刷新数据
curl -X POST http://localhost:8010/tools/force_update -d '{}'

# 查看系统状态
curl -X POST http://localhost:8010/tools/get_pipeline_status -d '{}'

# 测试 KG 多跳查询
curl -X POST http://localhost:8010/tools/kg_lookup -H "Content-Type: application/json" -d '{"entity_name":"MindSpore"}'

# 评测
cd E:\agent-system\pipeline
python eval/run_eval.py          # 对比评测
python eval/llm_judge_eval.py    # LLM-as-Judge
python eval/negative_test.py     # 负样本测试
python eval/e2e_demo.py          # 端到端演示
```
