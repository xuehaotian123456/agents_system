# Agent 双引擎系统 — 开发文档

> 给新 Claude 会话看的速览文档。读完就能上手开发。

---

## 一、一句话概览

**GraphRAG 增强的 Agent 双引擎系统**：LangGraph Pipeline（爬取→入库→GraphRAG 检索）+ 自研 CC-Harness AgentLoop（动态决策→多跳推理→决策链可视化），A2A 协议互通。

核心亮点：**知识图谱多跳推理 + Agentic 检索循环 + 社区全局检索 + LLM-as-Judge 评测 + Agent 决策链可追溯**。

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
│   ├── a2a_server.py         # A2A Server (16工具, 端口8010)
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
    │   │   ├── agentic_retriever.py  # Agentic 循环 (改写+HyDE+反思+重试)
│   │   ├── knowledge_graph.py     # KG: 多跳扩散 + 路径查找 + 社区检测/摘要
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

> ⚠️ 路径说明: 本文档的绝对路径 (E:gent-system) 为原始开发机 (Windows)。
> 在其他机器上 clone 后, 请将命令中的绝对路径替换为本地路径 (命令中已尽量使用相对路径)。

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

社区机制 (微软 GraphRAG):
  强边子图 (全局阈值 + 每实体 top-5 稀疏化)
  → 标签传播社区检测 (11,529 社区)
  → LLM 批量社区摘要 (45 个, 每批 10)
  → global_search 工具: query 实体 → 定位社区 → 返回主题摘要
```

### Agent 四节点 Pipeline

```
PLANNER ─→ RETRIEVER ─→ REFLECTOR ─→ SUMMARIZER → END
   │           │              │              │
   │     KG多跳扩散      置信度判定     透传GRAPH标记
   │     工具≤3个       <0.5重试       可信度标注
   └── LLM失败→规则兜底──→ 全失败→降级直达──→ 诚实拒答
```

### 数据质量治理 (两级质量门)

```
多源数据 (Gitee Issues + Docs + 掘金 + 博客园)
  → 第一级: 规则过滤 (免费: 广告词/长度/纯外链)
  → 第二级: LLM 质量门控 (LangGraph 条件路由)
       CRAWL → RULE_FILTER → LLM_GATE ──条件边──┬→ INGEST (入库)
                                                 ├→ DEMOTE (可信度×0.4 降权入库)
                                                 └→ SKIP (丢弃, 理由留痕)
  → 判定结果按内容 hash 缓存 (跨轮次复用, 零重复 LLM 成本)
  → 来源可信度打分 (official_doc=1.0 → rss=0.3) + 冲突消解
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

### 当前结果 (50 条查询, 7206 chunks, 15150 KG 实体, 11仓库扩语料后干净库实测)

> 口径: 纯向量 = 仅 ChromaDB top-5；GraphRAG = 三路 RRF 融合 + 查询自适应权重 + Reranker top-5。**两边同 k 对比**。
> 数据源: 干净向量库 (7,206 embeddings 1:1, 评测前后 count 一致; 内容级去重 + 英文停用词 + 元数据停用词清洗 KG)。

| 指标 | 纯向量 RAG | GraphRAG | 提升 |
|:---|:---|:---|:---|
| Recall@5 | 76.3% | 78.7% | **+2.3%** |
| 实体关联类 | 85.8% | 88.3% | +2.5% |
| 时效查询类 | 84.4% | 91.7% | +7.3% |
| 多跳推理类 | 75.0% | 83.3% | **+8.3%** |
| 报错溯源类 | 94.2% | 94.2% | 持平 |
| 事实查找类 | 64.2% | 59.2% | -5% |
| 对比分析类 | 41.7% | 47.2% | +5.5% (扩语料后转正) |

**诚实结论**: 在 7.2k chunks 规模上三路融合 Recall@5 +2.3%（图路臂存在 ±0.5pp 运行间波动，此为最近一次实测）。三组语料规模的增益方向稳定为正且随规模上升，多跳推理类从 +4% 涨到 +8.3%，对比分析类由 -4% 转正为 +5.5%——语料规模越大图路增量信息越多，权重差假说成立，优势集中在实体关联（+5.9%）与时效查询（+7.3%）。这符合预期——小语料下向量检索已能命中关键词，且 BGE-Reranker 主导最终排序；GraphRAG 的真实价值在语义相关性（LLM-as-Judge 评估）和 KG 多跳推理链（作为 Agent 工具提供推理路径），而非关键词召回。曾尝试分数级融合调 λ，实测提升 <0.5% 且有测试集过拟合风险，已撤回。

### 评测局限性 (面试时需说明)

- expected_entities 关键词匹配是弱标注 proxy，不是人工相关性别定
- LLM-as-Judge 为语义口径评测方法（代码已修复截断/分制 bug 待重跑；结果本地生成不入库）
- **报错溯源类在两种口径下差异的解释**: 关键词口径高分（报错码字面匹配，94.2%），语义口径逐文档均分天然低——报错答案需多文档拼接（现象/原因/修复分布在不同文档），单文档"完全可答"几乎不可能，这正说明评测应采用集合级可答率而非逐文档均分
- 50 条查询仍不够统计显著，建议 200+
- 负样本测试（30 条）验证了诚实拒答能力（90% 拒答率 27/30, 幻觉陷阱 8/8 拦截）
- 全量评测数据需运行 force_update 拉取后复现（种子数据仅保证 demo 可跑）

### 重大教训: 数据污染追查 (面试故事素材)

早期评测曾出现**虚假的高提升数字**（重复入库污染纯向量基线所致，假数字已全部撤回）：① md5.txt 写入用 Windows 默认 GBK、读取用 UTF-8，中文文件名导致缓存静默失效，每次 load_articles 全量重复 add_documents（ChromaDB 膨胀 15 倍根因）；② `_recover_from_store` 的 metadata 过滤用子串匹配，'juejin_Cursor.md' 会串档匹配 'sched_juejin_Cursor.md' 等文件。重复文档压低纯向量基线 top-5 质量，制造了虚假提升。修复后重跑得到上表真实数字。**这个排查故事比任何漂亮数字都更能证明你的工程能力。**

---

## 六、当前数据规模

| 指标 | 数值 |
|:---|:---|
| 文章数 | 363 篇 (内容去重后) |
| Chunks | 7,206 |
| KG 实体 | 15150 (停用词清洗 + 11仓库扩语料后) |
| ChromaDB | 7,206 embeddings (1:1) |
| URL 去重 | 251 条 |
| 数据源 | Gitee (MindSpore + Paddle) + 掘金 + 博客园 + HN + OSChina |

---

## 七、面试话术

**这个项目的亮点**：

1. **GraphRAG 三路融合 + 多跳推理 + 社区全局检索 + Agentic 循环**：Vector + BM25 + KG 三路召回，RRF 分数级融合（查询自适应权重）+ 可信度加权 + BGE-Reranker 精排。rag_search 内置 **Agentic 检索循环**（查询改写 + HyDE 假设答案 + 充分性反思 + 不足自动重试，决策留痕）。KG 支持 BFS 多跳实体扩散（2-3 hops）、实体最短路径查找、**标签传播社区检测（kNN 稀疏化防坍缩，11,529 社区）+ LLM 社区摘要（45 个）+ 全局检索工具**——对齐微软 GraphRAG 社区机制。评测（同 k 口径, 干净库）验证多跳推理类 Recall +8.3%，增益随语料规模单调上升（+1.3%→+2.5%→+2.8%）。

2. **双引擎分层架构**：LangGraph StateGraph 处理 Pipeline，自研 CC-Harness AgentLoop 处理对话。各用最合适的范式。

3. **Agent 决策链完全可见**：Tracer 记录每一步思考→工具调用→结果→反思，`to_markdown()` 输出人类可读的决策报告。面试时跑 `e2e_demo.py` 直接展示。

4. **完整的评测体系**：对比评测（纯向量 vs GraphRAG）+ LLM-as-Judge 评测 + 负样本测试（诚实拒答），不是"看起来效果不错"而是有数据支撑。

5. **数据质量治理（两级质量门）**：规则层免费过滤格式垃圾（广告/外链/过短），LangGraph 质量门控节点做 LLM 语义判定（抓规则层抓不住的八卦/软文/无关内容），条件边把每篇文档路由到 入库/降权/丢弃 三通道，判定理由留痕 + 内容 hash 缓存。可信度打分（1.0→0.3）+ 冲突消解 + 三级离线容错。

6. **工程实践**：三级异常降级、结构化输出自动修复、ChromaDB 膨胀治理（440MB→88MB）、会话持久化。

### 规模

- Pipeline: ~3500 行 Python，4 节点 + 14 工具，11 仓库爬虫，15150 KG 实体
- Interaction: ~7500 行 Python，45 模块，零 LangChain
- 评测: 50 条查询 + 30 条负样本 + LLM-as-Judge

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
