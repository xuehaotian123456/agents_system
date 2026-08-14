# Agent 双引擎系统 — GraphRAG 增强的技术知识 Agent

> LangGraph Pipeline + 自研 CC-Harness AgentLoop 双引擎，A2A 协议互通。
> 核心亮点：**知识图谱多跳推理 + LLM-as-Judge 评测 + Agent 决策链可追溯**。

---

## 快速开始（30 秒）

```bash
# 1. 环境
conda activate rag_env            # Python 3.11

# 2. 配置 API Key
#    在 pipeline/.env 中设置 DASHSCOPE_API_KEY

# 3. 一键初始化（种子数据 → 向量库 → 知识图谱）
cd pipeline
python scripts/init_demo.py

# 4. 启动服务
python -m uvicorn a2a_server:app --host 0.0.0.0 --port 8010      # Pipeline
cd ../interaction
python -m uvicorn server.app:app --host 0.0.0.0 --port 8020      # Interaction

# 5. 看效果
cd ../pipeline
python eval/e2e_demo.py
```

---

## 系统架构

```
用户 → Web UI (8020)
         │
    ┌────▼───────────────────┐     A2A      ┌────────────────────────┐
    │ CC-Harness AgentLoop   │◄───────────►│ LangGraph Pipeline      │
    │ (动态决策, Think→Act→  │   15 工具   │ (四节点 DAG)            │
    │  Observe, 零 LangChain)│             │ Planner→Retriever→      │
    │                        │             │ Reflector→Summarizer    │
    │ Tracer 决策链可视化     │             │                         │
    │ ContextEngine 分层压缩  │             │ GraphRAG 三路混合检索    │
    └────────────────────────┘             │ Vector+BM25+KG 多跳     │
                                           └────────────────────────┘
```

| 层 | 引擎 | 职责 |
|:---|:---|:---|
| Pipeline | LangGraph StateGraph | 爬取 → 数据治理 → 入库 → GraphRAG 检索 |
| Interaction | CC-Harness AgentLoop | 对话 → 动态决策 → 工具调用 → 决策链报告 |

---

## 核心能力

### GraphRAG 三路混合检索
- **Vector**: ChromaDB 语义相似度 (top_k=10)
- **BM25**: jieba 分词 + 词频匹配 (top_k=10)
- **Graph**: KG 多跳扩散 → 关联 chunk (top_k=5)
- 三路去重 → 可信度加权 → BGE-Reranker 精排 → top_k=3

### 知识图谱多跳推理
- 15150 实体 / 50万+ 共现边（jieba.posseg 抽取 + 共现矩阵 + IDF 过滤）
- `multi_hop_expand`: BFS 1-3 跳实体扩散，返回完整推理链
- `find_path(A, B)`: 实体间最短关联路径

### Agent 动态决策
- ReAct 循环: 思考 → 工具调用 → 观察 → 反思
- 结构化输出 + 自动修复（Pydantic Schema 约束 + 常见错误纠正）
- 多 Provider 路由 + 模型降级链
- 三级异常降级: 单工具失败 → 全失败降级 → LLM 失败规则兜底

### 评测体系
| 方法 | 指标 | 规模 |
|:---|:---|:---|
| 对比评测 | Recall@5 / Hit Rate / MRR | 50 条查询 (6 类 × 3 难度) |
| LLM-as-Judge | Answerability (0-1) | 50 条查询 |
| 负样本测试 | 诚实拒答率 | 15 条 (5 类陷阱) |

**当前结果** (同 k 口径, 11仓库扩语料后干净库实测): GraphRAG 比纯向量 RAG 总体 Recall@5 **+2.3%**（76.3%→78.7%，图路臂有 ±0.5pp 运行间波动），多跳推理类 **+8.3%**（75%→83.3%），对比分析类 +5.5%（由负转正）；增益方向随语料规模稳定为正；负样本诚实拒答率 **90%**（27/30），幻觉陷阱 8/8 拦截。

> **诚实说明**: 小语料下三路融合对关键词召回提升有限（BGE-Reranker 主导最终排序），GraphRAG 的真实价值在于实体关联查询、KG 多跳推理链（Agent 工具）与语义相关性评估（LLM-as-Judge）。早期版本的更高提升数字是"重复入库污染基线"的假象（MD5 GBK 编码 + 子串匹配串档），已修复并重测——详见 commit 历史与 CLAUDE.md。
> ⚠️ 评测数据依赖全量语料（363 篇文章）。clone 后先运行 `curl -X POST http://localhost:8010/tools/force_update -d '{}'` 拉取数据，再跑评测复现。

---

## 数据策略

**Git 仓库只含种子数据（15 篇精选），全量数据由爬虫生成。**

| 数据 | 位置 | 策略 |
|:---|:---|:---|
| 种子文章 (15 篇) | `pipeline/data/seed_articles/` | ✅ 进 git，clone 后 30 秒可 demo（注意：仅演示用，跑不出全量评测口径） |
| 全量文章 (~363 篇) | `pipeline/data/articles/` | ❌ gitignore，`force_update` 生成 |
| ChromaDB 向量库 | `pipeline/data/chroma_db/` | ❌ gitignore，`init_demo.py` 重建 |
| 知识图谱状态 | `pipeline/data/kg_state.json` | ❌ gitignore，7 秒可重建 |
| 爬取状态 | `pipeline/data/crawl_state.json` | ❌ gitignore，运行时状态 |

**全量数据拉取**:

```bash
# 服务启动后执行（约 10-15 分钟，受 Gitee API 节流影响）
curl -X POST http://localhost:8010/tools/force_update -d '{}'

# 数据源配置: pipeline/config/data_source.yaml
#   enable_gitee=true   (MindSpore + PaddlePaddle, 国内直连)
#   enable_github=false (需翻墙, 可开启)
#   enable_juejin=true  enable_cnblogs=true
```

---

## 评测运行

```bash
cd pipeline
python eval/e2e_demo.py          # 端到端演示 (面试跑这个)
python eval/run_eval.py          # 纯向量 vs GraphRAG 对比
python eval/negative_test.py     # 负样本诚实拒答测试
python eval/llm_judge_eval.py    # LLM-as-Judge 评测
python eval/generate_report.py   # 生成完整评测报告
```

---

## 项目规模

- **~23,000 行 Python**，100 个 .py 文件
- Interaction 层: ~10,000 行（harness 核心引擎），零 LangChain Agent 依赖
- Pipeline 层: 四节点 + 21 A2A 工具 + 11 仓库爬虫
- 评测: 50 查询 + 30 负样本 + LLM-as-Judge

## 技术文档

- [CLAUDE.md](CLAUDE.md) — 开发速览（架构、环境、面试话术）
- [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) — 技术规格书（组件详解、评测方法）
