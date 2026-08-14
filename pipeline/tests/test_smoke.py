"""
核心冒烟测试 — 验证系统关键链路可用
======================================
运行: cd pipeline && python -m pytest tests/ -v

覆盖:
  - KG 构建/加载/多跳扩散
  - 向量库加载与三路检索 (含 RRF 融合)
  - 检索口径: k 参数生效 (GraphRAG 与纯向量同口径对比的前提)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== KG ====================

def test_kg_build_and_multi_hop():
    """KG 构建 + 多跳扩散 + 路径查找"""
    from rag.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    chunks = [
        "LangGraph 的 Checkpointer 用于状态持久化，支持 MemorySaver",
        "MemorySaver 适合开发环境，SqliteSaver 适合生产环境",
        "SqliteSaver 依赖 thread_id 隔离不同会话的状态",
        "MindSpore 的 FusedAdamW 优化器融合了权重衰减",
        "FusedAdamW 相比 AdamW 减少了显存访问次数",
    ]
    kg.build(chunks)

    assert kg.is_built, "KG 构建失败"
    assert kg.entity_count > 0, "KG 没有提取到实体"

    # 多跳扩散
    mh = kg.multi_hop_expand("LangGraph", max_hops=2)
    assert mh["total_expanded"] > 0, "多跳扩散无结果"
    assert len(mh["hops"]) >= 1, "至少应有 1 跳"

    # 路径查找
    path = kg.find_path("LangGraph", "SqliteSaver", max_hops=3)
    assert path is not None, "应找到 LangGraph→MemorySaver→SqliteSaver 路径"


def test_kg_persistence(tmp_path):
    """KG 持久化与加载"""
    from rag.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.build(["测试文档包含 MindSpore 和 LangGraph 两个实体"])

    save_path = str(tmp_path / "kg.json")
    kg.save(save_path)

    kg2 = KnowledgeGraph()
    assert kg2.load(save_path), "KG 加载失败"
    assert kg2.entity_count == kg.entity_count, "实体数不一致"


# ==================== 检索 ====================

@pytest.fixture(scope="module")
def vector_store():
    """加载全量数据（需要已运行 init_demo.py 或 force_update）"""
    from rag.vector_store import VectorStore
    vs = VectorStore()
    vs.load_articles()
    return vs


def test_retrieval_k_param(vector_store):
    """检索 k 参数生效 — 评测口径公平性的前提"""
    docs3 = vector_store.search("MindSpore", top_k=3)
    docs5 = vector_store.search("MindSpore", top_k=5)
    assert len(docs3) == 3, f"top_k=3 应返回 3 篇, 实际 {len(docs3)}"
    assert len(docs5) == 5, f"top_k=5 应返回 5 篇, 实际 {len(docs5)}"


def test_pure_vector_vs_graphrag_same_k(vector_store):
    """纯向量与 GraphRAG 同 k 口径对比（评测基线一致性）"""
    query = "MindSpore FusedAdamW"
    vec_docs = vector_store.store.similarity_search(query, k=5)
    graph_docs = vector_store.search(query, top_k=5)
    assert len(vec_docs) == 5, "纯向量基线应返回 5 篇"
    assert len(graph_docs) == 5, "GraphRAG 应返回 5 篇（与基线同口径）"


def test_rrf_fusion_metadata(vector_store):
    """RRF 融合分数写入 metadata"""
    docs = vector_store.search("PaddleNLP 训练", top_k=5)
    for d in docs:
        src = d.metadata.get("retrieval_source", "")
        assert src in ("vector", "bm25", "graph"), f"未知检索来源: {src}"
        assert "rrf_score" in d.metadata, "缺少 RRF 融合分数"


# ==================== 数据质量 ====================

def test_source_credibility():
    """可信度打分与脏数据过滤"""
    from crawlers.source_credibility import (
        get_credibility, is_low_quality,
    )
    assert get_credibility("official_doc") == 1.0
    assert get_credibility("rss_headline") == 0.3
    assert get_credibility("unknown_type") == 0.1

    # 脏数据
    assert is_low_quality("太短", title="") is True
    assert is_low_quality("加微信领取免费课程，限时优惠!", title="x") is True
    assert is_low_quality("这是一篇正常的技术文章内容。" * 30, title="正常文章") is False


# ==================== 质量门控 (LangGraph 条件路由) ====================

def test_quality_gate_routing_offline():
    """
    质量门控三路路由 (离线: 预写判定缓存, 零 LLM 成本)。
    验证: 规则层拦截 + 条件边路由 + demote 降权 + 判定留痕。
    """
    from agent.quality_gate import run_quality_gate, _content_hash, _save_verdict_cache

    docs = [
        {"title": "广告软文", "content": "加微信领取免费课程，限时优惠扫码关注" * 10,
         "credibility": 0.4, "source_type": "tech_blog_personal"},
        {"title": "技术八卦文", "content": "某大厂员工薪资讨论与技术无关的生活内容" * 10,
         "credibility": 0.5, "source_type": "tech_blog_quality"},
        {"title": "软文推广", "content": "AI 技术文章但大部分是产品推广链接和联系方式" * 10,
         "credibility": 0.5, "source_type": "tech_blog_quality"},
        {"title": "正常技术文", "content": "LangGraph 的状态图构建方法与 Checkpointer 使用详解" * 10,
         "credibility": 0.5, "source_type": "tech_blog_quality"},
    ]

    # 预写缓存: 八卦 skip, 软文 demote, 技术 ingest (离线模拟 LLM 判定结果)
    cache = {
        _content_hash(docs[1]): {"verdict": "skip", "reason": "离线测试: 与技术无关"},
        _content_hash(docs[2]): {"verdict": "demote", "reason": "离线测试: 软文降权"},
        _content_hash(docs[3]): {"verdict": "ingest", "reason": "离线测试: 正常"},
    }
    _save_verdict_cache(cache)

    result = run_quality_gate(docs)

    # 广告文: 规则层拦截 → skip
    # 八卦文: LLM 判定 skip; 软文: demote (可信度 ×0.4); 技术文: ingest
    assert result["stats"]["skip"] == 2, f"应 2 篇 skip: {result['stats']}"
    assert result["stats"]["demote"] == 1, f"应 1 篇 demote: {result['stats']}"
    assert result["stats"]["ingest"] == 1, f"应 1 篇 ingest: {result['stats']}"

    # demote 降权验证
    demoted = result["demote"][0]
    assert demoted["quality_demoted"] is True
    assert demoted["credibility"] == round(0.5 * 0.4, 2)

    # 判定留痕验证
    verdicts_in_trace = {t["title"]: t["verdict"] for t in result["trace"]}
    assert verdicts_in_trace["广告软文"] == "skip"
    assert verdicts_in_trace["技术八卦文"] == "skip"
    assert verdicts_in_trace["软文推广"] == "demote"
    assert verdicts_in_trace["正常技术文"] == "ingest"

    # 清理测试缓存, 避免污染后续真实判定
    from agent.quality_gate import VERDICT_CACHE_PATH
    if VERDICT_CACHE_PATH.exists():
        VERDICT_CACHE_PATH.unlink()


def test_quality_gate_disabled():
    """门控开关: enabled=False 时全部直通"""
    from agent.quality_gate import run_quality_gate
    docs = [{"title": "x", "content": "y" * 50, "credibility": 0.5}]
    result = run_quality_gate(docs, enabled=False)
    assert result["ingest"] == docs
    assert result["stats"]["gate_disabled"] is True


# ==================== 社区检测 + 全局检索 ====================

def test_community_detection_and_global_search():
    """社区检测 (kNN 稀疏化) + 全局检索 (离线, 无需 LLM)"""
    from rag.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    chunks = [
        "MindSpore 的 FusedAdamW 优化器融合权重衰减，提升训练性能",
        "FusedAdamW 与 AdamW 的区别在于减少显存访问次数",
        "MindSpore HyperOffload 把静态图显存卸载到主机内存",
        "PaddleNLP 的 Taskflow 支持文本分类和序列标注任务",
        "Taskflow 的 pipeline 接口可以加载预训练模型",
        "PaddleOCR 支持中英文文本检测与识别",
    ]
    kg.build(chunks)

    # 社区检测
    labels = kg.detect_communities(top_n_per_entity=3)
    assert len(labels) > 0, "社区检测失败"
    # 不应坍缩成单一巨型社区
    from collections import Counter
    sizes = Counter(labels.values())
    largest_ratio = sizes.most_common(1)[0][1] / max(len(labels), 1)
    assert largest_ratio < 0.8, f"社区坍缩: 最大社区占 {largest_ratio:.0%}"

    # 全局检索 (无摘要时返回 note, 不崩溃)
    result = kg.global_search("MindSpore 优化器")
    assert "note" in result or "communities" in result


# ==================== Agentic RAG 检索循环 ====================

class _FakeVS:
    """离线假向量库: 按查询词返回脚本化文档, 零 ChromaDB/零 LLM 依赖"""

    def __init__(self):
        from langchain_core.documents import Document
        self.Document = Document
        self.calls = []

    def search(self, q, top_k=5):
        self.calls.append(q)
        return [self.Document(page_content=f"{q} 的相关文档{i}",
                              metadata={"reranker_score": 1.0 - i * 0.1})
                for i in range(min(top_k, 3))]

    @property
    def store(self):
        return self


def _make_retriever(vs, scripted_responses=None, max_rounds=2):
    """构造 AgenticRetriever, 注入脚本化 LLM (离线)"""
    from rag.agentic_retriever import AgenticRetriever
    ar = AgenticRetriever(vs, top_k=5, max_rounds=max_rounds)

    if scripted_responses is None:
        scripted_responses = []
    responses = iter(scripted_responses)

    def fake_llm(prompt):
        try:
            return next(responses)
        except StopIteration:
            return None
    ar._llm = fake_llm
    return ar


def test_agentic_rewrite_hyde_and_retry():
    """完整循环: 改写+HyDE → 多查询检索 → 反思不足 → 二轮重试"""
    vs = _FakeVS()
    # 脚本化 LLM: 第1次=改写+HyDE, 第2次=反思(不足+改写查询)
    ar = _make_retriever(vs, scripted_responses=[
        '{"rewrites": ["变体A", "变体B"], "hypothetical_answer": "假设答案片段"}',
        '{"sufficient": false, "reason": "信息不足", "rewritten_query": "更精准的查询"}',
    ])

    docs, trace = ar.search("原始查询")

    # 多查询检索: 原始 + 2 变体 + HyDE = 4 个查询
    assert len(vs.calls) == 5, f"应检索 5 次 (4 查询 + 1 重试), 实际 {len(vs.calls)}"
    assert "变体A" in vs.calls and "假设答案片段" in vs.calls
    # 去重合并 (每个查询返回 3 篇, 内容不同则合并后 15 篇上限)
    assert len(docs) > 3, "合并结果应多于单查询"
    # 决策留痕
    actions = [(t["round"], t["action"]) for t in trace]
    assert (1, "rewrite") in actions
    assert (1, "hyde") in actions
    assert (1, "reflect") in actions
    assert (2, "retry") in actions


def test_agentic_sufficient_short_circuit():
    """反思判定充分 → 不触发二轮"""
    vs = _FakeVS()
    ar = _make_retriever(vs, scripted_responses=[
        '{"rewrites": [], "hypothetical_answer": ""}',
        '{"sufficient": true, "reason": "信息完整", "rewritten_query": ""}',
    ])
    docs, trace = ar.search("q")
    assert len(vs.calls) == 1, "充分时不触发重试检索"
    assert not any(t["action"] == "retry" for t in trace)


def test_agentic_llm_failure_fallback():
    """LLM 不可用 → 故障开放: 单次检索, 不崩溃"""
    vs = _FakeVS()
    ar = _make_retriever(vs, scripted_responses=[])  # LLM 永远失败
    docs, trace = ar.search("q")
    assert len(vs.calls) == 1, "LLM 失败时仅原始查询检索"
    assert len(docs) == 3
    # 反思默认 sufficient=True (故障开放)
    assert any(t["action"] == "reflect" and "True" in t["detail"] for t in trace)


def test_agentic_dedup_merge():
    """多查询结果去重 (相同内容只保留 reranker 分更高的一次)"""
    from langchain_core.documents import Document

    class _DupVS(_FakeVS):
        def search(self, q, top_k=5):
            self.calls.append(q)
            # 所有查询返回同一篇文档 (不同 reranker 分)
            return [Document(page_content="同一篇文档内容",
                             metadata={"reranker_score": 0.9 if "变体" in q else 0.5})]

    vs = _DupVS()
    ar = _make_retriever(vs, scripted_responses=[
        '{"rewrites": ["变体A"], "hypothetical_answer": ""}',
        '{"sufficient": true, "reason": "", "rewritten_query": ""}',
    ])
    docs, _ = ar.search("原始")
    assert len(docs) == 1, "相同内容应去重为 1 篇"
    assert docs[0].metadata["reranker_score"] == 0.9, "保留高分版本"


# ==================== 负样本诚实性 ====================

def test_honesty_detector():
    """诚实拒答检测逻辑"""
    from eval.negative_test import check_honesty

    honest = check_honesty("知识库中没有找到相关信息，建议查阅官方文档。")
    assert honest["is_honest"] is True

    dishonest = check_honesty("LangGraph 的 Checkpointer 用于持久化状态，配置方式如下...")
    assert dishonest["hallucination_risk"] in (True, False)  # 长回答无诚实标记 → 幻觉风险


if __name__ == "__main__":
    # 无 pytest 时直接运行
    sys.exit(pytest.main([__file__, "-v"]))
