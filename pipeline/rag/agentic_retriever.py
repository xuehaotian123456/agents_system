"""
Agentic RAG 检索循环 — Self-RAG 风格
=====================================
把"单次检索"升级为"检索→反思→改写→再检索"的 Agent 循环。

Round 1:
  1. 查询改写: LLM 生成 2 个改写变体 + 1 个 HyDE 假设答案
     (HyDE: Hypothetical Document Embeddings — 用假设答案检索,
      假设答案与真实文档的语义分布更接近)
  2. 多查询检索: 原 query + 2 变体 + HyDE → 各自 top-k → 合并去重
  3. 充分性反思: LLM 判断合并结果能否回答问题 (Self-RAG 的反思机制)
  4. 充分 → 返回; 不充分 → Round 2

Round 2 (仅当不足):
  5. 用反思建议的 rewritten_query 再检索一轮
  6. 两轮结果合并返回

成本: 每查询 2-3 次 LLM 调用 (改写 + 反思 [+1 次改写])

设计要点:
- 反思留痕: 每轮 sufficient/rewritten_query/reason 记入日志, 决策可审计
- 故障开放: LLM 不可用 → 降级为单次检索, 不阻塞
"""

from __future__ import annotations

from typing import Optional

from utils.logger_handler import logger

REWRITE_PROMPT = """你是检索查询优化专家。用户查询经过改写能显著提升检索效果。

用户查询: {query}

请以JSON格式输出:
{{
  "rewrites": ["改写查询1", "改写查询2"],
  "hypothetical_answer": "假设答案片段: 如果语料库中有答案, 它最可能长什么样 (50字以内, 包含关键术语)"
}}"""

REFLECT_PROMPT = """你是检索质量评估员。判断以下检索结果能否充分回答用户问题。

用户问题: {query}
检索到的文档 (截断):
{docs_text}

请以JSON格式输出:
{{
  "sufficient": true/false,
  "reason": "一句话理由(30字内)",
  "rewritten_query": "如果不足, 给出更精准的改写查询(30字内)"
}}"""


class AgenticRetriever:
    """
    Agentic RAG 检索循环。

    使用:
        ar = AgenticRetriever(vs)
        docs, trace = ar.search(query)
        # trace: [{round, action, detail}] 决策留痕
    """

    def __init__(self, vector_store, top_k: int = 5, max_rounds: int = 2):
        self.vs = vector_store
        self.top_k = top_k
        self.max_rounds = max_rounds

    def _llm(self, prompt: str) -> Optional[str]:
        from model.factory import robust_llm_call
        try:
            resp = robust_llm_call(prompt)
            return resp.content if hasattr(resp, 'content') else str(resp)
        except Exception as e:
            logger.warning(f"[AgenticRAG] LLM 调用失败: {e}")
            return None

    @staticmethod
    def _extract_json(text: str) -> dict:
        import json
        import re
        for candidate in [text.strip()]:
            try:
                return json.loads(candidate)
            except Exception:
                pass
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}

    def _multi_query_retrieve(self, queries: list[str]) -> list:
        """多查询检索 + 合并去重 (按 reranker_score 排序)"""
        merged: dict[str, object] = {}
        for q in queries:
            try:
                docs = self.vs.search(q, top_k=self.top_k)
            except Exception:
                docs = self.vs.store.similarity_search(q, k=self.top_k)
            for d in docs:
                key = d.page_content[:100]
                score = d.metadata.get("reranker_score",
                                       d.metadata.get("rrf_score", 0.0))
                if key not in merged or score > merged[key][1]:
                    merged[key] = (d, score)
        ranked = sorted(merged.values(), key=lambda x: -x[1])
        return [d for d, _ in ranked]

    def search(self, query: str) -> tuple[list, list[dict]]:
        """
        执行 Agentic 检索循环。

        Returns:
            (docs, trace) — 合并去重后的文档列表 + 决策留痕
        """
        trace: list[dict] = []

        # ── Round 1: 查询改写 + HyDE ──
        queries = [query]
        rewrite_raw = self._llm(REWRITE_PROMPT.format(query=query))
        rewrite = self._extract_json(rewrite_raw) if rewrite_raw else {}
        rewrites = rewrite.get("rewrites", [])
        hyde = rewrite.get("hypothetical_answer", "")

        if rewrites:
            queries.extend(rewrites[:2])
            trace.append({"round": 1, "action": "rewrite",
                          "detail": f"改写: {rewrites[:2]}"})
        if hyde:
            queries.append(hyde)
            trace.append({"round": 1, "action": "hyde",
                          "detail": f"假设答案: {hyde[:60]}"})

        docs = self._multi_query_retrieve(queries)
        logger.info(f"[AgenticRAG] Round1: {len(queries)} 个查询 → {len(docs)} 篇去重文档")

        # ── 充分性反思 ──
        docs_text = "\n\n".join(
            f"[{i+1}] {d.page_content[:200]}" for i, d in enumerate(docs[:8]))
        reflect_raw = self._llm(REFLECT_PROMPT.format(query=query, docs_text=docs_text))
        reflection = self._extract_json(reflect_raw) if reflect_raw else {}
        sufficient = reflection.get("sufficient", True)  # 故障开放: 默认充分
        trace.append({"round": 1, "action": "reflect",
                      "detail": f"sufficient={sufficient}, {reflection.get('reason', '')[:60]}"})

        # ── Round 2: 不足时用反思的改写查询再检索 ──
        if not sufficient and len(trace) // 3 < self.max_rounds:
            rq = reflection.get("rewritten_query", "")
            if rq and rq != query:
                round2_docs = self._multi_query_retrieve([rq])
                # 合并两轮 (第二轮排前, 理由: 更精准的查询)
                docs = round2_docs + [d for d in docs if d not in round2_docs]
                trace.append({"round": 2, "action": "retry",
                              "detail": f"改写查询: {rq[:60]} → +{len(round2_docs)} 篇"})
                logger.info(f"[AgenticRAG] Round2 补充检索: '{rq[:50]}'")

        return docs, trace
