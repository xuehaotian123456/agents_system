"""混合检索: 向量 + BM25 + 知识图谱 + BGE-Reranker（三路融合）"""
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from utils.logger_handler import logger
import jieba
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    三路混合检索器: Vector + BM25 + Graph（可选）→ 可信度加权 → Reranker 重排。

    检索流程:
    1. 向量路: ChromaDB 语义相似度
    2. BM25 路: 关键词词频匹配
    3. 图检索路: 实体 → 邻居 → 关联 chunk（Graph-RAG 核心）
    4. 三路去重合并 → 可信度加权 → BGE-Reranker 精排
    """

    def __init__(
        self,
        vector_store,
        chunks: List[str],
        docs: List[Document],
        top_k_retrieve: int = 10,
        top_k_rerank: int = 3,
        alpha: float = 0.5,
        graph_retriever=None,  # GraphRetriever 实例
    ):
        self.vector_store = vector_store
        self.docs = docs
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.alpha = alpha
        self.graph_retriever = graph_retriever

        # BM25
        tokens = [list(jieba.cut(c)) for c in chunks]
        self.bm25 = BM25Okapi(tokens)
        self.chunks = chunks
        logger.info(f"BM25 索引: {len(chunks)} 个文档块")

        # Reranker
        self.reranker = None
        model_path = str(Path(__file__).parent.parent / "models" / "bge-reranker-base")
        try:
            if os.path.exists(model_path):
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(model_path)
                logger.info("BGE-Reranker 加载成功(本地)")
        except Exception:
            logger.warning("Reranker 不可用，将跳过重排序")

    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        三路混合检索 (RRF 融合)。

        Args:
            query: 用户查询
            k: 最终返回数量 (默认 top_k_rerank)

        Returns:
            融合重排后的 top-k Document 列表
        """
        final_k = k if k is not None else self.top_k_rerank

        # ── 第一路: 向量检索 ──
        vec_docs = self.vector_store.similarity_search(query, k=self.top_k_retrieve)
        for d in vec_docs:
            d.metadata["retrieval_source"] = "vector"

        # ── 第二路: BM25 关键词 ──
        qtokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(qtokens)
        bm25_top = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:self.top_k_retrieve]
        bm25_docs = []
        for i, score in bm25_top:
            if i < len(self.docs):
                doc = self.docs[i]
                doc.metadata["retrieval_source"] = "bm25"
                doc.metadata["bm25_score"] = float(score)
                bm25_docs.append(doc)

        # ── 第三路: 图检索（Graph-RAG 核心）──
        graph_docs = []
        if self.graph_retriever and self.graph_retriever.is_available:
            graph_docs = self.graph_retriever.search(query)
            logger.info(f"[HybridRetriever] 图检索: {len(graph_docs)} 个文档块")

        # ── RRF (Reciprocal Rank Fusion) 分数融合 ──
        # score(d) = Σ 1/(60 + rank_i(d))
        # alpha 加权: 向量路 × alpha, BM25 路 × (1-alpha), 图路固定权重 1.0
        # 融合后取 top-k 候选进 Reranker
        RRF_K = 60
        rrf_scores: dict[str, float] = {}
        doc_by_key: dict[str, Document] = {}

        def _add_route(docs: List[Document], weight: float):
            for rank, d in enumerate(docs):
                key = d.page_content[:100]
                if key not in doc_by_key:
                    doc_by_key[key] = d
                rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (RRF_K + rank + 1)

        _add_route(vec_docs, self.alpha)              # 向量路
        _add_route(bm25_docs, 1.0 - self.alpha)       # BM25 路
        _add_route(graph_docs, 1.0)                   # 图路

        merged = [doc_by_key[k] for k in rrf_scores]
        for d in merged:
            d.metadata["rrf_score"] = round(rrf_scores[d.page_content[:100]], 5)

        # ── 可信度加权 ──
        for doc in merged:
            credibility = doc.metadata.get("credibility", 0.5)
            weight = 0.7 + 0.3 * credibility
            doc.metadata["credibility_weighted"] = round(
                doc.metadata.get("rrf_score", 0) * weight, 6)

        # ── Reranker 精排 ──
        if self.reranker and len(merged) > final_k:
            pairs = [[query, d.page_content] for d in merged]
            scores = self.reranker.predict(pairs)
            # Reranker 分数 × 可信度因子
            for doc, score in zip(merged, scores):
                credibility = doc.metadata.get("credibility", 0.5)
                doc.metadata["reranker_score"] = float(score * (0.7 + 0.3 * credibility))
            ranked = sorted(
                zip(merged, [d.metadata.get("reranker_score", 0) for d in merged]),
                key=lambda x: x[1], reverse=True,
            )
            result = [d for d, _ in ranked[:final_k]]
            logger.info(
                f"[HybridRetriever] 三路融合(RRF): vec={len(vec_docs)} bm25={len(bm25_docs)} "
                f"graph={len(graph_docs)} → merged={len(merged)} → top{final_k}={len(result)}"
            )
            return result

        # 无 Reranker 时按可信度加权 RRF 排序
        merged.sort(key=lambda d: d.metadata.get("credibility_weighted", 0), reverse=True)
        result = merged[:final_k]
        logger.info(
            f"[HybridRetriever] 三路融合(RRF无Reranker): vec={len(vec_docs)} bm25={len(bm25_docs)} "
            f"graph={len(graph_docs)} → merged={len(result)}"
        )
        return result
