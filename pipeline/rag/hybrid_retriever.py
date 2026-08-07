"""混合检索: 向量 + BM25 + BGE-Reranker"""
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from utils.logger_handler import logger
import jieba
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vector_store, chunks: List[str], docs: List[Document],
                 top_k_retrieve=10, top_k_rerank=3, alpha=0.5):
        self.vector_store = vector_store
        self.docs = docs
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.alpha = alpha

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

    def search(self, query: str) -> List[Document]:
        # Vector
        vec_docs = self.vector_store.similarity_search(query, k=self.top_k_retrieve)

        # BM25
        qtokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(qtokens)
        bm25_top = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:self.top_k_retrieve]
        bm25_docs = [self.docs[i] for i, _ in bm25_top if i < len(self.docs)]

        # Merge
        seen = set()
        merged = []
        for d in vec_docs + bm25_docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(d)

        # Rerank
        if self.reranker and len(merged) > self.top_k_rerank:
            pairs = [[query, d.page_content] for d in merged]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)
            return [d for d, _ in ranked[:self.top_k_rerank]]

        return merged[:self.top_k_rerank]
