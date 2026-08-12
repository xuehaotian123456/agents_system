"""
图检索器 — 将 KnowledgeGraph 图检索包装为标准 Document 接口
============================================================
作为 HybridRetriever 的第三路召回（Vector + BM25 + Graph）。

接口与 HybridRetriever 一致: search(query) → list[Document]
"""

from __future__ import annotations

from typing import Optional, List

from langchain_core.documents import Document


class GraphRetriever:
    """
    基于知识图谱的检索器。

    使用方式:
        kg = get_kg()
        gr = GraphRetriever(kg, chunks, docs)
        graph_docs = gr.search("LangGraph 报错")

    注意:
        - 仅在 KG.is_built == True 时生效
        - 无 KG 时自动降级为空列表（不影响其他路）
    """

    def __init__(
        self,
        kg,  # KnowledgeGraph 实例
        chunks: list[str],       # 全局 chunk 文本
        docs: list[Document],    # 全局 Document 列表
        top_k: int = 5,
    ):
        self.kg = kg
        self.chunks = chunks
        self.docs = docs
        self.top_k = top_k

    @property
    def is_available(self) -> bool:
        return self.kg is not None and self.kg.is_built

    def search(self, query: str) -> list[Document]:
        """
        图检索：从 query 抽取实体 → BFS 多层扩散 → 返回关联 Document。
        """
        if not self.is_available:
            return []

        try:
            # 使用 2-hop 扩散，覆盖更多关联文档
            mh_result = self.kg.multi_hop_expand(query, max_hops=2, top_per_hop=3)
            raw_results = self._multi_hop_to_chunks(mh_result)
        except Exception:
            # 降级到 1-hop
            try:
                raw_results = self.kg.graph_retrieve(query, self.chunks, self.top_k)
            except Exception:
                return []

        graph_docs = []
        for r in raw_results:
            cid = r["chunk_idx"]
            if cid < len(self.docs):
                doc = self.docs[cid]
                doc.metadata["retrieval_source"] = "graph"
                doc.metadata["graph_entity"] = r.get("entity", "")
                doc.metadata["graph_score"] = r.get("score", 0)
                graph_docs.append(doc)

        return graph_docs

    def _multi_hop_to_chunks(self, mh_result: dict) -> list[dict]:
        """将 multi_hop_expand 结果转为 chunk 列表"""
        chunk_scores: dict[int, float] = {}

        if not isinstance(mh_result, dict):
            return []

        # 从种子实体开始
        seed = mh_result.get("seed_entities", [])
        for entity in seed:
            entity_freq = self.kg.entity_freq.get(entity, 1)
            for cid in self.kg.entity_to_chunks.get(entity, []):
                chunk_scores[cid] = chunk_scores.get(cid, 0) + entity_freq

        # 逐跳加入，衰减权重
        for hop_data in mh_result.get("hops", []):
            hop = hop_data.get("hop", 1)
            decay = 1.0 / (hop + 1)  # Hop1=0.5, Hop2=0.33, Hop3=0.25
            for e in hop_data.get("entities", []):
                entity_freq = self.kg.entity_freq.get(e["entity"], 1)
                for cid in self.kg.entity_to_chunks.get(e["entity"], []):
                    chunk_scores[cid] = chunk_scores.get(cid, 0) + entity_freq * decay
                for nb in e.get("neighbors", []):
                    nb_freq = self.kg.entity_freq.get(nb["name"], 1)
                    nb_weight = nb_freq * (1 + nb.get("co_occur", 0)) * decay
                    for cid in self.kg.entity_to_chunks.get(nb["name"], []):
                        chunk_scores[cid] = chunk_scores.get(cid, 0) + nb_weight

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        results = []
        for cid, score in ranked:
            if cid < len(self.chunks):
                results.append({
                    "chunk_idx": cid,
                    "chunk_text": self.chunks[cid],
                    "entity": seed[0] if seed else "",
                    "score": round(score, 2),
                })
        return results
