"""轻量知识图谱 — jieba.posseg + 共现矩阵 + IDF + 一跳扩散"""
import re
import jieba
import jieba.posseg as pseg
from collections import Counter, defaultdict
from typing import List, Set, Optional
from utils.logger_handler import logger

_STOPWORDS = {
    "的", "了", "在", "是", "有", "和", "就", "不", "人", "都", "一",
    "很", "去", "能", "到", "说", "要", "会", "也", "着", "被",
    "可以", "使用", "需要", "一个", "进行", "这个", "那个",
    "什么", "怎么", "如何", "为什么", "哪些", "哪个",
    "目前", "现在", "通过", "根据", "对于", "关于",
    "包括", "具有", "存在", "比较", "非常", "特别",
    "分类", "属性", "说明", "备注", "参考", "来源",
    "大家", "以上", "以下", "相关", "其他", "各种", "不同",
    "主要", "所有", "部分", "方面", "产品", "适合",
    "点击", "查看", "链接", "评论", "阅读", "分享",
    "技术", "开发", "应用", "系统", "提供", "支持", "实现", "采用",
}

_ALLOWED_POS = {'n', 'nr', 'ns', 'nt', 'nz', 'eng', 'x', 'j', 'l'}

# 实体别名归一化映射（极简版，10 分钟写完）
# 解决 LLM 抽取实体时的大小写/缩写不一致问题
_ENTITY_NORMALIZE: dict[str, str] = {
    "langgraph": "LangGraph",
    "lang graph": "LangGraph",
    "langchain": "LangChain",
    "lang chain": "LangChain",
    "lc": "LangChain",
    "lg": "LangGraph",
    "vllm": "vLLM",
    "vllm": "vLLM",
    "openai": "OpenAI",
    "open ai": "OpenAI",
    "llm": "LLM",
    "rag": "RAG",
    "agentic rag": "Agentic RAG",
    "graphrag": "GraphRAG",
    "graph rag": "GraphRAG",
    "gpt": "GPT",
    "api": "API",
    "sdk": "SDK",
    "cli": "CLI",
    "ui": "UI",
    "html": "HTML",
    "css": "CSS",
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "NLP",
    "cv": "Computer Vision",
    "rl": "Reinforcement Learning",
}

class KnowledgeGraph:
    def __init__(self):
        self.entity_to_chunks: dict = defaultdict(list)
        self.co_occurrence: dict = defaultdict(lambda: defaultdict(int))
        self.chunk_entities: dict = defaultdict(list)
        self.entity_freq: Counter = Counter()
        self._built = False
        self._total_chunks = 0

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def entity_count(self) -> int:
        return len(self.entity_to_chunks)

    def build(self, chunks: List[str]):
        self._total_chunks = len(chunks)
        logger.info(f"[KG] 构建: {self._total_chunks} chunks")

        for i, text in enumerate(chunks):
            entities = self._extract(text)
            self.chunk_entities[i] = list(entities)
            for e in entities:
                self.entity_to_chunks[e].append(i)
                self.entity_freq[e] += 1
            elist = list(entities)
            for a in range(len(elist)):
                for b in range(a + 1, len(elist)):
                    self.co_occurrence[elist[a]][elist[b]] += 1
                    self.co_occurrence[elist[b]][elist[a]] += 1

        # IDF 过滤
        max_df = max(3, int(self._total_chunks * 0.3))
        removed = []
        for e in list(self.entity_to_chunks.keys()):
            if len(self.entity_to_chunks[e]) > max_df:
                removed.append(e)
                self._cleanup(e)

        self._built = True
        if removed:
            logger.info(f"[KG] IDF过滤: {len(removed)} 个泛化词(如 {removed[:5]})")
        logger.info(f"[KG] 构建完成: {self.entity_count} entities")
        self.save()  # 持久化到磁盘

    def _normalize_entity(self, name: str) -> str:
        """实体别名归一化: 大小写/缩写 → 标准名"""
        return _ENTITY_NORMALIZE.get(name.lower().strip(), name)

    def _extract(self, text: str) -> Set[str]:
        entities = set()
        for pair in pseg.cut(text):
            word, flag = pair.word.strip(), pair.flag
            if len(word) < 2 or word in _STOPWORDS:
                continue
            if flag not in _ALLOWED_POS and not flag.startswith('n'):
                continue
            if word.isdigit() or re.match(r'^\d+$', word):
                continue
            if re.match(r'^[#\-=*_.]+$', word):  # markdown 符号
                continue
            # 实体归一化
            entities.add(self._normalize_entity(word))
        return entities

    def _cleanup(self, entity: str):
        if entity in self.co_occurrence:
            for nb in list(self.co_occurrence[entity].keys()):
                self.co_occurrence[nb].pop(entity, None)
            del self.co_occurrence[entity]
        for cid in list(self.chunk_entities.keys()):
            if entity in self.chunk_entities[cid]:
                self.chunk_entities[cid].remove(entity)
        self.entity_to_chunks.pop(entity, None)
        self.entity_freq.pop(entity, None)

    def get_entity(self, name: str) -> dict | None:
        if not self._built:
            return None
        chunks = self.entity_to_chunks.get(name, [])
        if not chunks:
            for k in self.entity_to_chunks:
                if name in k:
                    name = k
                    chunks = self.entity_to_chunks[k]
                    break
        if not chunks:
            return None
        neighbors = self.co_occurrence.get(name, {})
        top = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "name": name, "freq": self.entity_freq.get(name, 0),
            "chunks": len(chunks),
            "related": [{"entity": n, "co_occur": c} for n, c in top],
        }

    def one_hop_expand(self, query: str, max_entities: int = 3) -> List[str]:
        if not self._built:
            return []
        q_entities = self._extract(query)
        expanded = set()
        for e in q_entities:
            neighbors = self.co_occurrence.get(e, {})
            for n, _ in sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:3]:
                if n not in q_entities:
                    expanded.add(n)
        return list(expanded)[:max_entities * 2]

    def multi_hop_expand(self, query: str, max_hops: int = 3, top_per_hop: int = 3) -> list[dict]:
        """
        多跳实体扩散 — GraphRAG 核心能力。

        从 query 实体出发，BFS 扩散 max_hops 层，每层取 co_occurrence
        最强的 top_per_hop 个邻居。返回完整的推理链路。

        示例:
            query="LangGraph 报错" -> entities=[LangGraph, 报错]
            Hop 1: LangGraph -> [Checkpointer, StateGraph, Node]
                    报错 -> [Traceback, 异常, 调试]
            Hop 2: Checkpointer -> [MemorySaver, SqliteSaver, thread_id]
                   StateGraph -> [Compile, Builder, Edge]
            ...
            输出推理链: LangGraph -> Checkpointer -> MemorySaver -> 断点恢复

        Args:
            query: 用户查询
            max_hops: 最大跳数 (1-3)
            top_per_hop: 每层每实体保留的邻居数

        Returns:
            [{hop, entity, neighbors: [{name, co_occur, freq}], expanded_from}]
            完整的多跳扩散记录，可用于可视化推理链
        """
        if not self._built:
            return []

        q_entities = list(self._extract(query))
        if not q_entities:
            return []

        hops_log: list[dict] = []
        frontier = set(q_entities)
        visited = set(q_entities)
        all_expanded = set(q_entities)

        for hop in range(1, max_hops + 1):
            hop_entities = []
            next_frontier = set()

            for entity in sorted(frontier):
                neighbors = self.co_occurrence.get(entity, {})
                top_neighbors = sorted(
                    neighbors.items(), key=lambda x: x[1], reverse=True
                )[:top_per_hop]

                hop_neighbors = []
                for nb_name, co_count in top_neighbors:
                    if nb_name in visited:
                        continue
                    visited.add(nb_name)
                    next_frontier.add(nb_name)
                    all_expanded.add(nb_name)
                    hop_neighbors.append({
                        "name": nb_name,
                        "co_occur": co_count,
                        "freq": self.entity_freq.get(nb_name, 0),
                    })

                if hop_neighbors:
                    hop_entities.append({
                        "entity": entity,
                        "freq": self.entity_freq.get(entity, 0),
                        "neighbors": hop_neighbors,
                    })

            if hop_entities:
                hops_log.append({
                    "hop": hop,
                    "entities": hop_entities,
                    "new_count": len(next_frontier),
                })

            frontier = next_frontier
            if not frontier:
                break

        # 构建推理链描述
        chain_text = self._build_chain_text(q_entities, hops_log)

        return {
            "query": query,
            "seed_entities": q_entities,
            "total_expanded": len(all_expanded),
            "max_hops": max_hops,
            "hops": hops_log,
            "expanded_entities": sorted(all_expanded),
            "chain_text": chain_text,  # 人类可读的推理链
        }

    def find_path(self, entity_a: str, entity_b: str, max_hops: int = 3) -> dict | None:
        """
        查找两个实体之间的最短关联路径。

        BFS 从 entity_a 出发，逐跳扩散直到命中 entity_b。
        面试展示利器：证明 KG 能做实体间的多跳推理。

        Args:
            entity_a: 起点实体
            entity_b: 终点实体
            max_hops: 最大搜索深度

        Returns:
            {path: [entity_a, ..., entity_b], hops: N, relations: [...]}
            或 None（找不到路径）
        """
        if not self._built:
            return None

        # 归一化
        entity_a = self._normalize_entity(entity_a)
        entity_b = self._normalize_entity(entity_b)

        if entity_a not in self.co_occurrence:
            return None
        if entity_b not in self.co_occurrence:
            return None
        if entity_a == entity_b:
            return {"path": [entity_a], "hops": 0, "relations": []}

        # BFS
        from collections import deque
        queue = deque([(entity_a, [entity_a], [])])
        visited = {entity_a}

        while queue:
            current, path, relations = queue.popleft()
            if len(path) > max_hops + 1:
                continue

            neighbors = self.co_occurrence.get(current, {})
            # 按共现强度排序，优先探索强关联
            for nb, co_count in sorted(neighbors.items(),
                                        key=lambda x: x[1], reverse=True):
                if nb == entity_b:
                    return {
                        "path": path + [nb],
                        "hops": len(path),
                        "relations": relations + [{
                            "from": current,
                            "to": nb,
                            "co_occur": co_count,
                            "depth": len(path),
                        }],
                    }
                if nb not in visited:
                    visited.add(nb)
                    queue.append((
                        nb,
                        path + [nb],
                        relations + [{
                            "from": current,
                            "to": nb,
                            "co_occur": co_count,
                            "depth": len(path),
                        }],
                    ))

        return None

    def _build_chain_text(self, seed: list[str], hops_log: list[dict]) -> str:
        """构建人类可读的推理链文本"""
        parts = [f"起点: {', '.join(seed)}"]
        for h in hops_log:
            hop_ents = []
            for e in h["entities"]:
                nb_names = [n["name"] for n in e["neighbors"][:3]]
                hop_ents.append(f"{e['entity']} -> [{', '.join(nb_names)}]")
            parts.append(f"  Hop {h['hop']}: {'; '.join(hop_ents)}")
        return "\n".join(parts)

    def graph_retrieve(self, query: str, chunks: list[str] = None, top_k: int = 5) -> list[dict]:
        """
        图检索通路（独立于向量/BM25 的第三路召回）。

        流程：
        1. 从 query 抽取实体
        2. 每个实体获取邻居实体（co_occurrence top-3）
        3. 收集这些实体关联的所有 chunk_idx
        4. 按 entity_freq × co_occurrence_count 加权排序
        5. 返回 [{chunk_idx, chunk_text, entity, score}, ...]

        Args:
            query: 用户查询
            chunks: 全局 chunk 文本列表（用于获取实际内容）
            top_k: 返回数

        Returns:
            排序后的检索结果
        """
        if not self._built:
            return []

        q_entities = list(self._extract(query))
        if not q_entities:
            return []

        # 收集候选 chunk + 分数
        chunk_scores: dict[int, float] = {}
        seen_entities = set()

        for entity in q_entities:
            # 查询实体本身的 chunk
            entity_chunks = self.entity_to_chunks.get(entity, [])
            entity_freq = self.entity_freq.get(entity, 1)
            for cid in entity_chunks:
                chunk_scores[cid] = chunk_scores.get(cid, 0) + entity_freq

            # 邻居实体
            neighbors = self.co_occurrence.get(entity, {})
            for neighbor, co_count in sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:3]:
                if neighbor in seen_entities or neighbor in q_entities:
                    continue
                seen_entities.add(neighbor)

                neighbor_chunks = self.entity_to_chunks.get(neighbor, [])
                neighbor_freq = self.entity_freq.get(neighbor, 1)
                weight = neighbor_freq * (1 + co_count)
                for cid in neighbor_chunks:
                    chunk_scores[cid] = chunk_scores.get(cid, 0) + weight

        # 排序取 top_k
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for cid, score in ranked:
            chunk_text = chunks[cid] if chunks and cid < len(chunks) else f"[chunk {cid}]"
            results.append({
                "chunk_idx": cid,
                "chunk_text": chunk_text,
                "entity": q_entities[0] if q_entities else "",
                "score": round(score, 2),
            })

        if results:
            logger.info(f"[KG GraphRetrieve] query='{query[:40]}' → {len(results)} chunks (entities: {q_entities[:3]})")
        return results

    def incremental_build(self, new_chunks: list[str], start_idx: int = 0):
        """
        增量构建 KG（不重建整个图谱，只对新 chunk 抽取实体并更新共现矩阵）。

        Args:
            new_chunks: 新增的 chunk 文本列表
            start_idx: 新 chunk 的起始索引（避免与已有 chunk 索引冲突）
        """
        logger.info(f"[KG] 增量构建: {len(new_chunks)} 新 chunks (起始索引 {start_idx})")

        for i, text in enumerate(new_chunks):
            cid = start_idx + i
            entities = self._extract(text)
            self.chunk_entities[cid] = list(entities)
            for e in entities:
                self.entity_to_chunks[e].append(cid)
                self.entity_freq[e] += 1
            elist = list(entities)
            for a in range(len(elist)):
                for b in range(a + 1, len(elist)):
                    self.co_occurrence[elist[a]][elist[b]] += 1
                    self.co_occurrence[elist[b]][elist[a]] += 1

        self._total_chunks += len(new_chunks)

        # IDF 过滤（仅对新实体做）
        max_df = max(3, int(self._total_chunks * 0.3))
        for e in list(self.entity_to_chunks.keys()):
            if len(self.entity_to_chunks[e]) > max_df:
                self._cleanup(e)

        self.save()
        logger.info(f"[KG] 增量构建完成: {self.entity_count} entities (总计 {self._total_chunks} chunks)")

    def search(self, keyword: str, limit: int = 10) -> list:
        results = []
        for e, freq in self.entity_freq.items():
            if keyword in e:
                results.append({"name": e, "freq": freq, "chunks": len(self.entity_to_chunks.get(e, []))})
        results.sort(key=lambda x: x["freq"], reverse=True)
        return results[:limit]

    # ==================== 持久化 ====================

    def save(self, path: str = ""):
        """保存 KG 到磁盘"""
        import json, os
        if not path:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "kg_state.json")
        data = {
            "entity_to_chunks": {k: list(v) for k, v in self.entity_to_chunks.items()},
            "co_occurrence": {k: dict(v) for k, v in self.co_occurrence.items()},
            "entity_freq": dict(self.entity_freq),
            "total_chunks": self._total_chunks,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"[KG] 已保存: {self.entity_count} entities → {path}")

    def load(self, path: str = "") -> bool:
        """从磁盘加载 KG"""
        import json, os
        if not path:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "kg_state.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entity_to_chunks = defaultdict(list, {k: v for k, v in data.get("entity_to_chunks", {}).items()})
            self.co_occurrence = defaultdict(lambda: defaultdict(int))
            for k, v in data.get("co_occurrence", {}).items():
                self.co_occurrence[k] = defaultdict(int, v)
            self.entity_freq = Counter(data.get("entity_freq", {}))
            self._total_chunks = data.get("total_chunks", 0)
            self._built = True
            logger.info(f"[KG] 已加载: {self.entity_count} entities (from {path})")
            return True
        except Exception as e:
            logger.warning(f"[KG] 加载失败: {e}")
            return False


_global_kg: Optional[KnowledgeGraph] = None

def get_kg() -> KnowledgeGraph:
    global _global_kg
    if _global_kg is None:
        _global_kg = KnowledgeGraph()
        _global_kg.load()  # 启动时尝试从磁盘恢复
    return _global_kg
