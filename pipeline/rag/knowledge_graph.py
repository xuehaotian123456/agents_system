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
    # ── 英文功能词 (KG 垃圾实体主来源: 'in' 出现 424 次污染共现矩阵) ──
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "with", "without", "from", "by", "as",
    "if", "then", "else", "when", "while", "not", "no", "yes", "so", "such",
    "than", "into", "onto", "over", "under", "about", "after", "before",
    "we", "you", "he", "she", "they", "them", "their", "our", "your", "his",
    "her", "my", "me", "us", "i", "do", "does", "did", "done", "doing",
    "can", "could", "will", "would", "should", "shall", "may", "might",
    "must", "have", "has", "had", "having", "get", "got", "make", "made",
    "use", "used", "using", "also", "only", "just", "very", "too", "more",
    "most", "some", "any", "all", "each", "every", "both", "few", "many",
    "much", "other", "another", "same", "different", "one", "two", "three",
    "first", "last", "new", "old", "good", "bad", "well", "now", "here",
    "there", "where", "what", "which", "who", "whom", "whose", "why", "how",
    "up", "down", "out", "off", "through", "between", "among", "during",
    "per", "via", "etc", "e.g", "i.e", "vs", "com", "www", "http", "https",
    # ── 爬虫元数据词 (来源 header 污染共现矩阵的假邻居) ──
    "gitee", "github", "community", "develop", "issue", "issues", "repo",
    "repository", "readme", "docs", "md", "source", "来源", "可信度", "标签",
    "platform", "fork", "star", "branch", "commit", "release", "version",
    "hackernews", "oschina", "trending", "rss", "头条", "热榜", "转载",
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
        # 社区检测 (微软 GraphRAG 机制)
        self.communities: dict = {}              # {entity: community_id}
        self.community_summaries: dict = {}      # {cid: {summary, top_entities, size}}

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def entity_count(self) -> int:
        return len(self.entity_to_chunks)

    def build(self, chunks: List[str]):
        # ★ 必须重置: 否则重复 build 会累积旧实体/边 (此前 KG 实体数漂移的原因之一)
        self.entity_to_chunks.clear()
        self.co_occurrence.clear()
        self.chunk_entities.clear()
        self.entity_freq.clear()

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
            # 停用词大小写不敏感 (Gitee/gitee 都过滤)
            if len(word) < 2 or word in _STOPWORDS or word.lower() in _STOPWORDS:
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

    # ==================== 社区检测 + 社区摘要 (微软 GraphRAG 机制) ====================

    def detect_communities(self, max_iter: int = 10,
                           edge_weight_pct: int = 75,
                           top_n_per_entity: int = 5) -> dict[str, int]:
        """
        标签传播社区检测 (纯 Python, 零第三方依赖)。

        ★ 关键设计: 在稀疏化的强边子图上运行。
        原始共现图是毛线球 (泛词桥接一切, 直接传播会坍缩成 1 个大社区
        ——实测 11,469 社区里 3,563 实体塌进社区0)。两步稀疏化:
          1. 全局权重阈值: 只保留 w >= P75 分位数的边 (剪噪音边)
          2. 每实体 top-N: 每个实体最多保留最强的 top_n_per_entity 条边
             (kNN 稀疏化, 防止 hub 实体桥接所有社区)

        Args:
            max_iter: 传播最大迭代轮数
            edge_weight_pct: 边权重分位数阈值 (75 = 只保留 top 25% 强边)
            top_n_per_entity: 每实体保留的最强邻居数 (防 hub 桥接)

        Returns:
            {entity: community_id}
        """
        # ── 强边子图 (全局阈值 + per-entity top-N 稀疏化) ──
        all_weights = [w for nbrs in self.co_occurrence.values() for w in nbrs.values()]
        if not all_weights:
            return {}
        threshold = max(4, sorted(all_weights)[int(len(all_weights) * edge_weight_pct / 100)])
        logger.info(f"[KG社区] 强边阈值 w>={threshold} (P{edge_weight_pct}, 总边 {len(all_weights)})")

        strong_edges: dict[str, list] = {}   # entity -> [(nb, w)]
        for e, nbrs in self.co_occurrence.items():
            kept = [(nb, w) for nb, w in nbrs.items() if w >= threshold]
            kept.sort(key=lambda x: -x[1])
            strong_edges[e] = kept[:top_n_per_entity]
        n_edges = sum(len(v) for v in strong_edges.values())
        logger.info(f"[KG社区] 稀疏化后边数: {n_edges} (每实体 top-{top_n_per_entity})")

        # ── 标签传播 ──
        labels = {e: i for i, e in enumerate(self.co_occurrence.keys())}
        entities = list(labels.keys())

        for _ in range(max_iter):
            changed = 0
            for e in entities:
                neighbor_labels: dict[int, float] = {}
                for nb, w in strong_edges.get(e, []):
                    if nb in labels:
                        neighbor_labels[labels[nb]] = neighbor_labels.get(labels[nb], 0.0) + w
                if not neighbor_labels:
                    continue
                new_label = max(neighbor_labels, key=neighbor_labels.get)
                if labels[e] != new_label:
                    labels[e] = new_label
                    changed += 1
            if changed == 0:
                break

        # 社区重编号 (连续 id, 按大小排序)
        from collections import Counter as _Counter
        cid_counts = _Counter(labels.values())
        ordered = sorted(cid_counts, key=lambda c: -cid_counts[c])
        remap = {old: new for new, old in enumerate(ordered)}
        labels = {e: remap[c] for e, c in labels.items()}

        self.communities = labels
        n_communities = len(set(labels.values()))
        logger.info(f"[KG社区] 检测完成: {n_communities} 个社区 "
                    f"(top: {[(c, n) for c, n in cid_counts.most_common(5)]})")
        return labels

    def build_community_summaries(self, chunks: list[str] | None = None,
                                  min_size: int = 5, max_communities: int = 60) -> dict:
        """
        社区摘要: 每个社区用 LLM 生成一段语义摘要 (微软 GraphRAG 机制)。

        Args:
            chunks: 全局 chunk 文本 (用于找代表性片段)
            min_size: 小于此实体数的社区跳过
            max_communities: 最多摘要的社区数 (成本控制)

        Returns:
            {community_id: {"summary", "top_entities", "size"}}
        """
        if not getattr(self, "communities", None):
            self.detect_communities()

        # ── 社区分组 ──
        from collections import defaultdict as _dd
        comm_entities: dict[int, list] = _dd(list)
        for e, c in self.communities.items():
            comm_entities[c].append(e)

        # 过滤 + 排序 (大社区优先)
        valid = [(c, ents) for c, ents in comm_entities.items() if len(ents) >= min_size]
        valid.sort(key=lambda x: -len(x[1]))
        valid = valid[:max_communities]
        logger.info(f"[KG社区] 摘要目标: {len(valid)} 个社区 (min_size={min_size})")

        # ── 代表性 chunk 预计算 ──
        chunk_by_comm: dict[int, list[str]] = _dd(list)
        if chunks:
            for e, c in self.communities.items():
                for cid in self.entity_to_chunks.get(e, [])[:2]:
                    if cid < len(chunks):
                        chunk_by_comm[c].append(chunks[cid])

        # ── LLM 批量摘要 (每批 10 社区) ──
        summaries: dict = {}
        batch_size = 10
        try:
            from model.factory import robust_llm_call
        except Exception:
            robust_llm_call = None

        for i in range(0, len(valid), batch_size):
            batch = valid[i:i + batch_size]
            if robust_llm_call is None:
                for c, ents in batch:
                    summaries[c] = {
                        "summary": f"社区主题: {', '.join(ents[:10])}",
                        "top_entities": ents[:10], "size": len(ents)}
                continue

            prompt_parts = []
            for idx, (c, ents) in enumerate(batch):
                top_ents = sorted(ents, key=lambda e: self.entity_freq.get(e, 0), reverse=True)[:10]
                snippets = (chunk_by_comm.get(c, [])[:2])
                snippet_text = " | ".join(s[:120] for s in snippets)
                prompt_parts.append(
                    f"[社区{idx}] 实体: {', '.join(top_ents)}\n  代表片段: {snippet_text}")

            prompt = (
                "你是知识图谱社区分析师。以下是实体共现图检测出的社区, "
                "每个社区代表一组紧密关联的技术概念。请为每个社区写一句"
                "60字以内的中文摘要, 概括该社区的知识主题。\n\n"
                + "\n".join(prompt_parts) +
                "\n\n以JSON数组输出: [{\"index\":0,\"summary\":\"...\"}, ...]")

            try:
                resp = robust_llm_call(prompt)
                raw = resp.content if hasattr(resp, 'content') else str(resp)
                import json as _json, re as _re
                try:
                    data = _json.loads(raw.strip())
                except Exception:
                    m = _re.search(r'\[[\s\S]*\]', raw)
                    data = _json.loads(m.group(0)) if m else []
                parsed = {}
                for item in data if isinstance(data, list) else []:
                    if isinstance(item, dict) and "index" in item:
                        parsed[int(item["index"])] = item.get("summary", "")
                for idx, (c, ents) in enumerate(batch):
                    top_ents = sorted(ents, key=lambda e: self.entity_freq.get(e, 0), reverse=True)[:10]
                    summaries[c] = {
                        "summary": parsed.get(idx, f"社区主题: {', '.join(top_ents[:6])}"),
                        "top_entities": top_ents, "size": len(ents)}
            except Exception as e:
                logger.warning(f"[KG社区] 摘要 LLM 失败 (批次{i}): {e}")
                for c, ents in batch:
                    summaries[c] = {
                        "summary": f"社区主题: {', '.join(ents[:8])}",
                        "top_entities": ents[:8], "size": len(ents)}

        self.community_summaries = summaries
        self.save()
        logger.info(f"[KG社区] 摘要完成: {len(summaries)} 个社区摘要")
        return summaries

    def global_search(self, query: str, top_k: int = 3) -> dict:
        """
        全局检索 (Global Search): 按社区索引回答"整体性"问题。

        query 实体 → 定位所属社区 → 按命中实体权重排序 → 返回
        社区摘要 + 代表实体 + 代表 chunk 索引。

        Args:
            query: 用户查询
            top_k: 返回社区数

        Returns:
            {"communities": [{id, summary, top_entities, match_count}],
             "matched_entities": [...]}
        """
        if not getattr(self, "community_summaries", None):
            return {"communities": [], "matched_entities": [], "note": "社区摘要未构建, 先运行 build_community_summaries()"}

        q_entities = list(self._extract(query))
        if not q_entities:
            return {"communities": [], "matched_entities": []}

        # 命中社区计数 (按实体频次加权)
        comm_scores: dict[int, float] = {}
        matched = []
        for e in q_entities:
            c = self.communities.get(e)
            if c is not None and c in self.community_summaries:
                comm_scores[c] = comm_scores.get(c, 0) + self.entity_freq.get(e, 1)
                matched.append(e)

        ranked = sorted(comm_scores, key=lambda c: -comm_scores[c])[:top_k]
        communities = []
        for c in ranked:
            info = self.community_summaries[c]
            communities.append({
                "id": c,
                "summary": info.get("summary", ""),
                "top_entities": info.get("top_entities", []),
                "size": info.get("size", 0),
                "match_count": round(comm_scores[c], 1),
            })
        return {"communities": communities, "matched_entities": matched[:10]}

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
            # 社区结构
            "communities": {k: v for k, v in self.communities.items()},
            "community_summaries": self.community_summaries,
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
            # 社区结构
            self.communities = {k: int(v) for k, v in data.get("communities", {}).items()}
            self.community_summaries = {
                int(k): v for k, v in data.get("community_summaries", {}).items()}
            self._built = True
            logger.info(f"[KG] 已加载: {self.entity_count} entities, "
                        f"{len(self.community_summaries)} 社区摘要 (from {path})")
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
