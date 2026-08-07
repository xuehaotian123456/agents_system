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
            entities.add(word)
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
