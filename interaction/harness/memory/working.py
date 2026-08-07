"""
工作记忆 (Working Memory)

会话内短期记忆，自动追踪重要事实、用户偏好和关键实体。

设计:
    - 容量有限（默认 7 条），满时自动淘汰最不重要的
    - 每条记忆有重要性评分，LLM 调用越多越重要
    - 在 system prompt 中注入当前工作记忆
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryItem:
    """单条工作记忆"""
    content: str                           # 记忆内容
    category: str = "fact"                 # fact / preference / entity / decision
    importance: float = 0.5                # 重要性 (0-1)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0                  # 被访问次数
    source: str = ""                       # 来源（哪条消息提取的）


class WorkingMemory:
    """
    工作记忆

    使用方式:
        wm = WorkingMemory(max_items=7)

        # Agent 发现重要信息时记录
        wm.add("用户偏好静音扫地机器人", category="preference", importance=0.9)
        wm.add("讨论的技术栈: LangGraph + ChromaDB", category="entity")

        # 重要事实自动提升优先级
        wm.access("用户偏好静音扫地机器人")  # importance += 0.05

        # 注入到 system prompt
        prompt = wm.to_prompt_text()
        # → "## 当前已知信息\n- [偏好] 用户偏好静音扫地机器人\n- ..."
    """

    def __init__(self, max_items: int = 7):
        self.max_items = max_items
        self._items: list[MemoryItem] = []

    def add(self, content: str, category: str = "fact",
            importance: float = 0.5, source: str = "") -> MemoryItem:
        """
        添加一条记忆

        - 如果已存在相似内容，更新重要性（+= importance）
        - 如果容量已满，淘汰重要性最低的
        """
        # 去重检查
        for item in self._items:
            if self._is_similar(item.content, content):
                item.importance = min(1.0, item.importance + importance)
                item.access_count += 1
                return item

        # 创建新记忆
        mem = MemoryItem(
            content=content,
            category=category,
            importance=importance,
            source=source,
        )

        # 容量管理
        if len(self._items) >= self.max_items:
            # 淘汰最不重要的
            self._items.sort(key=lambda x: x.importance)
            removed = self._items.pop(0)
            # 如果新记忆比被淘汰的还不重要，放弃添加
            if importance <= removed.importance:
                return removed

        self._items.append(mem)
        return mem

    def access(self, query: str) -> Optional[MemoryItem]:
        """访问记忆（提升重要性）"""
        for item in self._items:
            if self._is_similar(item.content, query):
                item.importance = min(1.0, item.importance + 0.05)
                item.access_count += 1
                return item
        return None

    def search(self, query: str) -> list[MemoryItem]:
        """搜索相关记忆（简单关键词匹配）"""
        query_lower = query.lower()
        results = []
        for item in self._items:
            if any(word in item.content.lower() for word in query_lower.split()):
                results.append(item)
        return sorted(results, key=lambda x: x.importance, reverse=True)

    def forget(self, content: str):
        """删除一条记忆"""
        self._items = [i for i in self._items if not self._is_similar(i.content, content)]

    def clear(self):
        """清空所有记忆"""
        self._items.clear()

    def to_prompt_text(self) -> str:
        """将工作记忆转为 prompt 可用的文本"""
        if not self._items:
            return ""

        sorted_items = sorted(self._items, key=lambda x: x.importance, reverse=True)

        lines = ["## 当前已知信息（工作记忆）"]
        for item in sorted_items:
            cat_label = {
                "fact": "📌",
                "preference": "⭐",
                "entity": "🔗",
                "decision": "✅",
            }.get(item.category, "📌")
            lines.append(f"- {cat_label} [{item.category}] {item.content}")

        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        """序列化"""
        return [
            {
                "content": i.content,
                "category": i.category,
                "importance": i.importance,
                "access_count": i.access_count,
                "source": i.source,
            }
            for i in self._items
        ]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "WorkingMemory":
        """反序列化"""
        wm = cls()
        for d in data:
            wm.add(
                content=d["content"],
                category=d.get("category", "fact"),
                importance=d.get("importance", 0.5),
                source=d.get("source", ""),
            )
        return wm

    def _is_similar(self, a: str, b: str) -> bool:
        """简单相似度判断"""
        # 基于共同字符的 Jaccard 相似度
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return False
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union > 0.6

    @property
    def size(self) -> int:
        return len(self._items)

    def __len__(self) -> int:
        return len(self._items)
