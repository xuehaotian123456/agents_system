"""
向量记忆 (Vector Memory)

跨会话长期记忆，基于 ChromaDB 语义检索。
关键事实持久化到向量库，对话间可检索。

使用场景:
    - 用户偏好跨会话保持（"用户上次说要静音的"）
    - 已解决问题缓存（"这个问题之前讨论过"）
    - 知识积累（Agent 学到的知识跨会话复用）

与普通 RAG 的区别:
    RAG: 检索外部文档
    VectorMemory: 检索 Agent 自身的记忆（与用户交互中积累的）
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VectorMemoryItem:
    """向量记忆条目"""
    id: str
    content: str
    category: str = "general"              # general / preference / knowledge / decision
    importance: float = 0.5
    session_id: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorMemory:
    """
    向量记忆

    使用方式:
        vm = VectorMemory(persist_dir="./memory_db")

        # 存储记忆
        await vm.store("用户偏好 Python 技术栈", category="preference", importance=0.9)

        # 检索记忆
        memories = await vm.search("技术栈偏好", limit=5)
        for m in memories:
            print(f"[{m.category}] {m.content} (importance: {m.importance})")

        # 注入到 context
        context = vm.to_context_text("技术栈偏好")
    """

    def __init__(
        self,
        persist_dir: str = "./memory_db",
        collection_name: str = "agent_memory",
        embedding_function=None,  # 可传入自定义 embedding
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._embedding_fn = embedding_function

        # 延迟初始化 Chroma（避免启动时下载模型）
        self._collection = None
        self._initialized = False

    async def _ensure_init(self):
        """延迟初始化"""
        if self._initialized:
            return

        import chromadb
        import os

        # 如果有 DashScope embedding，使用之
        if self._embedding_fn is None and os.getenv("DASHSCOPE_API_KEY"):
            try:
                from tools.rag_tool import _DashScopeEmbedding
                from openai import AsyncOpenAI
                import httpx
                client = AsyncOpenAI(
                    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    http_client=httpx.AsyncClient(timeout=httpx.Timeout(30.0)),
                )
                self._embedding_fn = _DashScopeEmbedding(client, "text-embedding-v1")
            except Exception:
                pass

        client = chromadb.PersistentClient(path=self.persist_dir)

        if self._embedding_fn:
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
            )
        else:
            # Fallback: 使用 Chroma 默认 embedding（会下载 ONNX 模型）
            self._collection = client.get_or_create_collection(name=self.collection_name)

        self._initialized = True

    # ==================== CRUD ====================

    async def store(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        session_id: str = "",
        metadata: dict | None = None,
    ) -> str:
        """
        存储一条记忆

        Returns:
            记忆 ID
        """
        await self._ensure_init()

        # 生成唯一 ID
        mem_id = hashlib.md5(
            f"{content[:100]}{time.time()}".encode()
        ).hexdigest()[:12]

        # 向量存储
        import asyncio
        def _add():
            self._collection.add(
                ids=[mem_id],
                documents=[content],
                metadatas=[{
                    "category": category,
                    "importance": importance,
                    "session_id": session_id,
                    "created_at": time.time(),
                    **(metadata or {}),
                }],
            )
        await asyncio.to_thread(_add)

        return mem_id

    async def search(
        self,
        query: str,
        limit: int = 5,
        category_filter: str | None = None,
        min_importance: float = 0.0,
    ) -> list[VectorMemoryItem]:
        """
        语义检索记忆

        Args:
            query: 查询文本
            limit: 返回数量
            category_filter: 过滤分类
            min_importance: 最低重要性阈值

        Returns:
            相关记忆列表
        """
        await self._ensure_init()

        import asyncio

        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        def _query():
            return self._collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

        results = await asyncio.to_thread(_query)

        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0

                if meta.get("importance", 0.5) < min_importance:
                    continue

                memories.append(VectorMemoryItem(
                    id=results["ids"][0][i] if results["ids"] else "",
                    content=doc,
                    category=meta.get("category", "general"),
                    importance=meta.get("importance", 0.5),
                    session_id=meta.get("session_id", ""),
                    metadata={"distance": dist, **meta},
                ))

        return memories

    async def forget(self, mem_id: str):
        """删除一条记忆"""
        await self._ensure_init()
        import asyncio

        def _delete():
            self._collection.delete(ids=[mem_id])

        await asyncio.to_thread(_delete)

    async def clear(self):
        """清空所有记忆"""
        await self._ensure_init()
        import asyncio

        def _clear():
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)

        await asyncio.to_thread(_clear)

    # ==================== 上下文整合 ====================

    def to_context_text(self, query: str, limit: int = 3) -> str:
        """
        搜索相关记忆并转为可注入上下文的文本

        同步包装，方便在 Session.build_messages 中调用。
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            memories = asyncio.run(self.search(query, limit=limit))
        else:
            # 在已有事件循环中，创建新任务（简化处理）
            return ""

        if not memories:
            return ""

        lines = ["## 相关历史记忆"]
        for m in memories:
            cat_emoji = {"preference": "⭐", "knowledge": "📚", "decision": "✅"}.get(m.category, "📌")
            lines.append(f"- {cat_emoji} {m.content}")

        return "\n".join(lines)

    # ==================== 统计 ====================

    async def stats(self) -> dict:
        """获取记忆库统计"""
        await self._ensure_init()
        import asyncio

        def _get():
            return self._collection.get(include=["metadatas"])

        data = await asyncio.to_thread(_get)

        categories = {}
        for meta in (data.get("metadatas") or []):
            cat = meta.get("category", "general") if meta else "general"
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_memories": len(data.get("ids", [])),
            "by_category": categories,
            "collection": self.collection_name,
        }
