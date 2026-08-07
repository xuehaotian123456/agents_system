"""
CC-Harness Agent - 异步缓存组件
================================
基于 redis.asyncio 实现对话缓存。

为什么用 Redis 而不是内存缓存？
- 多实例部署时共享缓存
- 服务重启后缓存不丢失
- 支持分布式限流和分布式锁

缓存策略：
1. 对话缓存：高频问题的 LLM 回答缓存 → 减少模型调用
2. 向量检索缓存：相同 query 的检索结果缓存 → 减少 embedding 计算
3. 会话缓存：Session 对象序列化存储 → 支持分布式会话

使用方式：
    cache = AsyncCache(redis_url="redis://localhost:6379")
    # 存入缓存，1小时过期
    await cache.set("key", "value", ttl=3600)
    # 读取缓存
    value = await cache.get("key")
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import redis.asyncio as aioredis


class AsyncCache:
    """
    异步 Redis 缓存

    重要：使用 redis.asyncio（异步客户端），不能用同步 redis-py！
    同步 Redis 会阻塞事件循环，导致所有并发请求卡住。
    """

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self):
        """建立 Redis 连接（在 FastAPI 启动事件中调用）"""
        self._redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        # 测试连接
        await self._redis.ping()

    async def disconnect(self):
        """关闭 Redis 连接（在 FastAPI 关闭事件中调用）"""
        if self._redis:
            await self._redis.close()

    @property
    def client(self) -> aioredis.Redis:
        """获取 Redis 客户端（如果未连接则抛出异常）"""
        if self._redis is None:
            raise RuntimeError("Redis 未连接，请先调用 connect()")
        return self._redis

    # ==================== 基础操作 ====================

    async def get(self, key: str) -> Optional[str]:
        """读取缓存"""
        return await self.client.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600):
        """写入缓存（带过期时间）"""
        await self.client.setex(key, ttl, value)

    async def delete(self, key: str):
        """删除缓存"""
        await self.client.delete(key)

    # ==================== 高级场景 ====================

    async def cache_llm_response(self, query: str, response: str, ttl: int = 3600):
        """
        缓存 LLM 回答（对高频重复问题有效）

        key 设计：prefix:query_hash
        例如："llm_cache:abc123def"

        注意：只缓存标准化问答（如 RAG 检索结果），不缓存个性化回答。
        """
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
        key = f"llm_cache:{query_hash}"
        await self.set(key, response, ttl=ttl)

    async def get_cached_llm_response(self, query: str) -> Optional[str]:
        """读取缓存的 LLM 回答"""
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
        key = f"llm_cache:{query_hash}"
        return await self.get(key)

    async def cache_session(self, session_id: str, session_data: dict, ttl: int = 86400):
        """
        缓存会话数据（分布式会话共享）

        存入 Redis 后，其他 FastAPI 实例也可以读取同一会话。
        """
        key = f"session:{session_id}"
        await self.set(key, json.dumps(session_data, ensure_ascii=False), ttl=ttl)

    async def get_cached_session(self, session_id: str) -> Optional[dict]:
        """读取缓存的会话数据"""
        key = f"session:{session_id}"
        data = await self.get(key)
        return json.loads(data) if data else None
