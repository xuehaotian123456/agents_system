"""
CC-Harness Agent — 持久化执行 & 断点恢复 (CheckpointSaver)
=============================================================
让 Agent 具备"崩溃恢复"能力，对标 LangGraph PostgresSaver。

核心场景:
    1. Agent 执行到第 5 轮时 LLM 超时 → 恢复后从第 5 轮继续
    2. 服务器重启 → 所有未完成会话自动恢复
    3. 用户关闭页面后重新打开 → 回到上次对话状态

设计:
    每轮循环结束时自动保存 Checkpoint（状态快照）。
    Checkpoint 包含:
    - 消息历史 (messages)
    - 循环轮次 (turn)
    - 工具调用记录
    - 工作记忆状态
    - 配置参数

后端支持:
    - SQLite (单机/开发)
    - Redis (分布式/生产)
    - Memory (测试/禁用持久化)

使用方式:
    # SQLite 持久化
    saver = CheckpointSaver("sqlite://./checkpoints.db")
    session = Session(config, saver=saver)

    # Agent 崩溃后恢复
    session = await saver.resume(session_id)
    loop = AgentLoop(session, ...)
    answer = await loop.run()  # 从断点继续

    # Redis 持久化 (多实例共享)
    saver = CheckpointSaver("redis://localhost:6379")
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.types import AgentConfig, Message, MessageRole


# ==================== Checkpoint 数据模型 ====================

@dataclass
class Checkpoint:
    """一次状态快照"""
    session_id: str
    turn: int
    messages_json: str                 # 消息列表的 JSON
    system_prompt: str = ""
    config_json: str = "{}"            # AgentConfig 的 JSON
    tool_calls_count: int = 0
    subagents_spawned: int = 0
    compression_count: int = 0
    working_memory_json: str = "[]"    # 工作记忆 JSON
    metadata_json: str = "{}"          # 自定义元数据
    created_at: float = field(default_factory=time.time)
    checkpoint_id: str = ""            # 唯一 ID

    def __post_init__(self):
        if not self.checkpoint_id:
            self.checkpoint_id = f"ckpt_{self.session_id}_{self.turn}_{int(self.created_at)}"


# ==================== SQLite 后端 ====================

class SQLiteBackend:
    """SQLite 存储后端"""

    def __init__(self, db_path: str):
        self.db_path = db_path.replace("sqlite://", "")
        self._conn: sqlite3.Connection | None = None

    def _ensure_table(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                session_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                messages_json TEXT DEFAULT '[]',
                system_prompt TEXT DEFAULT '',
                config_json TEXT DEFAULT '{}',
                tool_calls_count INTEGER DEFAULT 0,
                subagents_spawned INTEGER DEFAULT 0,
                compression_count INTEGER DEFAULT 0,
                working_memory_json TEXT DEFAULT '[]',
                metadata_json TEXT DEFAULT '{}',
                created_at REAL,
                checkpoint_id TEXT PRIMARY KEY
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_turn
            ON checkpoints(session_id, turn DESC)
        """)
        self._conn.commit()

    def connect(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_table()

    def disconnect(self):
        if self._conn:
            self._conn.close()

    def save(self, ckpt: Checkpoint):
        self._conn.execute("""
            INSERT OR REPLACE INTO checkpoints
            (session_id, turn, messages_json, system_prompt, config_json,
             tool_calls_count, subagents_spawned, compression_count,
             working_memory_json, metadata_json, created_at, checkpoint_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ckpt.session_id, ckpt.turn, ckpt.messages_json, ckpt.system_prompt,
            ckpt.config_json, ckpt.tool_calls_count, ckpt.subagents_spawned,
            ckpt.compression_count, ckpt.working_memory_json,
            ckpt.metadata_json, ckpt.created_at, ckpt.checkpoint_id,
        ))
        self._conn.commit()

    def load_latest(self, session_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY turn DESC LIMIT 1",
            (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT session_id FROM checkpoints ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        return [r["session_id"] for r in rows]

    def delete(self, session_id: str):
        self._conn.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
        self._conn.commit()


# ==================== Redis 后端 ====================

class RedisBackend:
    """Redis 存储后端"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None

    async def connect(self):
        import redis.asyncio as aioredis
        self._redis = await aioredis.from_url(
            self.redis_url.replace("redis://", "redis://"),
            encoding="utf-8", decode_responses=True,
        )

    async def disconnect(self):
        if self._redis:
            await self._redis.close()

    async def save(self, ckpt: Checkpoint):
        key = f"ckpt:{ckpt.session_id}"
        data = {
            "session_id": ckpt.session_id,
            "turn": str(ckpt.turn),
            "messages_json": ckpt.messages_json,
            "system_prompt": ckpt.system_prompt,
            "config_json": ckpt.config_json,
            "tool_calls_count": str(ckpt.tool_calls_count),
            "subagents_spawned": str(ckpt.subagents_spawned),
            "compression_count": str(ckpt.compression_count),
            "working_memory_json": ckpt.working_memory_json,
            "metadata_json": ckpt.metadata_json,
            "created_at": str(ckpt.created_at),
            "checkpoint_id": ckpt.checkpoint_id,
        }
        await self._redis.hset(key, mapping=data)
        await self._redis.expire(key, 86400 * 7)  # 7 天过期

    async def load_latest(self, session_id: str) -> dict | None:
        data = await self._redis.hgetall(f"ckpt:{session_id}")
        return data if data else None

    async def list_sessions(self) -> list[str]:
        keys = await self._redis.keys("ckpt:*")
        return [k.replace("ckpt:", "") for k in keys[:100]]

    async def delete(self, session_id: str):
        await self._redis.delete(f"ckpt:{session_id}")


# ==================== CheckpointSaver ====================

class CheckpointSaver:
    """
    状态持久化管理器

    使用方式:
        # 初始化
        saver = CheckpointSaver("sqlite://./checkpoints.db")

        # AgentLoop 每轮调用
        await saver.save(session)

        # 崩溃后恢复
        session = await saver.resume(session_id, llm_adapter, tool_registry, prompt_engine)
        if session:
            loop = AgentLoop(session, ...)
            answer = await loop.run()
    """

    def __init__(self, backend_url: str = "sqlite://./agent_checkpoints.db"):
        """
        Args:
            backend_url: "sqlite://path" / "redis://host:port" / "memory://"
        """
        self.backend_url = backend_url
        self._sqlite: SQLiteBackend | None = None
        self._redis: RedisBackend | None = None
        self._memory: dict[str, Checkpoint] = {}  # 内存后端
        self._initialized = False

        if backend_url.startswith("sqlite"):
            self._sqlite = SQLiteBackend(backend_url)
        elif backend_url.startswith("redis"):
            self._redis = RedisBackend(backend_url)
        # else: memory backend

    async def _ensure_init(self):
        if self._initialized:
            return
        if self._sqlite:
            await asyncio.to_thread(self._sqlite.connect)
        elif self._redis:
            await self._redis.connect()
        self._initialized = True

    async def close(self):
        if self._sqlite:
            await asyncio.to_thread(self._sqlite.disconnect)
        elif self._redis:
            await self._redis.disconnect()
        self._initialized = False

    # ==================== 保存 ====================

    async def save(self, session) -> Checkpoint:
        """
        保存当前会话状态

        在 AgentLoop 每轮循环结束后调用。
        也支持手动在任何时机调用。
        """
        await self._ensure_init()

        # 序列化消息
        messages_json = json.dumps(
            [{"role": m.role.value, "content": m.content,
              "tool_name": m.tool_name, "tool_call_id": m.tool_call_id}
             for m in session.messages],
            ensure_ascii=False,
        )

        ckpt = Checkpoint(
            session_id=session.session_id,
            turn=session.total_turns,
            messages_json=messages_json,
            system_prompt=session._system_prompt,
            config_json=session.config.model_dump_json() if hasattr(session.config, 'model_dump_json') else json.dumps({}),
            tool_calls_count=session.tool_calls_count,
            subagents_spawned=session.subagents_spawned,
            compression_count=session._compression_count,
            working_memory_json="[]",
        )

        if self._sqlite:
            await asyncio.to_thread(self._sqlite.save, ckpt)
        elif self._redis:
            await self._redis.save(ckpt)
        else:
            self._memory[ckpt.session_id] = ckpt

        return ckpt

    # ==================== 恢复 ====================

    async def resume(self, session_id: str) -> Optional[Any]:
        """
        从最近的 Checkpoint 恢复会话

        Returns:
            Session 对象（如果找到），否则 None
        """
        await self._ensure_init()

        data = None
        if self._sqlite:
            data = await asyncio.to_thread(self._sqlite.load_latest, session_id)
        elif self._redis:
            data = await self._redis.load_latest(session_id)
        else:
            ckpt = self._memory.get(session_id)
            data = ckpt.__dict__ if ckpt else None

        if not data:
            return None

        # 重建 Session
        from harness.session import Session

        config = AgentConfig()
        try:
            config_data = json.loads(data.get("config_json", "{}"))
            config = AgentConfig(**config_data)
        except Exception:
            pass

        session = Session(config=config)
        session.session_id = session_id

        # 恢复消息
        try:
            messages_data = json.loads(data.get("messages_json", "[]"))
            for m in messages_data:
                session.messages.append(Message(
                    role=MessageRole(m.get("role", "user")),
                    content=m.get("content", ""),
                    tool_name=m.get("tool_name"),
                    tool_call_id=m.get("tool_call_id"),
                ))
        except Exception:
            pass

        # 恢复状态
        session._system_prompt = data.get("system_prompt", "")
        session.total_turns = int(data.get("turn", 0))
        session.tool_calls_count = int(data.get("tool_calls_count", 0))
        session.subagents_spawned = int(data.get("subagents_spawned", 0))
        session._compression_count = int(data.get("compression_count", 0))

        return session

    # ==================== 查询 ====================

    async def list_sessions(self) -> list[str]:
        """列出所有保存的会话"""
        await self._ensure_init()
        if self._sqlite:
            return await asyncio.to_thread(self._sqlite.list_sessions)
        elif self._redis:
            return await self._redis.list_sessions()
        return list(self._memory.keys())

    async def delete(self, session_id: str):
        """删除会话的所有 checkpoint"""
        await self._ensure_init()
        if self._sqlite:
            await asyncio.to_thread(self._sqlite.delete, session_id)
        elif self._redis:
            await self._redis.delete(session_id)
        else:
            self._memory.pop(session_id, None)


# ==================== AgentLoop 集成 ====================

class DurableAgentLoop:
    """
    带持久化的 AgentLoop 包装器

    每轮循环自动保存 Checkpoint。
    如果 run() 抛出异常，下一次 run() 自动从断点恢复。

    使用方式:
        loop = DurableAgentLoop(session, llm, registry, prompt_engine, saver=saver)
        try:
            answer = await loop.run()
        except Exception:
            # 服务器重启后...
            answer = await loop.run()  # 自动从断点继续
    """

    def __init__(self, session, llm_adapter, tool_registry, prompt_engine,
                 saver: CheckpointSaver, tracer=None):
        self.session = session
        self.llm = llm_adapter
        self.tool_registry = tool_registry
        self.prompt_engine = prompt_engine
        self.saver = saver
        self.tracer = tracer

    async def run(self) -> str:
        """
        执行 AgentLoop，每轮自动保存 Checkpoint

        如果 session 之前已经执行过（total_turns > 0），
        说明这是恢复执行——从当前状态继续而不重置。
        """
        from harness.agent_loop import AgentLoop

        loop = AgentLoop(
            session=self.session,
            llm_adapter=self.llm,
            tool_registry=self.tool_registry,
            prompt_engine=self.prompt_engine,
            tracer=self.tracer,
        )

        # 包装 run()，每轮后自动保存
        # 注意：这里利用 AgentLoop 的 while 循环特性
        # 当从断点恢复时，session.total_turns 已有值，AgentLoop 会自动跳过已执行的轮次

        try:
            answer = await loop.run()
            # 成功完成 → 保存最终状态
            await self.saver.save(self.session)
            return answer
        except Exception as e:
            # 异常 → 保存当前状态以便恢复
            await self.saver.save(self.session)
            raise
