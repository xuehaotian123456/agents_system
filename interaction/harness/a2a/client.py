"""
A2A Client — 连接远程 Agent

通过 A2A 协议发现远程 Agent 能力，发送任务，接收流式结果。

使用方式:
    client = A2AClient("http://localhost:8010")
    agent = await client.discover()  # 获取远程 Agent 的能力描述

    # 方式1: 直接对话
    async for token in client.stream_task("帮我搜索 Python 相关文章"):
        print(token, end="")

    # 方式2: 将远程工具注册到本地 ToolRegistry
    tools = await client.list_tools()
    for tool in tools:
        local_registry.register(A2AToolAdapter(tool, client))
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx


@dataclass
class A2AToolDef:
    """远程 Agent 暴露的工具定义"""
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)


@dataclass
class RemoteAgent:
    """远程 Agent 信息（从 Agent Card 解析）"""
    name: str
    description: str
    url: str
    tools: list[A2AToolDef] = field(default_factory=list)
    capabilities: dict = field(default_factory=dict)


class A2AClient:
    """
    A2A 客户端

    使用方式:
        client = A2AClient("http://localhost:8010")
        agent = await client.discover()
        print(f"发现远程 Agent: {agent.name}, 工具: {[t.name for t in agent.tools]}")
    """

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._agent_card: dict | None = None

    async def discover(self) -> RemoteAgent:
        """获取远程 Agent 的 Agent Card"""
        try:
            # 标准 A2A 端点
            r = await self.http.get(f"{self.base_url}/.well-known/agent.json")
            r.raise_for_status()
            card = r.json()
        except Exception:
            # Fallback: 简单健康检查端点
            r = await self.http.get(f"{self.base_url}/health")
            r.raise_for_status()
            card = {
                "name": "Remote Agent",
                "description": f"Agent at {self.base_url}",
                "url": self.base_url,
                "tools": [],
            }

        self._agent_card = card

        tools = []
        for t in card.get("tools", []):
            tools.append(A2AToolDef(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("parameters", {})),
            ))

        return RemoteAgent(
            name=card.get("name", "Unknown"),
            description=card.get("description", ""),
            url=card.get("url", self.base_url),
            tools=tools,
            capabilities=card.get("capabilities", {}),
        )

    async def list_tools(self) -> list[A2AToolDef]:
        """列出远程 Agent 的工具"""
        if not self._agent_card:
            await self.discover()
        tools = []
        for t in (self._agent_card or {}).get("tools", []):
            tools.append(A2AToolDef(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("parameters", {})),
            ))
        return tools

    async def send_task(self, query: str, stream: bool = True) -> dict:
        """发送任务到远程 Agent（同步）"""
        r = await self.http.post(
            f"{self.base_url}/tasks",
            json={"query": query, "stream": stream},
        )
        r.raise_for_status()
        return r.json()

    async def stream_task(self, query: str) -> AsyncIterator[dict]:
        """发送任务到远程 Agent（流式 SSE）"""
        async with self.http.stream(
            "POST",
            f"{self.base_url}/tasks",
            json={"query": query, "stream": True},
        ) as resp:
            current_event = None
            current_data = ""

            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    current_data = line[6:]
                elif line == "" and current_event:
                    try:
                        yield {"event": current_event, "data": json.loads(current_data)}
                    except json.JSONDecodeError:
                        pass
                    current_event = None
                    current_data = ""

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        """调用远程 Agent 的指定工具"""
        r = await self.http.post(
            f"{self.base_url}/tools/{tool_name}",
            json=args,
        )
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self.http.aclose()
