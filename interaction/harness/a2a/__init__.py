"""
CC-Harness Agent — A2A 协议集成 (Agent-to-Agent)
==================================================
Google A2A 开放协议 — 让不同框架的 Agent 可以互相通信。

协议要点:
    - Agent Card: 每个 Agent 暴露自己的能力描述 (端点 + 工具)
    - Task: 通过 POST /tasks 发送任务，SSE 流式返回结果
    - 框架无关: LangGraph/CrewAI/CC-Harness 都可以通过 A2A 互操作

使用场景:
    CC-Harness (主控)  →  A2A  →  DevPilot (LangGraph) 的爬虫工具
                                   CC-Harness 的评测结果可以返回给 DevPilot 展示
"""

from harness.a2a.client import A2AClient, RemoteAgent, A2AToolDef
from harness.a2a.server import A2AServer, create_a2a_card

__all__ = ["A2AClient", "RemoteAgent", "A2AToolDef", "A2AServer", "create_a2a_card"]
