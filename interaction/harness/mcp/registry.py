"""
MCP Server Registry — 多 MCP Server 管理

场景:
    一个 Agent 可能需要同时连接多个 MCP Server:
    - GitHub MCP Server: 提供 github_search, github_create_issue 等工具
    - 数据库 MCP Server: 提供 db_query, db_schema 等工具
    - 企业内部 MCP Server: 提供 jira_search, wiki_read 等工具

    Server Registry 统一管理所有 MCP Server 的连接、工具发现和调用路由。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from harness.mcp.client import MCPClient, MCPToolDef


@dataclass
class MCPServerConfig:
    """MCP Server 连接配置"""
    name: str                             # 唯一名称（如 "github", "database"）
    description: str = ""
    transport: str = "stdio"              # stdio / http
    command: str = ""                     # stdio 模式：可执行文件路径
    args: list[str] = field(default_factory=list)   # stdio 模式：命令行参数
    env: dict[str, str] = field(default_factory=dict)  # stdio 模式：环境变量
    url: str = ""                         # HTTP 模式：MCP Server URL
    auto_connect: bool = True             # 启动时自动连接
    tool_prefix: str = ""                 # 工具名前缀（避免不同 Server 的工具名冲突）


@dataclass
class MCPServerState:
    """MCP Server 运行时状态"""
    config: MCPServerConfig
    client: MCPClient | None = None
    connected: bool = False
    tools: list[MCPToolDef] = field(default_factory=list)
    error: str = ""


class MCPServerRegistry:
    """
    MCP Server 注册中心

    使用方式:
        registry = MCPServerRegistry()

        # 注册 GitHub MCP Server
        registry.register(MCPServerConfig(
            name="github",
            transport="stdio",
            command="npx",
            args=["-y", "@anthropic/mcp-server-github"],
            env={"GITHUB_TOKEN": "ghp_xxx"},
            tool_prefix="github_",
        ))

        # 注册内部数据库 MCP Server
        registry.register(MCPServerConfig(
            name="internal_db",
            transport="http",
            url="http://db-server:3001/mcp",
        ))

        # 连接所有 Server
        await registry.connect_all()

        # 获取所有 MCP 工具
        all_tools = registry.get_all_tools()
        # → [MCPToolDef(name="github_search", ...), MCPToolDef(name="db_query", ...)]

        # 调用工具
        result = await registry.call_tool("github_search", {"query": "bug"})
    """

    def __init__(self):
        self._servers: dict[str, MCPServerState] = {}

    # ==================== 注册 ====================

    def register(self, config: MCPServerConfig):
        """注册 MCP Server"""
        if config.name in self._servers:
            raise ValueError(f"MCP Server '{config.name}' 已注册")

        self._servers[config.name] = MCPServerState(config=config)
        print(f"[MCP Registry] 注册 Server: {config.name} (transport={config.transport})")

    def unregister(self, name: str):
        """注销 MCP Server"""
        self._servers.pop(name, None)

    # ==================== 连接管理 ====================

    async def connect_all(self):
        """连接所有 Server（并行）"""
        tasks = [
            self._connect_one(name, state)
            for name, state in self._servers.items()
            if state.config.auto_connect
        ]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (name, _), result in zip(self._servers.items(), results):
                if isinstance(result, Exception):
                    self._servers[name].error = str(result)

    async def connect_one(self, name: str):
        """连接单个 Server"""
        if name not in self._servers:
            raise ValueError(f"MCP Server '{name}' 未注册")
        await self._connect_one(name, self._servers[name])

    async def _connect_one(self, name: str, state: MCPServerState):
        """内部连接逻辑"""
        config = state.config

        try:
            # 创建 Client
            if config.transport == "stdio":
                client = MCPClient.stdio(config.command, config.args, config.env)
            elif config.transport == "http":
                client = MCPClient.http(config.url)
            else:
                raise ValueError(f"不支持的传输方式: {config.transport}")

            # 连接 + 初始化握手
            await client.connect()
            state.client = client

            # 发现工具
            tools = await client.list_tools()

            # 应用工具名前缀
            if config.tool_prefix:
                for t in tools:
                    t.name = f"{config.tool_prefix}{t.name}"

            state.tools = tools
            state.connected = True
            state.error = ""

            print(f"[MCP Registry] ✅ {name}: 已连接, 发现 {len(tools)} 个工具")

        except Exception as e:
            state.connected = False
            state.error = str(e)
            print(f"[MCP Registry] ❌ {name}: 连接失败 - {e}")

    async def disconnect_all(self):
        """断开所有 Server"""
        for state in self._servers.values():
            if state.client:
                await state.client.disconnect()
                state.connected = False

    # ==================== 工具发现 ====================

    def get_all_tools(self) -> list[MCPToolDef]:
        """获取所有已连接 Server 的工具"""
        tools = []
        for state in self._servers.values():
            if state.connected:
                tools.extend(state.tools)
        return tools

    def get_server_tools(self, server_name: str) -> list[MCPToolDef]:
        """获取指定 Server 的工具"""
        if server_name in self._servers:
            return self._servers[server_name].tools
        return []

    def find_tool(self, tool_name: str) -> tuple[str, MCPToolDef] | None:
        """查找工具所在的 Server"""
        for name, state in self._servers.items():
            if state.connected:
                for tool in state.tools:
                    if tool.name == tool_name:
                        return (name, tool)
        return None

    # ==================== 工具调用 ====================

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        调用 MCP 工具（自动路由到正确的 Server）

        Returns:
            {"success": bool, "content": str, "error": str}
        """
        found = self.find_tool(tool_name)
        if not found:
            return {"success": False, "content": "", "error": f"工具 '{tool_name}' 未在任何 MCP Server 中找到"}

        server_name, tool_def = found
        state = self._servers[server_name]

        if not state.client:
            return {"success": False, "content": "", "error": f"Server '{server_name}' 未连接"}

        try:
            result = await state.client.call_tool(tool_name, arguments)

            # 提取文本内容
            text_parts = []
            for item in result.content:
                if item.get("type") == "text":
                    text_parts.append(item["text"])

            return {
                "success": not result.is_error,
                "content": "\n".join(text_parts),
                "error": "MCP tool returned error" if result.is_error else "",
            }
        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}

    # ==================== 状态查询 ====================

    def status(self) -> dict:
        """获取所有 Server 的状态"""
        return {
            name: {
                "connected": state.connected,
                "tools_count": len(state.tools),
                "error": state.error,
                "tools": [t.name for t in state.tools],
            }
            for name, state in self._servers.items()
        }

    @property
    def connected_count(self) -> int:
        return sum(1 for s in self._servers.values() if s.connected)

    @property
    def total_tools(self) -> int:
        return sum(len(s.tools) for s in self._servers.values() if s.connected)
