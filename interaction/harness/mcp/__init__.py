"""
CC-Harness Agent — MCP 协议集成 (Model Context Protocol)
=========================================================
Anthropic MCP (Model Context Protocol) 标准协议集成。

MCP 是连接 AI 模型与外部工具/数据源的开放标准协议。
通过 MCP，Agent 可以无缝对接任何实现了 MCP 协议的服务。

核心组件：
- MCPClient: JSON-RPC 客户端，连接 MCP Server
- MCPServerRegistry: 管理多个 MCP Server 连接
- MCPToolAdapter: 将 MCP Tool 包装为 CC-Harness BaseTool

协议参考: https://spec.modelcontextprotocol.io/
"""

from harness.mcp.client import MCPClient, MCPToolDef, MCPResourceDef
from harness.mcp.registry import MCPServerRegistry, MCPServerConfig
from harness.mcp.adapter import MCPToolAdapter, register_mcp_tools

__all__ = [
    "MCPClient",
    "MCPToolDef",
    "MCPResourceDef",
    "MCPServerRegistry",
    "MCPServerConfig",
    "MCPToolAdapter",
    "register_mcp_tools",
]
