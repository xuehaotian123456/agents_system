"""
MCP Tool Adapter — 将 MCP 工具无缝接入 CC-Harness ToolRegistry

核心能力：
    把 MCP Tool 自动包装为 CC-Harness 的 BaseTool，
    使其与本地工具在同一个 ToolRegistry 中统一管理和调用。

使用方式:
    # 从 MCP Server Registry 自动注册所有工具
    mcp_registry = MCPServerRegistry()
    mcp_registry.register(MCPServerConfig(...))
    await mcp_registry.connect_all()

    tool_registry = ToolRegistry()
    await register_mcp_tools(tool_registry, mcp_registry)

    # 现在 Agent 可以像调用本地工具一样调用 MCP 工具
    result = await tool_registry.execute("github_search", {"query": "bug"})
"""

from __future__ import annotations

from typing import Any

from harness.mcp.client import MCPToolDef
from harness.mcp.registry import MCPServerRegistry
from harness.types import ToolResult
from tools.base import BaseTool


class MCPToolAdapter(BaseTool):
    """
    MCP 工具适配器

    将 MCPToolDef 包装为 CC-Harness 的 BaseTool，
    可注册到 ToolRegistry 中与本地工具统一调用。

    使用方式:
        mcp_tool_def = MCPToolDef(name="github_search", description="...", input_schema={...})
        adapted = MCPToolAdapter(mcp_tool_def, mcp_registry)
        tool_registry.register(adapted)

        # 调用
        result = await adapted.execute(query="bug in main.py")
    """

    def __init__(self, tool_def: MCPToolDef, mcp_registry: MCPServerRegistry):
        self._tool_def = tool_def
        self._mcp_registry = mcp_registry

    @property
    def name(self) -> str:
        return self._tool_def.name

    @property
    def description(self) -> str:
        return self._tool_def.description

    @property
    def parameters(self) -> dict[str, str]:
        """从 JSON Schema 提取参数描述"""
        schema = self._tool_def.input_schema
        properties = schema.get("properties", {})
        return {
            name: prop.get("description", f"参数: {name}")
            for name, prop in properties.items()
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        执行 MCP 工具调用

        将 CC-Harness 的 execute(**kwargs) 转为 MCP 的 call_tool(name, arguments)
        """
        try:
            result = await self._mcp_registry.call_tool(self.name, kwargs)

            if result["success"]:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    content=result["content"],
                    metadata={"source": "mcp"},
                )
            else:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    content="",
                    error=result["error"],
                )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="",
                error=f"MCP工具调用异常: {type(e).__name__}: {e}",
            )


async def register_mcp_tools(
    tool_registry,
    mcp_registry: MCPServerRegistry,
    prefix: str = "",
) -> int:
    """
    批量将 MCP 工具注册到 CC-Harness ToolRegistry

    Args:
        tool_registry: CC-Harness 工具注册表
        mcp_registry: MCP Server 注册中心
        prefix: 工具名前缀过滤（只注册匹配前缀的工具，空字符串=全部）

    Returns:
        注册的工具数量

    Example:
        tool_registry = ToolRegistry()
        count = await register_mcp_tools(tool_registry, mcp_registry)
        print(f"已注册 {count} 个 MCP 工具")
        # Agent 现在可以使用所有 MCP 工具了
    """
    count = 0
    all_tools = mcp_registry.get_all_tools()

    for tool_def in all_tools:
        if prefix and not tool_def.name.startswith(prefix):
            continue

        adapted = MCPToolAdapter(tool_def, mcp_registry)
        tool_registry.register(adapted)
        count += 1

    return count
