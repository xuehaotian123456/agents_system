"""
CC-Harness Agent 工具注册表
===========================
管理所有已注册的工具，提供查询、执行、动态筛选功能。

CC 的设计要点：
- 工具注册是动态的：不同 Session 可以有不同工具集
- 工具筛选：根据上下文智能决定展示哪些工具给 LLM
              ── 避免把所有工具全部塞进 prompt（减少 token 消耗、提高选择准确率）
"""

from __future__ import annotations

from typing import Any, Optional

from harness.types import ToolResult
from tools.base import BaseTool


class ToolRegistry:
    """
    工具注册表

    职责：
    1. 注册/注销工具
    2. 按名称查询工具
    3. 执行工具调用
    4. 列出所有可用工具
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """
        注册工具

        Args:
            tool: 工具实例

        Raises:
            ValueError: 同名工具已存在
        """
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已注册，不能重复注册")
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        """注销工具"""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[BaseTool]:
        """按名称获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """列出所有已注册的工具"""
        return list(self._tools.values())

    def list_tool_names(self) -> list[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    async def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """
        执行工具调用

        这是 AgentLoop 调用工具的入口。

        Args:
            tool_name: 工具名称
            args: 参数字典

        Returns:
            ToolResult: 执行结果

        错误处理：
        - 工具不存在 → ToolResult(success=False, error="工具未注册")
        - 工具执行失败 → ToolResult(success=False, error=str(e))
        - 成功 → ToolResult(success=True, content=结果文本)
        """
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                error=f"工具 '{tool_name}' 未注册。可用工具：{', '.join(self.list_tool_names())}",
            )

        try:
            result = await tool.execute(**args)
            return result
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                error=f"工具执行异常：{type(e).__name__}: {e}",
            )

    def filter_for_context(self, query: str = "", max_tools: int = 8) -> list[BaseTool]:
        """
        根据上下文筛选工具（CC 特色：动态工具选择）

        当前实现为简单返回所有工具，生产环境可根据 query 语义匹配相关工具。

        Args:
            query: 用户查询（用于语义匹配）
            max_tools: 最大返回工具数

        Returns:
            筛选后的工具列表
        """
        tools = self.list_tools()
        if len(tools) <= max_tools:
            return tools
        return tools[:max_tools]
