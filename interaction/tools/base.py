"""
CC-Harness Agent 工具系统 - 基类定义
=====================================
所有工具必须继承 BaseTool 并实现 execute 方法。

设计对比：
  LangChain: @tool 装饰器 + BaseTool 抽象类，参数通过 pydantic Field 定义
  CC路线:    BaseTool 抽象类，参数用字典描述（更灵活，不绑定 Pydantic Schema）

工具是 Agent 的能力单元，有以下约束：
1. 每个工具有唯一名称（name），Agent 通过名称调用
2. execute 返回 ToolResult，失败不抛异常，而是 result.success=False
3. 工具可以是同步的（会在 asyncio.to_thread 中执行）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from harness.types import ToolResult


class BaseTool(ABC):
    """
    工具基类

    子类需要实现：
    - name: 工具名称（类属性）
    - description: 工具描述（类属性）
    - parameters: 参数说明字典（类属性）
    - execute(**kwargs) -> ToolResult: 工具执行逻辑
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, str] = {}

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        执行工具

        Args:
            **kwargs: 工具参数（由 LLM 的输出 JSON 解析而来）

        Returns:
            ToolResult: 执行结果（始终返回，不抛异常）

        实现要求：
        - 必须 catch 所有异常，转为 ToolResult(success=False, error=str(e))
        - 不能抛出未捕获的异常（会中断 AgentLoop）
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于注册表和调试）"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
