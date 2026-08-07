"""
CC-Harness Agent 核心类型定义
==============================
基于 Pydantic v2 定义所有数据结构。

设计决策（vs LangGraph）：
- 不使用 TypedDict + Annotated[operator.add]，改用 Pydantic BaseModel
- 所有状态统一封装在 Session 对象中，不散落在共享 State 字典
- 结构化输出用 Pydantic 模型约束，避免裸字典

对比 LangGraph AgentState：
  LangGraph: TypedDict + Annotated[Sequence[BaseMessage], operator.add]
  CC路线:   Pydantic BaseModel + 显式方法管理消息列表

优势：类型安全、IDE 自动补全、序列化/反序列化更方便（存入 Redis）
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ==================== 消息模型 ====================

class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"              # 工具调用结果
    SUBAGENT = "subagent"       # 子 Agent 返回的摘要（区别于 tool）


class Message(BaseModel):
    """
    统一消息模型（替代 LangChain 的 HumanMessage / AIMessage / ToolMessage）

    设计意图：
    - LangChain 为每种消息类型定义了独立类，继承链复杂
    - CC 路线用单一 Message 类 + role 枚举区分，简洁且方便序列化
    """
    role: MessageRole
    content: str
    tool_name: Optional[str] = None       # TOOL/SUBAGENT 消息附带的工具名
    tool_call_id: Optional[str] = None    # 工具调用唯一 ID（关联请求和响应）
    timestamp: float = Field(default_factory=time.time)
    token_count: Optional[int] = None     # 该消息的 token 估算（用于上下文压缩）

    def to_openai_format(self) -> dict[str, Any]:
        """
        转换为 OpenAI Chat Completions API 格式

        这是 LLM 适配层的标准输入格式。
        TOOL 角色消息会被转为 OpenAI 的 tool 消息格式。
        SUBAGENT 消息会被转为 user 消息（作为上下文注入）。
        """
        role_map = {
            MessageRole.SYSTEM: "system",
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.TOOL: "user",      # 不用 OpenAI 原生 tool role（我们不使用 function calling 协议），
                                           # 作为 user 消息注入让模型自然理解工具结果
            MessageRole.SUBAGENT: "user",  # 子Agent摘要作为user上下文注入
        }
        msg: dict[str, Any] = {"role": role_map[self.role], "content": self.content}

        # 不使用原生 tool role，不需要 tool_call_id

        return msg

    def estimate_tokens(self) -> int:
        """
        粗略 token 估算（中文约 1.5 字符/token，英文约 4 字符/token）
        精确计数需要 tiktoken，但为保持零依赖（避免 Azure 下载超时），使用估算
        """
        text = self.content
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


# ==================== Agent 动作模型（结构化输出） ====================

class ActionType(str, Enum):
    """
    Agent 可执行的动作类型

    核心设计：LLM 输出不是自由文本，而是被约束为以下三种动作之一。
    这替代了 LangGraph 的节点 + 条件边的概念：
      LangGraph:  节点函数 → 返回状态更新 → 条件边决定下一节点
      CC路线:     LLM 输出结构化动作 → AgentLoop 根据动作类型调度
    """
    FINAL_ANSWER = "final_answer"     # 直接回答用户，结束本轮
    TOOL_CALL = "tool_call"           # 调用一个工具
    SPAWN_SUBAGENT = "spawn_subagent" # 派生子 Agent 处理子任务


class ToolCall(BaseModel):
    """
    工具调用请求（Pydantic 模型约束 LLM 输出的 JSON Schema）

    CC 设计要点：一次只调用一个工具（不是 OpenAI 的 parallel tool calls）。
    理由：
    - 简化 AgentLoop 逻辑
    - 每个工具结果可以立即注入上下文，影响下一步决策
    - 避免并行工具调用结果之间互相干扰
    """
    tool_name: str = Field(description="要调用的工具名称，必须是已注册的工具")
    args: dict[str, Any] = Field(default_factory=dict, description="工具参数")


class SubagentTask(BaseModel):
    """
    子 Agent 任务定义

    与 tool_call 的区别：
    - tool_call：执行单一功能，结果直接返回
    - spawn_subagent：启动独立 AgentLoop，拥有独立上下文，返回摘要
    """
    task_description: str = Field(description="子Agent要完成的任务描述")
    subagent_type: str = Field(default="general", description="子Agent类型：general/rag/code")
    max_turns: int = Field(default=5, description="子Agent最大循环轮次")


class AgentAction(BaseModel):
    """
    Agent 动作（LLM 结构化输出的顶层模型）

    这是 AgentLoop 解析 LLM 响应的标准格式。
    LLM 被要求输出符合此 Schema 的 JSON。

    对比：
    - LangGraph: 节点内手动解析 LLM 输出 → 返回 dict → 条件边
    - CC路线:   LLM 直接输出结构化 JSON → AgentLoop 统一解析 → switch-case 调度
    """
    action_type: ActionType = Field(description="动作类型")
    thought: str = Field(default="", description="思考过程（ReAct 的 Thought 步骤）")
    tool_call: Optional[ToolCall] = Field(default=None, description="工具调用（action_type=tool_call 时必填）")
    subagent_task: Optional[SubagentTask] = Field(default=None, description="子Agent任务（action_type=spawn_subagent 时必填）")
    answer: Optional[str] = Field(default=None, description="最终回答（action_type=final_answer 时必填）")

    @field_validator("action_type")
    @classmethod
    def check_action_consistency(cls, v: ActionType, info: Any) -> ActionType:
        """运行时校验：动作类型与对应字段的一致性"""
        return v


# ==================== 工具系统类型 ====================

class ToolResult(BaseModel):
    """
    工具执行结果

    设计要点：
    - success: 工具是否执行成功（失败不会中断 AgentLoop，而是注入失败信息继续）
    - content: 工具返回的文本内容，会被注入到 Agent 上下文
    - metadata: 附加信息（如检索耗时、文档数量），供调试和监控
    """
    tool_name: str
    success: bool
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ==================== Agent 配置 ====================

class AgentConfig(BaseModel):
    """
    Agent 运行时配置

    替代 LangGraph 的 .compile() 时传入的 config。
    所有配置集中在一个对象中，便于序列化和跨进程传递。
    """
    max_turns: int = Field(default=15, description="最大循环轮次（防止死循环）")
    model: str = Field(default="qwen3.5-flash", description="LLM 模型名")
    temperature: float = Field(default=0.0, description="生成温度（Agent 任务建议 0）")
    max_context_tokens: int = Field(default=8000, description="上下文窗口 token 上限（触发压缩）")
    system_prompt: str = Field(default="", description="自定义系统提示词")
    enable_subagents: bool = Field(default=True, description="是否允许派生子Agent")
    enable_tool_reuse: bool = Field(default=False, description="同一轮内是否允许重复调用同一工具")
    abort_signal: Optional[Any] = Field(default=None, description="外部中断信号（WebSocket断开时设置）")
