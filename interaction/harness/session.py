"""
CC-Harness Agent 会话管理模块
==============================
Session 是 CC 路线最核心的抽象，替代 LangGraph 的 Checkpoint + 共享 State。

核心职责：
1. 管理 Agent 对话上下文（消息历史）
2. 上下文字段压缩（token 超限时自动摘要历史）
3. 工具调用结果写入上下文
4. 子 Agent 返回摘要写入上下文
5. 会话暂停/恢复/终止

设计对比：
  LangGraph: Checkpoint 系统把整个 State 序列化到数据库
             State 是所有节点共享的全局字典
  CC路线:    Session 对象封装所有状态
             每个 Session 独立，天然支持多租户隔离
             上下文压缩是内置能力，不需要开发者自己实现
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from harness.types import AgentConfig, Message, MessageRole


class Session:
    """
    Agent 会话对象

    生命周期：
    1. 创建 → 设置系统提示词
    2. 用户发消息 → append_user_message
    3. Agent 调用工具 → append_tool_result
    4. Agent 返回答案 → append_assistant_message
    5. 上下文超限 → compress() 自动摘要
    6. 会话结束 → 可序列化存入 Redis

    使用示例：
        session = Session(config)
        session.set_system_prompt("你是知识库助手...")
        session.append_user_message("什么是Agentic RAG？")
        # AgentLoop 会在每轮循环前检查是否需要压缩上下文
    """

    def __init__(self, config: AgentConfig | None = None):
        self.session_id: str = str(uuid.uuid4())[:8]
        self.config: AgentConfig = config or AgentConfig()
        self.messages: list[Message] = []
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.total_turns: int = 0           # 累计循环轮次
        self.tool_calls_count: int = 0       # 累计工具调用次数
        self.subagents_spawned: int = 0     # 累计派生子Agent次数
        self._aborted: bool = False          # 外部中断标记
        self._compression_count: int = 0     # 累计压缩次数
        self._system_prompt: str = ""

    # ==================== 消息管理 ====================

    def set_system_prompt(self, prompt: str):
        """设置系统提示词（一般只在会话开始时调用一次）"""
        self._system_prompt = prompt

    def append_user_message(self, content: str):
        """追加用户消息"""
        self.messages.append(Message(role=MessageRole.USER, content=content))
        self.last_active = time.time()

    def append_assistant_message(self, content: str):
        """追加 AI 助手消息（非工具调用的最终回答）"""
        self.messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        self.last_active = time.time()

    def append_tool_result(self, tool_name: str, result: str, tool_call_id: str = "", success: bool = True):
        """
        追加工具调用结果

        CC 设计：工具结果以 TOOL 角色注入上下文。
        与 LangChain ToolMessage 的区别：
        - LangChain: ToolMessage 是独立类型，与 AIMessage/HumanMessage 不同处理
        - CC路线:    统一 Message 模型，role 区分类型，序列化和处理更统一

        Args:
            tool_name: 工具名称
            result: 工具返回的文本内容
            tool_call_id: 调用 ID（关联请求与响应）
            success: 是否成功（失败的工具结果也会被注入，让 LLM 知道失败了）
        """
        prefix = f"[工具 {tool_name} 执行{'成功' if success else '失败'}]\n"
        self.messages.append(Message(
            role=MessageRole.TOOL,
            content=prefix + result,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        ))
        self.tool_calls_count += 1
        self.last_active = time.time()

    def append_subagent_result(self, task: str, summary: str):
        """
        追加子 Agent 返回的摘要

        与工具结果的区别：
        - 工具结果：完整的、结构化的返回值
        - 子Agent摘要：压缩后的关键信息，不保留子Agent的完整对话历史
                      这是 CC 的核心设计：子Agent 上下文不污染主Agent
        """
        self.messages.append(Message(
            role=MessageRole.SUBAGENT,
            content=f"[子Agent完成任务]\n任务：{task}\n结果摘要：{summary}",
        ))
        self.subagents_spawned += 1
        self.last_active = time.time()

    def append_thought(self, thought: str):
        """
        追加 Agent 思考过程（可选，用于调试和透明度）

        不作为独立消息存入 messages，而是追加到最后一条 assistant 消息中。
        如果前端需要展示思考过程，可以在 WebSocket 推送时拆分。
        """
        # 简单实现：作为 assistant 消息的 reasoning_content 前缀
        pass

    # ==================== 上下文压缩 ====================

    def estimate_total_tokens(self) -> int:
        """估算当前上下文总 token 数"""
        system_tokens = int(len(self._system_prompt) / 1.5)  # 中文约 1.5 字符/token
        message_tokens = sum(m.estimate_tokens() for m in self.messages)
        return system_tokens + message_tokens

    def should_compress(self) -> bool:
        """判断是否需要压缩上下文（使用 ContextEngine）"""
        # 简单判断：token 超限
        if self.estimate_total_tokens() > self.config.max_context_tokens:
            return True
        # 或者消息数过多
        if len(self.messages) > 20:
            return True
        return False

    async def compress_async(self, llm_adapter=None) -> str:
        """
        上下文压缩 (v2)：使用 ContextEngine 做 LLM 驱动的分层摘要。

        与旧版 compress() 的区别：
        - 旧版：规则截断，丢失细节
        - v2：LLM 分层摘要，早期→深度摘要，中期→轻量摘要，近期→保留

        Args:
            llm_adapter: LLM 适配器（可选，不传则降级为规则压缩）

        Returns:
            压缩描述文本
        """
        from harness.context_engine import ContextEngine, TokenBudget

        engine = ContextEngine(
            budget=TokenBudget.for_model(self.config.model),
            recent_window=4,
        )

        new_messages, result = await engine.compress(
            self.messages,
            llm_adapter=llm_adapter,
        )

        if result.compressed:
            self.messages = new_messages
            self._compression_count += 1
            return (
                f"压缩完成: {result.original_tokens} → {result.final_tokens} tokens "
                f"(-{result.reduction_ratio}%), {result.summaries_generated} 条摘要, "
                f"耗时 {result.duration_ms:.0f}ms"
            )
        else:
            return f"无需压缩 (当前 {result.original_tokens} tokens, 窗口 {engine.budget.total})"

    def compress(self) -> str:
        """
        上下文压缩（同步兼容接口，降级为规则压缩）

        生产环境应使用 compress_async() 获得 LLM 驱动的摘要质量。
        此方法保留向后兼容，在无法调用 LLM 时使用规则压缩。
        """
        if len(self.messages) <= 6:
            return "消息数不足6条，无需压缩"

        from harness.context_engine import ContextEngine, TokenBudget

        engine = ContextEngine(
            budget=TokenBudget.for_model(self.config.model),
            recent_window=4,
        )

        # 同步调用异步方法（简化处理）
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # 有运行中的事件循环，不能直接用 asyncio.run
            # 降级为规则压缩
            return self._rule_compress()
        except RuntimeError:
            # 无运行中的事件循环
            return self._rule_compress()

    def _rule_compress(self) -> str:
        """规则压缩（无 LLM 依赖的降级方案）"""
        keep_recent = 4
        old_messages = self.messages[:-keep_recent]
        recent_messages = self.messages[-keep_recent:]

        old_content = " | ".join([
            m.content[:100] for m in old_messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ])

        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[对话历史摘要 - 第{self._compression_count + 1}次压缩] {old_content[:500]}",
        )

        self.messages = [summary_msg] + recent_messages
        self._compression_count += 1

        return f"规则压缩完成：{len(old_messages)}条早期消息 → 1条摘要，保留{len(recent_messages)}条近期消息"

    # ==================== 生命周期 ====================

    def can_continue(self) -> bool:
        """
        判断 AgentLoop 是否可以继续循环

        终止条件：
        1. 超过最大轮次 → 防止死循环
        2. 外部中断信号 → WebSocket 断开时调用 abort()
        3. 工具调用次数超限 → 防止 tool_call 死循环
        """
        if self._aborted:
            return False
        if self.total_turns >= self.config.max_turns:
            return False
        if self.tool_calls_count >= self.config.max_turns * 2:
            return False
        return True

    def abort(self):
        """外部中断（用户关闭网页、取消请求等）"""
        self._aborted = True

    def reset_abort(self):
        """重置中断标记（下次对话重新开始）"""
        self._aborted = False

    def increment_turn(self):
        """增加循环计数（AgentLoop 每轮调用一次）"""
        self.total_turns += 1

    # ==================== Prompt 组装 ====================

    def build_messages(self) -> list[Message]:
        """
        组装发送给 LLM 的完整消息列表

        消息顺序：
        1. System prompt（Agent 角色定义 + 工具说明 + 格式约束）
        2. 已压缩的历史摘要（如果有）
        3. 对话历史（含工具调用结果）
        """
        result: list[Message] = []

        # 1. System prompt
        if self._system_prompt:
            result.append(Message(role=MessageRole.SYSTEM, content=self._system_prompt))

        # 2. 对话历史（已包含压缩摘要和工具结果）
        result.extend(self.messages)

        return result
