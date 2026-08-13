"""
CC-Harness Agent - 核心 AgentLoop 引擎 (v2)
============================================
这是整个框架的心脏，替代 LangGraph 的 StateGraph + Node + Edge 体系。

v2 新增:
- AgentTracer 集成 — 结构化追踪替代 print，每一步决策可回放
- 流式回调 — on_event 实时推送，前端逐阶段展示
- 上下文压缩升级 — 委托给 ContextEngine (LLM 驱动分层摘要)
- 详细统计 — token 消耗、延迟分布、工具成功率

设计哲学（CC 路线）：
  LangGraph:   预定义图结构（节点 + 条件边）→ 编译 → invoke
              流程在开发时确定，运行时沿图遍历
  CC AgentLoop: 异步 while 循环
               每轮：思考(LLM) → 行动(工具/子Agent) → 观察(结果写入上下文) → 反思(下一轮)
               流程由 LLM 在运行时动态决策

ReAct 循环（Reasoning + Acting）：
  1. Thought: LLM 分析当前状态，决定下一步
  2. Action: 执行工具调用 OR 输出最终答案 OR 派生子Agent
  3. Observation: 将结果写入会话上下文
  4. Loop: 回到 Thought，直到输出最终答案
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from harness.types import (
    ActionType,
    AgentAction,
    AgentConfig,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)
from harness.session import Session
from harness.llm_adapter import LLMAdapter
from harness.prompt_engine import PromptEngine
from harness.subagent import SubagentSpawner, SubagentResult
from harness.tracer import AgentTracer, TraceEventType
from tools.registry import ToolRegistry


class AgentLoop:
    """
    Agent 主循环引擎 (v2)

    使用示例：
        session = Session(config)
        session.set_system_prompt(prompt_engine.build_system_prompt(config))
        session.append_user_message("什么是Agentic RAG？")

        tracer = AgentTracer(verbose=True)

        loop = AgentLoop(session, llm_adapter, tool_registry, prompt_engine, tracer=tracer)
        answer = await loop.run()

        # 获取完整轨迹
        print(tracer.to_json())
    """

    def __init__(
        self,
        session: Session,
        llm_adapter: LLMAdapter,
        tool_registry: ToolRegistry,
        prompt_engine: PromptEngine,
        tracer: AgentTracer | None = None,
    ):
        self.session = session
        self.llm = llm_adapter
        self.tool_registry = tool_registry
        self.prompt_engine = prompt_engine
        self.tracer = tracer or AgentTracer(verbose=True)

        self.subagent_spawner = SubagentSpawner(
            tool_registry=tool_registry,
            llm_adapter=llm_adapter,
            prompt_engine=prompt_engine,
        )

        # 运行时统计
        self.start_time: float = 0
        self.end_time: float = 0

    async def run(self) -> str:
        """
        启动 AgentLoop

        Returns:
            Agent 最终回答文本

        循环流程（每轮）：
        1. 检查是否可以继续（最大轮次/中断信号）
        2. 检查是否需要压缩上下文
        3. 组装 prompt → 调用 LLM
        4. 解析 LLM 输出为 AgentAction
        5. 根据 action_type 执行对应操作
        6. 如果是 final_answer → 退出循环
        """
        self.start_time = time.time()

        # 确保 system prompt 已设置
        if not self.session._system_prompt:
            self.session.set_system_prompt(
                self.prompt_engine.build_system_prompt(self.session.config)
            )

        # ── 会话开始 ──
        self.tracer.session_start(
            session_id=self.session.session_id,
            model=self.llm.model,
            max_turns=self.session.config.max_turns,
        )

        # ============ 主循环 ============
        while self.session.can_continue():
            self.session.increment_turn()
            turn = self.session.total_turns

            # token 估算
            token_estimate = self.session.estimate_total_tokens()
            self.tracer.turn_start(turn, token_estimate=token_estimate)

            # ---- Step 1: 上下文压缩检查 ----
            if self.session.should_compress():
                result = await self.session.compress_async(llm_adapter=self.llm)
                new_estimate = self.session.estimate_total_tokens()
                self.tracer.context_compress(
                    before_tokens=token_estimate,
                    after_tokens=new_estimate,
                    reduction_ratio=round((1 - new_estimate / max(token_estimate, 1)) * 100, 1),
                )

            # ---- Step 2: 组装消息并调用 LLM ----
            messages = self.session.build_messages()

            try:
                action = await self.llm.generate_structured(messages, AgentAction)
            except Exception as e:
                self.tracer.error(f"LLM调用失败: {e}", exception=str(e))
                return f"抱歉，AI 服务暂时不可用。错误：{e}"

            # 记录 LLM 思考
            self.tracer.llm_thinking(action.thought)
            self.tracer.llm_action(
                action_type=action.action_type.value,
                thought=action.thought,
                details=self._action_details(action),
            )

            # ---- Step 3: 执行动作 ----
            if action.action_type == ActionType.FINAL_ANSWER:
                answer = action.answer or "（未生成回答）"
                self.session.append_assistant_message(answer)
                self.end_time = time.time()

                self.tracer.final_answer(answer)
                self.tracer.session_end(answer=answer, total_turns=turn)
                return answer

            elif action.action_type == ActionType.TOOL_CALL:
                result_text, tool_success = await self._execute_tool_call(action.tool_call)
                self.session.append_tool_result(
                    tool_name=action.tool_call.tool_name if action.tool_call else "unknown",
                    result=result_text if result_text else "工具返回空结果",
                    tool_call_id=str(turn),
                    success=tool_success,
                )

            elif action.action_type == ActionType.SPAWN_SUBAGENT:
                if not self.session.config.enable_subagents:
                    self.tracer.error("子Agent功能已被禁用", recoverable=True)
                    self.session.append_tool_result(
                        tool_name="spawn_subagent",
                        result="子Agent功能已被禁用",
                        success=False,
                    )
                elif action.subagent_task:
                    result = await self._execute_subagent(action.subagent_task)
                    self.session.append_subagent_result(
                        task=action.subagent_task.task_description,
                        summary=result.summary,
                    )
                else:
                    self.tracer.error("子Agent任务定义不完整", recoverable=True)
                    self.session.append_tool_result(
                        tool_name="spawn_subagent",
                        result="子Agent任务定义不完整",
                        success=False,
                    )

        # ---- 循环超限 ----
        self.end_time = time.time()
        fallback = "抱歉，处理您的问题时超出了最大推理步数。请尝试简化问题或换个方式提问。"
        self.session.append_assistant_message(fallback)

        self.tracer.error("超出最大推理步数", recoverable=False)
        self.tracer.session_end(answer=fallback, total_turns=self.session.total_turns)
        return fallback

    # ==================== 工具调用执行 ====================

    async def _execute_tool_call(self, tool_call: Optional[ToolCall]) -> tuple[str, bool]:
        """
        执行工具调用。

        Returns:
            (结果文本, 是否成功) — 失败状态必须真实传递给会话记录
        """
        if tool_call is None:
            self.tracer.error("工具调用请求为空", recoverable=True)
            return "工具调用请求为空", False

        tool_name = tool_call.tool_name
        args = tool_call.args

        self.tracer.tool_call(tool_name, args)

        t_start = time.time()
        result: ToolResult = await self.tool_registry.execute(tool_name, args)
        latency_ms = (time.time() - t_start) * 1000

        self.tracer.tool_result(
            tool_name=tool_name,
            result_preview=result.content[:300] if result.success else f"失败: {result.error}",
            success=result.success,
            latency_ms=latency_ms,
        )

        return (result.content, True) if result.success else (f"工具执行失败：{result.error}", False)

    # ==================== 子 Agent 执行 ====================

    async def _execute_subagent(self, task) -> "SubagentResult":
        """执行子 Agent 任务"""
        self.tracer.subagent_spawn(
            task=task.task_description,
            subagent_type=task.subagent_type if hasattr(task, 'subagent_type') else "general",
        )

        result = await self.subagent_spawner.spawn_and_run(
            task=task,
            parent_session=self.session,
        )

        self.tracer.subagent_result(
            task=task.task_description,
            summary=result.summary,
            success=result.success,
            turns=result.turns,
        )

        return result

    # ==================== 辅助方法 ====================

    def _action_details(self, action: AgentAction) -> dict:
        """提取 Action 的详细信息"""
        details = {}
        if action.tool_call:
            details["tool_name"] = action.tool_call.tool_name
            details["tool_args"] = json.dumps(action.tool_call.args, ensure_ascii=False)
        if action.subagent_task:
            details["subagent_task"] = action.subagent_task.task_description[:100]
        if action.answer:
            details["answer_preview"] = action.answer[:150]
        return details
