"""
CC-Harness Agent - Subagent 子智能体系统
=========================================
CC 路线的多 Agent 方案：层级式 Subagent 模型。

架构：
  主 Agent (Orchestrator)
      │
      ├── spawn → Subagent 1 (独立 Session + AgentLoop)
      ├── spawn → Subagent 2 (独立 Session + AgentLoop)
      └── 收集摘要 → 汇总生成最终答案

与 LangGraph 多 Agent 的区别：
  LangGraph:  多个 StateGraph 通过共享 State 通信
              → 子图与父图共享状态，容易互相干扰
  CC路线:     每个 Subagent 拥有独立 Session
              → 上下文隔离，子Agent 不污染父Agent 记忆
              → 返回摘要而非完整对话历史（节省 token）

与工具调用的区别：
  tool_call:       执行单一功能 → 结果直接返回 → 无独立循环
  spawn_subagent:  启动独立 AgentLoop → 多轮思考+工具调用 → 返回摘要
"""

from __future__ import annotations

import asyncio
from typing import Any

from harness.types import AgentAction, AgentConfig, Message, MessageRole, SubagentTask, ToolResult


class SubagentResult:
    """子 Agent 执行结果"""

    def __init__(self, task: str, summary: str, success: bool, turns: int):
        self.task = task
        self.summary = summary
        self.success = success
        self.turns = turns


class SubagentSpawner:
    """
    子 Agent 调度器

    职责：
    1. 根据 SubagentTask 创建子 Agent 的 Session 和 AgentLoop
    2. 执行子 Agent（独立上下文）
    3. 收集结果摘要返还给主 Agent

    注意：
    - 子 Agent 也可以 spawn 更深层子 Agent（递归支持，但有深度限制）
    - 当前实现为防止无限递归，限制最大深度为 2
    """

    def __init__(self, tool_registry, llm_adapter, prompt_engine, max_depth: int = 2):
        self.tool_registry = tool_registry
        self.llm_adapter = llm_adapter
        self.prompt_engine = prompt_engine
        self.max_depth = max_depth
        self._current_depth = 0

    async def spawn_and_run(
        self,
        task: SubagentTask,
        parent_session=None,
    ) -> SubagentResult:
        """
        派生子 Agent 并等待完成

        Args:
            task: 子任务定义
            parent_session: 父 Session（用于继承部分配置）

        Returns:
            SubagentResult: 子 Agent 执行结果
        """
        if self._current_depth >= self.max_depth:
            return SubagentResult(
                task=task.task_description,
                summary=f"达到最大子Agent深度限制({self.max_depth})，任务未执行",
                success=False,
                turns=0,
            )

        self._current_depth += 1

        try:
            # 创建子 Agent 配置
            sub_config = AgentConfig(
                max_turns=task.max_turns,
                model=self.llm_adapter.model,
                temperature=0.0,
                enable_subagents=(self._current_depth < self.max_depth),
            )

            # 构建子 Agent 专用的 system prompt
            sub_system_prompt = self._build_subagent_prompt(task)

            # 创建独立 Session
            from harness.session import Session
            sub_session = Session(config=sub_config)
            sub_session.set_system_prompt(sub_system_prompt)
            sub_session.append_user_message(task.task_description)

            # 启动独立 AgentLoop
            from harness.agent_loop import AgentLoop
            sub_loop = AgentLoop(
                session=sub_session,
                llm_adapter=self.llm_adapter,
                tool_registry=self.tool_registry,
                prompt_engine=self.prompt_engine,
            )

            result_text = await sub_loop.run()

            # 生成摘要
            summary = await self._summarize(task.task_description, result_text)

            return SubagentResult(
                task=task.task_description,
                summary=summary,
                success=True,
                turns=sub_session.total_turns,
            )

        except Exception as e:
            return SubagentResult(
                task=task.task_description,
                summary=f"子Agent执行异常：{e}",
                success=False,
                turns=0,
            )
        finally:
            self._current_depth -= 1

    def _build_subagent_prompt(self, task: SubagentTask) -> str:
        """
        构建子 Agent 专用的系统提示词

        子 Agent 的 prompt 更聚焦：只告诉它要完成什么子任务。
        不会把主 Agent 的全套能力说明塞给它。
        """
        return f"""你是一个专业子Agent，负责完成一项具体子任务。

## 任务
{task.task_description}

## 行为规范
- 专注于当前子任务，不要尝试扩展到其他领域
- 使用可用工具获取所需信息
- 完成任务后立即返回结果，不要进行不必要的探索
- 结果应该简洁、准确、直接回答子任务要求
"""

    async def _summarize(self, task: str, full_result: str) -> str:
        """
        将子 Agent 的完整回答压缩为摘要

        为什么要压缩？
        - 子Agent 的完整对话可能有数千 token
        - 主Agent 只需要知道关键结论，不需要全部过程
        - 压缩后的摘要注入主Agent上下文，节省token预算

        生产环境会调用 LLM 做摘要，当前实现为截断。
        """
        if len(full_result) <= 500:
            return full_result

        # 调用 LLM 生成摘要
        try:
            prompt = (
                f"将以下任务结果压缩为 200 字以内的摘要，保留关键信息和结论。\n"
                f"任务：{task}\n"
                f"结果：{full_result[:1500]}\n\n摘要："
            )
            resp = await self.llm_adapter.client.chat.completions.create(
                model=self.llm_adapter.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return full_result[:500] + "..."
