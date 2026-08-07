"""
CC-Harness Agent — 多 Agent 协作模式 (v2)
==========================================
三种工业级多 Agent 协作模式：

1. Debate (辩论模式)
   多个 Agent 独立回答同一问题 → 交叉评审 → 综合
   适用场景：需要严谨判断的决策场景（安全审查、合规检查）

2. Map-Reduce (分发归并模式)
   主 Agent 拆解任务 → 并行派发子 Agent → 汇总综合
   适用场景：大规模文档分析、多视角调研

3. Hierarchy (层级委托模式)
   主 Agent → 专业 Agent → 工具 Agent（链式委托）
   适用场景：复杂多步骤任务、需要专业分工的场景

对比 LangGraph 多 Agent:
    LangGraph: 多个 StateGraph 通过共享 State 通信 → 子图和父图耦合
    CC 路线:  每个 Agent 独立 Session → 上下文隔离 → 只返回摘要
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.session import Session
from harness.agent_loop import AgentLoop
from harness.llm_adapter import LLMAdapter
from harness.prompt_engine import PromptEngine
from harness.types import AgentConfig
from tools.registry import ToolRegistry


# ==================== 共享类型 ====================

@dataclass
class CollaborationResult:
    """多 Agent 协作的产出"""
    answer: str                              # 最终综合答案
    mode: str                                # debate / map_reduce / hierarchy
    sub_results: list["SubResult"] = field(default_factory=list)
    total_turns: int = 0
    total_tool_calls: int = 0
    success: bool = True
    error: str = ""


@dataclass
class SubResult:
    """单个子 Agent 的产出"""
    agent_name: str
    answer: str
    success: bool
    turns: int
    tool_calls: int


# ==================== 1. Debate 辩论模式 ====================

DEBATE_SYSTEM_PROMPT = """你是一个专业辩论 Agent，立场是：{stance}

## 任务
对以下问题给出你的独立分析和判断：
{question}

## 规则
1. 基于你的专业立场给出最严谨的分析
2. 使用知识库检索获取事实支撑
3. 如果你的立场有弱点，诚实指出
4. 输出结构清晰：论点 → 论据 → 结论
"""

DEBATE_SYNTHESIS_PROMPT = """你是一个公正的裁决者。以下是对同一问题的 {n} 个不同视角的分析：

{opinions}

## 任务
综合以上所有视角，生成一份平衡、客观的最终判断。要求：
1. 指出各方一致认可的共识
2. 指出各方分歧点和各自论据
3. 给出综合判断（如有倾向，说明理由）
"""


class DebateOrchestrator:
    """
    辩论模式编排器

    使用方式:
        orch = DebateOrchestrator(llm, tool_registry, prompt_engine)
        result = await orch.debate(
            question="公司应不应该采用微服务架构？",
            stances=["激进派（全力推行）", "保守派（谨慎评估）", "务实派（因地制宜）"],
            rounds=2,
        )
    """

    def __init__(self, llm_adapter: LLMAdapter, tool_registry: ToolRegistry,
                 prompt_engine: PromptEngine):
        self.llm = llm_adapter
        self.tool_registry = tool_registry
        self.prompt_engine = prompt_engine

    async def debate(
        self,
        question: str,
        stances: list[str] | None = None,
        rounds: int = 2,
    ) -> CollaborationResult:
        """
        执行辩论

        Args:
            question: 辩论问题
            stances: 各方立场（默认 3 方：正方/反方/中立方）
            rounds: 辩论轮次

        Returns:
            综合判断
        """
        if stances is None:
            stances = ["支持方（分析优点）", "反对方（分析风险）", "中立方（客观评估）"]

        sub_results = []

        # Round 1: 各方独立分析
        for i, stance in enumerate(stances):
            config = AgentConfig(max_turns=3, model=self.llm.model, enable_subagents=False)
            session = Session(config=config)
            session.set_system_prompt(DEBATE_SYSTEM_PROMPT.format(stance=stance, question=question))
            session.append_user_message(question)

            loop = AgentLoop(session, self.llm, self.tool_registry, self.prompt_engine)
            answer = await loop.run()

            sub_results.append(SubResult(
                agent_name=f"debater_{i+1}_{stance[:10]}",
                answer=answer,
                success=True,
                turns=session.total_turns,
                tool_calls=session.tool_calls_count,
            ))

        # Round 2+ (可选): 交叉评审
        for r in range(1, rounds):
            for i, stance in enumerate(stances):
                # 让当前 agent 看到其他人的观点
                other_opinions = "\n".join([
                    f"[{sr.agent_name}]: {sr.answer[:300]}"
                    for j, sr in enumerate(sub_results) if j != i
                ])

                cross_prompt = (
                    f"以下是其他方的观点：\n{other_opinions}\n\n"
                    f"请针对以上观点进行回驳或补充，强化你（{stance}）的论证。"
                )

                config = AgentConfig(max_turns=3, model=self.llm.model)
                session = Session(config=config)
                session.set_system_prompt(DEBATE_SYSTEM_PROMPT.format(stance=stance, question=question))
                session.append_user_message(cross_prompt)

                loop = AgentLoop(session, self.llm, self.tool_registry, self.prompt_engine)
                rebuttal = await loop.run()

                # 追加到子结果
                sub_results[i].answer += f"\n\n--- 第{r+1}轮回驳 ---\n{rebuttal}"

        # 综合裁决
        opinions = "\n\n---\n\n".join([
            f"## {sr.agent_name}\n{sr.answer}"
            for sr in sub_results
        ])

        synthesis_prompt = DEBATE_SYNTHESIS_PROMPT.format(n=len(stances), opinions=opinions[:6000])

        # 用简单的 LLM 调用做综合
        resp = await self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0,
            max_tokens=1500,
        )
        final_answer = resp.choices[0].message.content.strip()

        return CollaborationResult(
            answer=final_answer,
            mode="debate",
            sub_results=sub_results,
            total_turns=sum(sr.turns for sr in sub_results),
            total_tool_calls=sum(sr.tool_calls for sr in sub_results),
        )


# ==================== 2. Map-Reduce 分发归并模式 ====================

MAP_PROMPT = """你是一个任务执行 Agent。请完成以下子任务：

## 子任务
{subtask}

## 全局上下文
{context}

请只聚焦于你的子任务，给出简洁、准确的回答。
"""

REDUCE_PROMPT = """你是一个综合汇总 Agent。以下是对同一问题的多个子任务结果：

{sub_results}

## 原始问题
{question}

## 任务
将以上所有子任务结果整合为一份完整的、结构化的最终回答。要求：
1. 按逻辑顺序组织信息
2. 去除重复内容
3. 补充跨子任务的关联分析
4. 如果有矛盾之处，指出并给出建议
"""


class MapReduceOrchestrator:
    """
    Map-Reduce 编排器

    使用方式:
        orch = MapReduceOrchestrator(llm, tool_registry, prompt_engine)

        # 自动拆解（LLM 分拆）
        result = await orch.auto_map_reduce("分析公司的技术债务情况")

        # 手动指定子任务
        result = await orch.map_reduce(
            "分析微服务架构的优劣",
            subtasks=["性能影响", "开发效率", "运维复杂度", "扩展性"],
        )
    """

    def __init__(self, llm_adapter: LLMAdapter, tool_registry: ToolRegistry,
                 prompt_engine: PromptEngine):
        self.llm = llm_adapter
        self.tool_registry = tool_registry
        self.prompt_engine = prompt_engine

    async def map_reduce(
        self,
        question: str,
        subtasks: list[str],
    ) -> CollaborationResult:
        """
        手动指定子任务的 Map-Reduce

        Args:
            question: 总问题
            subtasks: 子任务描述列表

        Returns:
            综合结果
        """
        # Map 阶段：并行执行子任务
        async def run_subtask(task: str, idx: int) -> SubResult:
            config = AgentConfig(max_turns=4, model=self.llm.model, enable_subagents=False)
            session = Session(config=config)
            session.set_system_prompt(MAP_PROMPT.format(subtask=task, context=question))
            session.append_user_message(task)

            loop = AgentLoop(session, self.llm, self.tool_registry, self.prompt_engine)
            answer = await loop.run()

            return SubResult(
                agent_name=f"worker_{idx+1}",
                answer=answer,
                success=True,
                turns=session.total_turns,
                tool_calls=session.tool_calls_count,
            )

        # 并行执行
        tasks = [run_subtask(t, i) for i, t in enumerate(subtasks)]
        sub_results = await asyncio.gather(*tasks)

        # Reduce 阶段：汇总
        formatted = "\n\n---\n\n".join([
            f"## 子任务: {subtasks[i]}\n{sr.answer}"
            for i, sr in enumerate(sub_results)
        ])

        reduce_prompt = REDUCE_PROMPT.format(
            sub_results=formatted[:8000],
            question=question,
        )

        resp = await self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[{"role": "user", "content": reduce_prompt}],
            temperature=0,
            max_tokens=2000,
        )
        final_answer = resp.choices[0].message.content.strip()

        return CollaborationResult(
            answer=final_answer,
            mode="map_reduce",
            sub_results=list(sub_results),
            total_turns=sum(sr.turns for sr in sub_results),
            total_tool_calls=sum(sr.tool_calls for sr in sub_results),
        )

    async def auto_map_reduce(self, question: str, num_workers: int = 3) -> CollaborationResult:
        """
        LLM 自动拆解任务的 Map-Reduce

        LLM 分析问题 → 拆解为 N 个子任务 → Map → Reduce
        """
        # Step 1: LLM 拆解
        split_prompt = (
            f"将以下复杂问题拆解为 {num_workers} 个独立的子任务，每个子任务可以并行执行。\n"
            f"问题：{question}\n"
            f"输出格式：每行一个子任务，以 '- ' 开头。"
        )

        resp = await self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[{"role": "user", "content": split_prompt}],
            temperature=0,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()

        # 解析子任务
        subtasks = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                subtasks.append(line[2:])
            elif line and len(subtasks) < num_workers:
                subtasks.append(line)

        # 确保至少有 num_workers 个
        if len(subtasks) < num_workers:
            # Fallback: 均匀分拆
            subtasks = [f"{question} (方面 {i+1}/{num_workers})" for i in range(num_workers)]

        return await self.map_reduce(question, subtasks[:num_workers])


# ==================== 3. Hierarchy 层级委托模式 ====================

HIERARCHY_ORCHESTRATOR_PROMPT = """你是一个总协调 Agent。你的职责是：
1. 分析用户请求，确定需要哪些专业 Agent
2. 将子任务委托给专业 Agent
3. 收集所有结果后，综合成最终答案

## 可用工具
- rag_search: 知识库检索
- spawn_subagent: 派生子 Agent（指定专业类型：analyst/researcher/critic）

## 行为规范
- 复杂问题先规划再执行
- 对需要深度分析的问题，优先委托给专业 Agent
- 最终答案应综合所有子 Agent 的结果
"""


class HierarchyOrchestrator:
    """
    层级委托编排器

    主 Agent 拆解任务 → 派发专业 Agent → 收集 → 综合

    使用方式:
        orch = HierarchyOrchestrator(llm, tool_registry, prompt_engine)
        result = await orch.execute(
            "分析 LangGraph vs AgentLoop 的架构差异，给出选型建议"
        )
    """

    def __init__(self, llm_adapter: LLMAdapter, tool_registry: ToolRegistry,
                 prompt_engine: PromptEngine):
        self.llm = llm_adapter
        self.tool_registry = tool_registry
        self.prompt_engine = prompt_engine

    async def execute(self, question: str, specialist_types: list[str] | None = None) -> CollaborationResult:
        """
        层级委托执行

        Args:
            question: 用户问题
            specialist_types: 专业 Agent 类型列表 (默认: analyst, researcher, critic)

        Returns:
            综合结果
        """
        if specialist_types is None:
            specialist_types = ["analyst", "researcher"]

        sub_results = []

        # 并行启动所有专业 Agent
        async def run_specialist(spec_type: str) -> SubResult:
            specialist_prompt = self._get_specialist_prompt(spec_type, question)
            config = AgentConfig(max_turns=5, model=self.llm.model, enable_subagents=True)
            session = Session(config=config)
            session.set_system_prompt(specialist_prompt)
            session.append_user_message(question)

            loop = AgentLoop(session, self.llm, self.tool_registry, self.prompt_engine)
            answer = await loop.run()

            return SubResult(
                agent_name=f"specialist_{spec_type}",
                answer=answer,
                success=True,
                turns=session.total_turns,
                tool_calls=session.tool_calls_count,
            )

        tasks = [run_specialist(t) for t in specialist_types]
        sub_results = await asyncio.gather(*tasks)

        # 主 Agent 综合
        specialist_outputs = "\n\n---\n\n".join([
            f"## {sr.agent_name}\n{sr.answer}"
            for sr in sub_results
        ])

        synthesis_prompt = (
            f"## 用户问题\n{question}\n\n"
            f"## 专业 Agent 分析结果\n{specialist_outputs[:8000]}\n\n"
            f"## 任务\n综合以上专业分析，给出完整的最终回答。要求：结构清晰、有理有据、给出明确建议。"
        )

        resp = await self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0,
            max_tokens=2000,
        )
        final_answer = resp.choices[0].message.content.strip()

        return CollaborationResult(
            answer=final_answer,
            mode="hierarchy",
            sub_results=list(sub_results),
            total_turns=sum(sr.turns for sr in sub_results),
            total_tool_calls=sum(sr.tool_calls for sr in sub_results),
        )

    def _get_specialist_prompt(self, spec_type: str, context: str) -> str:
        """获取专业 Agent 的系统提示词"""
        prompts = {
            "analyst": (
                "你是一个技术分析专家。擅长从架构、性能、可维护性等维度进行深入分析。\n"
                f"背景：{context}\n请给出专业的分析，包括技术细节和 trade-off。"
            ),
            "researcher": (
                "你是一个调研专家。擅长收集和整理信息，发现不同方案之间的联系和差异。\n"
                f"背景：{context}\n请从多个来源收集信息，给出全面的调研结果。"
            ),
            "critic": (
                "你是一个严格的评审专家。擅长发现方案中的问题、风险和边界条件。\n"
                f"背景：{context}\n请以批判性思维审视问题，指出潜在风险和遗漏。"
            ),
            "writer": (
                "你是一个技术写作者。擅长将复杂信息整理为清晰、易读的文档。\n"
                f"背景：{context}\n请生成结构化的、面向读者的高质量内容。"
            ),
        }
        return prompts.get(spec_type, prompts["analyst"])


# ==================== 便捷工厂 ====================

def create_collaboration(
    mode: str,
    llm_adapter: LLMAdapter,
    tool_registry: ToolRegistry,
    prompt_engine: PromptEngine,
) -> DebateOrchestrator | MapReduceOrchestrator | HierarchyOrchestrator:
    """工厂函数：根据模式创建编排器"""
    if mode == "debate":
        return DebateOrchestrator(llm_adapter, tool_registry, prompt_engine)
    elif mode == "map_reduce":
        return MapReduceOrchestrator(llm_adapter, tool_registry, prompt_engine)
    elif mode == "hierarchy":
        return HierarchyOrchestrator(llm_adapter, tool_registry, prompt_engine)
    else:
        raise ValueError(f"未知的协作模式: {mode}。支持: debate, map_reduce, hierarchy")
