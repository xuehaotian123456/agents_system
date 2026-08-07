"""
CC-Harness Agent - 上下文引擎 (Context Engine)
================================================
管理 Agent 上下文窗口的智能压缩与 Token 预算分配。

核心问题：
    LLM 上下文窗口有限（8k/32k/128k），对话越长越容易超限。
    简单截断会丢失早期重要信息，LLM 遗忘关键上下文。

解决方案（三层摘要金字塔）：
    ┌──────────────────────────────────────┐
    │  L1: 近期消息 (窗口)                   │  ← 保留原文，不压缩
    │      最近 N 条消息 / 最近 M tokens     │
    ├──────────────────────────────────────┤
    │  L2: 中期消息 (轻量摘要)                │  ← LLM 逐条压缩为 1-2 句
    │      每条消息压缩为关键信息              │
    ├──────────────────────────────────────┤
    │  L3: 早期消息 (深度摘要)                │  ← 多轮合并为一段摘要
    │      整个会话段落合并为 200 字摘要       │
    └──────────────────────────────────────┘

Token 预算模型:
    ┌──────────┬──────────┬──────────┬──────────┐
    │ System   │ Tools    │ History  │ Answer   │
    │ Prompt   │ Defs     │ (压缩后)  │ Reserve  │
    │ ~500     │ ~800     │ ~4000    │ ~2700    │
    └──────────┴──────────┴──────────┴──────────┘
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.types import Message, MessageRole


# ==================== Token 预算 ====================

@dataclass
class TokenBudget:
    """
    Token 预算分配器

    把上下文窗口切分为几个区域，每个区域有各自的 token 上限。
    压缩时优先压缩 History 区（对话历史），不动 System Prompt 和 Tools。
    """
    total: int = 8000                               # 上下文总窗口
    system_reserve: int = 800                       # System Prompt 保留
    tools_reserve: int = 1000                       # 工具定义保留
    history_budget: int = 4500                      # 对话历史预算
    answer_reserve: int = 1700                      # 回答预留

    @classmethod
    def for_model(cls, model_name: str) -> "TokenBudget":
        """根据模型自动设置预算"""
        budgets = {
            "qwen-turbo":      (8000,  500,  600, 4500, 2400),
            "qwen-plus":       (32000, 800, 1000, 20000, 10200),
            "qwen-max":        (32768, 800, 1000, 20000, 10968),
            "deepseek-chat":   (65536, 800, 1000, 50000, 13736),
            "gpt-4o":          (128000, 800, 1000, 100000, 26200),
            "gpt-4o-mini":     (128000, 800, 1000, 100000, 26200),
        }
        if model_name in budgets:
            t, s, tools, hist, ans = budgets[model_name]
            return cls(total=t, system_reserve=s, tools_reserve=tools, history_budget=hist, answer_reserve=ans)

        # 默认：8k 窗口
        return cls()

    @property
    def available_for_history(self) -> int:
        """对话历史当前可用 token"""
        return max(0, self.history_budget)


# ==================== 摘要层级 ====================

class SummaryLevel:
    """摘要层级常量"""
    FULL = 0         # 保留原文
    LIGHT = 1        # 轻量压缩（保留关键信息，~50字/条）
    DEEP = 2         # 深度压缩（段落摘要，~200字）


@dataclass
class CompressionResult:
    """压缩操作的结果"""
    compressed: bool                          # 是否执行了压缩
    original_tokens: int                      # 压缩前 token 数
    final_tokens: int                         # 压缩后 token 数
    reduction_ratio: float                    # 压缩比
    summaries_generated: int = 0              # 生成的摘要数
    duration_ms: float = 0.0                  # 压缩耗时


# ==================== 上下文引擎 ====================

class ContextEngine:
    """
    上下文引擎

    职责：
    1. 监控对话历史的 token 消耗
    2. 在超限前触发分层压缩
    3. 管理摘要消息的注入和替换

    使用方式：
        engine = ContextEngine(budget=TokenBudget.for_model("qwen-plus"))
        session.messages = await engine.compress(session.messages, llm_adapter)
    """

    def __init__(
        self,
        budget: TokenBudget | None = None,
        recent_window: int = 4,          # 保留最近 N 条消息不压缩
        light_summary_max_chars: int = 80,  # 轻量摘要最大字符数
        deep_summary_max_chars: int = 300,  # 深度摘要最大字符数
    ):
        self.budget = budget or TokenBudget()
        self.recent_window = recent_window
        self.light_summary_max_chars = light_summary_max_chars
        self.deep_summary_max_chars = deep_summary_max_chars

        # 统计
        self.total_compressions: int = 0
        self.total_tokens_saved: int = 0

    # ==================== 主入口 ====================

    async def compress(
        self,
        messages: list[Message],
        llm_adapter=None,
        force: bool = False,
    ) -> tuple[list[Message], CompressionResult]:
        """
        压缩对话历史

        Args:
            messages: 当前消息列表
            llm_adapter: LLM 适配器（用于生成摘要，可选——不传则用规则压缩）
            force: 强制压缩（忽略 token 检查）

        Returns:
            (压缩后的消息列表, 压缩结果)
        """
        start = time.time()

        # 计算当前 token 使用量
        current_tokens = sum(m.estimate_tokens() for m in messages)

        # 检查是否需要压缩
        if not force and current_tokens <= self.budget.available_for_history:
            return messages, CompressionResult(
                compressed=False,
                original_tokens=current_tokens,
                final_tokens=current_tokens,
                reduction_ratio=0.0,
            )

        # 消息太少，不值得压缩
        if len(messages) <= self.recent_window + 2:
            return messages, CompressionResult(
                compressed=False,
                original_tokens=current_tokens,
                final_tokens=current_tokens,
                reduction_ratio=0.0,
            )

        # ── 分层压缩 ──
        # 1. 保留最近 N 条（L0: FULL）
        recent = messages[-self.recent_window:]
        remaining = messages[:-self.recent_window]

        # 2. 中期消息轻量压缩（L1: LIGHT）
        if len(remaining) > 2:
            mid_start = max(0, len(remaining) - 6)
            early = remaining[:mid_start]
            middle = remaining[mid_start:]

            # 深度压缩早期消息
            deep_summaries = []
            if early:
                if llm_adapter:
                    deep_text = await self._deep_summarize(early, llm_adapter)
                else:
                    deep_text = self._rule_deep_summarize(early)
                deep_summaries.append(self._make_summary_msg(deep_text, "早期对话"))

            # 轻量压缩中期消息
            if llm_adapter:
                light_text = await self._light_summarize(middle, llm_adapter)
            else:
                light_text = self._rule_light_summarize(middle)
            mid_summary = [self._make_summary_msg(light_text, "近期对话")]
        else:
            deep_summaries = []
            if llm_adapter:
                light_text = await self._light_summarize(remaining, llm_adapter)
            else:
                light_text = self._rule_light_summarize(remaining)
            mid_summary = [self._make_summary_msg(light_text, "历史对话")]

        # 3. 重建消息列表
        new_messages = deep_summaries + mid_summary + recent

        new_tokens = sum(m.estimate_tokens() for m in new_messages)
        self.total_compressions += 1
        self.total_tokens_saved += (current_tokens - new_tokens)

        result = CompressionResult(
            compressed=True,
            original_tokens=current_tokens,
            final_tokens=new_tokens,
            reduction_ratio=round((1 - new_tokens / max(current_tokens, 1)) * 100, 1),
            summaries_generated=len(deep_summaries) + len(mid_summary),
            duration_ms=(time.time() - start) * 1000,
        )

        return new_messages, result

    # ==================== 摘要生成 ====================

    async def _light_summarize(self, messages: list[Message], llm_adapter) -> str:
        """轻量摘要：逐条提取关键信息"""
        if len(messages) <= 2:
            return " | ".join([m.content[:100] for m in messages
                               if m.role in (MessageRole.USER, MessageRole.ASSISTANT)])

        # 用 LLM 做轻量摘要
        convo_text = "\n".join([
            f"[{m.role.value}]: {m.content[:200]}"
            for m in messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL)
        ])

        prompt = (
            f"将以下对话压缩为 150 字以内的摘要，保留关键问题和结论：\n\n{convo_text[:2000]}"
        )

        try:
            resp = await llm_adapter.client.chat.completions.create(
                model=llm_adapter.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return self._rule_light_summarize(messages)

    async def _deep_summarize(self, messages: list[Message], llm_adapter) -> str:
        """深度摘要：整段对话合并为一段"""
        convo_text = "\n".join([
            f"[{m.role.value}]: {m.content[:300]}"
            for m in messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL)
        ])

        prompt = (
            f"将以下对话历史压缩为 200 字以内的摘要，概括讨论的主题、关键决策和结论：\n\n{convo_text[:3000]}"
        )

        try:
            resp = await llm_adapter.client.chat.completions.create(
                model=llm_adapter.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return self._rule_deep_summarize(messages)

    # ==================== 规则降级（无 LLM 时） ====================

    def _rule_light_summarize(self, messages: list[Message]) -> str:
        """规则压缩：提取每条消息的前 80 字符"""
        parts = []
        for m in messages:
            if m.role == MessageRole.USER:
                parts.append(f"❓ 用户问: {m.content[:self.light_summary_max_chars]}")
            elif m.role == MessageRole.ASSISTANT:
                parts.append(f"🤖 助手答: {m.content[:self.light_summary_max_chars]}")
            elif m.role == MessageRole.TOOL:
                parts.append(f"🔧 调用 {m.tool_name}: {m.content[:60]}")
        return " | ".join(parts)

    def _rule_deep_summarize(self, messages: list[Message]) -> str:
        """规则压缩：提取用户问题 + 最后一条助手回复"""
        user_qs = [m.content[:100] for m in messages if m.role == MessageRole.USER]
        last_answer = ""
        for m in reversed(messages):
            if m.role == MessageRole.ASSISTANT:
                last_answer = m.content[:150]
                break

        qs_text = "; ".join(user_qs[:5])
        return f"用户问过: {qs_text}。最终回答: {last_answer}"

    def _make_summary_msg(self, text: str, label: str = "") -> Message:
        """创建摘要消息"""
        prefix = f"[{label}摘要]" if label else "[历史摘要]"
        return Message(
            role=MessageRole.SYSTEM,
            content=f"{prefix} {text}",
        )

    # ==================== Token 监控 ====================

    def estimate_total(self, system_prompt: str, messages: list[Message]) -> int:
        """估算总 token 用量"""
        from harness.llm_adapter import estimate_tokens
        sys_tokens = estimate_tokens(system_prompt)
        msg_tokens = sum(m.estimate_tokens() for m in messages)
        return sys_tokens + msg_tokens

    def should_compress(self, system_prompt: str, messages: list[Message]) -> bool:
        """判断是否需要压缩"""
        total = self.estimate_total(system_prompt, messages)
        return total > self.budget.total * 0.75  # 超过 75% 就触发压缩

    def get_stats(self) -> dict:
        """获取压缩统计"""
        return {
            "total_compressions": self.total_compressions,
            "total_tokens_saved": self.total_tokens_saved,
            "budget_total": self.budget.total,
            "budget_history": self.budget.history_budget,
        }
