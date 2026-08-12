"""
CC-Harness Agent - 执行追踪器 (Agent Tracer)
=============================================
记录 Agent 每一步决策的完整轨迹，替代零散的 print()。

设计目标：
1. 结构化 — 每一步都是强类型事件，可被程序解析
2. 实时 — 事件产生时立即推送回调，前端实时展示
3. 可回放 — 完整轨迹可序列化为 JSON，事后复现 Agent 的全部思考过程
4. 可审计 — 记录 LLM 调用、工具调用、延迟、Token 消耗等

与 LangFuse / Weights & Biases 的区别：
    这些是外部可观测性平台；Tracer 是内置的轻量级追踪器，
    输出可直接喂给 LangFuse，也可独立使用。

事件类型:
    SESSION_START    — 会话开始
    TURN_START       — 新一轮循环开始
    LLM_THINKING     — LLM 返回思考过程
    LLM_ACTION       — LLM 决定执行什么动作
    TOOL_CALL        — 开始调用工具
    TOOL_RESULT      — 工具返回结果
    SUBAGENT_SPAWN   — 派生子 Agent
    SUBAGENT_RESULT  — 子 Agent 返回
    CONTEXT_COMPRESS — 上下文压缩
    FINAL_ANSWER     — 最终回答
    SESSION_END      — 会话结束
    ERROR            — 异常
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ==================== 事件类型 ====================

class TraceEventType(str, Enum):
    SESSION_START = "session_start"
    TURN_START = "turn_start"
    LLM_THINKING = "llm_thinking"
    LLM_ACTION = "llm_action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SUBAGENT_SPAWN = "subagent_spawn"
    SUBAGENT_RESULT = "subagent_result"
    CONTEXT_COMPRESS = "context_compress"
    FINAL_ANSWER = "final_answer"
    SESSION_END = "session_end"
    ERROR = "error"


# ==================== 事件数据模型 ====================

@dataclass
class TraceEvent:
    """单条追踪事件"""
    type: TraceEventType
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    turn: int = 0                              # 当前循环轮次
    elapsed_ms: float = 0.0                    # 从会话开始到现在的累计时间

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "turn": self.turn,
            "elapsed_ms": round(self.elapsed_ms, 1),
            **self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ==================== 回调类型 ====================

TraceCallback = Callable[[TraceEvent], None]
"""追踪回调: (event: TraceEvent) -> None"""


# ==================== AgentTracer ====================

class AgentTracer:
    """
    Agent 执行追踪器

    使用方式：
        tracer = AgentTracer()
        tracer.on_event(lambda e: print(e.to_json()))  # 控制台输出

        tracer.session_start(session_id="abc123")
        tracer.turn_start(1)
        tracer.llm_thinking("需要查询知识库...")
        tracer.tool_call("rag_search", {"query": "..."})
        tracer.tool_result("rag_search", "查到3篇文档", True, 120)
        tracer.final_answer("根据知识库...")
        tracer.session_end()

        # 获取完整轨迹
        print(tracer.to_json())          # JSON 字符串
        print(tracer.summary())          # 人类可读摘要

        # 导出给 LangFuse
        langfuse.trace(tracer.to_langfuse_format())
    """

    def __init__(self, verbose: bool = True):
        self.events: list[TraceEvent] = []
        self._callbacks: list[TraceCallback] = []
        self._session_start: float = 0.0
        self._current_turn: int = 0
        self.verbose = verbose

        # 聚合统计
        self.llm_call_count: int = 0
        self.tool_call_count: int = 0
        self.subagent_count: int = 0
        self.error_count: int = 0
        self.total_tool_latency_ms: float = 0.0

    # ==================== 回调注册 ====================

    def on_event(self, callback: TraceCallback):
        """注册事件回调（实时推送用）"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: TraceCallback):
        """移除回调"""
        self._callbacks.remove(callback)

    # ==================== 事件记录 ====================

    def _emit(self, event_type: TraceEventType, **data) -> TraceEvent:
        """记录并广播事件"""
        event = TraceEvent(
            type=event_type,
            turn=self._current_turn,
            elapsed_ms=(time.time() - self._session_start) * 1000 if self._session_start else 0,
            data=data,
        )

        self.events.append(event)

        # 控制台输出
        if self.verbose:
            self._console_print(event)

        # 回调广播
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass

        return event

    # ── 生命周期 ──

    def session_start(self, session_id: str, model: str = "", max_turns: int = 15, **meta):
        """会话开始"""
        self._session_start = time.time()
        self._emit(TraceEventType.SESSION_START,
                   session_id=session_id, model=model, max_turns=max_turns, **meta)

    def session_end(self, answer: str = "", total_turns: int = 0, **meta):
        """会话结束"""
        self._emit(TraceEventType.SESSION_END,
                   answer=answer[:200], total_turns=total_turns, **meta)

    # ── 循环轮次 ──

    def turn_start(self, turn: int, token_estimate: int = 0):
        """新一轮循环开始"""
        self._current_turn = turn
        self._emit(TraceEventType.TURN_START,
                   turn=turn, token_estimate=token_estimate)

    # ── LLM 交互 ──

    def llm_thinking(self, thought: str):
        """LLM 思考过程"""
        self.llm_call_count += 1
        self._emit(TraceEventType.LLM_THINKING,
                   thought=thought, thought_preview=thought[:120])

    def llm_action(self, action_type: str, thought: str = "", details: dict | None = None):
        """LLM 决定执行的动作"""
        self._emit(TraceEventType.LLM_ACTION,
                   action_type=action_type,
                   thought_preview=thought[:100] if thought else "",
                   details=details or {})

    # ── 工具调用 ──

    def tool_call(self, tool_name: str, args: dict):
        """开始调用工具"""
        self.tool_call_count += 1
        self._emit(TraceEventType.TOOL_CALL,
                   tool_name=tool_name, args=args)

    def tool_result(self, tool_name: str, result_preview: str, success: bool, latency_ms: float = 0):
        """工具调用结果"""
        self.total_tool_latency_ms += latency_ms
        self._emit(TraceEventType.TOOL_RESULT,
                   tool_name=tool_name,
                   result_preview=result_preview[:300],
                   success=success,
                   latency_ms=round(latency_ms, 1))

    # ── 子 Agent ──

    def subagent_spawn(self, task: str, subagent_type: str = "general"):
        """派生子 Agent"""
        self.subagent_count += 1
        self._emit(TraceEventType.SUBAGENT_SPAWN,
                   task=task[:200], subagent_type=subagent_type)

    def subagent_result(self, task: str, summary: str, success: bool, turns: int = 0):
        """子 Agent 返回"""
        self._emit(TraceEventType.SUBAGENT_RESULT,
                   task=task[:100], summary=summary[:300], success=success, turns=turns)

    # ── 上下文管理 ──

    def context_compress(self, before_tokens: int, after_tokens: int, reduction_ratio: float):
        """上下文压缩"""
        self._emit(TraceEventType.CONTEXT_COMPRESS,
                   before_tokens=before_tokens,
                   after_tokens=after_tokens,
                   reduction_ratio=reduction_ratio)

    # ── 最终结果 ──

    def final_answer(self, answer: str, citations: list[str] | None = None):
        """生成最终回答"""
        self._emit(TraceEventType.FINAL_ANSWER,
                   answer_preview=answer[:300],
                   answer_length=len(answer),
                   citations=citations or [])

    def error(self, message: str, exception: str = "", recoverable: bool = True):
        """记录错误"""
        self.error_count += 1
        self._emit(TraceEventType.ERROR,
                   message=message, exception=exception, recoverable=recoverable)

    # ==================== 输出 ====================

    def to_markdown(self, title: str = "Agent 决策链报告") -> str:
        """
        生成人类可读的 markdown 决策链报告。

        面试/演示直接打印，一眼看出 Agent 的动态推理过程。
        """
        if not self.events:
            return "*（无追踪事件）*"

        lines = []
        links = []
        link_counter = [0]  # 闭包引用

        def _ref_id():
            link_counter[0] += 1
            return link_counter[0]

        # ── 标题 ──
        lines.append(f"# {title}")
        lines.append("")

        # ── 概览表 ──
        s = self.summary()
        lines.append("## 会话概览")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|:---|---|")
        lines.append(f"| 模型 | `{self._model_used}` |")
        lines.append(f"| 推理轮次 | {s['total_turns']} 轮 |")
        lines.append(f"| LLM 调用 | {s['llm_calls']} 次 |")
        lines.append(f"| 工具调用 | {s['tool_calls']} 次 |")
        lines.append(f"| 子 Agent | {s['subagents']} 个 |")
        lines.append(f"| 总耗时 | {s['total_elapsed_ms']:.0f} ms |")
        if s['errors']:
            lines.append(f"| 异常 | {s['errors']} 次 |")
        lines.append("")

        # ── 决策链 ──
        lines.append("## 决策链")
        lines.append("")

        round_tools = {}   # turn -> [(tool_name, success, latency)]
        round_thoughts = {}  # turn -> [thoughts]
        round_errors = {}    # turn -> [errors]
        final_answer = ""

        for e in self.events:
            t = e.turn
            d = e.data

            if e.type == TraceEventType.LLM_THINKING:
                round_thoughts.setdefault(t, []).append(d.get("thought_preview", ""))
            elif e.type == TraceEventType.TOOL_RESULT:
                round_tools.setdefault(t, []).append((
                    d.get("tool_name", "?"),
                    d.get("success", True),
                    d.get("latency_ms", 0),
                    d.get("result_preview", ""),
                ))
            elif e.type == TraceEventType.ERROR:
                round_errors.setdefault(t, []).append(d.get("message", ""))
            elif e.type == TraceEventType.FINAL_ANSWER:
                final_answer = d.get("answer_preview", "")
            elif e.type == TraceEventType.CONTEXT_COMPRESS:
                lines.append(f"*[压缩] 上下文压缩: {d.get('before_tokens',0)} -> {d.get('after_tokens',0)} tokens (-{d.get('reduction_ratio',0)}%)*")
                lines.append("")

        # 逐轮输出
        for turn in sorted(set(list(round_thoughts.keys()) + list(round_tools.keys()))):
            lines.append(f"### Round {turn}")
            lines.append("")

            # 思考
            thoughts = round_thoughts.get(turn, [])
            for th in thoughts:
                lines.append(f"> **[*] 思考**: {th}")
                lines.append("")

            # 工具
            tools = round_tools.get(turn, [])
            for i, (tname, ok, lat, preview) in enumerate(tools):
                status = "[OK]" if ok else "[FAIL]"
                lines.append(f"**>> {status} {tname}** ({lat:.0f}ms)")
                lines.append("")
                preview_clean = preview[:250].replace("\n", " ")
                lines.append(f"> {preview_clean}")
                lines.append("")

            # 错误
            errs = round_errors.get(turn, [])
            for err in errs:
                lines.append(f"!! 异常: {err}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # ── 最终回答 ──
        if final_answer:
            lines.append("## 最终回答")
            lines.append("")
            lines.append(final_answer[:2000])
            lines.append("")

        # ── 统计 ──
        lines.append("## 统计")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|:---|---|")
        lines.append(f"| 总轮次 | {s['total_turns']} |")
        lines.append(f"| LLM 调用 | {s['llm_calls']} |")
        lines.append(f"| 工具调用 | {s['tool_calls']} |")
        lines.append(f"| 总耗时 | {s['total_elapsed_ms']:.0f}ms |")
        if s['tool_calls'] > 0:
            lines.append(f"| 平均工具延迟 | {s['avg_tool_latency_ms']:.1f}ms |")
        lines.append(f"| 事件总数 | {s['events_count']} |")
        lines.append("")

        return "\n".join(lines)

    def to_chain(self) -> list[dict]:
        """
        导出为结构化决策链，方便程序化消费。

        Returns:
            [{turn, thought, tool_name, tool_success, tool_latency_ms, tool_result}]
        """
        chain = []
        current = {"turn": 0, "thought": "", "tool_name": "",
                    "tool_success": True, "tool_latency_ms": 0, "tool_result": ""}

        for e in self.events:
            d = e.data
            if e.type == TraceEventType.TURN_START:
                if current["thought"] or current["tool_name"]:
                    chain.append(current)
                current = {"turn": e.turn, "thought": "", "tool_name": "",
                            "tool_success": True, "tool_latency_ms": 0, "tool_result": ""}
            elif e.type == TraceEventType.LLM_THINKING:
                current["thought"] = d.get("thought_preview", "")
            elif e.type == TraceEventType.TOOL_RESULT:
                current["tool_name"] = d.get("tool_name", "")
                current["tool_success"] = d.get("success", True)
                current["tool_latency_ms"] = d.get("latency_ms", 0)
                current["tool_result"] = d.get("result_preview", "")[:300]

        if current["thought"] or current["tool_name"]:
            chain.append(current)

        return chain

    @property
    def _model_used(self) -> str:
        """从事件中提取模型名"""
        for e in self.events:
            if e.type == TraceEventType.SESSION_START:
                return e.data.get("model", "unknown")
        return "unknown"

    def to_dict(self) -> dict:
        """导出完整轨迹为字典"""
        return {
            "events": [e.to_dict() for e in self.events],
            "summary": self.summary(),
        }

    def to_json(self, indent: int = 2) -> str:
        """导出完整轨迹为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary(self) -> dict:
        """生成统计摘要"""
        total_time = (time.time() - self._session_start) * 1000 if self._session_start else 0
        return {
            "total_turns": self._current_turn,
            "llm_calls": self.llm_call_count,
            "tool_calls": self.tool_call_count,
            "subagents": self.subagent_count,
            "errors": self.error_count,
            "total_elapsed_ms": round(total_time, 0),
            "avg_tool_latency_ms": round(
                self.total_tool_latency_ms / max(self.tool_call_count, 1), 1
            ),
            "events_count": len(self.events),
        }

    def to_langfuse_format(self) -> dict:
        """
        导出为 LangFuse 兼容格式

        LangFuse trace 结构：
        - trace: 一次完整对话
          - generation: LLM 调用
          - span: 工具调用 / 子 Agent
        """
        s = self.summary()
        return {
            "name": f"cc-harness-session",
            "metadata": s,
            "generations": [
                {"name": f"turn-{e.turn}", "model": e.data.get("model", ""),
                 "input": e.data.get("thought_preview", "")}
                for e in self.events if e.type == TraceEventType.LLM_THINKING
            ],
            "spans": [
                {"name": e.data.get("tool_name", "unknown"),
                 "input": e.data.get("args", {}),
                 "output": e.data.get("result_preview", "")}
                for e in self.events if e.type == TraceEventType.TOOL_CALL
            ],
        }

    # ==================== 控制台输出 ====================

    def _console_print(self, event: TraceEvent):
        """友好的控制台输出"""
        d = event.data

        if event.type == TraceEventType.SESSION_START:
            print(f"\n{'='*50}\n🚀 会话开始 | 模型: {d.get('model', '?')} | "
                  f"最大轮次: {d.get('max_turns', '?')}\n{'='*50}")

        elif event.type == TraceEventType.TURN_START:
            print(f"\n{'─'*40}\n🔄 Round {d.get('turn', '?')}")

        elif event.type == TraceEventType.LLM_THINKING:
            thought = d.get("thought_preview", "")
            print(f"   💭 思考: {thought}")

        elif event.type == TraceEventType.LLM_ACTION:
            print(f"   🎯 动作: {d.get('action_type', '?')}")

        elif event.type == TraceEventType.TOOL_CALL:
            args_str = json.dumps(d.get("args", {}), ensure_ascii=False)
            print(f"   🔧 调用工具: {d.get('tool_name', '?')}({args_str})")

        elif event.type == TraceEventType.TOOL_RESULT:
            status = "✅" if d.get("success") else "❌"
            print(f"   {status} 工具返回: {d.get('result_preview', '')[:100]} "
                  f"({d.get('latency_ms', 0)}ms)")

        elif event.type == TraceEventType.SUBAGENT_SPAWN:
            print(f"   🤖 派生子Agent: {d.get('task', '')[:80]}")

        elif event.type == TraceEventType.SUBAGENT_RESULT:
            status = "✅" if d.get("success") else "❌"
            print(f"   {status} 子Agent完成 ({d.get('turns', 0)}轮): {d.get('summary', '')[:100]}")

        elif event.type == TraceEventType.CONTEXT_COMPRESS:
            print(f"   📦 上下文压缩: {d.get('before_tokens', 0)} → "
                  f"{d.get('after_tokens', 0)} tokens (-{d.get('reduction_ratio', 0)}%)")

        elif event.type == TraceEventType.FINAL_ANSWER:
            print(f"   ✨ 最终回答 ({d.get('answer_length', 0)} 字符)")

        elif event.type == TraceEventType.SESSION_END:
            print(f"\n{'='*50}")
            s = self.summary()
            print(f"📊 会话统计: {s['total_turns']}轮, {s['llm_calls']}次LLM调用, "
                  f"{s['tool_calls']}次工具调用, {s['total_elapsed_ms']:.0f}ms")
            print(f"{'='*50}")

        elif event.type == TraceEventType.ERROR:
            print(f"   ❌ 错误: {d.get('message', '')}")


# ==================== 全局工厂 ====================

def create_tracer(verbose: bool = True, callbacks: list[TraceCallback] | None = None) -> AgentTracer:
    """创建 Tracer 实例"""
    t = AgentTracer(verbose=verbose)
    for cb in (callbacks or []):
        t.on_event(cb)
    return t
