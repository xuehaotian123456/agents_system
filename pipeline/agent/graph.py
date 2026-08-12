"""LangGraph 四节点图组装 + Checkpointer"""
import json
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState, initial_state
from agent.nodes import planner_node, retriever_node, reflector_node, summarizer_node
from utils.logger_handler import logger

# 全局 checkpointer 实例（支持会话隔离 + 中断恢复）
_memory_saver = MemorySaver()

def _should_search(state: AgentState) -> str:
    plan = state.get("plan", {}) or {}
    suggested = plan.get("suggested_tools", [])
    requires = plan.get("requires_search", True)
    # LLM 有时自相矛盾（requires_search=false 但给了工具列表）→ 有工具就执行
    should = requires or len(suggested) > 0
    logger.info(f"[Route] planner→{'retriever' if should else 'summarizer'} (requires={requires}, tools={suggested})")
    return "retriever" if should else "summarizer"

def _after_retrieve(state: AgentState) -> str:
    # 所有工具失败 → 触发降级标记，跳过 Refector 直接进 Summarizer
    if state.get("degradation_triggered") and not state.get("context", "").strip():
        logger.info("[Route] retriever→summarizer (degradation: 所有工具失败)")
        return "summarizer"
    return "reflector"

def _after_reflect(state: AgentState) -> str:
    if state.get("confidence", 0) >= 0.5 or state.get("retry_count", 0) >= state.get("max_retries", 2):
        return "summarizer"
    return "retriever"

def build_graph(checkpointer=None):
    """构建图（可注入外部 checkpointer，默认 MemorySaver）"""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("reflector", reflector_node)
    builder.add_node("summarizer", summarizer_node)

    builder.set_entry_point("planner")

    # Planner → Retriever 或 Summarizer（不需要搜索时直接回答）
    builder.add_conditional_edges("planner", _should_search, {
        "retriever": "retriever",
        "summarizer": "summarizer",
    })

    # Retriever → Reflector 或 Summarizer（所有工具失败时降级直达）
    builder.add_conditional_edges("retriever", _after_retrieve, {
        "reflector": "reflector",
        "summarizer": "summarizer",
    })

    # Reflector → Summarizer（通过）或 Retriever（重试）
    builder.add_conditional_edges("reflector", _after_reflect, {
        "summarizer": "summarizer",
        "retriever": "retriever",
    })

    builder.add_edge("summarizer", END)

    # ★ 注入 Checkpointer: 支持线程隔离 + 中断恢复
    return builder.compile(checkpointer=checkpointer or _memory_saver)

# 全局图实例（启动时编译，所有请求共用）
graph = build_graph()
