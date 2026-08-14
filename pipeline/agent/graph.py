"""LangGraph 四节点图组装 + Checkpointer"""
import json
import os
from pathlib import Path
from langgraph.graph import StateGraph, END
from agent.state import AgentState, initial_state
from agent.nodes import planner_node, retriever_node, reflector_node, summarizer_node
from utils.logger_handler import logger

# ── Checkpointer (断点续传) ──
# ★ 2026-08-14: MemorySaver(内存态, 重启即失) → SqliteSaver(跨重启持久化)。
# 断点状态落盘 pipeline/data/langgraph_checkpoints.sqlite,
# thread_id 隔离 + 服务重启后可从上次节点继续执行。
CHECKPOINT_DB = Path(__file__).parent.parent / "data" / "langgraph_checkpoints.sqlite"

_checkpointer = None

def _get_checkpointer():
    """懒加载 checkpointer: 优先 SqliteSaver(跨重启), 不可用降级 MemorySaver"""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        _checkpointer = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        logger.info(f"[Graph] SqliteSaver checkpointer 就绪 (跨重启持久化): {CHECKPOINT_DB}")
    except Exception as e:
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        logger.warning(f"[Graph] SqliteSaver 不可用({e}), 降级 MemorySaver(仅进程内)")
    return _checkpointer

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
    """构建图（可注入外部 checkpointer，默认 SqliteSaver 跨重启持久化）"""
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

    # ★ 注入 Checkpointer: thread_id 隔离 + 跨重启断点恢复
    return builder.compile(checkpointer=checkpointer or _get_checkpointer())

# 全局图实例（启动时编译，所有请求共用）
graph = build_graph()
