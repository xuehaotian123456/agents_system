"""LangGraph 四节点图组装"""
import json
from langgraph.graph import StateGraph, END
from agent.state import AgentState, initial_state
from agent.nodes import planner_node, retriever_node, reflector_node, summarizer_node
from utils.logger_handler import logger

def _should_search(state: AgentState) -> str:
    plan = state.get("plan", {}) or {}
    suggested = plan.get("suggested_tools", [])
    requires = plan.get("requires_search", True)
    # LLM 有时自相矛盾（requires_search=false 但给了工具列表）→ 有工具就执行
    should = requires or len(suggested) > 0
    logger.info(f"[Route] planner→{'retriever' if should else 'summarizer'} (requires={requires}, tools={suggested})")
    return "retriever" if should else "summarizer"

def _after_retrieve(state: AgentState) -> str:
    return "reflector"

def _after_reflect(state: AgentState) -> str:
    if state.get("confidence", 0) >= 0.5 or state.get("retry_count", 0) >= state.get("max_retries", 2):
        return "summarizer"
    return "retriever"

def build_graph() -> StateGraph:
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

    # Retriever → Reflector
    builder.add_edge("retriever", "reflector")

    # Reflector → Summarizer（通过）或 Retriever（重试）
    builder.add_conditional_edges("reflector", _after_reflect, {
        "summarizer": "summarizer",
        "retriever": "retriever",
    })

    builder.add_edge("summarizer", END)

    return builder.compile()

graph = build_graph()
