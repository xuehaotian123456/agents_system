"""LangGraph 四节点图组装 + Checkpointer"""
import json
import os
from pathlib import Path
from langgraph.graph import StateGraph, END
from agent.state import AgentState, initial_state
from agent.nodes import (
    planner_node, retriever_node, reflector_node, summarizer_node,
    retriever_vec_node, retriever_bm25_node, retriever_graph_node,
    merge_retrieval_node,
)
from utils.logger_handler import logger

# ── Checkpointer (断点续传) ──
# ★ 2026-08-14: MemorySaver(内存态, 重启即失) → SqliteSaver(跨重启持久化)。
# 断点状态落盘 pipeline/data/langgraph_checkpoints.sqlite,
# thread_id 隔离 + 服务重启后可从上次节点继续执行。
CHECKPOINT_DB = Path(__file__).parent.parent / "data" / "langgraph_checkpoints.sqlite"

_checkpointer = None
_checkpoint_ctx = None

def _get_checkpointer():
    """懒加载 checkpointer: 优先 SqliteSaver(跨重启), 不可用降级 MemorySaver"""
    global _checkpointer, _checkpoint_ctx
    if _checkpointer is not None:
        return _checkpointer
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        # langgraph 1.x: from_conn_string 返回上下文管理器, 进入后保持进程生命周期
        _checkpoint_ctx = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        _checkpointer = _checkpoint_ctx.__enter__()
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
    logger.info(f"[Route] planner→{'检索分支' if should else 'summarizer'} (requires={requires}, tools={suggested})")
    return "retrieval" if should else "summarizer"

def _retrieval_mode(state: AgentState) -> str:
    """
    检索模式路由:
      - parallel: 知识类查询 → 三路并行检索 (Fan-out/Fan-in)
      - tool_exec: 实时/编排类工具 (热榜/对比/爬取) → 顺序工具执行
    """
    plan = state.get("plan", {}) or {}
    suggested = plan.get("suggested_tools", [])
    realtime_tools = {"trending_list", "search_web", "fetch_article",
                      "daily_digest", "trend_report", "compare_tech"}
    if not suggested or "rag_search" in suggested or not any(t in realtime_tools for t in suggested):
        logger.info(f"[Route] 检索模式=parallel (三路并行 Fan-out)")
        return "parallel"
    logger.info(f"[Route] 检索模式=tool_exec (工具编排)")
    return "tool_exec"

def _after_retrieve(state: AgentState) -> str:
    # 所有检索/工具均失败 → 降级直达 Summarizer
    if state.get("degradation_triggered") and not state.get("context", "").strip():
        logger.info("[Route] 检索→summarizer (degradation: 全部失败)")
        return "summarizer"
    return "reflector"

def _after_reflect(state: AgentState) -> str:
    if state.get("confidence", 0) >= 0.5 or state.get("retry_count", 0) >= state.get("max_retries", 2):
        return "summarizer"
    # 重试回到 planner 重新规划 (并行分支重新扇出)
    return "planner"

def build_graph(checkpointer=None):
    """
    构建图（可注入外部 checkpointer，默认 SqliteSaver 跨重启持久化）。

    图结构 (并行 Fan-out/Fan-in):
        planner ──┬→ retriever_vec   ─┐
                  ├→ retriever_bm25  ─┼→ merge_retrieval ─┐
                  └→ retriever_graph ─┘                   ├→ reflector ─┬→ summarizer → END
                       (并行扇出)         (扇入融合)        │              └→ planner (重试)
                                                          └→ retriever (工具编排) ─┘
    """
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever_vec", retriever_vec_node)
    builder.add_node("retriever_bm25", retriever_bm25_node)
    builder.add_node("retriever_graph", retriever_graph_node)
    builder.add_node("merge_retrieval", merge_retrieval_node)
    builder.add_node("retriever", retriever_node)       # 工具编排路径
    builder.add_node("reflector", reflector_node)
    builder.add_node("summarizer", summarizer_node)

    builder.set_entry_point("planner")

    # Planner → 检索分支 / Summarizer
    builder.add_conditional_edges("planner", _should_search, {
        "retrieval": "dispatch",
        "summarizer": "summarizer",
    })

    # 检索模式分派: 并行扇出 vs 工具编排
    builder.add_node("dispatch", lambda s: s)
    builder.add_conditional_edges("dispatch", _retrieval_mode, {
        "parallel": "fanout",
        "tool_exec": "tool_exec_router",
    })
    # ★ Fan-out: 虚拟扇出节点同时派发三路并行检索
    builder.add_node("fanout", lambda s: s)
    builder.add_edge("fanout", "retriever_vec")
    builder.add_edge("fanout", "retriever_bm25")
    builder.add_edge("fanout", "retriever_graph")
    # ★ Fan-in: 三路并行结果汇入融合节点 (RRF + Reranker)
    builder.add_edge("retriever_vec", "merge_retrieval")
    builder.add_edge("retriever_bm25", "merge_retrieval")
    builder.add_edge("retriever_graph", "merge_retrieval")
    builder.add_edge("merge_retrieval", "reflector")

    # 工具编排路径: tool_exec_router 是虚拟节点, 由 retriever 承接
    builder.add_node("tool_exec_router", lambda s: s)
    builder.add_edge("tool_exec_router", "retriever")
    builder.add_conditional_edges("retriever", _after_retrieve, {
        "reflector": "reflector",
        "summarizer": "summarizer",
    })

    # Reflector → Summarizer / 重试回 planner
    builder.add_conditional_edges("reflector", _after_reflect, {
        "summarizer": "summarizer",
        "planner": "planner",
    })

    builder.add_edge("summarizer", END)

    # ★ 注入 Checkpointer: thread_id 隔离 + 跨重启断点恢复
    return builder.compile(checkpointer=checkpointer or _get_checkpointer())

# 全局图实例（启动时编译，所有请求共用）
graph = build_graph()
