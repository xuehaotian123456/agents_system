"""AgentState — LangGraph 状态定义"""
from typing import TypedDict, Annotated, Optional, List, Any
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    original_query: str
    plan: Optional[dict]
    retrieval_rounds: int
    tool_calls: list
    context: str
    answer: Optional[str]
    confidence: float
    retry_count: int
    max_retries: int
    session_id: str
    errors: list[dict]                # ★ 新增: 错误追踪 [{"node", "tool", "error", "timestamp"}]
    degradation_triggered: bool       # ★ 新增: 是否触发降级策略
    # 并行检索三路结果 (可序列化, checkpoint 兼容; 并行节点各写各的 key)
    vec_docs: list[dict]              # 向量路 [{content, source, credibility, ...}]
    bm25_docs: list[dict]             # BM25 路
    graph_docs: list[dict]            # 图路

def initial_state(query: str, session_id: str = "default") -> AgentState:
    return {
        "messages": [],
        "query": query,
        "original_query": query,
        "plan": None,
        "retrieval_rounds": 0,
        "tool_calls": [],
        "context": "",
        "answer": None,
        "confidence": 0.0,
        "retry_count": 0,
        "max_retries": 2,
        "session_id": session_id,
        "errors": [],
        "degradation_triggered": False,
        "vec_docs": [],
        "bm25_docs": [],
        "graph_docs": [],
    }
