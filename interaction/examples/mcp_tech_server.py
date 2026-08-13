"""
MCP 示例 Server — 技术雷达服务 (纯 stdio JSON-RPC, 零依赖)
==========================================================
这是一个完整的 MCP Server 实现，展示如何用 ~150 行纯 Python
实现 MCP 协议 (JSON-RPC 2.0 over stdio)。

协议流程:
  1. client 发送 initialize 请求
  2. server 返回 capabilities + serverInfo
  3. client 发送 notifications/initialized 通知
  4. client 发送 tools/list 获取工具列表
  5. client 发送 tools/call 调用工具

提供的工具:
  - tech_lookup:   查询技术关键词的简介和生态信息
  - tech_compare:  对比两个技术的适用场景
  - version:       查询项目版本号

运行 (MCP 标准方式):
  python examples/mcp_tech_server.py
  # 独立运行时通过 stdin 读取 JSON-RPC 请求，stdout 返回响应

测试:
  echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python examples/mcp_tech_server.py

参考: https://spec.modelcontextprotocol.io/specification/
"""

from __future__ import annotations

import json
import sys

# Windows: 强制 stdout 用 UTF-8（MCP 协议要求，避免 GBK 编码冲突）
if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "tech-radar-mcp"
SERVER_VERSION = "1.0.0"

# ==================== 工具数据 ====================

TECH_DB = {
    "langgraph": {
        "name": "LangGraph",
        "category": "Agent 框架",
        "description": "基于图状态机的 LLM Agent 编排框架，通过 StateGraph 定义节点和边",
        "ecosystem": ["LangChain", "LangSmith", "Checkpointer"],
        "use_case": "多步骤工作流、状态持久化、人工介入",
    },
    "graphrag": {
        "name": "GraphRAG",
        "category": "检索增强",
        "description": "结合知识图谱的 RAG 方案，通过实体关系增强检索召回",
        "ecosystem": ["Neo4j", "ChromaDB", "BM25"],
        "use_case": "实体关联查询、多跳推理、报错溯源",
    },
    "mcp": {
        "name": "MCP (Model Context Protocol)",
        "category": "协议标准",
        "description": "Anthropic 提出的模型上下文协议，标准化 LLM 与外部工具的连接",
        "ecosystem": ["Claude Desktop", "MCP Server", "JSON-RPC"],
        "use_case": "工具生态标准化、跨 Agent 工具复用",
    },
    "a2a": {
        "name": "A2A (Agent-to-Agent)",
        "category": "协议标准",
        "description": "Google 提出的 Agent 间互操作协议，标准化 Agent 能力发现与任务委派",
        "ecosystem": ["Agent Card", "Task API", "SSE"],
        "use_case": "多 Agent 协作、跨框架互操作",
    },
    "chromadb": {
        "name": "ChromaDB",
        "category": "向量数据库",
        "description": "轻量级嵌入式向量数据库，适合单机 RAG 场景",
        "ecosystem": ["LangChain", "LlamaIndex", "Sentence-Transformers"],
        "use_case": "中小规模向量检索、快速原型",
    },
    "reranker": {
        "name": "BGE-Reranker",
        "category": "检索精排",
        "description": "交叉编码器精排模型，对初步召回结果进行语义相关性重排",
        "ecosystem": ["BAAI", "CrossEncoder", "Sentence-Transformers"],
        "use_case": "混合检索精排、提升 Recall@k",
    },
}

COMPARE_PAIRS = {
    ("mcp", "a2a"): "MCP 连接 Agent 与工具（纵向），A2A 连接 Agent 与 Agent（横向）。两者互补：MCP 管工具生态，A2A 管 Agent 协作。",
    ("chromadb", "milvus"): "ChromaDB 嵌入式单机，适合原型和小规模；Milvus 分布式架构，适合生产级大规模向量检索。",
    ("langgraph", "langchain"): "LangChain 是组件库（模型封装/提示词/检索），LangGraph 是编排框架（状态图/循环/持久化）。",
    ("graphrag", "rag"): "传统 RAG 单跳检索语义相似文档；GraphRAG 通过实体关系多跳扩散，能发现隐含关联。",
}


# ==================== MCP 工具定义 ====================

TOOLS = [
    {
        "name": "tech_lookup",
        "description": "查询技术关键词的简介、生态和使用场景。入参 keyword（如 langgraph/graphrag/mcp）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "技术关键词（英文小写）"},
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "tech_compare",
        "description": "对比两个技术的适用场景差异。入参 tech_a, tech_b（如 mcp/a2a）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tech_a": {"type": "string", "description": "技术 A（英文小写）"},
                "tech_b": {"type": "string", "description": "技术 B（英文小写）"},
            },
            "required": ["tech_a", "tech_b"],
        },
    },
    {
        "name": "version",
        "description": "查询 MCP Server 版本信息。无入参",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ==================== 工具实现 ====================

def _impl_tech_lookup(keyword: str) -> str:
    info = TECH_DB.get(keyword.strip().lower())
    if not info:
        available = ", ".join(TECH_DB.keys())
        return f"未找到 '{keyword}'。可查询: {available}"
    return (
        f"{info['name']} [{info['category']}]\n"
        f"简介: {info['description']}\n"
        f"生态: {', '.join(info['ecosystem'])}\n"
        f"适用: {info['use_case']}"
    )


def _impl_tech_compare(tech_a: str, tech_b: str) -> str:
    key = (tech_a.strip().lower(), tech_b.strip().lower())
    if key in COMPARE_PAIRS:
        return COMPARE_PAIRS[key]
    key_r = (key[1], key[0])
    if key_r in COMPARE_PAIRS:
        return COMPARE_PAIRS[key_r]
    return f"暂无 '{tech_a}' vs '{tech_b}' 的对比数据"


def _impl_version() -> str:
    return f"{SERVER_NAME} v{SERVER_VERSION} (MCP {PROTOCOL_VERSION})"


def _call_tool(name: str, arguments: dict) -> list[dict]:
    """执行工具，返回 MCP content 列表"""
    try:
        if name == "tech_lookup":
            text = _impl_tech_lookup(arguments.get("keyword", ""))
        elif name == "tech_compare":
            text = _impl_tech_compare(
                arguments.get("tech_a", ""), arguments.get("tech_b", ""))
        elif name == "version":
            text = _impl_version()
        else:
            return [{"type": "text", "text": f"未知工具: {name}"}], True
        return [{"type": "text", "text": text}], False
    except Exception as e:
        return [{"type": "text", "text": f"工具执行异常: {e}"}], True


# ==================== JSON-RPC 请求处理 ====================

def handle_request(req: dict) -> dict | None:
    """
    处理单个 JSON-RPC 请求。返回 None 表示通知（无需响应）。
    """
    method = req.get("method", "")
    params = req.get("params", {})
    req_id = req.get("id")

    # 通知（无 id）→ 不返回响应
    if req_id is None:
        if method == "notifications/initialized":
            return None  # 握手完成通知，无需响应
        return None

    # ── initialize ──
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    # ── ping ──
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    # ── tools/list ──
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    # ── tools/call ──
    if method == "tools/call":
        content, is_error = _call_tool(params.get("name", ""), params.get("arguments", {}))
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": content, "isError": is_error},
        }

    # ── 未知方法 ──
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    """stdio 模式: 逐行读取 JSON-RPC 请求，逐行写响应"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
