"""
CC-Harness Agent — FastAPI Server
==================================
生产级异步 API 服务，支持:
- REST API (同步问答)
- SSE 流式推送 (逐 token 实时输出)
- WebSocket 双向通信
- 会话管理
- 健康检查 + 统计端点

启动方式:
    uvicorn server.app:app --host 0.0.0.0 --port 8020 --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

# Windows GBK 编码修复
if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from harness import (
    AgentLoop, Session, LLMAdapter, PromptEngine, AgentConfig,
    AgentTracer, create_tracer, create_llm_adapter,
)
from tools import ToolRegistry, RAGTool

load_dotenv()


# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    model: str = "qwen-plus"
    max_turns: int = 10
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    turns: int
    tool_calls: int
    elapsed_ms: float
    trace: dict | None = None


class EvalRequest(BaseModel):
    questions: list[dict]        # [{"question": "...", "ground_truth": "..."}]
    model: str = "qwen-plus"


# ==================== 全局状态 ====================

class AppState:
    """应用全局状态（在 lifespan 中初始化）"""

    def __init__(self):
        self.llm: LLMAdapter | None = None
        self.tool_registry: ToolRegistry | None = None
        self.prompt_engine: PromptEngine | None = None
        self.sessions: dict[str, Session] = {}  # session_id → Session
        self.tracers: dict[str, AgentTracer] = {}  # session_id → Tracer
        self.sessions_dir = Path(__file__).parent.parent / "data" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create_session(self, session_id: str | None, config: AgentConfig) -> tuple[str, Session, AgentTracer]:
        """获取或创建 Session"""
        sid = session_id or str(uuid.uuid4())[:8]

        # 尝试从内存获取
        if sid in self.sessions:
            return sid, self.sessions[sid], self.tracers[sid]

        # 尝试从磁盘加载
        loaded = self._load_from_disk(sid, config)
        if loaded:
            return loaded

        # 创建新会话
        session = Session(config=config)
        session.set_system_prompt(self.prompt_engine.build_system_prompt(config))
        tracer = create_tracer(verbose=False)
        self.sessions[sid] = session
        self.tracers[sid] = tracer
        return sid, session, tracer

    def save_session(self, sid: str):
        """持久化会话到磁盘"""
        if sid not in self.sessions:
            return
        session = self.sessions[sid]
        fpath = self.sessions_dir / f"{sid}.json"
        data = {
            "session_id": sid,
            "messages": [
                {"role": m.role.value, "content": m.content, "tool_name": m.tool_name}
                for m in session.messages
            ],
            "total_turns": session.total_turns,
            "tool_calls_count": session.tool_calls_count,
            "last_active": session.last_active,
            "first_query": next((m.content[:100] for m in session.messages if m.role.value == "user"), ""),
        }
        fpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_from_disk(self, sid: str, config: AgentConfig) -> tuple[str, Session, AgentTracer] | None:
        """从磁盘恢复会话"""
        fpath = self.sessions_dir / f"{sid}.json"
        if not fpath.exists():
            return None
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            session = Session(config=config)
            session.set_system_prompt(self.prompt_engine.build_system_prompt(config))
            for m in data.get("messages", []):
                from harness.types import Message, MessageRole
                role = MessageRole(m["role"])
                msg = Message(role=role, content=m["content"], tool_name=m.get("tool_name"))
                session.messages.append(msg)
            session.total_turns = data.get("total_turns", 0)
            session.tool_calls_count = data.get("tool_calls_count", 0)
            session.last_active = data.get("last_active", time.time())
            tracer = create_tracer(verbose=False)
            self.sessions[sid] = session
            self.tracers[sid] = tracer
            return sid, session, tracer
        except Exception:
            return None

    def list_sessions(self) -> list[dict]:
        """列出所有会话（从磁盘扫描）"""
        result = []
        for fpath in sorted(self.sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                result.append({
                    "session_id": data.get("session_id", fpath.stem),
                    "first_query": data.get("first_query", "")[:80],
                    "last_active": data.get("last_active", 0),
                    "turns": data.get("total_turns", 0),
                    "msg_count": len(data.get("messages", [])),
                })
            except Exception:
                pass
        return result

    def cleanup_old_sessions(self, max_age_sec: float = 3600):
        """清理过期会话"""
        now = time.time()
        expired = [
            sid for sid, s in self.sessions.items()
            if now - s.last_active > max_age_sec
        ]
        for sid in expired:
            del self.sessions[sid]
            del self.tracers[sid]


state = AppState()


# ==================== 生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭"""
    # 启动
    print("🚀 CC-Harness Agent Server 启动中...")

    # 初始化 LLM
    state.llm = create_llm_adapter(
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        fallback_models=["qwen-turbo"],
    )
    print(f"   LLM: {state.llm.model} ({state.llm.provider.value})")

    # 初始化工具注册表
    state.tool_registry = ToolRegistry()
    rag_tool = RAGTool(collection_name="cc_harness_server", persist_dir="./chroma_db", k=3)
    state.tool_registry.register(rag_tool)

    # 从 Pipeline 文章目录加载真实文章到本地 RAG
    pipeline_articles_dir = Path(__file__).parent.parent.parent / "pipeline" / "data" / "articles"
    rag_doc_count = 0
    if pipeline_articles_dir.exists():
        article_files = sorted(pipeline_articles_dir.glob("*.md"))
        # 分块加载：每块 ≤ 800 字符（embedding API 限 2048 tokens）
        chunks = []
        for fpath in article_files:
            try:
                text = fpath.read_text(encoding="utf-8")
                if len(text) < 100:
                    continue
                # 简单分块：按段落拆，每块最多 800 字符
                paras = text.split("\n\n")
                current = ""
                for p in paras:
                    p = p.strip()
                    if not p:
                        continue
                    if len(current) + len(p) < 800:
                        current += p + "\n\n"
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        current = p + "\n\n"
                if current.strip():
                    chunks.append(current.strip())
            except Exception:
                continue

        # 批量导入（RAGTool 内部会自动 embedding 并存入 ChromaDB）
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            rag_tool.add_documents(batch)
            rag_doc_count += len(batch)

        print(f"   RAG: {rag_doc_count} chunks 就绪 (来自 {len(article_files)} 篇 Pipeline 文章)")
    else:
        # Fallback
        test_docs = ["Agentic RAG 基于智能体实现动态检索"]
        rag_tool.add_documents(test_docs)
        print(f"   RAG: {len(test_docs)} 篇测试文档就绪 (Pipeline 文章目录未找到)")

    # 初始化 Prompt Engine
    state.prompt_engine = PromptEngine(state.tool_registry)

    # ── 自动发现 Pipeline A2A 工具 ──
    devpilot_url = os.getenv("DEVPILOT_A2A_URL", "http://localhost:8010")
    a2a_tools_registered = 0
    try:
        from harness.a2a.client import A2AClient
        a2a_client = A2AClient(devpilot_url, timeout=60)
        remote_agent = await a2a_client.discover()
        for tool_def in remote_agent.tools:
            # 跳过 Harness 本地已有的同名工具
            if state.tool_registry.get(tool_def.name):
                continue

            # 动态创建 A2A 工具适配器
            # 注意: 必须用工厂函数捕获 tool_def，避免 Python 循环闭包陷阱
            # (所有适配器会引用最后一个 tool_def)
            def _make_adapter(td, client):
                from tools.base import BaseTool
                from harness.types import ToolResult

                class _Adapter(BaseTool):
                    name = td.name
                    description = td.description

                    async def execute(self, **kwargs):
                        try:
                            result = await client.call_tool(td.name, kwargs)
                            if result.get("success"):
                                content = result.get("content", "")
                                print(f"[A2A-ADAPTER] {td.name} SUCCESS len={len(content)} preview={content[:100]}")
                                return ToolResult(tool_name=td.name, success=True, content=content)
                            err = result.get("error") or result.get("detail") or "未知错误(空响应)"
                            print(f"[A2A-ADAPTER] {td.name} FAILED: {err[:200]}")
                            return ToolResult(tool_name=td.name, success=False, content="", error=err)
                        except Exception as e:
                            import traceback
                            print(f"[A2A-ADAPTER] {td.name} EXCEPTION: {e}")
                            traceback.print_exc()
                            return ToolResult(tool_name=td.name, success=False, content="", error=f"{type(e).__name__}: {e}")

                return _Adapter()

            adapter = _make_adapter(tool_def, a2a_client)
            state.tool_registry.register(adapter)
            a2a_tools_registered += 1

        print(f"   A2A: {a2a_tools_registered} 个远程工具已注册 (from {devpilot_url})")
        for t in remote_agent.tools:
            print(f"      - {t.name}: {t.description[:60]}...")
        state._a2a_client = a2a_client  # 保持连接
    except Exception as e:
        print(f"   A2A: DevPilot 未连接 ({e}) — 仅本地工具可用")

    # ── 自动发现 MCP Server（Model Context Protocol）──
    mcp_tools_registered = 0
    mcp_servers_json = os.getenv("MCP_SERVERS", "")
    if not mcp_servers_json:
        # 默认配置：本地文件系统 MCP + 网络搜索 MCP
        mcp_servers_json = json.dumps([
            {"name": "filesystem", "transport": "stdio",
             "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        ])
    try:
        mcp_configs = json.loads(mcp_servers_json) if isinstance(mcp_servers_json, str) else mcp_servers_json
        from harness.mcp import MCPServerRegistry, MCPServerConfig, register_mcp_tools

        mcp_registry = MCPServerRegistry()
        for cfg in mcp_configs:
            if isinstance(cfg, dict):
                mcp_registry.register(MCPServerConfig(
                    name=cfg.get("name", "mcp"),
                    transport=cfg.get("transport", "stdio"),
                    command=cfg.get("command", ""),
                    args=cfg.get("args", []),
                    url=cfg.get("url", cfg.get("base_url", "")),
                ))

        await mcp_registry.connect_all()
        mcp_tools_registered = await register_mcp_tools(state.tool_registry, mcp_registry)
        state._mcp_registry = mcp_registry  # 保持连接

        print(f"   MCP: {mcp_tools_registered} 个外部工具已注册 (from {len(mcp_configs)} MCP Servers)")
    except Exception as e:
        print(f"   MCP: 未启用 ({e}) — MCP Server 不可用，不影响核心功能")

    print(f"✅ Server 就绪 — http://0.0.0.0:8020")
    print(f"   Web UI:  http://0.0.0.0:8020")
    print(f"   API 文档: http://0.0.0.0:8020/docs")

    yield

    # 关闭
    print("👋 CC-Harness Agent Server 关闭")


# ==================== FastAPI App ====================

app = FastAPI(
    title="CC-Harness Agent API",
    description="基于 Claude Code Harness 架构的轻量级 Agent 框架 API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
# 注意: allow_origins=["*"] 与 allow_credentials=True 是非法组合
# (浏览器会拒绝带凭证的跨域请求)。本地开发场景无需凭证，禁用 credentials。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 静态文件 (前端 UI)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """CC-Harness Agent Web UI"""
    from fastapi.responses import Response
    index_path = static_dir / "index.html"
    if index_path.exists():
        html = index_path.read_text(encoding="utf-8")
        return Response(content=html, media_type="text/html",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return "<h1>CC-Harness Agent API</h1><p>访问 <a href='/docs'>/docs</a> 查看 API 文档</p>"


# ==================== 健康检查 ====================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": state.llm.model if state.llm else "not initialized",
        "tools": state.tool_registry.list_tool_names() if state.tool_registry else [],
        "active_sessions": len(state.sessions),
    }


@app.get("/stats")
async def stats():
    """LLM 用量统计"""
    if state.llm:
        return state.llm.get_usage_summary()
    return {"error": "LLM not initialized"}


# ==================== 会话历史 API ====================

@app.get("/api/sessions")
async def list_sessions():
    """列出所有持久化的会话"""
    return {"sessions": state.list_sessions()}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取指定会话的详细信息（含完整消息历史）"""
    fpath = state.sessions_dir / f"{session_id}.json"
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="会话不存在")
    data = json.loads(fpath.read_text(encoding="utf-8"))
    return {
        "session_id": data.get("session_id"),
        "messages": [
            {"role": m["role"], "content": m["content"], "tool_name": m.get("tool_name")}
            for m in data.get("messages", [])
        ],
        "total_turns": data.get("total_turns", 0),
        "last_active": data.get("last_active", 0),
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    fpath = state.sessions_dir / f"{session_id}.json"
    if fpath.exists():
        fpath.unlink()
    if session_id in state.sessions:
        del state.sessions[session_id]
    if session_id in state.tracers:
        del state.tracers[session_id]
    return {"ok": True}


# ==================== 工具辅助 ====================

def _append_graph_markers(session, answer: str) -> str:
    """如果会话中的工具结果包含 [GRAPH:...] 或 [VIZ:...] 标记，自动追加到回答末尾"""
    import re
    markers = []
    for msg in session.messages:
        found = re.findall(r'\[(?:GRAPH|VIZ):[^\]]+\]', msg.content)
        if found:
            print(f"[GraphMarkers] 在消息中找到: {found}")
        markers.extend(found)
    if markers:
        print(f"[GraphMarkers] 追加 {len(markers)} 个标记")
        for m in markers:
            if m not in answer:
                answer += "\n\n" + m
    else:
        print(f"[GraphMarkers] 未找到标记 ({len(session.messages)} 条消息)")
    return answer


# ==================== REST Chat API ====================

@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """同步问答（非流式）"""
    config = AgentConfig(
        max_turns=req.max_turns,
        model=req.model,
        temperature=0.0,
        enable_subagents=False,
    )

    sid, session, tracer = state.get_or_create_session(req.session_id, config)
    session.append_user_message(req.query)

    loop = AgentLoop(
        session=session,
        llm_adapter=state.llm,
        tool_registry=state.tool_registry,
        prompt_engine=state.prompt_engine,
        tracer=tracer,
    )

    t_start = time.time()
    answer = await loop.run()
    elapsed = (time.time() - t_start) * 1000

    # 自动追加工具结果中的 GRAPH/VIZ 标记（LLM 常漏掉）
    answer = _append_graph_markers(session, answer)

    state.save_session(sid)  # 持久化
    return ChatResponse(
        answer=answer,
        session_id=sid,
        turns=session.total_turns,
        tool_calls=tracer.tool_call_count,
        elapsed_ms=round(elapsed, 0),
        trace=tracer.to_dict(),
    )


# ==================== SSE 流式 Chat API ====================

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式问答"""
    config = AgentConfig(
        max_turns=req.max_turns,
        model=req.model,
        temperature=0.0,
        enable_subagents=False,
    )

    sid, session, tracer = state.get_or_create_session(req.session_id, config)
    session.append_user_message(req.query)

    async def event_stream() -> AsyncIterator[str]:
        """SSE 事件流生成器"""
        loop = AgentLoop(
            session=session,
            llm_adapter=state.llm,
            tool_registry=state.tool_registry,
            prompt_engine=state.prompt_engine,
            tracer=tracer,
        )

        # 注册实时回调 → SSE event
        event_queue: asyncio.Queue = asyncio.Queue()

        def on_trace_event(event):
            """将 AgentTracer 事件转为 SSE 事件"""
            asyncio.ensure_future(event_queue.put(event.to_dict()))

        tracer.on_event(on_trace_event)

        # 发送 session_start
        yield f"event: session\ndata: {json.dumps({'session_id': sid, 'model': req.model}, ensure_ascii=False)}\n\n"

        # 后台执行 AgentLoop
        async def run_agent():
            try:
                answer = await loop.run()
                answer = _append_graph_markers(session, answer)
                state.save_session(sid)  # 持久化
                await event_queue.put({"type": "done", "answer": answer, "session_id": sid})
            except Exception as e:
                await event_queue.put({"type": "error", "message": str(e)})

        task = asyncio.ensure_future(run_agent())

        # 流式推送事件
        while True:
            try:
                event_data = await asyncio.wait_for(event_queue.get(), timeout=300)
            except asyncio.TimeoutError:
                yield f"event: timeout\ndata: {json.dumps({'message': '请求超时'})}\n\n"
                break

            event_type = event_data.get("type", "unknown")

            if event_type == "done":
                yield f"event: done\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                break
            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                break
            elif event_type in ("llm_thinking", "llm_action"):
                yield f"event: thinking\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            elif event_type in ("tool_call", "tool_result"):
                yield f"event: tool\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            elif event_type == "final_answer":
                yield f"event: answer\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            else:
                yield f"event: trace\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"

        await task
        tracer.remove_callback(on_trace_event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== WebSocket ====================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    """WebSocket 双向通信"""
    await ws.accept()
    await ws.send_json({"type": "connected", "session_id": session_id})

    config = AgentConfig(max_turns=10, model="qwen-plus")
    sid, session, tracer = state.get_or_create_session(session_id, config)

    try:
        while True:
            data = await ws.receive_json()
            query = data.get("query", "").strip()

            if not query:
                continue
            if query in ("/exit", "/quit"):
                await ws.send_json({"type": "bye"})
                break

            # 发送新问题
            session.append_user_message(query)

            loop = AgentLoop(
                session=session,
                llm_adapter=state.llm,
                tool_registry=state.tool_registry,
                prompt_engine=state.prompt_engine,
                tracer=tracer,
            )

            # 注册实时回调
            def on_event(event):
                asyncio.ensure_future(ws.send_json(event.to_dict()))

            tracer.on_event(on_event)

            answer = await loop.run()

            tracer.remove_callback(on_event)

            await ws.send_json({
                "type": "done",
                "answer": answer,
                "session_id": sid,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})


# ==================== 评测端点 ====================

@app.post("/api/eval")
async def eval_endpoint(req: EvalRequest):
    """批量评测"""
    if not state.llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    from harness.eval import BenchmarkSuite, QAPair, EvalRunner, LLMJudge

    suite = BenchmarkSuite("api-eval")
    for i, q in enumerate(req.questions):
        suite.add(QAPair(
            id=f"q{i+1}",
            question=q.get("question", ""),
            ground_truth=q.get("ground_truth", ""),
        ))

    async def agent_factory():
        config = AgentConfig(max_turns=5, model=req.model)
        session = Session(config=config)
        session.set_system_prompt(state.prompt_engine.build_system_prompt(config))
        return session, state.llm, state.tool_registry

    judge = LLMJudge(state.llm) if state.llm else None
    runner = EvalRunner(suite, agent_factory, llm_judge=judge)
    result = await runner.run()

    return result.to_dict()


# ==================== 工具端点 ====================

@app.get("/api/tools")
async def list_tools():
    """列出所有已注册工具"""
    tools = state.tool_registry.list_tools() if state.tool_registry else []
    return {
        "tools": [t.to_dict() for t in tools],
        "count": len(tools),
    }


# ==================== 会话端点 ====================

@app.get("/api/sessions")
async def list_sessions():
    """列出活跃会话"""
    return {
        "count": len(state.sessions),
        "sessions": [
            {
                "id": sid,
                "turns": s.total_turns,
                "messages": len(s.messages),
                "tokens": s.estimate_total_tokens(),
            }
            for sid, s in state.sessions.items()
        ],
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id in state.sessions:
        del state.sessions[session_id]
        if session_id in state.tracers:
            del state.tracers[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8020,
        reload=True,
        log_level="info",
    )
