"""
A2A Server — 将 Agent 暴露为 A2A 兼容的服务

让 CC-Harness Agent 可以通过 A2A 协议被其他 Agent 调用。

使用方式:
    # 在 FastAPI 中挂载
    server = A2AServer(agent_loop, tool_registry, name="CC-Harness")
    server.mount(app)  # 自动添加 /.well-known/agent.json + /tasks + /tools/*

    # 或独立启动
    server.serve(port=8080)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional


def create_a2a_card(name: str, description: str, url: str,
                    tools: list[dict], capabilities: dict | None = None) -> dict:
    """创建标准的 A2A Agent Card"""
    return {
        "name": name,
        "description": description,
        "url": url,
        "version": "2.0.0",
        "capabilities": capabilities or {"streaming": True},
        "tools": tools,
    }


class A2AServer:
    """
    A2A Server 端

    将 CC-Harness Agent 的能力通过 A2A 协议暴露出去。
    其他框架的 Agent 可以通过标准协议调用这个 Agent。

    使用方式:
        server = A2AServer(agent_loop, tool_registry,
                          name="DevPilot Research Agent",
                          description="技术调研助手，支持爬虫和知识图谱查询")

        # 方式1: 挂载到已有 FastAPI
        server.mount(app)

        # 方式2: 独立服务
        server.serve(port=8010)
    """

    def __init__(self, name: str = "CC-Harness Agent",
                 description: str = "",
                 agent_runner=None,    # async callable(query) → answer
                 tool_registry=None,
                 tools_metadata: list[dict] | None = None):
        self.name = name
        self.description = description
        self._runner = agent_runner
        self._tool_registry = tool_registry
        self._tools_metadata = tools_metadata or []

    def mount(self, app):
        """挂载到 FastAPI app"""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse

        server = self

        @app.get("/.well-known/agent.json")
        async def agent_card():
            tools = server._tools_metadata
            if server._tool_registry:
                tools = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"type": "object", "properties": t.parameters},
                    }
                    for t in server._tool_registry.list_tools()
                ]
            return create_a2a_card(
                name=server.name,
                description=server.description,
                url=str(app.url_path_for("agent_card")).replace("/.well-known/agent.json", ""),
                tools=tools,
            )

        @app.post("/tasks")
        async def create_task(request: Request):
            body = await request.json()
            query = body.get("query", "")
            use_stream = body.get("stream", True)

            if not use_stream or server._runner is None:
                # 同步模式
                result = await server._runner(query) if server._runner else "Agent not configured"
                answer = result if isinstance(result, str) else result.get("answer", str(result))
                return JSONResponse({"answer": answer, "status": "completed"})

            # 流式模式
            async def event_stream():
                try:
                    yield f"event: status\ndata: {json.dumps({'status': 'started'})}\n\n"

                    if server._runner:
                        result = await server._runner(query)
                        answer = result if isinstance(result, str) else result.get("answer", str(result))
                    else:
                        answer = "Agent not configured"

                    # 模拟流式输出
                    import re
                    chunks = re.split(r'(\s+)', answer)
                    for chunk in chunks:
                        if chunk:
                            yield f"event: text\ndata: {json.dumps({'text': chunk})}\n\n"

                    yield f"event: done\ndata: {json.dumps({'answer': answer, 'status': 'completed'})}\n\n"

                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        @app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            return JSONResponse({"task_id": task_id, "status": "not_implemented"})

        @app.post("/tools/{tool_name}")
        async def call_tool(tool_name: str, request: Request):
            if not server._tool_registry:
                return JSONResponse({"error": "No tool registry"}, status_code=500)

            body = await request.json()
            result = await server._tool_registry.execute(tool_name, body)
            return JSONResponse({
                "success": result.success,
                "content": result.content,
                "error": result.error,
            })

    def serve(self, host: str = "0.0.0.0", port: int = 8010):
        """独立启动 A2A 服务"""
        import uvicorn
        from fastapi import FastAPI
        app = FastAPI(title=self.name)
        self.mount(app)
        uvicorn.run(app, host=host, port=port)
