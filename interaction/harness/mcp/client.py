"""
MCP Client — JSON-RPC 2.0 客户端

MCP (Model Context Protocol) 使用 JSON-RPC 2.0 作为消息协议。
传输层支持两种方式：
1. stdio — 启动子进程，通过标准输入/输出通信
2. HTTP — 通过 HTTP + SSE 通信（Streamable HTTP）

协议方法：
    tools/list        — 列出服务器提供的工具
    tools/call        — 调用工具
    resources/list    — 列出资源
    resources/read    — 读取资源
    prompts/list      — 列出提示词模板
    prompts/get       — 获取提示词模板

参考: https://spec.modelcontextprotocol.io/specification/
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional


# ==================== 数据模型 ====================

@dataclass
class MCPToolDef:
    """MCP Tool 定义（从 tools/list 响应解析）"""
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)  # JSON Schema


@dataclass
class MCPResourceDef:
    """MCP Resource 定义"""
    uri: str
    name: str
    description: str = ""
    mime_type: str = ""


@dataclass
class MCPToolResult:
    """MCP 工具调用结果"""
    content: list[dict] = field(default_factory=list)  # [{"type": "text", "text": "..."}]
    is_error: bool = False


# ==================== JSON-RPC 2.0 协议 ====================

class JSONRPCError(Exception):
    """JSON-RPC 错误"""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC Error {code}: {message}")


class MCPClient:
    """
    MCP JSON-RPC 2.0 客户端

    支持两种传输方式：
    1. stdio — 启动子进程通信
    2. http — HTTP + SSE 通信

    使用方式:
        # stdio 模式
        client = MCPClient.stdio("python", ["mcp_server.py"])
        tools = await client.list_tools()
        result = await client.call_tool("search", {"query": "hello"})

        # HTTP 模式
        client = MCPClient.http("http://localhost:3000/mcp")
        tools = await client.list_tools()
    """

    def __init__(self, transport: str = "stdio"):
        self.transport = transport
        self._request_id = 0
        self._initialized = False
        self._server_capabilities: dict = {}

        # stdio 模式
        self._process: subprocess.Popen | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

        # HTTP 模式
        self._base_url: str = ""
        self._session = None  # aiohttp session

        # 缓存
        self._tools_cache: list[MCPToolDef] | None = None
        self._resources_cache: list[MCPResourceDef] | None = None

    # ==================== 工厂方法 ====================

    @classmethod
    def stdio(cls, command: str, args: list[str] | None = None,
              env: dict | None = None) -> "MCPClient":
        """创建 stdio 传输的 MCP Client"""
        client = cls(transport="stdio")
        client._command = command
        client._args = args or []
        client._env = {**os.environ, **(env or {})}
        return client

    @classmethod
    def http(cls, base_url: str, headers: dict | None = None) -> "MCPClient":
        """创建 HTTP 传输的 MCP Client"""
        client = cls(transport="http")
        client._base_url = base_url.rstrip("/")
        client._headers = headers or {}
        return client

    # ==================== 连接管理 ====================

    async def connect(self):
        """建立连接并完成 MCP 初始化握手"""
        if self._initialized:
            return

        if self.transport == "stdio":
            await self._connect_stdio()
        elif self.transport == "http":
            await self._connect_http()

        # MCP 初始化握手
        try:
            result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},       # 声明客户端支持工具调用
                    "resources": {},   # 声明客户端支持资源访问
                },
                "clientInfo": {
                    "name": "cc-harness-agent",
                    "version": "2.0.0",
                },
            })
            self._server_capabilities = result.get("capabilities", {})
            self._initialized = True

            # 发送 initialized 通知
            await self._send_notification("notifications/initialized", {})

        except Exception as e:
            await self.disconnect()
            raise RuntimeError(f"MCP 初始化失败: {e}")

    async def disconnect(self):
        """断开连接"""
        self._initialized = False
        self._tools_cache = None
        self._resources_cache = None

        if self.transport == "stdio":
            if self._writer:
                self._writer.close()
                await self._writer.wait_closed()
            if self._process:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()

        elif self.transport == "http":
            if self._session:
                await self._session.close()

    async def _connect_stdio(self):
        """建立 stdio 连接"""
        self._process = subprocess.Popen(
            [self._command] + self._args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
        )
        # 注意：asyncio subprocess 更合适，这里简化处理
        # 生产环境应使用 asyncio.create_subprocess_exec

    async def _connect_http(self):
        """建立 HTTP 连接"""
        try:
            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={**self._headers, "Content-Type": "application/json"},
                timeout=httpx.Timeout(60.0),
            )
        except ImportError:
            raise ImportError("HTTP 模式需要 httpx 库: pip install httpx")

    # ==================== 核心协议方法 ====================

    async def list_tools(self) -> list[MCPToolDef]:
        """获取 MCP Server 提供的工具列表"""
        if self._tools_cache is not None:
            return self._tools_cache

        result = await self._send_request("tools/list", {})
        tools_data = result.get("tools", [])

        tools = []
        for t in tools_data:
            tools.append(MCPToolDef(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            ))

        self._tools_cache = tools
        return tools

    async def call_tool(self, name: str, arguments: dict) -> MCPToolResult:
        """调用 MCP 工具"""
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        content = result.get("content", [])
        is_error = result.get("isError", False)

        return MCPToolResult(content=content, is_error=is_error)

    async def list_resources(self) -> list[MCPResourceDef]:
        """获取资源列表"""
        if self._resources_cache is not None:
            return self._resources_cache

        result = await self._send_request("resources/list", {})
        resources_data = result.get("resources", [])

        resources = []
        for r in resources_data:
            resources.append(MCPResourceDef(
                uri=r["uri"],
                name=r.get("name", ""),
                description=r.get("description", ""),
                mime_type=r.get("mimeType", ""),
            ))

        self._resources_cache = resources
        return resources

    async def read_resource(self, uri: str) -> dict:
        """读取 MCP 资源"""
        return await self._send_request("resources/read", {"uri": uri})

    # ==================== JSON-RPC 通信 ====================

    async def _send_request(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并等待响应"""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        if self.transport == "stdio":
            return await self._stdio_request(request)
        elif self.transport == "http":
            return await self._http_request(request)
        else:
            raise ValueError(f"不支持的传输方式: {self.transport}")

    async def _send_notification(self, method: str, params: dict):
        """发送 JSON-RPC 通知（无需响应）"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        if self.transport == "stdio":
            self._stdio_send(notification)
        elif self.transport == "http":
            await self._http_request(notification)

    # ── stdio 传输 ──

    async def _stdio_request(self, request: dict) -> dict:
        """通过 stdio 发送请求"""
        self._stdio_send(request)

        # 读取响应
        if self._process and self._process.stdout:
            line = self._process.stdout.readline()
            if not line:
                raise ConnectionError("MCP Server 连接已关闭")

            response = json.loads(line.decode("utf-8").strip())

            if "error" in response:
                err = response["error"]
                raise JSONRPCError(err["code"], err["message"], err.get("data"))

            return response.get("result", {})
        raise ConnectionError("MCP Server 进程未启动")

    def _stdio_send(self, message: dict):
        """通过 stdio 发送消息"""
        if self._process and self._process.stdin:
            payload = json.dumps(message, ensure_ascii=False) + "\n"
            self._process.stdin.write(payload.encode("utf-8"))
            self._process.stdin.flush()

    # ── HTTP 传输 ──

    async def _http_request(self, request: dict) -> dict:
        """通过 HTTP 发送 JSON-RPC 请求"""
        if not hasattr(self, '_http_client'):
            raise RuntimeError("HTTP 客户端未初始化，请先调用 connect()")

        resp = await self._http_client.post("/mcp", json=request)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            err = data["error"]
            raise JSONRPCError(err["code"], err["message"], err.get("data"))

        return data.get("result", {})

    # ==================== 查询 ====================

    @property
    def server_capabilities(self) -> dict:
        """MCP Server 的能力声明"""
        return self._server_capabilities

    @property
    def supports_tools(self) -> bool:
        return "tools" in self._server_capabilities

    @property
    def supports_resources(self) -> bool:
        return "resources" in self._server_capabilities

    async def ping(self) -> bool:
        """检查 MCP Server 是否存活"""
        try:
            await self._send_request("ping", {})
            return True
        except Exception:
            return False
