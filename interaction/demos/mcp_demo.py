"""
MCP 协议端到端 Demo — Agent 通过 MCP 连接外部工具服务器
========================================================
演示完整链路:
  1. 启动 MCP Server 子进程 (stdio 传输)
  2. MCPClient 建立连接 + initialize 握手
  3. tools/list 发现服务器工具
  4. tools/call 调用工具
  5. MCP 工具注册进 CC-Harness ToolRegistry (与本地工具统一管理)

运行:
    cd E:/agent-system/interaction
    python demos/mcp_demo.py

面试话术:
    "我实现了 MCP 协议的完整链路 —— 用纯 Python 写了 MCP Server
    (JSON-RPC 2.0 over stdio) 和 MCP Client (asyncio 子进程 + 握手 +
    工具发现 + 调用路由)，MCP 工具通过 Adapter 模式无缝接入 Agent
    的 ToolRegistry，与本地工具统一调度。"
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows GBK 修复
if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

SEP = "=" * 60


async def demo_mcp_end_to_end():
    from harness.mcp.client import MCPClient
    from harness.mcp.registry import MCPServerRegistry, MCPServerConfig
    from harness.mcp.adapter import register_mcp_tools

    # ==================== 1. 启动 MCP Server ====================
    print(f"\n{SEP}")
    print("  1. 注册 MCP Server (stdio 传输)")
    print(f"{SEP}")

    server_script = str(Path(__file__).parent.parent / "examples" / "mcp_tech_server.py")

    registry = MCPServerRegistry()
    registry.register(MCPServerConfig(
        name="tech_radar",
        description="技术雷达 MCP 服务 — 提供技术查询/对比工具",
        transport="stdio",
        command=sys.executable,          # python
        args=[server_script],            # examples/mcp_tech_server.py
        tool_prefix="",                  # 不加前缀
    ))

    # ==================== 2. 连接 + 握手 ====================
    print(f"\n{SEP}")
    print("  2. 连接 + MCP initialize 握手")
    print(f"{SEP}")

    await registry.connect_all()
    status = registry.status()

    for name, s in status.items():
        state = "✅ 已连接" if s["connected"] else f"❌ 失败: {s['error']}"
        print(f"  {name}: {state}")

    if registry.connected_count == 0:
        print("\n  [失败] 无法连接 MCP Server，请检查 examples/mcp_tech_server.py")
        return

    # ==================== 3. 工具发现 ====================
    print(f"\n{SEP}")
    print("  3. tools/list — 发现 MCP 工具")
    print(f"{SEP}")

    all_tools = registry.get_all_tools()
    for t in all_tools:
        print(f"  - {t.name}: {t.description[:60]}")

    # ==================== 4. 工具调用 ====================
    print(f"\n{SEP}")
    print("  4. tools/call — 调用 MCP 工具")
    print(f"{SEP}")

    # 调用 1: tech_lookup
    result = await registry.call_tool("tech_lookup", {"keyword": "graphrag"})
    print(f"  [tech_lookup('graphrag')]")
    print(f"  {result['content']}")
    print()

    # 调用 2: tech_compare
    result = await registry.call_tool("tech_compare", {"tech_a": "mcp", "tech_b": "a2a"})
    print(f"  [tech_compare('mcp', 'a2a')]")
    print(f"  {result['content']}")
    print()

    # 调用 3: version
    result = await registry.call_tool("version", {})
    print(f"  [version()]")
    print(f"  {result['content']}")

    # ==================== 5. 注册进 ToolRegistry ====================
    print(f"\n{SEP}")
    print("  5. MCP 工具 → CC-Harness ToolRegistry")
    print(f"{SEP}")

    from tools.registry import ToolRegistry
    tool_registry = ToolRegistry()

    count = await register_mcp_tools(tool_registry, registry)
    print(f"  已注册 {count} 个 MCP 工具到 ToolRegistry")

    # 展示与本地工具的统一调用
    all_local = tool_registry.list_tools() if hasattr(tool_registry, 'list_tools') else []
    local_names = [t.name if hasattr(t, 'name') else str(t) for t in all_local]
    print(f"  ToolRegistry 当前工具: {local_names}")

    # ==================== 6. 断开 ====================
    print(f"\n{SEP}")
    print("  6. 断开连接")
    print(f"{SEP}")
    await registry.disconnect_all()
    print("  ✅ 已断开，MCP Server 子进程已终止")


async def main():
    print(f"\n{'='*60}")
    print("  MCP (Model Context Protocol) 端到端演示")
    print("  Agent ↔ MCP Server 工具互操作")
    print(f"{'='*60}")

    await demo_mcp_end_to_end()

    print(f"\n{'='*60}")
    print("  演示完成。技术栈:")
    print("  - JSON-RPC 2.0 over stdio (MCP 标准传输)")
    print("  - initialize 握手 + 能力协商")
    print("  - tools/list 工具发现 + tools/call 调用")
    print("  - Adapter 模式接入 Agent ToolRegistry")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
