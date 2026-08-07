"""
CC-Harness 技术调研 Agent — 端到端 Demo
=========================================
展示 CC-Harness 作为技术调研 Agent 的完整能力：

架构:
    用户提问
        │
        ▼
    CC-Harness Agent (主控, AgentLoop 动态决策)
        │
        ├── 本地工具: rag_search
        ├── A2A → DevPilot Agent (爬虫 + KG)
        │      ├── trending_list (掘金热榜)
        │      ├── kg_lookup (知识图谱)
        │      ├── fetch_article (爬文章)
        │      ├── compare_tech (技术对比)
        │      └── code_example (代码搜索)
        │
        ├── 可选: Map-Reduce 多 Agent 分工
        │
        └── → 生成结构化调研报告

启动方式:
    # 终端 1: 启动 DevPilot A2A Server
    cd tech_agent
    python a2a_server.py --port 8010

    # 终端 2: 启动 CC-Harness API
    cd cc-harness-agent
    uvicorn server.app:app --port 8020

    # 终端 3: 运行 Demo
    cd cc-harness-agent
    python demos/research_demo.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def demo_a2a_discovery():
    """Demo 1: A2A 发现远程 Agent"""
    from harness.a2a import A2AClient

    print("=" * 60)
    print("🔍 Demo 1: A2A 发现 DevPilot Agent")
    print("=" * 60)

    client = A2AClient("http://localhost:8010")

    try:
        agent = await client.discover()
        print(f"  Agent: {agent.name}")
        print(f"  描述: {agent.description[:100]}")
        print(f"  工具数: {len(agent.tools)}")
        for t in agent.tools:
            print(f"    - {t.name}: {t.description[:60]}")
        print("  ✅ A2A 连接成功!")
    except Exception as e:
        print(f"  ⚠️ DevPilot A2A 未启动: {e}")
        print("  请先运行: cd tech_agent && python a2a_server.py --port 8010")

    await client.close()


async def demo_direct_tool_call():
    """Demo 2: 通过 A2A 直接调用 DevPilot 工具"""
    from harness.a2a import A2AClient

    print("\n" + "=" * 60)
    print("🔧 Demo 2: A2A 直接调用 DevPilot 工具")
    print("=" * 60)

    client = A2AClient("http://localhost:8010")

    try:
        # 调用热榜
        print("\n  📋 调用 trending_list...")
        result = await client.call_tool("trending_list", {"source": "juejin"})
        content = result.get("content", "")
        if content:
            print(f"  返回: {content[:200]}...")
            print("  ✅ 工具调用成功!")
        else:
            print(f"  ⚠️ 返回空: {result}")

        # 调用 KG
        print("\n  🕸️ 调用 kg_lookup(Python)...")
        result = await client.call_tool("kg_lookup", {"entity_name": "Python"})
        content = result.get("content", "")
        if content:
            print(f"  返回: {content[:200]}...")
            print("  ✅ KG查询成功!")
        else:
            print(f"  ⚠️ 返回空: {result}")

    except Exception as e:
        print(f"  ⚠️ 调用失败: {e}")

    await client.close()


async def demo_research_agent():
    """Demo 3: 完整技术调研流程"""
    from harness.agent_profile import AgentProfile, AgentBuilder

    print("\n" + "=" * 60)
    print("📊 Demo 3: 完整技术调研 Agent")
    print("=" * 60)

    try:
        profile = AgentProfile.from_yaml("profiles/tech_researcher.yaml")
        agent = await AgentBuilder.build(profile)
        print(f"  Agent: {agent.profile.name}")
        print(f"  技能: {agent.profile.skills}")
        print(f"  状态: {agent.status()}")
    except Exception as e:
        print(f"  ⚠️ Profile 构建失败: {e}")
        return

    # 调研问题
    questions = [
        "帮我调研一下 Rust 在 2026 年前端工具链领域的最新进展，给出关键项目和趋势分析",
    ]

    for q in questions:
        print(f"\n{'─' * 50}")
        print(f"❓ 调研问题: {q}")
        print("─" * 50)

        try:
            answer = await agent.run(q)
            print(f"\n📝 调研报告:\n{answer[:800]}...")
        except Exception as e:
            print(f"  ❌ 执行失败: {e}")


async def demo_compare_workflow():
    """Demo 4: 调研 + 对比 + 报告（CC-Harness Map-Reduce 模式）"""
    from harness.agent_profile import AgentProfile, AgentBuilder
    from harness.multi_agent import MapReduceOrchestrator
    from harness.prompt_engine import PromptEngine
    from tools import ToolRegistry, RAGTool

    print("\n" + "=" * 60)
    print("🔄 Demo 4: 多 Agent 协作调研（Map-Reduce）")
    print("=" * 60)

    try:
        profile = AgentProfile.from_yaml("profiles/tech_researcher.yaml")
        agent = await AgentBuilder.build(profile)
        print(f"  主 Agent: {agent.profile.name} 就绪")
    except Exception as e:
        print(f"  ⚠️ 构建失败: {e}")
        return

    # 如果 DevPilot A2A 可用，连接它
    from harness.a2a import A2AClient
    a2a_available = False
    try:
        a2a = A2AClient("http://localhost:8010")
        remote = await a2a.discover()
        print(f"  A2A: 已连接 {remote.name} ({len(remote.tools)} 工具)")
        a2a_available = True
    except Exception:
        print("  A2A: DevPilot 未启动，仅使用本地工具")

    # Map-Reduce 调研
    question = "对比 Rust 和 Go 在云原生后端开发中的适用场景、生态系统和性能表现"

    subtasks = [
        "Rust 在云原生后端中的生态（框架、库、工具链）和使用场景",
        "Go 在云原生后端中的生态（框架、库、工具链）和使用场景",
        "Rust vs Go 性能对比数据和社区发展趋势",
    ]

    print(f"\n  主问题: {question}")
    print(f"  拆解为 {len(subtasks)} 个子任务（Map 阶段）:")
    for i, st in enumerate(subtasks, 1):
        print(f"    {i}. {st}")

    # 使用 AgentBuilder 的组件做 Map-Reduce
    orch = MapReduceOrchestrator(
        llm_adapter=agent.llm,
        tool_registry=agent.tool_registry,
        prompt_engine=PromptEngine(agent.tool_registry),
    )

    try:
        print("\n  ⏳ 执行 Map-Reduce...")
        result = await orch.map_reduce(question, subtasks)
        print(f"\n  📝 综合报告 (Reduce 阶段):\n{result.answer[:1000]}...")
        print(f"\n  ✅ 完成: {len(result.sub_results)} 个子任务, {result.total_turns} 轮, {result.total_tool_calls} 次工具调用")
    except Exception as e:
        print(f"  ❌ Map-Reduce 失败: {e}")

    if a2a_available:
        await a2a.close()


# ==================== 主入口 ====================

async def main():
    print("""
╔══════════════════════════════════════════════════════╗
║     CC-Harness 技术调研 Agent — 端到端 Demo          ║
║                                                      ║
║  CC-Harness (主控) ←→ A2A ←→ DevPilot (数据源)       ║
║                                                      ║
║  前提: 先启动 DevPilot A2A Server                     ║
║        cd tech_agent && python a2a_server.py          ║
╚══════════════════════════════════════════════════════╝
""")

    demos = [
        ("A2A 发现", demo_a2a_discovery),
        ("工具调用", demo_direct_tool_call),
        ("调研 Agent", demo_research_agent),
        ("Map-Reduce", demo_compare_workflow),
    ]

    for name, func in demos:
        print(f"\n▶ 运行 Demo: {name}")
        try:
            await func()
        except Exception as e:
            print(f"  ❌ Demo '{name}' 失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())
