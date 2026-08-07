"""
CC-Harness Agent - 启动入口
============================
演示纯 CC 风格 AgentLoop 的完整运行流程。

启动方式：
    python main.py

与 agentic_rag_lab 的区别：
    那里：LangGraph StateGraph → 5 个节点 + 条件边 → 编译 → invoke
    这里：Session + AgentLoop → while 循环 → 每轮 LLM 动态决策

对比运行这两个项目可以帮助你深入理解两种架构的差异。
"""

import asyncio
import os
import sys

# 修复 Windows GBK 终端 emoji 编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from harness import AgentLoop, Session, LLMAdapter, PromptEngine, AgentConfig
from tools import ToolRegistry, RAGTool


async def main():
    """主函数：初始化组件 → 准备知识库 → 启动 AgentLoop"""

    load_dotenv()

    print("=" * 60)
    print("🤖 CC-Harness Agent 启动（自研框架，无 LangChain/LangGraph）")
    print("=" * 60)

    # ==================== 1. 初始化组件 ====================
    # LLM 适配器（百炼 DashScope，qwen3.5-flash）
    llm = LLMAdapter()

    # 工具注册表（CC 风格：RAG 是一个 Tool，不是 Graph 节点）
    tool_registry = ToolRegistry()

    # RAG 工具（内部封装完整 Agentic RAG 循环）
    rag_tool = RAGTool(
        collection_name="cc_harness_kb",
        persist_dir="./chroma_db",
        k=2,
        max_rewrites=1,
    )
    tool_registry.register(rag_tool)

    # Prompt 引擎（根据注册的工具动态生成系统提示词）
    prompt_engine = PromptEngine(tool_registry)

    print(f"📋 已注册工具：{tool_registry.list_tool_names()}")

    # ==================== 2. 准备知识库 ====================
    # 灌入测试文档（首次运行）
    test_docs = [
        "Agentic RAG 基于智能体实现动态检索，支持多次迭代查询和动态决策",
        "普通RAG是固定流水线，只会执行一次检索然后生成答案",
        "LangGraph 使用状态机实现Agent工作流，通过节点和条件边定义行为",
        "GraphRAG抽取实体关系构建知识图谱，和Agentic RAG不属于同一维度概念",
        "Claude Code采用AgentLoop异步循环引擎，以ReAct模式实现智能体自主规划",
    ]
    rag_tool.add_documents(test_docs)
    print(f"📚 知识库已就绪，共 {len(test_docs)} 篇文档")

    # ==================== 3. 创建 Session 和 AgentLoop ====================
    config = AgentConfig(
        max_turns=3,            # 简单问题最多 3 轮
        model="qwen-plus",      # 无推理开销，响应快
        temperature=0.0,
        enable_subagents=False,
    )

    session = Session(config=config)
    session.set_system_prompt(prompt_engine.build_system_prompt(config))

    # ==================== 4. 测试问题 ====================
    test_questions = [
        "Agentic RAG 和普通RAG有什么区别？",
        "Claude Code的AgentLoop与LangGraph有什么不同？",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ 用户问题：{question}")
        print("=" * 60)

        # 每次问题创建新 Session（新对话）
        session = Session(config=config)
        session.set_system_prompt(prompt_engine.build_system_prompt(config))
        session.append_user_message(question)

        loop = AgentLoop(
            session=session,
            llm_adapter=llm,
            tool_registry=tool_registry,
            prompt_engine=prompt_engine,
        )

        answer = await loop.run()

        print(f"\n🤖 最终回答：\n{answer}")

    print(f"\n{'='*60}")
    print("✅ CC-Harness Agent 演示完成")


if __name__ == "__main__":
    asyncio.run(main())
