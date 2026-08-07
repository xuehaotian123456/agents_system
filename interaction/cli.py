"""
CC-Harness Agent - 交互式命令行
================================
仿 Claude Code 体验：终端输入问题，滚动显示思考过程，输出最终结果。

启动方式：
    /Users/a1234/miniconda3/envs/myenv/bin/python cli.py

使用：
    ❯ 你的问题
    （观察 Agent 思考 → 工具调用 → 生成答案的完整过程）
    ❯ 下一个问题（保留上下文）
    ❯ /exit 退出
"""

import asyncio
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from harness import AgentLoop, Session, LLMAdapter, PromptEngine, AgentConfig
from tools import ToolRegistry, RAGTool


async def main():
    load_dotenv()

    # ── 初始化组件 ──
    llm = LLMAdapter()
    tool_registry = ToolRegistry()

    rag_tool = RAGTool(
        collection_name="cc_harness_kb",
        persist_dir="./chroma_db",
        k=2, max_rewrites=1,
    )
    tool_registry.register(rag_tool)

    prompt_engine = PromptEngine(tool_registry)

    # 灌入知识库
    test_docs = [
        "Agentic RAG 基于智能体实现动态检索，支持多次迭代查询和动态决策",
        "普通RAG是固定流水线，只会执行一次检索然后生成答案",
        "LangGraph 使用状态机实现Agent工作流，通过节点和条件边定义行为",
        "GraphRAG抽取实体关系构建知识图谱，和Agentic RAG不属于同一维度概念",
        "Claude Code采用AgentLoop异步循环引擎，以ReAct模式实现智能体自主规划",
    ]
    rag_tool.add_documents(test_docs)

    # ── 会话配置 ──
    config = AgentConfig(
        max_turns=3,
        model="qwen-plus",
        temperature=0.0,
        enable_subagents=False,
    )

    print("\n" + "=" * 60)
    print("🤖 CC-Harness Agent — 交互式终端")
    print("   输入问题开始对话，输入 /exit 退出")
    print("=" * 60 + "\n")

    # ── 交互循环 ──
    while True:
        try:
            user_input = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "/q"):
            print("👋 再见")
            break

        # 每个问题创建新 Session
        session = Session(config=config)
        session.set_system_prompt(prompt_engine.build_system_prompt(config))
        session.append_user_message(user_input)

        loop = AgentLoop(
            session=session,
            llm_adapter=llm,
            tool_registry=tool_registry,
            prompt_engine=prompt_engine,
        )

        answer = await loop.run()

        print(f"\n{answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
