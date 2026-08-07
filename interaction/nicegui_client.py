"""
CC-Harness Agent — NiceGUI 客户端
==================================
Python 驱动的现代 UI，自动打开浏览器窗口。

启动方式：
    /Users/a1234/miniconda3/envs/myenv/bin/python nicegui_client.py

交互：
- 底部输入框打字，Enter 发送
- 聊天气泡展示对话
- 可展开「查看思考过程」看 Agent 内部推理
- 最终回答 Markdown 渲染
"""

from __future__ import annotations

import sys
import io
import asyncio
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from nicegui import ui

from harness import AgentLoop, Session, LLMAdapter, PromptEngine, AgentConfig
from tools import ToolRegistry, RAGTool

# ==================== 知识库文档 ====================

TEST_DOCS = [
    "Agentic RAG 基于智能体实现动态检索，支持多次迭代查询和动态决策",
    "普通RAG是固定流水线，只会执行一次检索然后生成答案",
    "LangGraph 使用状态机实现Agent工作流，通过节点和条件边定义行为",
    "GraphRAG抽取实体关系构建知识图谱，和Agentic RAG不属于同一维度概念",
    "Claude Code采用AgentLoop异步循环引擎，以ReAct模式实现智能体自主规划",
]


def _init_tools() -> tuple[ToolRegistry, PromptEngine, AgentConfig]:
    """每个请求创建独立的工具集和配置"""
    tool_registry = ToolRegistry()
    rag_tool = RAGTool(collection_name="cc_harness_kb", persist_dir="./chroma_db", k=2)
    tool_registry.register(rag_tool)
    rag_tool.add_documents(TEST_DOCS)

    prompt_engine = PromptEngine(tool_registry)
    config = AgentConfig(max_turns=3, model="qwen-plus", temperature=0.0)

    return tool_registry, prompt_engine, config


# ==================== 主页面 ====================

@ui.page("/")
async def index():
    # ── 页面标题 ──
    with ui.row().classes("w-full max-w-4xl mx-auto mt-6 px-4 items-center gap-3"):
        ui.label("🤖").classes("text-3xl")
        with ui.column().classes("gap-0"):
            ui.label("CC-Harness Agent").classes("text-xl font-bold")
            ui.label("基于 Claude Code 架构的自研 Agent 框架").classes("text-sm text-gray-500")

    ui.separator().classes("max-w-4xl mx-auto")

    # ── 聊天容器 ──
    chat_area = ui.column().classes("w-full max-w-4xl mx-auto px-4 mb-32")
    placeholder = ui.label("👋 在下方输入问题开始对话").classes("text-gray-400 text-center mt-20")

    # 当前任务引用
    task_ref: Optional[asyncio.Task] = None

    async def do_send():
        nonlocal task_ref

        question = input_box.value.strip()
        if not question:
            return
        if task_ref and not task_ref.done():
            ui.notify("请等待当前问题完成", type="warning")
            return

        input_box.value = ""
        send_btn.disable()
        stop_btn.visible = True
        placeholder.set_visibility(False)

        # ── 用户消息气泡 ──
        with chat_area:
            with ui.chat_message(name="你", sent=True).classes("w-full"):
                ui.label(question)

        # ── Agent 消息区域 ──
        with chat_area:
            agent_msg = ui.chat_message(name="Agent", sent=False).classes("w-full")
            with agent_msg:
                spinner = ui.spinner(size="sm")

                thinking_exp = ui.expansion("💭 思考过程", value=False).classes("w-full")
                with thinking_exp:
                    thinking_lbl = ui.label("").classes("text-xs text-gray-500 font-mono whitespace-pre-wrap")

                answer_lbl = ui.markdown("").classes("text-base")

        # 滚到底部
        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")

        # ── 初始化组件 + 运行 Agent ──
        tool_registry, prompt_engine, config = _init_tools()

        # 捕获 stdout
        log_buf = io.StringIO()

        class CaptureWriter:
            def __init__(self, orig, buf):
                self.orig = orig
                self.buf = buf
            def write(self, s):
                self.orig.write(s)
                self.buf.write(s)
            def flush(self):
                self.orig.flush()

        writer = CaptureWriter(sys.stdout, log_buf)

        async def agent_task():
            try:
                session = Session(config=config)
                session.set_system_prompt(prompt_engine.build_system_prompt(config))
                session.append_user_message(question)

                llm_adapter = LLMAdapter()
                loop = AgentLoop(
                    session=session,
                    llm_adapter=llm_adapter,
                    tool_registry=tool_registry,
                    prompt_engine=prompt_engine,
                )

                sys.stdout = writer
                answer = await loop.run()
                sys.stdout = sys.__stdout__

                # 思考过程
                log = log_buf.getvalue().strip()
                if log:
                    thinking_lbl.set_text(log[-3000:])
                else:
                    thinking_exp.set_visibility(False)

                # 最终回答
                spinner.set_visibility(False)
                answer_lbl.set_content(answer)

            except asyncio.CancelledError:
                sys.stdout = sys.__stdout__
                answer_lbl.set_content("*已停止*")
            except Exception as e:
                sys.stdout = sys.__stdout__
                answer_lbl.set_content(f"❌ 错误：{e}")
            finally:
                spinner.set_visibility(False)
                send_btn.enable()
                stop_btn.visible = False
                ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")

        # 定期刷新思考过程
        async def refresh_think():
            while task_ref and not task_ref.done():
                log = log_buf.getvalue().strip()
                if log:
                    thinking_lbl.set_text(log[-3000:])
                await asyncio.sleep(0.5)

        task_ref = asyncio.create_task(agent_task())
        asyncio.create_task(refresh_think())

    def stop():
        if task_ref and not task_ref.done():
            task_ref.cancel()

    # ── 底部输入栏 ──
    with ui.element("div").classes("fixed bottom-0 left-0 right-0 bg-white border-t py-3 px-4 z-10"):
        with ui.row().classes("w-full max-w-4xl mx-auto items-center gap-2"):
            input_box = (
                ui.input(placeholder="输入问题，Enter 发送...")
                .props("outlined rounded dense")
                .classes("flex-grow")
            )
            send_btn = ui.button("发送", icon="send", color="blue")
            stop_btn = ui.button("停止", icon="stop", color="red")
            stop_btn.visible = False

    send_btn.on("click", lambda: asyncio.ensure_future(do_send()))
    stop_btn.on("click", stop)


# ==================== 启动 ====================

if __name__ == "__main__":
    ui.run(
        title="🤖 CC-Harness Agent",
        favicon="🤖",
        window_size=(1100, 800),
        dark=False,
        reload=False,
        native=True,  # 桌面原生窗口（不依赖浏览器）
    )
