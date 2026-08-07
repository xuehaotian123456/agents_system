"""
CC-Harness Agent — 桌面客户端
==============================
基于 Python Tkinter 的独立窗口应用，零额外依赖。
- 窗口上半：滚动显示 Agent 思考过程和最终回答
- 窗口下半：输入框 + 发送按钮
- 异步 LLM 调用不阻塞 UI

启动方式：
    /Users/a1234/miniconda3/envs/myenv/bin/python client.py
"""

from __future__ import annotations

import asyncio
import threading
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tkinter as tk
from tkinter import scrolledtext, font
from queue import Queue, Empty

from dotenv import load_dotenv

from harness import AgentLoop, Session, LLMAdapter, PromptEngine, AgentConfig
from tools import ToolRegistry, RAGTool

load_dotenv()

TEST_DOCS = [
    "Agentic RAG 基于智能体实现动态检索，支持多次迭代查询和动态决策",
    "普通RAG是固定流水线，只会执行一次检索然后生成答案",
    "LangGraph 使用状态机实现Agent工作流，通过节点和条件边定义行为",
    "GraphRAG抽取实体关系构建知识图谱，和Agentic RAG不属于同一维度概念",
    "Claude Code采用AgentLoop异步循环引擎，以ReAct模式实现智能体自主规划",
]


# ==================== 消息队列（async → UI 线程桥梁） ====================

class LogWriter:
    """
    捕获 AgentLoop 的 print 输出，通过 Queue 发送到 UI 线程。
    这样异步 AgentLoop 运行时不会阻塞 UI，思考过程实时显示。
    """

    def __init__(self, queue: Queue):
        self.queue = queue
        self._original_stdout = sys.stdout

    def write(self, text: str):
        if text.strip():
            self.queue.put(("log", text))
        self._original_stdout.write(text)

    def flush(self):
        self._original_stdout.flush()


# ==================== Tkinter UI ====================

def run_agent(question: str, output_queue: Queue):
    """在后台线程运行 AgentLoop。所有组件在事件循环内创建，避免跨循环冲突"""

    async def _run():
        # ★ 关键：所有异步组件必须在同一个事件循环内创建
        _llm = LLMAdapter()
        _tool_registry = ToolRegistry()
        _rag_tool = RAGTool(collection_name="cc_harness_kb", persist_dir="./chroma_db", k=2)
        _tool_registry.register(_rag_tool)
        _prompt_engine = PromptEngine(_tool_registry)

        # 首次运行灌入知识库
        _rag_tool.add_documents(TEST_DOCS)

        _config = AgentConfig(max_turns=3, model="qwen-plus", temperature=0.0)

        session = Session(config=_config)
        session.set_system_prompt(_prompt_engine.build_system_prompt(_config))
        session.append_user_message(question)

        loop = AgentLoop(
            session=session,
            llm_adapter=_llm,
            tool_registry=_tool_registry,
            prompt_engine=_prompt_engine,
        )

        # 重定向 print → Queue → UI
        log_writer = LogWriter(output_queue)
        original_stdout = sys.stdout
        sys.stdout = log_writer

        try:
            answer = await loop.run()
            output_queue.put(("answer", answer))
        except Exception as e:
            output_queue.put(("answer", f"❌ 错误：{e}"))
        finally:
            sys.stdout = original_stdout

    asyncio.run(_run())


class AgentClient:
    """桌面客户端窗口"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 CC-Harness Agent")
        self.root.geometry("900x700")

        # 设置等宽字体
        self.text_font = font.Font(family="Menlo", size=12)
        self.answer_font = font.Font(family="Menlo", size=12, weight="bold")

        self._setup_ui()
        self._is_running = False

        # 消息队列：后台线程 → UI 线程
        self.queue: Queue = Queue()

    def _setup_ui(self):
        """构建 UI 布局"""

        # ── 主容器 ──
        main = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=4)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── 上方：对话显示区 ──
        top_frame = tk.Frame(main, height=500)
        main.add(top_frame)

        top_label = tk.Label(top_frame, text="💬 对话", font=("Menlo", 11, "bold"), anchor="w")
        top_label.pack(fill=tk.X, padx=5, pady=(5, 0))

        self.display = scrolledtext.ScrolledText(
            top_frame,
            font=self.text_font,
            wrap=tk.WORD,
            bg="#ffffff",        # 白色背景
            fg="#333333",        # 深灰文字
            insertbackground="#333333",
            state=tk.DISABLED,
        )
        self.display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 标签样式（浅色主题）
        self.display.tag_configure("question", foreground="#1565c0", font=("Menlo", 13, "bold"))  # 深蓝
        self.display.tag_configure("thinking", foreground="#2e7d32")    # 深绿
        self.display.tag_configure("tool", foreground="#e65100")        # 深橙
        self.display.tag_configure("error", foreground="#c62828")       # 红色
        self.display.tag_configure("answer", foreground="#111111", font=("Menlo", 13, "bold"))  # 黑色加粗
        self.display.tag_configure("divider", foreground="#999999")     # 浅灰分割线

        # ── 下方：输入区 ──
        bottom_frame = tk.Frame(main, height=120)
        main.add(bottom_frame)

        input_label = tk.Label(bottom_frame, text="⌨️ 输入问题（Ctrl+Enter 发送）", font=("Menlo", 11, "bold"), anchor="w")
        input_label.pack(fill=tk.X, padx=5, pady=(5, 0))

        self.input_box = tk.Text(
            bottom_frame,
            font=("Menlo", 13),
            height=3,
            wrap=tk.WORD,
            bg="#f5f5f5",
            fg="#333333",
            insertbackground="#333333",
        )
        self.input_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.input_box.bind("<Control-Return>", self._on_send)
        self.input_box.bind("<Command-Return>", self._on_send)  # macOS

        # 发送按钮
        btn_frame = tk.Frame(bottom_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.send_btn = tk.Button(
            btn_frame, text="▶ 发送 (Ctrl+Enter)",
            font=("Menlo", 11),
            bg="#1565c0", fg="#ffffff",
            activebackground="#1976d2", activeforeground="#ffffff",
            relief=tk.FLAT, padx=20, pady=4,
            command=self._on_send,
        )
        self.send_btn.pack(side=tk.RIGHT)

        # 关闭窗口时终止事件循环
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_send(self, event=None):
        """点击发送按钮"""
        if self._is_running:
            return

        question = self.input_box.get("1.0", tk.END).strip()
        if not question:
            return

        # 清空输入框
        self.input_box.delete("1.0", tk.END)

        # 禁用发送
        self._is_running = True
        self.send_btn.config(text="⏳ 思考中...", state=tk.DISABLED)

        # 在显示区打印问题
        self._append_text(f"\n❯ {question}\n\n", "question")
        self._append_text("─" * 60 + "\n", "divider")

        # 启动后台线程
        thread = threading.Thread(
            target=run_agent,
            args=(question, self.queue),
            daemon=True,
        )
        thread.start()

        # 开始轮询队列
        self._poll_queue()

    def _poll_queue(self):
        """定时轮询消息队列，更新 UI"""
        try:
            while True:
                msg_type, content = self.queue.get_nowait()

                if msg_type == "log":
                    self._append_text(content, "thinking")
                elif msg_type == "answer":
                    self._append_text(f"\n{content}\n", "answer")
                    self._append_text("─" * 60 + "\n", "divider")
                    self._is_running = False
                    self.send_btn.config(text="▶ 发送 (Ctrl+Enter)", state=tk.NORMAL)
        except Empty:
            pass

        # 继续轮询（每 200ms）
        if self._is_running:
            self.root.after(200, self._poll_queue)

    def _append_text(self, text: str, tag: str):
        """向显示区追加带颜色的文本"""
        self.display.config(state=tk.NORMAL)
        self.display.insert(tk.END, text, tag)
        self.display.see(tk.END)  # 自动滚动到底部
        self.display.config(state=tk.DISABLED)

    def _on_close(self):
        """关闭窗口"""
        self.root.destroy()

    def run(self):
        """启动客户端主循环"""
        self._append_text("🤖 CC-Harness Agent 已就绪\n", "answer")
        self._append_text("在下方输入问题，Ctrl+Enter 发送\n", "thinking")
        self._append_text("─" * 60 + "\n", "divider")
        self.root.mainloop()


# ==================== 启动入口 ====================

if __name__ == "__main__":
    client = AgentClient()
    client.run()
