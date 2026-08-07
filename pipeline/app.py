"""DevPilot — 智能技术助手 Streamlit UI"""
import streamlit as st
import time
import json
import os
import random
from pathlib import Path
from collections import Counter
import re

st.set_page_config(page_title="DevPilot 技术助手", page_icon="🤖", layout="wide")

def render_with_graph(content: str):
    """渲染含 [GRAPH:path] 标记的内容，将路径替换为交互式图谱"""
    parts = re.split(r'\[GRAPH:(.*?)\]', content)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # 文本部分
            if part.strip():
                st.markdown(part)
        else:
            # 图谱路径
            try:
                with open(part, "r", encoding="utf-8") as f:
                    html = f.read()
                st.components.v1.html(html, height=420, scrolling=True)
            except Exception:
                st.caption(f"(图谱加载失败: {part})")

# ==================== Sidebar ====================
with st.sidebar:
    st.title("🤖 DevPilot")
    st.markdown("**LangGraph 四节点闭环**: Planner → Retriever → Reflector → Summarizer")
    st.divider()

    st.subheader("💡 试试这些")
    st.caption("点击即可提问 ↓")

    st.caption("📚 **知识 + 代码**")
    if st.button("Python async/await 怎么用？", use_container_width=True):
        st.session_state.next_prompt = "Python 异步编程 async/await 怎么用？给出代码示例"
    if st.button("搜索 async 代码示例", use_container_width=True):
        st.session_state.next_prompt = "帮我找一下 async await 的代码示例"

    st.caption("🔀 **技术对比**")
    if st.button("React 和 Vue 对比", use_container_width=True):
        st.session_state.next_prompt = "帮我对比一下 React 和 Vue"

    st.caption("🔥 **实时热榜**")
    if st.button("掘金今天热榜有什么？", use_container_width=True):
        st.session_state.next_prompt = "帮我看看掘金今天的热榜"
    if st.button("🌍 综合技术热榜（多源）", use_container_width=True):
        st.session_state.next_prompt = "帮我出一份综合技术热榜，包含 GitHub、HackerNews、开源中国"

    st.caption("📊 **趋势分析**")
    if st.button("今日技术摘要", use_container_width=True):
        st.session_state.next_prompt = "给我生成一份今日技术摘要"
    if st.button("推送今日技术摘要（多源+图片+链接）", use_container_width=True):
        st.session_state.next_prompt = "帮我爬取最新文章，生成一份带图片和链接的今日技术摘要并推送"

    st.caption("🕸️ **知识图谱（交互式图）**")
    if st.button("Python 关联哪些技术？", use_container_width=True):
        st.session_state.next_prompt = "帮我查一下 Python 在知识图谱中和哪些技术关联最强"

    st.caption("🛠️ **动态工具**")
    if st.button("创建 AI 趋势跟踪工具", use_container_width=True):
        st.session_state.next_prompt = "帮我创建一个名为 ai_tracker 的自定义工具，关键词是 AI、大模型、LLM、Agent，用来跟踪AI最新动态"
    if st.button("展示全局知识图谱", use_container_width=True):
        st.session_state.next_prompt = "展示知识图谱中最重要的30个实体及其关系"

    # 全局图谱按钮
    if st.button("🗺️ 查看全局知识图谱（交互式）", use_container_width=True):
        try:
            from services.graph_viz import build_global_graph
            fpath = build_global_graph()
            if fpath:
                with open(fpath, "r", encoding="utf-8") as f:
                    html = f.read()
                st.components.v1.html(html, height=520, scrolling=True)
            else:
                st.caption("KG未构建")
        except Exception as e:
            st.caption(f"加载失败: {e}")

    st.divider()

    # ===== 热词趋势 =====
    st.subheader("📊 技术热词趋势")
    try:
        from services.trends import analyze_trends
        trends = analyze_trends()
        if trends["keywords"]:
            # 词云渲染
            top_words = trends["keywords"][:20]
            max_freq = top_words[0][1] if top_words else 1
            random.seed(42)
            html_parts = ['<div style="line-height:2.2;text-align:center;padding:8px;background:#f8f9fa;border-radius:8px">']
            for word, freq in top_words:
                size = max(12, min(36, int(14 + (freq / max_freq) * 22)))
                colors = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c","#e67e22","#2980b9"]
                color = random.choice(colors)
                html_parts.append(
                    f'<span style="font-size:{size}px;color:{color};margin:4px;display:inline-block">{word}</span>'
                )
            html_parts.append('</div>')
            st.markdown("".join(html_parts), unsafe_allow_html=True)

            # 最近几天趋势
            daily = trends.get("daily_trends", [])[:3]
            if daily:
                st.caption("---")
                for d in daily:
                    kw_str = ", ".join([w for w, c in d["keywords"][:4]])
                    st.caption(f"📅 {d['date']}: {kw_str}")
        else:
            st.caption("暂无趋势数据，爬取文章后自动生成")
    except Exception:
        st.caption("趋势分析暂不可用")

    st.divider()
    # System status
    vs = st.session_state.get("vs")
    kg = st.session_state.get("kg")
    if vs:
        st.metric("📚 文档块", len(vs.all_chunks))
    if kg and kg.is_built:
        st.metric("🕸️ KG 实体", kg.entity_count)
    article_count = len([f for f in os.listdir("data/articles") if f.endswith(".md")]) if os.path.exists("data/articles") else 0
    st.caption(f"📄 {article_count} 篇文章 · 掘金+博客园+种子数据")
    st.caption("🔍 混合检索: 向量 + BM25 + BGE-Reranker")
    st.caption("🕸️ KG: jieba.posseg + 共现矩阵 + IDF")

# ==================== Main ====================
st.title("🤖 DevPilot — 智能技术助手")
st.caption("Planner 规划 → Retriever 检索 → Reflector 反思 → Summarizer 总结 · 完整决策轨迹可见")
st.divider()

# ==================== Init ====================
@st.cache_resource(show_spinner=False)
def init_system():
    """系统初始化 — 使用缓存避免重复加载"""
    from rag.vector_store import VectorStore
    vs = VectorStore()
    vs.load_articles()

    from rag.knowledge_graph import get_kg
    kg = get_kg()
    if vs.all_chunks and not kg.is_built:
        kg.build(vs.all_chunks)

    # ── 启动后台调度器 ──
    try:
        from services.scheduler import start_scheduler
        start_scheduler()
    except Exception as e:
        print(f"[App] 调度器启动跳过: {e}")

    return vs, kg

if "agent_initialized" not in st.session_state:
    progress = st.progress(0, "加载向量库...")
    status = st.empty()

    try:
        progress.progress(10, "加载系统组件...")
        vs, kg = init_system()
        progress.progress(90, "完成")

        st.session_state.vs = vs
        st.session_state.kg = kg
        st.session_state.agent_initialized = True

        progress.progress(100, "就绪")
        progress.empty()
        status.success(f"✅ 就绪 — {len(vs.all_chunks)} chunks · {kg.entity_count} KG实体 · {len([f for f in __import__('os').listdir('data/articles') if f.endswith('.md')])} 篇文章")

    except Exception as e:
        progress.empty()
        st.error(f"初始化失败: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "next_prompt" not in st.session_state:
    st.session_state.next_prompt = None

# ==================== Render History ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("trace"):
            for step in msg["trace"]:
                if step["type"] == "plan":
                    with st.expander(f"🧠 Planner: {step.get('content', '')[:60]}", expanded=False):
                        st.json(step.get("data", {}))
                elif step["type"] == "retrieve":
                    with st.expander(f"🔍 Retriever: 检索 {step.get('rounds', 1)} 轮", expanded=False):
                        st.text(step.get("content", "")[:500])
                elif step["type"] == "reflect":
                    with st.expander(f"🪞 Reflector: confidence={step.get('confidence', 0):.2f}", expanded=False):
                        st.text(step.get("content", ""))
        render_with_graph(msg["content"])

# ==================== Handle Query ====================
def run_agent(prompt: str):
    from agent.state import initial_state
    from agent.graph import graph

    state = initial_state(prompt)
    trace = []

    with st.chat_message("assistant"):
        plan_placeholder = st.empty()
        answer_placeholder = st.empty()

        with st.spinner("🤔 Agent 思考中..."):
            try:
                config = {"configurable": {"thread_id": state["session_id"]}}
                result = graph.invoke(state, config)

                # Trace: plan
                plan = result.get("plan", {})
                trace.append({"type": "plan", "data": plan, "content": plan.get("intent", "")})

                # Trace: retrieve
                tool_calls = result.get("tool_calls", [])
                context = result.get("context", "")
                trace.append({"type": "retrieve", "rounds": result.get("retrieval_rounds", 0),
                              "content": f"工具调用: {len(tool_calls)} 次\n\n检索上下文: {context[:500]}"})

                # Trace: reflect
                trace.append({"type": "reflect", "confidence": result.get("confidence", 0),
                              "content": f"置信度: {result.get('confidence', 0):.2f}, 重试次数: {result.get('retry_count', 0)}"})

                answer = result.get("answer", "未能生成回答。")

                # Display
                with plan_placeholder.container():
                    st.caption(f"🧠 Plan: {plan.get('intent', '分析中')[:80]}")
                    st.caption(f"🔍 Retriever: 完成 ({result.get('retrieval_rounds', 0)} 轮)")
                    st.caption(f"🪞 Reflector: 置信度 {result.get('confidence', 0):.2f}")

                render_with_graph(answer)

            except Exception as e:
                answer = f"执行异常: {str(e)}"
                answer_placeholder.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "trace": trace
    })

# ==================== Input ====================
if st.session_state.get("next_prompt"):
    prompt = st.session_state.next_prompt
    st.session_state.next_prompt = None
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    run_agent(prompt)
    st.rerun()

prompt = st.chat_input("输入技术问题，观察 Agent 的 Plan → Retrieve → Reflect → Summarize 过程...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    run_agent(prompt)
    st.rerun()
