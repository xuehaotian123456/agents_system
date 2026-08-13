"""
DevPilot A2A Server — 将 DevPilot LangGraph Agent 暴露为 A2A 服务
==================================================================
启动: python a2a_server.py --port 8010
效果: CC-Harness 可以通过 A2A Client 调用 DevPilot 的爬虫和 KG 工具
"""

import sys
import os
import asyncio

# Windows GBK 编码修复 — 确保所有 print/日志能输出 emoji 和中文
if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import re

app = FastAPI(title="DevPilot A2A Agent", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==================== 工具元数据 ====================

A2A_TOOLS = [
    # ── 原有工具 (9个) ──
    {"name": "rag_search", "description": "从已爬取的技术文章中搜索知识。入参 query", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "trending_list", "description": "获取掘金/博客园/GitHub/HackerNews/OSChina 实时技术热榜。入参 source (juejin/cnblogs/github/hackernews/oschina/all)", "inputSchema": {"type": "object", "properties": {"source": {"type": "string", "default": "juejin"}}}},
    {"name": "kg_lookup", "description": "查询知识图谱中技术关键词的关联实体。支持多跳扩散推理链。入参 entity_name。查询关联/报错/依赖类问题时自动返回 2-hop 推理链路", "inputSchema": {"type": "object", "properties": {"entity_name": {"type": "string"}}}},
    {"name": "global_search", "description": "全局检索: 基于社区摘要索引回答整体性问题。入参 query。返回相关社区主题摘要与代表实体，适合宏观问题", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "fetch_article", "description": "爬取并保存指定URL的技术文章。入参 url", "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}},
    {"name": "search_web", "description": "搜索网络上的技术内容。入参 query", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "code_example", "description": "从知识库中搜索代码示例。入参 keyword", "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string"}}}},
    {"name": "compare_tech", "description": "对比两个技术/框架。入参 tech_a, tech_b", "inputSchema": {"type": "object", "properties": {"tech_a": {"type": "string"}, "tech_b": {"type": "string"}}}},
    {"name": "daily_digest", "description": "生成今日技术摘要报告。无入参，返回 Markdown 格式摘要", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "trend_report", "description": "生成技术热词趋势分析报告。无入参", "inputSchema": {"type": "object", "properties": {}}},
    # ── 新增: 邮件 & 调度控制 (6个) ──
    {"name": "send_digest_email", "description": "生成并发送每日技术摘要邮件。入参 to_email（收件人邮箱）。若SMTP未配置会返回need_smtp=true，Harness应引导用户先提供邮箱", "inputSchema": {"type": "object", "properties": {"to_email": {"type": "string"}}}},
    {"name": "get_smtp_help", "description": "根据邮箱地址返回SMTP服务器信息和授权码获取链接。入参 email。Harness 调用此工具获取授权码页面URL展示给用户。", "inputSchema": {"type": "object", "properties": {"email": {"type": "string"}}}},
    {"name": "configure_smtp", "description": "配置SMTP邮件发送。只需提供 email（发件邮箱）和 password（授权码），系统会自动识别SMTP服务器地址和端口。", "inputSchema": {"type": "object", "properties": {"email": {"type": "string"}, "password": {"type": "string"}, "host": {"type": "string", "description": "可选，不填则自动检测"}, "port": {"type": "integer", "description": "可选，不填则自动检测"}}}},
    {"name": "configure_daily_digest", "description": "配置每日定时摘要邮件。入参 email（收件人）、time（HH:MM格式，如08:00）、enabled（true/false）。修改后立即生效。", "inputSchema": {"type": "object", "properties": {"email": {"type": "string"}, "time": {"type": "string", "default": "08:00"}, "enabled": {"type": "boolean", "default": True}}}},
    {"name": "get_pipeline_status", "description": "获取 Pipeline 运行状态：最近爬取时间、文章总数、调度器状态、每日摘要配置、SMTP 配置状态。", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "force_update", "description": "强制执行一次全量增量爬取（多源 → 去重 → 入库 → 更新向量库 → 更新 KG）。", "inputSchema": {"type": "object", "properties": {}}},
]


# ==================== A2A 端点 ====================

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": "DevPilot Research Agent",
        "description": "技术调研助手 — 提供实时爬虫、知识图谱查询、代码搜索、技术对比。基于 LangGraph + 掘金/博客园数据。",
        "url": "http://localhost:8010",
        "version": "1.0.0",
        "capabilities": {"streaming": True, "tools": True, "stream_granularity": "node-level"},
        "tools": A2A_TOOLS,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "name": "DevPilot A2A", "tools": len(A2A_TOOLS)}


@app.post("/tasks")
async def create_task(request: Request):
    """A2A 任务端点 — 完整 LangGraph Agent 调用"""
    body = await request.json()
    query = body.get("query", "")
    use_stream = body.get("stream", True)

    # 会话隔离: 每个请求使用唯一 thread_id（支持中断恢复）
    thread_id = body.get("session_id", "a2a_default")

    if not use_stream:
        try:
            from agent.state import initial_state
            from agent.graph import graph
            state = initial_state(query, session_id=thread_id)
            result = graph.invoke(state, {"configurable": {"thread_id": thread_id}})
            answer = result.get("answer", "无法生成回答")
            errors = result.get("errors", [])
            return JSONResponse({"answer": answer, "status": "completed",
                                 "errors": len(errors), "thread_id": thread_id})
        except Exception as e:
            return JSONResponse({"answer": f"错误: {e}", "status": "error"})

    # 流式模式（SSE 实时决策事件流: status → plan → tool → answer → done）
    # 注意: 这里流的是 Pipeline 节点进度与决策过程，不是 token 级流式输出。
    async def event_stream():
        try:
            yield f"event: status\ndata: {json.dumps({'status': 'started', 'agent': 'DevPilot LangGraph', 'thread_id': thread_id, 'stream_granularity': 'node-level'})}\n\n"

            from agent.state import initial_state
            from agent.graph import graph
            state = initial_state(query, session_id=thread_id)
            result = graph.invoke(state, {"configurable": {"thread_id": thread_id}})

            plan = result.get("plan", {})
            yield f"event: plan\ndata: {json.dumps({'intent': plan.get('intent', ''), 'tools': plan.get('suggested_tools', [])})}\n\n"

            for tc in result.get("tool_calls", []):
                yield f"event: tool\ndata: {json.dumps({'name': tc.get('tool', '?'), 'result': str(tc.get('result', ''))[:200]})}\n\n"

            answer = result.get("answer", "无法生成回答")
            # 完整回答作为单个事件发送（不做伪 token 切块）
            yield f"event: answer\ndata: {json.dumps({'answer': answer, 'answer_length': len(answer)})}\n\n"

            yield f"event: done\ndata: {json.dumps({'status': 'completed', 'confidence': result.get('confidence', 0)})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


# ==================== 工具辅助 ====================

def _safe_json_content(content: str) -> str:
    """清理字符串中的孤立代理字符（surrogates），防止 JSON 序列化崩溃"""
    try:
        # 尝试编码为 UTF-8 再解码，surrogates 会触发报错
        content.encode('utf-8')
        return content
    except UnicodeEncodeError:
        # 逐字符过滤，干掉 surrogates
        return ''.join(c for c in content if not (0xD800 <= ord(c) <= 0xDFFF))


# ==================== 智能缓存: Harness 查询时按需更新 ====================

def _ensure_fresh_data(max_age_minutes: int = 60) -> str:
    """
    确保数据新鲜——若距上次全量爬取超过 max_age_minutes，触发增量更新。
    Harness 每次查询时调用此函数，实现"间隔短直接用缓存，间隔长自动更新"。
    """
    try:
        from services.crawl_state import get_crawl_state
        state = get_crawl_state()

        last = state.get_last_crawl_time()  # 最近全量爬取时间
        if last:
            from datetime import datetime
            last_time = datetime.fromisoformat(last)
            age = (datetime.now() - last_time).total_seconds() / 60
            if age < max_age_minutes:
                return ""  # 缓存新鲜，无需更新

        # 需要更新
        from services.scheduler import _incremental_crawl_job
        result = _incremental_crawl_job()
        return f"（自动增量更新: {result['message']}）"
    except Exception as e:
        return f"（数据刷新跳过: {e}）"


@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    """直接调用单个工具"""
    try:
        body = await request.json()
        from agent.tools import ALL_TOOLS
        tools_map = {t.name: t for t in ALL_TOOLS}

        # ── 新增工具: send_digest_email ──
        if tool_name == "send_digest_email":
            to_email = body.get("to_email", "")
            if not to_email:
                return JSONResponse({"success": False, "error": "缺少 to_email 参数"})
            from services.mailer import send_daily_digest
            result = send_daily_digest(to_email)
            return JSONResponse({"success": result["success"], "content": result["message"],
                                 "article_count": result.get("article_count", 0)})

        # ── 新增工具: get_smtp_help ──
        if tool_name == "get_smtp_help":
            email = body.get("email", "")
            if not email:
                return JSONResponse({"success": False, "error": "请提供邮箱地址"})
            from services.mailer import _detect_smtp
            info = _detect_smtp(email)
            if not info:
                return JSONResponse({"success": True,
                    "content": f"未识别邮箱 {email} 的SMTP配置。请手动提供SMTP服务器地址和端口。\n"
                               f"常见: QQ邮箱 smtp.qq.com:465 / 163邮箱 smtp.163.com:465 / Gmail smtp.gmail.com:587",
                    "detected": False})
            return JSONResponse({"success": True,
                "content": f"📧 {info['name']}\n"
                           f"   SMTP 服务器: {info['host']}:{info['port']}\n"
                           f"   🔗 获取授权码: {info['auth_help']}\n"
                           f"   📋 步骤: {info['auth_steps']}",
                "detected": True, "provider": info})

        # ── 新增工具: configure_smtp ──
        if tool_name == "configure_smtp":
            email = body.get("email", "")
            password = body.get("password", "")
            host = body.get("host", "")
            port = body.get("port", 0)

            if not email or not password:
                return JSONResponse({"success": False,
                    "error": "请提供 email（发件邮箱）和 password（授权码）",
                    "hint": "例如: \"xxx@qq.com\" 和你的授权码"})

            # 自动检测 SMTP 服务器
            if not host:
                from services.mailer import _detect_smtp
                info = _detect_smtp(email)
                if info:
                    host = info["host"]
                    port = port or info["port"]
                else:
                    return JSONResponse({"success": False,
                        "error": f"无法自动识别 {email} 的SMTP服务器。请手动提供 host 和 port。\n"
                                 f"常见: QQ邮箱 smtp.qq.com:465 / 163邮箱 smtp.163.com:465 / Gmail smtp.gmail.com:587"})

            from services.crawl_state import get_crawl_state
            state = get_crawl_state()
            state.set_smtp_config(host=host, port=port or 465, user=email,
                                  password=password, from_addr=email)
            return JSONResponse({"success": True,
                "content": f"✅ SMTP 已配置\n发件人: {email}\n服务器: {host}:{port or 465}\n\n现在可以发送邮件了。",
                "smtp_configured": True})

        # ── 新增工具: configure_daily_digest ──
        if tool_name == "configure_daily_digest":
            email = body.get("email", "")
            time_str = body.get("time", "08:00")
            enabled = body.get("enabled", True)
            if not email:
                return JSONResponse({"success": False, "error": "缺少 email 参数"})
            from services.crawl_state import get_crawl_state
            state = get_crawl_state()
            state.set_digest_config(email=email, time=time_str, enabled=enabled)
            from services.scheduler import reschedule_daily_digest
            reschedule_daily_digest(time_str)
            status = "启用" if enabled else "已保存(未启用)"
            return JSONResponse({"success": True,
                                 "content": f"✅ 每日摘要已配置: {email} | {time_str} | {status}\n"
                                            f"系统将在每天 {time_str} 自动发送技术摘要到 {email}。"
                                            f"\n如需修改，随时告诉我。",
                                 "config": {"email": email, "time": time_str, "enabled": enabled}})

        # ── 新增工具: get_pipeline_status ──
        if tool_name == "get_pipeline_status":
            from services.crawl_state import get_crawl_state
            from services.scheduler import get_scheduler_status
            state = get_crawl_state()
            crawl_status = state.get_status_summary()
            sched_status = get_scheduler_status()
            smtp_cfg = state.get_smtp_config()
            smtp_status = {
                "configured": state.has_smtp_configured(),
                "host": smtp_cfg.get("host", "") or "(未配置)",
                "user": smtp_cfg.get("user", "") or "(未配置)",
            }
            return JSONResponse({"success": True,
                                 "content": json.dumps({
                                     **crawl_status,
                                     "smtp": smtp_status,
                                     "scheduler": sched_status,
                                 }, ensure_ascii=False, indent=2),
                                 "status": {**crawl_status, "smtp": smtp_status, "scheduler": sched_status}})

        # ── 新增工具: force_update ──
        if tool_name == "force_update":
            from services.scheduler import _incremental_crawl_job
            result = _incremental_crawl_job()
            return JSONResponse({"success": result["success"], "content": result["message"],
                                 "new_articles": result.get("new_articles", 0)})

        # ── 原有工具 ──
        # 智能缓存: 仅对读库类工具自动检查数据新鲜度
        # trending_list 是实时 API 调用，不需要预爬取（反而会触发 API 限流）
        if tool_name in ("rag_search", "daily_digest", "trend_report"):
            note = _ensure_fresh_data(max_age_minutes=60)
            if note:
                print(f"[A2A] {note}")

        tool = tools_map.get(tool_name)
        if not tool:
            return JSONResponse({"success": False, "error": f"工具 '{tool_name}' 不存在"})

        result = tool.invoke(body)
        content = result.content if hasattr(result, 'content') else str(result)

        # 自动刷新已在后台完成，工具结果本身即是最新数据，不再拼接状态信息以免误导 LLM

        content = _safe_json_content(content)
        return JSONResponse({"success": True, "content": content})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.get("/graphs/{filename}")
async def serve_graph(filename: str):
    """提供知识图谱 HTML 文件（供 Web UI 嵌入）"""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "graphs")
    # 安全检查：只允许 .html 文件
    safe_name = os.path.basename(filename)
    if not safe_name.endswith(".html"):
        return JSONResponse({"error": "only .html files"}, status_code=403)
    fpath = os.path.join(base, safe_name)
    if not os.path.isfile(fpath):
        return JSONResponse({"error": f"not found: {safe_name}"}, status_code=404)
    # pyvis 写入中文 heading 时可能使用 GBK 编码，优先尝试
    html_content = ""
    for enc in ["utf-8", "gbk", "gb2312"]:
        try:
            with open(fpath, "r", encoding=enc) as f:
                html_content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if not html_content:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()
    return Response(content=html_content, media_type="text/html")


@app.get("/tools")
async def list_tools():
    return {"tools": A2A_TOOLS}


@app.on_event("startup")
async def startup_event():
    """A2A Server 启动时初始化调度器（KG 懒加载）"""
    try:
        from services.scheduler import start_scheduler
        start_scheduler()
        print("[A2A Server] ✅ 后台调度器已启动")
    except Exception as e:
        print(f"[A2A Server] 调度器启动跳过: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
