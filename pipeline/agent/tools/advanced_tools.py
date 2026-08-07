"""高级工具: 技术对比、每日摘要、代码搜索、动态工具创建"""
import json
import os
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool
from utils.logger_handler import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ===== Daily Digest =====

@tool(description="生成今日技术摘要报告。无入参，自动分析最近爬取的文章，提取Top 5热点话题、关键词趋势和推荐阅读。用于快速了解今日技术动态。")
def daily_digest() -> str:
    """从最近爬取的文章生成每日技术摘要（增强版：含链接、摘要、来源分组）"""
    try:
        from services.trends import analyze_trends
        from services.crawl_state import get_crawl_state
        from datetime import datetime

        trends = analyze_trends()
        state = get_crawl_state()

        if not trends["keywords"]:
            return "暂无数据生成摘要。请先等待系统自动爬取或使用 force_update 手动刷新。"

        top_kw = trends["keywords"][:15]
        daily = trends.get("daily_trends", [])
        today_str = datetime.now().strftime("%Y-%m-%d")

        lines = [
            f"# 📰 DevPilot 每日技术摘要",
            f"**{today_str}** · 基于 {trends['article_count']} 篇文章 · {state.get_article_count()} 篇入库\n",
        ]

        # ── 热点话题 ──
        lines.append("## 🔥 热点话题 Top 10")
        for i, (word, freq) in enumerate(top_kw[:10], 1):
            bar = "█" * min(20, freq // 30)
            lines.append(f"{i:2d}. **{word}** {bar} ({freq})")
        lines.append("")

        # ── 最新文章（从 crawl_state 获取带链接的） ──
        recent = state.get_recent_articles(hours=48)
        if recent:
            lines.append(f"## 📄 最新收录文章 ({len(recent)} 篇)")
            lines.append("")
            # 按来源分组
            from collections import defaultdict
            by_source = defaultdict(list)
            for r in recent:
                src = r.get("source", "其他")
                by_source[src].append(r)

            source_icons = {
                "掘金": "🔥", "博客园": "📘", "GitHub Trending": "⬛",
                "GitHub": "⬛", "HackerNews": "🟠", "OSChina": "🟢",
            }
            for src, articles in by_source.items():
                icon = source_icons.get(src, "📄")
                lines.append(f"### {icon} {src} ({len(articles)} 篇)")
                for a in articles[:5]:
                    title = a.get("title", "")[:80]
                    url = a.get("url", "")
                    brief = a.get("brief", "")
                    crawled = a.get("crawled_at", "")[:10]
                    if url:
                        lines.append(f"- [{title}]({url})")
                    else:
                        lines.append(f"- {title}")
                    if brief:
                        lines.append(f"  > {brief[:150]}")
                lines.append("")
        else:
            # Fallback: from trends timeline
            if trends.get("timeline"):
                lines.append("## 📖 推荐阅读")
                for item in trends["timeline"][:8]:
                    lines.append(f"- [{item['date']}] {item['title'][:80]}")
                lines.append("")

        # ── 每日趋势 ──
        if daily:
            today = daily[0]
            kw_str = " · ".join([f"{w}" for w, c in today["keywords"][:8]])
            lines.append(f"## 📅 今日技术热词")
            lines.append(f"> {kw_str}")
            lines.append("")

        # ── 系统状态 ──
        status = state.get_status_summary()
        lines.append("## ⚙️ 系统状态")
        lines.append(f"- 📚 已入库文章: {status['total_articles_crawled']} 篇 ({len(recent)} 篇 48h 内)")
        lines.append(f"- 🕐 最后爬取: {status.get('last_full_crawl', '未知')[:19] if status.get('last_full_crawl') else '未知'}")
        lines.append(f"- 📬 每日推送: {'✅ 已启用' if state.get_digest_config().get('enabled') else '❌ 未启用'}")
        lines.append("")
        lines.append("> 💡 使用 `send_digest_email` 将此摘要发送到邮箱")
        lines.append("> 💡 使用 `fetch_article` 获取任意文章完整内容")

        # ── 可视化 ──
        try:
            from services.trends import generate_trend_html
            viz_path = generate_trend_html()
            if viz_path:
                lines.append(f"\n[VIZ:{viz_path}]")
        except Exception:
            pass

        return "\n".join(lines)
    except Exception as e:
        return f"生成摘要失败: {e}"

# ===== Tech Compare =====

@tool(description="对比两个技术/框架，基于知识图谱和知识库数据。入参 tech_a 和 tech_b（两个技术名词），返回多维度对比分析。例如 compare_tech('React', 'Vue')。")
def compare_tech(tech_a: str, tech_b: str) -> str:
    """基于KG共现数据对比两个技术"""
    try:
        from rag.knowledge_graph import get_kg
        kg = get_kg()
        if not kg.is_built:
            return "知识图谱未构建。"

        info_a = kg.get_entity(tech_a)
        info_b = kg.get_entity(tech_b)

        lines = [f"# {tech_a} vs {tech_b} 技术对比\n"]

        # KG 数据
        lines.append("## 📊 知识图谱数据")
        if info_a:
            lines.append(f"**{tech_a}**: 频次 {info_a['freq']}, 覆盖 {info_a['chunks']} 个文档块")
            related_a = [r['entity'] for r in info_a.get('related', [])[:5]]
            lines.append(f"  关联技术: {', '.join(related_a)}")
        else:
            lines.append(f"**{tech_a}**: 知识库中无数据")

        if info_b:
            lines.append(f"**{tech_b}**: 频次 {info_b['freq']}, 覆盖 {info_b['chunks']} 个文档块")
            related_b = [r['entity'] for r in info_b.get('related', [])[:5]]
            lines.append(f"  关联技术: {', '.join(related_b)}")
        else:
            lines.append(f"**{tech_b}**: 知识库中无数据")

        # 共现关系
        if info_a and info_b:
            co = kg.co_occurrence.get(tech_a, {}).get(tech_b, 0)
            if co > 0:
                lines.append(f"\n🔗 {tech_a} 与 {tech_b} 在同一文档块中共现 {co} 次")
            else:
                lines.append(f"\n🔗 {tech_a} 与 {tech_b} 在知识库中无直接共现")

        # 从知识库搜索相关文章
        try:
            from agent.tools.rag_tools import rag_search
            result = rag_search.invoke({"query": f"{tech_a} 和 {tech_b} 对比区别"})
            content = result.content if hasattr(result, 'content') else str(result)
            if "资料不足" not in content and len(content) > 50:
                lines.append(f"\n## 📚 知识库参考\n{content[:600]}")
        except Exception:
            pass

        return "\n".join(lines)
    except Exception as e:
        return f"对比失败: {e}"

# ===== Code Search =====

@tool(description="从知识库中搜索代码示例。入参 keyword（技术关键词），返回相关的代码片段和用法示例。")
def code_example(keyword: str) -> str:
    """搜索代码示例"""
    try:
        articles_dir = DATA_DIR / "articles"
        if not articles_dir.exists():
            return "暂无文章数据。"

        results = []
        for fpath in articles_dir.glob("*.md"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                # 找代码块: ```...```
                import re
                blocks = re.findall(r'```(\w+)?\n(.*?)```', content, re.DOTALL)
                for lang, code in blocks:
                    if keyword.lower() in code.lower() or keyword.lower() in (lang or "").lower():
                        results.append({
                            "file": fpath.name,
                            "lang": lang or "text",
                            "code": code.strip()[:500]
                        })
                        if len(results) >= 5:
                            break

                # 也搜行内代码 `...`
                if len(results) < 3:
                    inline = re.findall(r'`([^`]{10,200}?)`', content)
                    for snippet in inline:
                        if keyword.lower() in snippet.lower():
                            results.append({
                                "file": fpath.name,
                                "lang": "inline",
                                "code": snippet.strip()[:200]
                            })
                            if len(results) >= 5:
                                break

                if len(results) >= 5:
                    break
            except Exception:
                continue

        if not results:
            return f"未找到与 '{keyword}' 相关的代码示例。试试更具体的关键词如 'async'、'docker'、'useState'。"

        lines = [f"🔍 '{keyword}' 代码示例 ({len(results)} 个):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n### {i}. {r['file'][:40]} ({r['lang']})")
            lines.append(f"```{r['lang']}\n{r['code']}\n```")
        return "\n".join(lines)
    except Exception as e:
        return f"代码搜索失败: {e}"

# ===== Dynamic Tool Creator =====

_CUSTOM_TOOLS_FILE = DATA_DIR / "custom_tools.json"

def _load_custom_tools() -> list[dict]:
    if _CUSTOM_TOOLS_FILE.exists():
        with open(_CUSTOM_TOOLS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_custom_tools(tools: list[dict]):
    _CUSTOM_TOOLS_FILE.parent.mkdir(exist_ok=True)
    with open(_CUSTOM_TOOLS_FILE, "w", encoding="utf-8") as f:
        json.dump(tools, f, ensure_ascii=False, indent=2)

@tool(description="创建一个自定义搜索工具。入参 name（工具名）、description（用途描述）、keywords（搜索关键词列表）。创建后可通过 custom_search 工具调用。例如创建一个名为'vue3_tracker'的工具，用于跟踪Vue3最新动态。")
def create_tool(name: str, description: str, keywords: str) -> str:
    """创建自定义工具（关键词以逗号分隔）"""
    try:
        tools = _load_custom_tools()
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]

        # 去重
        for t in tools:
            if t["name"] == name:
                t["description"] = description
                t["keywords"] = kw_list
                _save_custom_tools(tools)
                return f"✅ 工具 '{name}' 已更新（{len(kw_list)} 个关键词: {', '.join(kw_list)}）"

        tools.append({"name": name, "description": description, "keywords": kw_list,
                       "created": datetime.now().strftime("%Y-%m-%d %H:%M")})
        _save_custom_tools(tools)
        return f"✅ 工具 '{name}' 已创建（{len(kw_list)} 个关键词: {', '.join(kw_list)}）\n\n使用方式: 在对话中说'调用 {name}' 或直接问相关问题，Agent 会自动搜索。"
    except Exception as e:
        return f"创建失败: {e}"

@tool(description="执行自定义搜索工具。入参 tool_name（工具名）和 query（可选搜索查询），返回基于该工具关键词的搜索结果。")
def custom_search(tool_name: str, query: str = "") -> str:
    """执行自定义工具搜索"""
    try:
        tools = _load_custom_tools()
        tool_info = None
        for t in tools:
            if t["name"] == tool_name:
                tool_info = t
                break

        if not tool_info:
            available = [t["name"] for t in tools]
            return f"未找到工具 '{tool_name}'。可用自定义工具: {', '.join(available) if available else '暂无'}"

        keywords = tool_info["keywords"]
        search_query = query if query else " ".join(keywords)

        # 在文章库中搜索
        articles_dir = DATA_DIR / "articles"
        results = []
        for fpath in articles_dir.glob("*.md"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                score = sum(1 for kw in keywords if kw.lower() in content.lower())
                if score > 0:
                    # 找标题
                    title = ""
                    for line in content.split("\n"):
                        if line.startswith("# "):
                            title = line[2:].strip()[:60]
                            break
                    results.append({"file": fpath.name, "title": title or fpath.stem, "score": score,
                                    "snippet": content[:300]})
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)

        if not results:
            return f"工具 '{tool_name}' (关键词: {', '.join(keywords)}) 未找到匹配文章。"

        lines = [f"🔍 '{tool_name}' 搜索结果 ({len(results)} 篇):"]
        for i, r in enumerate(results[:5], 1):
            lines.append(f"\n{i}. **{r['title'][:50]}** (匹配度: {r['score']}/{len(keywords)})")
            lines.append(f"   文件: {r['file']}")
        return "\n".join(lines)
    except Exception as e:
        return f"自定义搜索失败: {e}"

@tool(description="爬取最新文章并生成一份信息流日报（含标题、摘要、图片、链接）。无入参，自动从掘金/博客园/GitHub/OSChina多源采集，生成HTML格式的每日技术摘要并保存。可用于'推送今日摘要'、'生成日报'等场景。")
def push_daily_digest() -> str:
    try:
        from services.digest_mail import crawl_all_sources, generate_digest
        articles = crawl_all_sources()
        html = generate_digest(articles)
        out = str(PROJECT_ROOT / "data" / f"digest_{datetime.now().strftime('%Y%m%d')}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)

        # 如果配置了SMTP，尝试发送
        mail_status = ""
        try:
            from services.digest_mail import load_smtp, send_email
            smtp = load_smtp()
            if smtp:
                to = os.environ.get("DIGEST_EMAIL_TO", "")
                if to:
                    send_email(html, to, smtp)
                    mail_status = "，已发送邮件"
        except Exception:
            pass

        return f"✅ 今日技术摘要已生成{mail_status}\n📄 共收录 {len(articles)} 篇文章\n📊 来源: {len(set(a['source'] for a in articles))} 个平台\n📎 报告: {out}"
    except Exception as e:
        return f"生成失败: {e}"

def list_custom_tools() -> str:
    """列出所有自定义工具"""
    tools = _load_custom_tools()
    if not tools:
        return "暂无自定义工具。使用 create_tool 创建你的第一个工具吧！"
    lines = ["🛠️ 自定义工具列表:"]
    for t in tools:
        lines.append(f"  • **{t['name']}**: {t['description']} (关键词: {', '.join(t['keywords'])})")
    return "\n".join(lines)
