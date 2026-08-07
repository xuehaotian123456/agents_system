"""爬虫工具"""
import os
from pathlib import Path
from langchain_core.tools import tool
from utils.logger_handler import logger

@tool(description="实时获取掘金技术热榜文章列表。无需参数，返回当前热门技术文章标题、摘要和链接。")
def trending_list(source: str = "juejin") -> str:
    try:
        if source == "all":
            from crawlers.multi_source import fetch_all_trending
            articles = fetch_all_trending(limit_per_source=5)
            if not articles:
                return "暂未获取到综合热榜数据。"
            lines = [f"🌍 多源技术热榜 Top {len(articles)}:\n"]
            for i, a in enumerate(articles[:15], 1):
                src_icon = {"HackerNews": "🟠", "GitHub Trending": "⬛", "OSChina": "🟢"}.get(a.get("source", ""), "📄")
                lines.append(f"\n{i}. {src_icon} [{a.get('source','')}] **{a.get('title','')[:80]}**")
                if a.get("brief"):
                    lines.append(f"   {a['brief'][:120]}")
                if a.get("url"):
                    lines.append(f"   链接: {a['url']}")
            return "\n".join(lines)

        if source == "hackernews":
            from crawlers.multi_source import fetch_hackernews
            articles = fetch_hackernews(limit=8)
            lines = [f"🟠 HackerNews Top {len(articles)}:\n"]
            for i, a in enumerate(articles, 1):
                lines.append(f"\n{i}. **{a['title'][:80]}**")
                if a.get("brief"):
                    lines.append(f"   {a['brief'][:120]}")
                lines.append(f"   链接: {a['url']}")
            return "\n".join(lines)

        if source == "github":
            from crawlers.multi_source import fetch_github_trending
            articles = fetch_github_trending(limit=8)
            lines = [f"⬛ GitHub Trending Top {len(articles)}:\n"]
            for i, a in enumerate(articles, 1):
                lines.append(f"\n{i}. **{a['title'][:80]}**")
                if a.get("brief"):
                    lines.append(f"   {a['brief'][:120]}")
                lines.append(f"   链接: {a['url']}")
            return "\n".join(lines)

        if source == "oschina":
            from crawlers.multi_source import fetch_oschina
            articles = fetch_oschina(limit=8)
            lines = [f"🟢 开源中国 Top {len(articles)}:\n"]
            for i, a in enumerate(articles, 1):
                lines.append(f"\n{i}. **{a['title'][:80]}**")
                if a.get("brief"):
                    lines.append(f"   {a['brief'][:120]}")
                lines.append(f"   链接: {a['url']}")
            return "\n".join(lines)

        if source == "juejin":
            from crawlers.juejin import fetch_juejin_hot
            articles = fetch_juejin_hot(limit=10)
            if not articles:
                return "暂未获取到掘金热榜数据。"

            lines = [f"🔥 掘金实时技术热榜 Top {len(articles)}:\n"]
            for i, a in enumerate(articles[:10], 1):
                title = a.get('title', '')
                brief = a.get('brief', '')
                if title:
                    lines.append(f"\n{i}. **{title[:80]}**")
                    if brief:
                        lines.append(f"   {brief[:120]}")
                else:
                    lines.append(f"\n{i}. {a['url']}")
                lines.append(f"   🔗 {a['url']}")

            lines.append(f"\n💡 使用 fetch_article 工具获取任意文章的完整内容。")
            return "\n".join(lines)

        elif source == "cnblogs":
            from crawlers.cnblogs import fetch_cnblogs_rss
            articles = fetch_cnblogs_rss(limit=10)
            if not articles:
                return "暂未获取到博客园RSS数据。"
            lines = [f"🔥 博客园技术热榜 Top {len(articles)}:"]
            for i, a in enumerate(articles, 1):
                lines.append(f"\n{i}. **{a.get('title', '')[:60]}**")
                lines.append(f"   作者: {a.get('author', '')}")
                lines.append(f"   链接: {a.get('url', '')}")
            return "\n".join(lines)

        return f"未知数据源: {source}，可选: juejin / cnblogs"
    except Exception as e:
        return f"获取热榜失败: {e}"

@tool(description="爬取并保存指定 URL 的技术文章。入参 url（文章链接），自动识别掘金/博客园，下载文章内容并存入本地知识库。返回文章摘要。")
def fetch_article(url: str) -> str:
    try:
        content = None
        title = ""

        if "juejin.cn/post/" in url:
            from crawlers.juejin import fetch_juejin_article
            article_id = url.split("/post/")[-1].split("?")[0].split("#")[0]
            article = fetch_juejin_article(article_id)
            if article:
                title = article.get('title', '')
                content = f"# {title}\n\n> 来源: 掘金  |  作者: {article.get('author', '')}\n\n{article.get('content', '')}"
        elif "cnblogs.com" in url:
            from crawlers.cnblogs import fetch_cnblogs_article
            content = fetch_cnblogs_article(url)
            for line in (content or "").split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()[:80]
                    break
        else:
            return f"暂不支持的链接格式: {url}\n支持: 掘金文章、博客园文章"

        if not content:
            return "无法获取文章内容。"

        data_dir = Path(__file__).parent.parent.parent / "data" / "articles"
        os.makedirs(data_dir, exist_ok=True)
        safe_name = "".join(c for c in (title or "article") if c.isalnum() or c in (' ', '-')).strip()[:40]
        fpath = data_dir / f"fetched_{safe_name}.md"
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)

        brief = content[content.find("\n\n"):][:300] if "\n\n" in content else content[:300]
        return f"✅ 文章已获取并保存\n标题: {title}\n摘要: {brief.strip()[:200]}...\n文件: {fpath.name}"
    except Exception as e:
        return f"文章获取失败: {e}"
