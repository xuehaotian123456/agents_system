"""博客园 RSS 爬虫"""
from datetime import datetime
import httpx
import feedparser
from bs4 import BeautifulSoup
from utils.logger_handler import logger

CNBLOGS_RSS = "https://feed.cnblogs.com/blog/sitehome/rss"

def fetch_cnblogs_rss(limit: int = 10) -> list[dict]:
    """获取博客园首页 RSS 文章"""
    try:
        resp = httpx.get(CNBLOGS_RSS, timeout=15)
        feed = feedparser.parse(resp.text)

        articles = []
        for entry in feed.entries[:limit]:
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:200]
            articles.append({
                "id": entry.get("id", ""),
                "title": entry.get("title", ""),
                "brief": summary,
                "author": entry.get("author", "未知"),
                "url": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": "博客园",
                "crawled_at": datetime.now().isoformat(),
            })
        logger.info(f"博客园 RSS 获取成功: {len(articles)} 条")
        return articles
    except Exception as e:
        logger.error(f"博客园 RSS 获取失败: {e}")
        return []

def fetch_cnblogs_article(url: str) -> str | None:
    """获取博客园单篇文章正文"""
    try:
        resp = httpx.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.find("title")
        title_text = title.get_text().split(" - ")[0] if title else ""

        body = soup.find("div", id="cnblogs_post_body")
        if not body:
            body = soup.find("div", class_="post-body")
        if not body:
            return None

        text = body.get_text("\n", strip=True)
        return f"# {title_text}\n\n> 来源: 博客园\n\n{text[:5000]}"
    except Exception as e:
        logger.error(f"博客园文章获取失败: {e}")
        return None
