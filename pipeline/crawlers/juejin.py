"""掘金爬虫 — API 获取 ID 列表 + 网页爬取文章内容"""
from datetime import datetime
import re
import httpx
from bs4 import BeautifulSoup
from utils.logger_handler import logger

BASE = "https://api.juejin.cn"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_juejin_hot(limit: int = 10) -> list[dict]:
    """获取掘金文章 ID 列表"""
    try:
        url = f"{BASE}/recommend_api/v1/article/recommend_all_feed"
        payload = {"id_type": 2, "sort_type": 200, "cursor": "0", "limit": limit}
        resp = httpx.post(url, json=payload, timeout=15)
        data = resp.json()

        articles = []
        for item in data.get("data", []):
            info = item.get("item_info", {})
            aid = str(info.get("article_id", ""))
            if aid:
                article_info = info.get("article_info", {})
                title = article_info.get("title", "")
                brief = article_info.get("brief_content", "")[:200]
                articles.append({
                    "id": aid,
                    "url": f"https://juejin.cn/post/{aid}",
                    "title": title,
                    "brief": brief,
                    "source": "掘金",
                    "crawled_at": datetime.now().isoformat(),
                })

        logger.info(f"掘金文章列表: {len(articles)} 篇")
        return articles
    except Exception as e:
        logger.error(f"掘金列表获取失败: {e}")
        return []

def fetch_juejin_article(article_id: str) -> dict | None:
    """网页爬取掘金文章详情"""
    try:
        url = f"https://juejin.cn/post/{article_id}"
        resp = httpx.get(url, headers=HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # 标题
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
            # 去掉 " - 掘金" 后缀
            title = re.sub(r'\s*[-|–—]\s*掘金\s*$', '', title)

        # 文章内容
        article = soup.find("article")
        if not article:
            return None

        # 提取图片
        images = []
        for img in article.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            if src and not src.endswith(".svg") and "avatar" not in src and "icon" not in src:
                # 补全协议
                if src.startswith("//"):
                    src = "https:" + src
                images.append(src)

        # 提取纯文本，跳过脚本和样式
        for tag in article.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        content = article.get_text("\n", strip=True)
        if not content or len(content) < 100:
            return None

        # 去掉开头的作者信息噪音
        lines = content.split("\n")
        clean_lines = []
        started = False
        for line in lines:
            if len(line) > 50 and not any(kw in line for kw in ["阅读", "点赞", "收藏", "举报"]):
                started = True
            if started:
                clean_lines.append(line)

        content = "\n\n".join(clean_lines)

        return {
            "id": article_id,
            "title": title,
            "content": content,
            "url": url,
            "source": "掘金",
            "images": images[:5],  # 前 5 张图
            "crawled_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"掘金文章获取失败 {article_id}: {e}")
        return None
