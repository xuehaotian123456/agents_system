"""多源爬虫: HackerNews / GitHub Trending / OSChina"""
from datetime import datetime
import re
import httpx
from bs4 import BeautifulSoup
from utils.logger_handler import logger

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ==================== HackerNews ====================

def fetch_hackernews(limit: int = 10) -> list[dict]:
    """HackerNews 首页热门 (via hnrss RSS)"""
    try:
        url = "https://hnrss.org/frontpage"
        resp = httpx.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "xml")

        articles = []
        for item in soup.find_all("item")[:limit]:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            link = item.find("link").get_text(strip=True) if item.find("link") else ""
            desc = item.find("description")
            description = desc.get_text(strip=True)[:200] if desc else ""

            articles.append({
                "id": link, "title": title[:100], "url": link,
                "brief": description, "source": "HackerNews",
                "crawled_at": datetime.now().isoformat(),
            })

        logger.info(f"HackerNews: {len(articles)} 篇")
        return articles
    except Exception as e:
        logger.warning(f"HackerNews 获取失败: {e}")
        return []

# ==================== GitHub Trending ====================

def fetch_github_trending(limit: int = 10) -> list[dict]:
    """GitHub Trending 今日热门仓库"""
    try:
        url = "https://github.com/trending"
        resp = httpx.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        articles = []
        for repo in soup.find_all("article", class_="Box-row")[:limit]:
            h2 = repo.find("h2")
            if not h2:
                continue
            name = h2.get_text(strip=True).replace("\n", "").strip()
            desc_el = repo.find("p")
            desc = desc_el.get_text(strip=True)[:200] if desc_el else ""
            lang_el = repo.find("span", itemprop="programmingLanguage")
            lang = lang_el.get_text(strip=True) if lang_el else ""
            stars_el = repo.find("span", class_="d-inline-block float-sm-right")
            stars = stars_el.get_text(strip=True) if stars_el else ""

            articles.append({
                "id": name, "title": f"{name} ({lang})" if lang else name,
                "url": f"https://github.com/{name}",
                "brief": f"⭐ {stars} | {desc}", "source": "GitHub Trending",
                "crawled_at": datetime.now().isoformat(),
            })

        logger.info(f"GitHub Trending: {len(articles)} 篇")
        return articles
    except Exception as e:
        logger.warning(f"GitHub Trending 获取失败: {e}")
        return []

# ==================== OSChina ====================

def fetch_oschina(limit: int = 10) -> list[dict]:
    """开源中国资讯 RSS"""
    try:
        url = "https://www.oschina.net/news/rss"
        resp = httpx.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "xml")

        articles = []
        for item in soup.find_all("item")[:limit]:
            title = item.find("title").get_text(strip=True) if item.find("title") else ""
            link = item.find("link").get_text(strip=True) if item.find("link") else ""
            desc = item.find("description")
            description = desc.get_text(strip=True)[:200] if desc else ""

            articles.append({
                "id": link, "title": title[:100], "url": link,
                "brief": description, "source": "OSChina",
                "crawled_at": datetime.now().isoformat(),
            })

        logger.info(f"OSChina: {len(articles)} 篇")
        return articles
    except Exception as e:
        logger.warning(f"OSChina 获取失败: {e}")
        return []

# ==================== 综合热榜 ====================

def fetch_all_trending(limit_per_source: int = 5) -> list[dict]:
    """综合获取多源热榜"""
    all_articles = []
    for fetcher in [fetch_hackernews, fetch_github_trending, fetch_oschina]:
        try:
            articles = fetcher(limit_per_source)
            all_articles.extend(articles)
        except Exception:
            pass
    return all_articles
