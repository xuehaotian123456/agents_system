"""
爬取状态跟踪 + URL 去重
========================
职责:
1. 记录每篇文章的 url / title / source / crawled_at / content_hash
2. 新文章入库前查重 —— URL 已存在则跳过
3. 跟踪每个 source 的最后爬取时间
4. 支持"增量模式"——只爬新文章
5. 支持"全量模式"——强制重新爬取所有源

持久化: data/crawl_state.json
"""

from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class CrawlState:
    """
    爬取状态管理器

    使用方式:
        state = CrawlState()

        # 爬取前检查
        if state.should_crawl("juejin", max_age_minutes=60):
            articles = fetch_juejin_hot()

        # 入库前去重
        for article in articles:
            if state.is_duplicate(article["url"]):
                continue
            state.mark_crawled(article)
            save_to_kb(article)
    """

    def __init__(self, state_path: str | None = None):
        if state_path is None:
            project_root = Path(__file__).parent.parent
            state_path = str(project_root / "data" / "crawl_state.json")

        self.state_path = Path(state_path)
        self._data: dict = self._load()

    # ==================== 持久化 ====================

    def _load(self) -> dict:
        """从 JSON 加载状态"""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "articles": {},        # {url: {title, source, crawled_at, content_hash}}
            "last_crawl": {},      # {source_name: "2026-08-06T12:00:00"}
            "last_full_crawl": None,  # 最近一次全量爬取时间
            "repo_state": {},      # GitHub 仓库增量状态: {"owner/repo": {"last_issue_number": N, "last_doc_sha": "..."}}
            "smtp_config": {       # 用户通过 Harness 对话配置的 SMTP（优先于 .env）
                "host": "",
                "port": 465,
                "user": "",
                "password": "",
                "from_addr": "",
            },
            "daily_digest_config": {  # 每日摘要配置
                "enabled": False,
                "email": "",
                "time": "08:00",
            },
        }

    def _save(self):
        """持久化到 JSON 文件"""
        os.makedirs(self.state_path.parent, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ==================== 去重检查 ====================

    def is_duplicate(self, url: str) -> bool:
        """检查 URL 是否已爬取过"""
        return url in self._data["articles"]

    def is_content_duplicate(self, content: str) -> bool:
        """通过内容 hash 检查是否重复（比 URL 去重更稳）"""
        content_hash = self._hash_content(content)
        for record in self._data["articles"].values():
            if record.get("content_hash") == content_hash:
                return True
        return False

    # ==================== 标记已爬 ====================

    def mark_crawled(self, article: dict):
        """
        标记一篇文章已爬取

        Args:
            article: 文章字典，必须包含 url 字段，可选 title / source / content
        """
        url = article.get("url", "")
        if not url:
            return

        content = article.get("content", "") or article.get("brief", "")
        content_hash = self._hash_content(content)

        self._data["articles"][url] = {
            "title": article.get("title", "")[:200],
            "source": article.get("source", "unknown"),
            "crawled_at": datetime.now().isoformat(),
            "content_hash": content_hash,
        }
        self._save()

    def mark_batch_crawled(self, articles: list[dict]):
        """批量标记"""
        for a in articles:
            self.mark_crawled(a)

    # ==================== 爬取时机判断 ====================

    def should_crawl(self, source: str, max_age_minutes: int = 60) -> bool:
        """
        判断是否需要重新爬取某个源

        Args:
            source: 数据源名称 (juejin / cnblogs / github / hackernews / oschina)
            max_age_minutes: 缓存有效期（分钟），默认 60 分钟

        Returns:
            True 表示需要爬取，False 表示缓存仍有效
        """
        last = self._data["last_crawl"].get(source)
        if not last:
            return True  # 从未爬过

        try:
            last_time = datetime.fromisoformat(last)
            age = (datetime.now() - last_time).total_seconds() / 60
            return age >= max_age_minutes
        except (ValueError, TypeError):
            return True

    def should_full_crawl(self, max_age_hours: int = 6) -> bool:
        """判断是否需要全量爬取"""
        last = self._data.get("last_full_crawl")
        if not last:
            return True
        try:
            last_time = datetime.fromisoformat(last)
            age_hours = (datetime.now() - last_time).total_seconds() / 3600
            return age_hours >= max_age_hours
        except (ValueError, TypeError):
            return True

    def should_refresh_from_cache(self, source: str, max_age_minutes: int = 60) -> bool:
        """判断是否可以直接用缓存（与 should_crawl 相反）"""
        return not self.should_crawl(source, max_age_minutes)

    # ==================== 更新爬取时间 ====================

    def mark_source_crawled(self, source: str):
        """更新某个源的最近爬取时间"""
        self._data["last_crawl"][source] = datetime.now().isoformat()
        self._save()

    def mark_full_crawl_done(self):
        """标记全量爬取完成"""
        self._data["last_full_crawl"] = datetime.now().isoformat()
        self._save()

    # ==================== 每日摘要配置 ====================

    def get_digest_config(self) -> dict:
        """获取每日摘要配置"""
        return self._data.get("daily_digest_config", {
            "enabled": False, "email": "", "time": "08:00"
        })

    def set_digest_config(self, email: str, time: str = "08:00", enabled: bool = True):
        """设置每日摘要配置"""
        self._data["daily_digest_config"] = {
            "enabled": enabled,
            "email": email,
            "time": time,
        }
        self._save()

    # ==================== SMTP 配置（用户通过 Harness 对话设置，优先于 .env） ====================

    def get_smtp_config(self) -> dict:
        """获取用户配置的 SMTP。返回空值表示未配置——此时应 fallback 到 .env"""
        return self._data.get("smtp_config", {})

    def set_smtp_config(self, host: str = "", port: int = 465, user: str = "",
                        password: str = "", from_addr: str = ""):
        """设置 SMTP 配置（通过 Harness 对话调用）"""
        self._data["smtp_config"] = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "from_addr": from_addr or user,
        }
        self._save()

    def has_smtp_configured(self) -> bool:
        """检查用户是否已通过对话配置了 SMTP"""
        cfg = self._data.get("smtp_config", {})
        return bool(cfg.get("host") and cfg.get("user") and cfg.get("password"))

    # ==================== GitHub 仓库增量状态 ====================

    def get_repo_state(self, owner: str, repo: str) -> dict:
        """获取指定仓库的增量爬取状态"""
        key = f"{owner}/{repo}"
        return self._data.get("repo_state", {}).get(key, {
            "last_issue_number": 0,
            "last_doc_sha": "",
            "last_crawl_time": None,
        })

    def update_repo_state(self, owner: str, repo: str, **kwargs):
        """更新仓库增量状态（issue_number, doc_sha 等）"""
        if "repo_state" not in self._data:
            self._data["repo_state"] = {}
        key = f"{owner}/{repo}"
        current = self._data["repo_state"].get(key, {})
        current.update(kwargs)
        current["last_crawl_time"] = datetime.now().isoformat()
        self._data["repo_state"][key] = current
        self._save()

    # ==================== 状态查询 ====================

    def get_last_crawl_time(self, source: str = "") -> Optional[str]:
        """获取最近爬取时间"""
        if source:
            return self._data["last_crawl"].get(source)
        return self._data.get("last_full_crawl")

    def get_article_count(self) -> int:
        """获取已爬取文章总数"""
        return len(self._data["articles"])

    def get_recent_articles(self, hours: int = 24) -> list[dict]:
        """获取最近 N 小时内爬取的文章"""
        cutoff = datetime.now().isoformat()
        recent = []
        for url, info in self._data["articles"].items():
            try:
                crawled_at = info.get("crawled_at", "")
                if crawled_at:
                    dt = datetime.fromisoformat(crawled_at)
                    age_hours = (datetime.now() - dt).total_seconds() / 3600
                    if age_hours <= hours:
                        recent.append({"url": url, **info})
            except (ValueError, TypeError):
                pass
        return sorted(recent, key=lambda x: x.get("crawled_at", ""), reverse=True)

    def get_status_summary(self) -> dict:
        """获取整体状态摘要（给 A2A get_pipeline_status 用）"""
        last_crawls = {}
        for src, ts in self._data["last_crawl"].items():
            try:
                dt = datetime.fromisoformat(ts)
                last_crawls[src] = {
                    "time": ts,
                    "minutes_ago": round((datetime.now() - dt).total_seconds() / 60, 1),
                }
            except Exception:
                last_crawls[src] = {"time": ts, "minutes_ago": "unknown"}

        return {
            "total_articles_crawled": len(self._data["articles"]),
            "last_full_crawl": self._data.get("last_full_crawl"),
            "last_crawl_by_source": last_crawls,
            "daily_digest": self._data.get("daily_digest_config", {}),
            "recent_24h": len(self.get_recent_articles(24)),
        }

    # ==================== 工具方法 ====================

    @staticmethod
    def _hash_content(content: str) -> str:
        """计算内容 SHA256 哈希（前 16 位）"""
        if not content:
            return ""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ==================== 全局单例 ====================

_crawl_state: Optional[CrawlState] = None


def get_crawl_state() -> CrawlState:
    """获取全局 CrawlState 单例"""
    global _crawl_state
    if _crawl_state is None:
        _crawl_state = CrawlState()
    return _crawl_state
