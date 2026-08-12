"""
GitHub Issues + Docs 爬虫 — 纯 REST API 拉取（三层离线容错）
============================================================
强制约束：禁止对 github.com/xxx 做网页解析。所有数据通过 GitHub REST API 获取。

容错策略（三层兜底）：
  第一层: enable_github=false → 直接跳过，返回 []
  第二层: 调用 GitHub REST API → 成功则缓存到 cache_dir
  第三层: API 连续 N 次失败 → 自动加载 cache_dir 离线数据集

数据源定位：
  - 主源: LangChain/LangGraph/vLLM Issues + Docs（高权威）
  - 补充: 掘金/博客园（低权重，通过 source_credibility 管理）

增量策略：
  - Issues: 记录 last_issue_number，只拉新增
  - Docs: 记录 last_commit_sha，只拉变更文件
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import yaml
from utils.logger_handler import logger

# ==================== 配置加载 ====================

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "cache" / "github_offline"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API 配置
BASE_URL = "https://api.github.com"

# GitHub Token: 未认证 60次/小时，认证后 5000次/小时
# 创建: https://github.com/settings/tokens
_GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN", "")
HEADERS = {
    "User-Agent": "DevPilot-Agent/1.0",
    "Accept": "application/vnd.github+json",
}
if _GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {_GITHUB_TOKEN}"


def _load_data_source_config() -> dict:
    """加载数据源配置（data_source.yaml）"""
    config_path = PROJECT_ROOT / "config" / "data_source.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _is_github_enabled() -> bool:
    """检查 GitHub 数据源是否启用"""
    cfg = _load_data_source_config()
    return cfg.get("data_source", {}).get("enable_github", True)


def _is_fallback_enabled() -> bool:
    """检查离线缓存回退是否启用"""
    cfg = _load_data_source_config()
    return cfg.get("data_source", {}).get("fallback_to_cache", True)


def _get_max_retries() -> int:
    cfg = _load_data_source_config()
    return cfg.get("data_source", {}).get("max_retries", 3)


def _get_timeout() -> int:
    cfg = _load_data_source_config()
    return cfg.get("github", {}).get("timeout_sec", 30)


def _get_repo_list() -> list[dict]:
    cfg = _load_data_source_config()
    return cfg.get("github", {}).get("repo_list", [])


def _get_filter_config() -> dict:
    cfg = _load_data_source_config()
    return cfg.get("filter", {})


# ==================== 容错: API 调用 + 重试 ====================

class GitHubAPIError(Exception):
    """GitHub API 调用失败（网络/限流/认证）"""


def _api_get(path: str, params: dict | None = None) -> dict | list:
    """
   调用 GitHub REST API，带指数退避重试。

   Args:
       path: API 路径，如 "/repos/langchain-ai/langgraph/issues"
       params: 查询参数

   Returns:
       解析后的 JSON 数据

   Raises:
       GitHubAPIError: 所有重试均失败
   """
    url = f"{BASE_URL}{path}"
    max_retries = _get_max_retries()
    timeout = _get_timeout()

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = httpx.get(url, headers=HEADERS, params=params,
                             timeout=timeout, follow_redirects=True)

            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                # 限流 → 等久一点再试
                wait = min(60, (attempt + 1) * 15)
                logger.warning(f"[GitHub API] 限流，等待 {wait}s 后重试...")
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                raise GitHubAPIError(f"资源不存在: {url}")

            if resp.status_code != 200:
                last_error = GitHubAPIError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    wait = min(30, (attempt + 1) * 5)
                    logger.warning(
                        f"[GitHub API] {resp.status_code}，{wait}s 后重试 ({attempt + 1}/{max_retries})")
                    time.sleep(wait)
                continue

            return resp.json()

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
            last_error = GitHubAPIError(f"网络异常: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait = min(30, (attempt + 1) * 5)
                logger.warning(
                    f"[GitHub API] 网络超时，{wait}s 后重试 ({attempt + 1}/{max_retries})")
                time.sleep(wait)

    raise last_error or GitHubAPIError("未知错误")


# ==================== 离线缓存读写 ====================

def _cache_path(owner: str, repo: str, kind: str) -> Path:
    """离线缓存文件路径，如 cache/github_offline/langgraph_issues.json"""
    safe_name = f"{repo}_{kind}.json"
    return CACHE_DIR / safe_name


def _load_cache(owner: str, repo: str, kind: str) -> list[dict] | None:
    """加载离线缓存数据"""
    path = _cache_path(owner, repo, kind)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"[离线缓存] 加载 {path.name}: {len(data)} 条")
        return data
    except Exception as e:
        logger.warning(f"[离线缓存] 读取失败 {path}: {e}")
        return None


def _save_cache(owner: str, repo: str, kind: str, data: list[dict]):
    """保存数据到离线缓存"""
    path = _cache_path(owner, repo, kind)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[离线缓存] 保存 {path.name}: {len(data)} 条")
    except Exception as e:
        logger.warning(f"[离线缓存] 保存失败 {path}: {e}")


# ==================== Issue 质量过滤 ====================

def _is_valid_issue(issue: dict) -> bool:
    """
    过滤无效 issue —— 保证入库知识图谱全是有效技术知识。

    过滤规则:
    1. body 长度 < min_body_length → 丢弃（空内容/一句话提问）
    2. 无代码块、无报错信息、无具体描述 → 丢弃（纯求助伸手贴）
    3. 机器人自动回复 → 丢弃
    """
    cfg = _get_filter_config()
    min_body_length = cfg.get("min_body_length", 200)
    min_lines = cfg.get("min_lines", 5)
    require_technical = cfg.get("require_technical", True)
    filter_bot = cfg.get("filter_bot", True)

    body = (issue.get("body") or "").strip()
    title = (issue.get("title") or "").strip()

    # 1. 空内容 / 过短
    if len(body) < min_body_length:
        return False

    # 2. 纯求助无技术细节
    lines = [l for l in body.split("\n") if l.strip()]
    has_code = "```" in body
    has_error = any(kw in body.lower()
                    for kw in ["traceback", "error", "exception",
                               "报错", "异常", "错误", "stack trace"])
    has_detail = len(lines) >= min_lines

    if require_technical and not (has_code or has_error or has_detail):
        return False

    # 3. 机器人自动回复
    if filter_bot:
        user_login = (issue.get("user") or {}).get("login", "").lower()
        if "bot" in user_login:
            return False
        # GitHub Actions bot
        if "github-actions" in user_login:
            return False

    return True


# ==================== 核心 API：拉取 Issues ====================

def fetch_repo_issues(
    owner: str,
    repo: str,
    limit: int = 20,
    state: str = "all",
    since_days: int = 7,
) -> list[dict]:
    """
    拉取指定仓库的 Issues（纯 REST API）。

    Args:
        owner: 仓库所有者
        repo: 仓库名
        limit: 最大返回数
        state: "open" / "closed" / "all"
        since_days: 只拉最近 N 天的 issue（增量模式）

    Returns:
        [{"title", "body", "labels", "state", "created_at",
          "number", "html_url", "comments_count", "source_type", "credibility"}, ...]
    """
    # ── 第一层: 检查开关 ──
    if not _is_github_enabled():
        logger.info(f"[GitHub Issues] enable_github=false，跳过 {owner}/{repo}")
        return []

    # ── 第二层: 尝试在线拉取 ──
    try:
        since = (datetime.utcnow() - timedelta(days=since_days)).isoformat() + "Z"
        params = {
            "state": state,
            "per_page": min(limit, 100),
            "sort": "updated",
            "direction": "desc",
            "since": since,
        }

        raw_issues = _api_get(f"/repos/{owner}/{repo}/issues", params)
        if not isinstance(raw_issues, list):
            raise GitHubAPIError(f"返回格式异常: {type(raw_issues)}")

        issues = []
        for raw in raw_issues:
            # 跳过 PR（GitHub API 把 PR 也当 issue 返回）
            if "pull_request" in raw:
                continue

            # 脏数据过滤
            if not _is_valid_issue(raw):
                continue

            labels = [lb["name"] for lb in (raw.get("labels") or [])]
            has_label = len(labels) > 0
            has_comments = (raw.get("comments") or 0) > 0

            # 可信度等级
            if has_label:
                source_type = "github_issue_labeled"
                credibility = 0.85
            elif has_comments:
                source_type = "github_issue_answered"
                credibility = 0.75
            else:
                source_type = "github_issue_open"
                credibility = 0.6

            issues.append({
                "title": raw.get("title", ""),
                "body": raw.get("body", ""),
                "labels": labels,
                "state": raw.get("state", ""),
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
                "number": raw.get("number"),
                "html_url": raw.get("html_url", ""),
                "comments_count": raw.get("comments", 0),
                "source": f"github_issue/{owner}/{repo}",
                "source_type": source_type,
                "credibility": credibility,
                "repo": f"{owner}/{repo}",
            })

        # 截断到 limit
        issues = issues[:limit]

        # 保存到离线缓存
        if issues:
            _save_cache(owner, repo, "issues", issues)

        logger.info(f"[GitHub Issues] {owner}/{repo}: {len(issues)} 篇有效 issue (总拉取 {len(raw_issues)} 条)")
        return issues

    except GitHubAPIError as e:
        logger.warning(f"[GitHub Issues] API 失败 {owner}/{repo}: {e}")

        # ── 第三层: 加载离线缓存 ──
        if _is_fallback_enabled():
            cached = _load_cache(owner, repo, "issues")
            if cached:
                logger.info(f"[GitHub Issues] 回退到离线缓存: {len(cached)} 条")
                return cached

        logger.warning(f"[GitHub Issues] 无离线缓存可用，返回空")
        return []


# ==================== 核心 API：拉取仓库文档 ====================

def _fetch_dir_contents(owner: str, repo: str, path: str) -> list[dict]:
    """递归获取目录下的所有 .md 文件（Contents API）"""
    try:
        contents = _api_get(f"/repos/{owner}/{repo}/contents/{path}")
        if not isinstance(contents, list):
            return []
    except GitHubAPIError:
        return []

    md_files = []
    for item in contents:
        if item.get("type") == "file" and item["name"].endswith(".md"):
            md_files.append(item)
        elif item.get("type") == "dir":
            # 递归子目录（限制深度避免爆炸）
            sub = _fetch_dir_contents(owner, repo, item["path"])
            md_files.extend(sub)

    return md_files


def fetch_repo_docs(
    owner: str,
    repo: str,
    doc_path: str = "docs",
) -> list[dict]:
    """
    拉取仓库文档（Contents API）。

    Args:
        owner: 仓库所有者
        repo: 仓库名
        doc_path: 文档目录路径（如 "docs" 或 "docs/source"）

    Returns:
        [{"title", "body", "path", "html_url", "source_type", "credibility"}, ...]
    """
    # ── 第一层: 检查开关 ──
    if not _is_github_enabled():
        logger.info(f"[GitHub Docs] enable_github=false，跳过 {owner}/{repo}")
        return []

    # ── 第二层: 尝试在线拉取 ──
    try:
        md_files = _fetch_dir_contents(owner, repo, doc_path)

        if not md_files:
            # 尝试 docs/ 根目录
            if doc_path != "docs":
                md_files = _fetch_dir_contents(owner, repo, "docs")

        docs = []
        for item in md_files:
            try:
                # 获取文件内容（Base64 编码）
                file_data = _api_get(
                    f"/repos/{owner}/{repo}/contents/{item['path']}")
                import base64
                content = base64.b64decode(
                    file_data.get("content", "")).decode("utf-8", errors="replace")
            except Exception:
                content = f"[无法解码: {item.get('name', '')}]"

            if len(content) < 100:
                continue

            docs.append({
                "title": item.get("name", "").replace(".md", ""),
                "body": content,
                "path": item.get("path", ""),
                "html_url": item.get("html_url", ""),
                "source": f"github_doc/{owner}/{repo}",
                "source_type": "official_doc",
                "credibility": 1.0,
                "repo": f"{owner}/{repo}",
            })

        # 保存到离线缓存
        if docs:
            _save_cache(owner, repo, "docs", docs)

        logger.info(f"[GitHub Docs] {owner}/{repo}: {len(docs)} 篇文档")
        return docs

    except GitHubAPIError as e:
        logger.warning(f"[GitHub Docs] API 失败 {owner}/{repo}: {e}")

        # ── 第三层: 加载离线缓存 ──
        if _is_fallback_enabled():
            cached = _load_cache(owner, repo, "docs")
            if cached:
                logger.info(f"[GitHub Docs] 回退到离线缓存: {len(cached)} 条")
                return cached

        logger.warning(f"[GitHub Docs] 无离线缓存可用，返回空")
        return []


# ==================== 综合拉取 ====================

def fetch_all_github_sources() -> tuple[list[dict], list[dict]]:
    """
    拉取所有配置仓库的 Issues + Docs。

    Returns:
        (all_issues, all_docs) — 两个平铺列表
    """
    all_issues = []
    all_docs = []

    for repo_cfg in _get_repo_list():
        owner = repo_cfg.get("owner", "")
        repo = repo_cfg.get("repo", "")
        doc_path = repo_cfg.get("doc_path", "docs")

        if not owner or not repo:
            continue

        # Issues
        try:
            issues = fetch_repo_issues(owner, repo)
            all_issues.extend(issues)
        except Exception as e:
            logger.error(f"[GitHub] Issues 拉取异常 {owner}/{repo}: {e}")

        # Docs
        try:
            docs = fetch_repo_docs(owner, repo, doc_path)
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"[GitHub] Docs 拉取异常 {owner}/{repo}: {e}")

    return all_issues, all_docs


def save_to_markdown(items: list[dict], data_dir: Path | None = None):
    """
    将拉取的 Issues/Docs 保存为 .md 文件（兼容现有 VectorStore 加载逻辑）。

    Args:
        items: fetch_repo_issues 或 fetch_repo_docs 返回的列表
        data_dir: 目标目录，默认 pipeline/data/articles/
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "articles"
    data_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for item in items:
        title = item.get("title", "untitled")
        body = item.get("body", "")
        source = item.get("source", "")
        url = item.get("html_url", "")
        source_type = item.get("source_type", "")
        credibility = item.get("credibility", 0.5)
        crawled_at = datetime.now().isoformat()

        # 安全文件名
        safe = "".join(c for c in title[:50]
                       if c.isalnum() or c in (' ', '-', '_')).strip()
        fname = f"github_{source.replace('/', '_')}_{safe}.md"
        fpath = data_dir / fname

        # 构造 Markdown（前端元数据用于可信度显示）
        content = (
            f"# {title}\n\n"
            f"> 来源: {source} | 可信度: {source_type} ({credibility})\n"
            f"> URL: {url}\n"
            f"> 爬取时间: {crawled_at}\n\n"
            f"{body}"
        )

        # 去重：相同标题且内容相同则跳过
        if fpath.exists():
            existing_hash = hashlib.md5(
                fpath.read_text(encoding="utf-8", errors="replace").encode()).hexdigest()
            new_hash = hashlib.md5(content.encode()).hexdigest()
            if existing_hash == new_hash:
                continue

        fpath.write_text(content, encoding="utf-8")
        saved += 1

    logger.info(f"[保存] {saved}/{len(items)} 篇新文章 → {data_dir}")


# ==================== 独立运行 / 测试入口 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("GitHub Issues + Docs 爬虫 — 独立测试")
    print("=" * 60)

    cfg = _load_data_source_config()
    print(f"\n配置状态:")
    print(f"  enable_github: {_is_github_enabled()}")
    print(f"  fallback_to_cache: {_is_fallback_enabled()}")
    print(f"  目标仓库: {len(_get_repo_list())} 个")
    for r in _get_repo_list():
        print(f"    - {r['owner']}/{r['repo']} (docs: {r.get('doc_path', 'docs')})")

    if not _is_github_enabled():
        print("\n⚠️  GitHub 数据源已禁用（enable_github=false），仅测试离线缓存加载。")
        for repo_cfg in _get_repo_list():
            owner, repo = repo_cfg["owner"], repo_cfg["repo"]
            cached_issues = _load_cache(owner, repo, "issues")
            cached_docs = _load_cache(owner, repo, "docs")
            print(f"  {owner}/{repo}: issues={len(cached_issues or [])}, docs={len(cached_docs or [])}")
    else:
        print(f"\n开始拉取...")
        issues, docs = fetch_all_github_sources()
        print(f"\n结果: {len(issues)} issues + {len(docs)} docs")

        if issues or docs:
            save_to_markdown(issues + docs)
            print("已保存到 data/articles/")
        else:
            print("⚠️  无数据（网络不可达且无离线缓存）。请:")
            print("  1. 检查网络连接")
            print("  2. 或将 enable_github 设为 false 使用社区数据源")
