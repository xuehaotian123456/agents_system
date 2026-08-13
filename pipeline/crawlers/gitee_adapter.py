"""
Gitee API 适配器 — 复用 GitHub Issues 爬虫的全部下游逻辑
========================================================
Gitee API v5 与 GitHub API 高度相似，本模块:
1. 替换 API 域名 + 请求参数格式
2. 输出 schema 与 github_issues.py 完全一致
3. 复用 source_credibility 可信度打分（新增 Gitee 来源分级）
4. 复用 save_to_markdown / 增量状态 / 离线缓存

限流策略:
- 全局节流: 每次 API 调用间隔 >= 2s，避免触发限流
- 文档递归深度限制: max 2 层，优先拉 README 和顶层 .md
- 403/429 智能退避: 识别限流响应，等待后重试

Gitee vs GitHub 差异处理:
- Issue number: string ("IDLSUI") -> 作为 str 存储
- html_url: https://gitee.com/{owner}/{repo}/issues/{number}
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

PROJECT_ROOT = Path(__file__).parent.parent

# 加载 .env 中的环境变量（GITEE_ACCESS_TOKEN 等）
try:
    from dotenv import load_dotenv
    for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
except ImportError:
    pass

# ==================== 配置 ====================

BASE_URL = "https://gitee.com/api/v5"

# ★ Gitee Access Token: 未认证 60次/小时，认证后 5000次/小时
# 创建: https://gitee.com/profile/personal_access_tokens
# 权限: 只勾选 user_info + projects（只读即可）
_GITEE_TOKEN = os.getenv("GITEE_ACCESS_TOKEN", "")
_has_token = bool(_GITEE_TOKEN)

HEADERS = {
    "User-Agent": "DevPilot-Agent/1.0",
    "Accept": "application/json",
}
_last_api_call = 0.0  # 全局限流时间戳

if _has_token:
    HEADERS["Authorization"] = f"Bearer {_GITEE_TOKEN}"
    # 有 Token 时降低节流间隔：5000次/小时 = 1.4次/秒，保守取 0.5s
    _RATE_LIMIT_INTERVAL = 0.5
else:
    _RATE_LIMIT_INTERVAL = 2.0

# ★ 文档递归深度上限：避免 MindSpore 这种 api_python 下有 50+ 子目录
_MAX_DOC_DEPTH = 2


def _load_config() -> dict:
    path = PROJECT_ROOT / "config" / "data_source.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_gitee_repos() -> list[dict]:
    cfg = _load_config()
    return cfg.get("gitee", {}).get("repo_list", [])


def _get_timeout() -> int:
    cfg = _load_config()
    return cfg.get("gitee", {}).get("timeout_sec", 30)


def _get_max_retries() -> int:
    cfg = _load_config()
    return cfg.get("data_source", {}).get("max_retries", 3)


# ==================== 全局限流 ====================

def _rate_limit_wait():
    """每次 API 调用前调用，确保不超频率限制"""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < _RATE_LIMIT_INTERVAL:
        time.sleep(_RATE_LIMIT_INTERVAL - elapsed)
    _last_api_call = time.time()


# ==================== API 调用 ====================

def _api_get(path: str, params: dict | None = None) -> dict | list:
    """Gitee API 调用，带全局节流 + 智能限流退避"""
    url = f"{BASE_URL}{path}"
    max_retries = _get_max_retries()
    timeout = _get_timeout()

    last_error = None
    for attempt in range(max_retries):
        try:
            _rate_limit_wait()  # ★ 节流：每次请求前等待
            resp = httpx.get(url, headers=HEADERS, params=params,
                             timeout=timeout, follow_redirects=True)

            # Gitee 未认证 API 超限时返回 403（而非标准 429）
            is_rate_limited = (
                resp.status_code == 429 or
                (resp.status_code == 403 and
                 any(kw in resp.text.lower() for kw in
                     ["rate", "exceeded", "limit", "频繁", "太频繁", "稍后再试"]))
            )

            if is_rate_limited:
                wait = min(180, 30 * (attempt + 1))
                logger.warning(
                    f"[Gitee API] 触发限流 (HTTP {resp.status_code})，"
                    f"等待 {wait}s 后重试 ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue

            if resp.status_code == 403:
                logger.warning(f"[Gitee API] 403 权限不足: {url}")
                return []

            if resp.status_code == 404:
                return []

            if resp.status_code != 200:
                last_error = Exception(
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    wait = min(30, 5 * (attempt + 1))
                    time.sleep(wait)
                continue

            return resp.json()

        except (httpx.TimeoutException, httpx.ConnectError,
                httpx.ReadError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = min(30, 5 * (attempt + 1))
                logger.warning(f"[Gitee API] 网络异常，{wait}s 后重试")
                time.sleep(wait)

    logger.warning(f"[Gitee API] 所有重试失败: {last_error}")
    return []


# ==================== Issue 质量过滤 ====================

def _is_valid_issue(issue: dict) -> bool:
    """Gitee issue 过滤（比 GitHub 更宽松，Gitee issue 平均较短）"""
    body = (issue.get("body") or "").strip()
    if len(body) < 100:
        return False
    lines = [l for l in body.split("\n") if l.strip()]
    has_code = "```" in body
    has_error = any(kw in body.lower()
                    for kw in ["traceback", "error", "exception",
                               "报错", "异常", "错误", "stack trace"])
    has_detail = len(lines) >= 3
    if not (has_code or has_error or has_detail):
        return False
    return True


# ==================== 核心: 拉取 Issues ====================

def fetch_gitee_issues(owner: str, repo: str, limit: int = 25,
                       state: str = "all", since_days: int = 60) -> list[dict]:
    """拉取 Gitee 仓库 Issues"""
    params = {
        "state": state,
        "per_page": min(limit, 100),
        "sort": "updated",
        "direction": "desc",
    }

    raw_issues = _api_get(f"/repos/{owner}/{repo}/issues", params)
    if not isinstance(raw_issues, list):
        return []

    issues = []
    for raw in raw_issues:
        if not isinstance(raw, dict):
            continue
        if not _is_valid_issue(raw):
            continue

        labels = [lb["name"] for lb in (raw.get("labels") or [])
                  if isinstance(lb, dict)]
        has_label = len(labels) > 0
        has_comments = (raw.get("comments") or 0) > 0

        if "bug" in labels or "Bug" in labels:
            source_type = "gitee_issue_labeled"
            credibility = 0.80
        elif has_label:
            source_type = "gitee_issue_labeled"
            credibility = 0.75
        elif has_comments:
            source_type = "gitee_issue_normal"
            credibility = 0.65
        else:
            source_type = "gitee_issue_normal"
            credibility = 0.60

        number = raw.get("number", "")

        issues.append({
            "title": raw.get("title", ""),
            "body": raw.get("body", ""),
            "labels": labels,
            "state": raw.get("state", ""),
            "created_at": raw.get("created_at", ""),
            "updated_at": raw.get("updated_at", ""),
            "number": str(number),
            "html_url": raw.get("html_url",
                f"https://gitee.com/{owner}/{repo}/issues/{number}"),
            "comments_count": raw.get("comments", 0),
            "source": f"gitee_issue/{owner}/{repo}",
            "source_type": source_type,
            "credibility": credibility,
            "repo": f"{owner}/{repo}",
            "platform": "gitee",
        })

    issues = issues[:limit]
    logger.info(
        f"[Gitee Issues] {owner}/{repo}: {len(issues)} 篇有效 "
        f"(总拉取 {len(raw_issues)} 条)")
    return issues


# ==================== 核心: 拉取文档 (限制深度) ====================

def _fetch_dir_contents(owner: str, repo: str, path: str,
                        depth: int = 0) -> list[dict]:
    """递归获取 Gitee 仓库目录下的 .md 文件（限制深度避免触发限流）"""
    if depth > _MAX_DOC_DEPTH:
        return []

    try:
        contents = _api_get(f"/repos/{owner}/{repo}/contents/{path}")
        if not isinstance(contents, list):
            return []
    except Exception:
        return []

    md_files = []
    for item in contents:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        name = item.get("name", "")

        if item_type == "file" and name.endswith(".md"):
            md_files.append(item)
        elif item_type == "dir" and not name.startswith("."):
            # 优先拉 README 和顶层文档目录，跳过深层 API 参考
            if depth < _MAX_DOC_DEPTH - 1 or name.lower() in (
                    "readme", "guide", "tutorial", "examples"):
                sub = _fetch_dir_contents(
                    owner, repo, item.get("path", ""), depth + 1)
                md_files.extend(sub)

    return md_files


def fetch_gitee_docs(owner: str, repo: str,
                     doc_path: str = "docs") -> list[dict]:
    """拉取 Gitee 仓库文档（限制递归深度，优先重要文件）"""
    try:
        md_files = _fetch_dir_contents(owner, repo, doc_path)
    except Exception as e:
        logger.warning(f"[Gitee Docs] 目录遍历失败 {owner}/{repo}/{doc_path}: {e}")
        md_files = []

    if not md_files and doc_path != "docs":
        try:
            md_files = _fetch_dir_contents(owner, repo, "docs")
        except Exception:
            pass

    docs = []
    for item in md_files[:20]:  # ★ 最多拉 20 个文档文件
        try:
            _rate_limit_wait()
            file_data = _api_get(
                f"/repos/{owner}/{repo}/contents/{item['path']}")
            if isinstance(file_data, dict):
                import base64
                content_b64 = file_data.get("content", "")
                if content_b64:
                    content = base64.b64decode(
                        content_b64).decode("utf-8", errors="replace")
                else:
                    content = ""
            else:
                content = ""
        except Exception:
            content = f"[无法解码: {item.get('name', '')}]"

        if len(content) < 100:
            continue

        docs.append({
            "title": item.get("name", "").replace(".md", ""),
            "body": content,
            "path": item.get("path", ""),
            "html_url": item.get("html_url", ""),
            "source": f"gitee_doc/{owner}/{repo}",
            "source_type": "gitee_repo_doc",
            "credibility": 0.95,
            "repo": f"{owner}/{repo}",
            "platform": "gitee",
        })

    logger.info(f"[Gitee Docs] {owner}/{repo}: {len(docs)} 篇文档")
    return docs


# ==================== 综合拉取 + 写入 ====================

def fetch_all_gitee_sources() -> tuple[list[dict], list[dict]]:
    """拉取所有配置的 Gitee 仓库 Issues + Docs"""
    all_issues = []
    all_docs = []

    for repo_cfg in _get_gitee_repos():
        owner = repo_cfg.get("owner", "")
        repo = repo_cfg.get("repo", "")
        doc_path = repo_cfg.get("doc_path", "docs")

        if not owner or not repo:
            continue

        try:
            issues = fetch_gitee_issues(owner, repo)
            all_issues.extend(issues)
        except Exception as e:
            logger.error(f"[Gitee] Issues 异常 {owner}/{repo}: {e}")

        try:
            docs = fetch_gitee_docs(owner, repo, doc_path)
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"[Gitee] Docs 异常 {owner}/{repo}: {e}")

    return all_issues, all_docs


def save_to_markdown(items: list[dict],
                     data_dir: Path | None = None):
    """保存为 .md 文件（与 github_issues.save_to_markdown 一致）"""
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
        platform = item.get("platform", "gitee")
        labels = item.get("labels", [])
        crawled_at = datetime.now().isoformat()

        safe = "".join(c for c in title[:50]
                       if c.isalnum() or c in (' ', '-', '_'))
        if not safe.strip():
            safe = f"issue_{abs(hash(title)) % 10000}"
        fname = f"{platform}_{source.replace('/', '_')}_{safe}.md"
        fpath = data_dir / fname

        label_str = f" | 标签: {', '.join(labels)}" if labels else ""

        content = (
            f"# {title}\n\n"
            f"> 来源: {source} | 可信度: {source_type} ({credibility})"
            f"{label_str}\n"
            f"> URL: {url}\n"
            f"> 爬取时间: {crawled_at}\n\n"
            f"{body}"
        )

        if fpath.exists():
            # ★ 比较时忽略"爬取时间"行: 内容没变就不重写文件。
            # 否则每次爬取都重写 → 文件 md5 变化 → 向量库重复入库
            # (ChromaDB 新旧版本并存, 曾致 2,936 个重复 chunk)
            def _strip_ts(text: str) -> str:
                return "\n".join(
                    ln for ln in text.split("\n")
                    if not ln.startswith("> 爬取时间"))

            h1 = hashlib.md5(
                _strip_ts(fpath.read_text(encoding="utf-8", errors="replace"))
                .encode()).hexdigest()
            h2 = hashlib.md5(_strip_ts(content).encode()).hexdigest()
            if h1 == h2:
                continue

        fpath.write_text(content, encoding="utf-8")
        saved += 1

    logger.info(f"[Gitee 保存] {saved}/{len(items)} 篇新文章 -> {data_dir}")
    return saved


# ==================== 独立运行 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Gitee Issues + Docs 爬虫 (带智能限流)")
    print("=" * 60)

    repos = _get_gitee_repos()
    print(f"\n目标仓库: {len(repos)} 个")
    for r in repos:
        print(f"  - {r['owner']}/{r['repo']} "
              f"(docs: {r.get('doc_path', 'docs')})")

    print(f"\n认证状态: {'已配置 Token (5000次/h)' if _has_token else '未认证 (60次/h) — 建议设置 GITEE_ACCESS_TOKEN'}")
    print(f"限流间隔: {_RATE_LIMIT_INTERVAL}s/次")
    print(f"文档递归深度上限: {_MAX_DOC_DEPTH} 层")
    print(f"预估耗时: 约 {len(repos) * (10 if _has_token else 30)}s\n")

    issues, docs = fetch_all_gitee_sources()
    print(f"\n结果: {len(issues)} issues + {len(docs)} docs")

    if issues or docs:
        saved = save_to_markdown(issues + docs)
        print(f"已保存 {saved} 篇到 data/articles/")
    else:
        print("未拉取到数据。请检查网络或仓库路径。")
