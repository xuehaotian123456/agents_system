"""
DevPilot 定时调度服务
====================
基于 APScheduler 的后台调度器，负责:

1. 定时增量爬取（默认每 6 小时）
   - 多源爬取 → URL 去重 → 保存文章 → 增量入库向量库 → 更新 KG
   - 只爬新文章，不重复处理已有内容

2. 每日摘要邮件（默认每天 08:00）
   - 生成多源技术日报 HTML → SMTP 发送
   - 收件人/时间通过 A2A 工具由 Harness 对话配置

3. 智能缓存
   - Harness 查询时，若距上次爬取 < 1 小时，直接返回缓存
   - 若 > 1 小时，先触发增量爬取再返回结果

集成方式:
    from services.scheduler import start_scheduler
    start_scheduler()  # 在 app.py 或 a2a_server.py 启动时调用
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ==================== 调度器状态 ====================

_scheduler = None
_scheduler_started = False


def _ensure_project_on_path():
    """确保项目根目录在 sys.path 中"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def start_scheduler():
    """
    启动后台调度器（线程安全，幂等——多次调用不会重复启动）。

    使用方式:
        # 在 app.py 或 a2a_server.py 中
        from services.scheduler import start_scheduler
        start_scheduler()

    调度任务:
        - 增量爬取: 每 6 小时自动执行
        - 每日摘要: 每天根据配置的时间发送
    """
    global _scheduler, _scheduler_started
    if _scheduler_started:
        return

    _ensure_project_on_path()

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        print("[Scheduler] APScheduler 未安装，跳过定时调度。pip install apscheduler")
        return

    _scheduler = BackgroundScheduler(
        timezone="Asia/Shanghai",
        job_defaults={"misfire_grace_time": 900, "coalesce": True},
    )

    # ---- 任务 1: 增量爬取（每 6 小时）----
    _scheduler.add_job(
        func=_incremental_crawl_job,
        trigger="interval",
        hours=6,
        id="incremental_crawl",
        name="增量爬取(6h)",
        next_run_time=None,  # 启动后立即不执行，等第一次 interval
    )

    # ---- 任务 2: 每日摘要邮件（时间从配置读取）----
    from services.crawl_state import get_crawl_state
    state = get_crawl_state()
    digest_config = state.get_digest_config()
    digest_time = digest_config.get("time", "08:00")
    hour, minute = digest_time.split(":")
    digest_enabled = digest_config.get("enabled", False)

    _scheduler.add_job(
        func=_daily_digest_job,
        trigger="cron",
        hour=int(hour),
        minute=int(minute),
        id="daily_digest",
        name=f"每日摘要({digest_time})",
    )

    _scheduler.start()
    _scheduler_started = True

    print(f"[Scheduler] [OK] 后台调度已启动")
    print(f"[Scheduler]   增量爬取: 每 6 小时")
    print(f"[Scheduler]   每日摘要: {digest_time} (启用={digest_enabled})")


def stop_scheduler():
    """停止调度器"""
    global _scheduler, _scheduler_started
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler_started = False
    print("[Scheduler] 已停止")


def get_scheduler_status() -> dict:
    """获取调度器运行状态"""
    global _scheduler, _scheduler_started
    if not _scheduler_started or not _scheduler:
        return {"running": False, "jobs": []}

    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time) if job.next_run_time else None,
        })
    return {"running": _scheduler.running, "jobs": jobs}


def reschedule_daily_digest(time_str: str):
    """
    重新安排每日摘要时间（当用户通过 Harness 修改配置时调用）

    Args:
        time_str: "HH:MM" 格式
    """
    global _scheduler
    if not _scheduler:
        return

    try:
        hour, minute = time_str.split(":")
        _scheduler.reschedule_job(
            "daily_digest",
            trigger="cron",
            hour=int(hour),
            minute=int(minute),
        )
        print(f"[Scheduler] 每日摘要时间已更新为 {time_str}")
    except Exception as e:
        print(f"[Scheduler] 重调度失败: {e}")


def trigger_manual_crawl() -> dict:
    """
    手动触发一次增量爬取（Harness 可通过 A2A 调用）

    Returns:
        {"success": bool, "new_articles": int, "message": str}
    """
    return _incremental_crawl_job()


def trigger_manual_digest() -> dict:
    """
    手动触发一次每日摘要生成和发送

    Returns:
        {"success": bool, "article_count": int, "message": str}
    """
    return _daily_digest_job()


# ==================== 质量门控 ====================

def _is_quality_gate_enabled() -> bool:
    """质量门控总开关 (data_source.yaml: enable_llm_quality_gate)"""
    try:
        import yaml
        cfg_path = PROJECT_ROOT / "config" / "data_source.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("data_source", {}).get("enable_llm_quality_gate", True)
    except Exception:
        pass
    return True


def _apply_quality_gate(candidates: list[dict], source_label: str = "") -> list[dict]:
    """
    对候选文档运行质量门控 (规则层 + LLM 语义判定)。

    Returns:
        通过门的文档列表 (ingest 原样 + demote 带 quality_demoted 标记)。
        skip 的文档被过滤, 判定留痕打印到日志。
    """
    if not candidates:
        return []

    try:
        from agent.quality_gate import run_quality_gate

        docs = [{
            "title": c.get("title", ""),
            "content": c.get("content", ""),
            "credibility": c.get("credibility", 0.5),
            "source_type": c.get("source_type", "unknown"),
            "_candidate": c,
        } for c in candidates]

        result = run_quality_gate(docs, enabled=_is_quality_gate_enabled())
        passed = []
        for d in result["ingest"] + result["demote"]:
            c = d["_candidate"]
            if d.get("quality_demoted"):
                c["quality_demoted"] = True
                c["credibility"] = d.get("credibility", c.get("credibility", 0.5))
            passed.append(c)

        stats = result.get("stats", {})
        if stats:
            print(f"  [QualityGate] {source_label}: ingest={stats.get('ingest')} "
                  f"demote={stats.get('demote')} skip={stats.get('skip')}")
        return passed
    except Exception as e:
        print(f"  [QualityGate] {source_label}: 门控执行失败, 全部放行: {e}")
        return candidates


# ==================== 内部任务函数 ====================

def _incremental_crawl_job() -> dict:
    """
    增量爬取任务
    - 爬取所有数据源
    - URL 去重
    - 新文章保存到 data/articles/
    - 增量更新向量库 (VectorStore)
    - 增量更新知识图谱 (KG)
    """
    _ensure_project_on_path()

    from services.crawl_state import get_crawl_state
    state = get_crawl_state()

    print(f"\n[Scheduler] [CRON] 增量爬取开始 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    new_articles = []
    data_dir = PROJECT_ROOT / "data" / "articles"
    os.makedirs(data_dir, exist_ok=True)

    # ── 掘金 ──
    try:
        from crawlers.juejin import fetch_juejin_hot, fetch_juejin_article
        ids = fetch_juejin_hot(limit=10)

        # 收集新文章详情, 先过质量门再写文件
        candidates = []
        for a in ids:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            detail = fetch_juejin_article(a['id'])
            if detail and detail.get('title'):
                candidates.append({"title": detail['title'], "url": url,
                                   "content": detail.get('content', '')})

        gated = _apply_quality_gate(candidates, source_label="掘金")
        for doc in gated:
            safe = ''.join(c for c in doc['title'][:40] if c.isalnum() or c in (' ', '-', '_')).strip()
            fpath = data_dir / f'sched_juejin_{safe}.md'
            demote_note = "\n> 质量门控: demote (语义判定降权)\n" if doc.get("quality_demoted") else ""
            if not fpath.exists():
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(f"# {doc['title']}\n\n> 来源: 掘金 | {doc['url']}\n"
                            f"> 爬取时间: {datetime.now().isoformat()}{demote_note}\n\n{doc['content']}")
            state.mark_crawled({"url": doc['url'], "title": doc['title'], "source": "掘金"})
            new_articles.append({"title": doc['title'], "url": doc['url'], "source": "掘金"})
        state.mark_source_crawled("juejin")
        print(f"  [Scheduler] 掘金: {len(gated)} 篇通过质量门 (候选 {len(candidates)})")
    except Exception as e:
        print(f"  [Scheduler] 掘金爬取失败: {e}")

    # ── 博客园 ──
    try:
        from crawlers.cnblogs import fetch_cnblogs_rss, fetch_cnblogs_article
        rss = fetch_cnblogs_rss(limit=10)

        candidates = []
        for a in rss:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            content = fetch_cnblogs_article(url)
            if content:
                candidates.append({"title": a['title'], "url": url, "content": content})

        gated = _apply_quality_gate(candidates, source_label="博客园")
        new_count = 0
        for doc in gated:
            safe = ''.join(c for c in doc['title'][:40] if c.isalnum() or c in (' ', '-', '_')).strip()
            fpath = data_dir / f'sched_cnblogs_{safe}.md'
            demote_note = "\n\n> 质量门控: demote (语义判定降权)" if doc.get("quality_demoted") else ""
            if not fpath.exists():
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(f"{doc['content']}{demote_note}\n\n> 爬取时间: {datetime.now().isoformat()}")
            state.mark_crawled({"url": doc['url'], "title": doc['title'], "source": "博客园"})
            new_articles.append({"title": doc['title'], "url": doc['url'], "source": "博客园"})
            new_count += 1
        state.mark_source_crawled("cnblogs")
        print(f"  [Scheduler] 博客园: {new_count} 篇通过质量门 (候选 {len(candidates)})")
    except Exception as e:
        print(f"  [Scheduler] 博客园爬取失败: {e}")

    # ── Gitee Issues + Docs（国内直连 API，主数据源）──
    try:
        from crawlers.gitee_adapter import fetch_all_gitee_sources, save_to_markdown as gitee_save
        gitee_issues, gitee_docs = fetch_all_gitee_sources()
        gitee_items = gitee_issues + gitee_docs

        # 质量门控: 官方文档 (credibility>=0.9) 走快通道免 LLM 判定 (省成本)
        official = [it for it in gitee_items if it.get("credibility", 0) >= 0.9]
        community = [it for it in gitee_items if it.get("credibility", 0) < 0.9]
        gate_docs = [{"title": it.get("title", ""), "content": it.get("body", ""),
                      "credibility": it.get("credibility", 0.5),
                      "source_type": it.get("source_type", "unknown"),
                      "_item": it} for it in community]
        gated = _apply_quality_gate(gate_docs, source_label="Gitee社区内容")

        # 官方文档 + 通过门的社区内容
        final_items = official + [g["_item"] for g in gated]
        gitee_new = 0
        for item in final_items:
            url = item.get("html_url", "")
            if not url or state.is_duplicate(url):
                continue
            state.mark_crawled({
                "url": url,
                "title": item.get("title", ""),
                "source": item.get("source", "gitee"),
            })
            new_articles.append({
                "title": item.get("title", ""),
                "url": url,
                "source": item.get("source", "gitee"),
            })
            gitee_new += 1
        if final_items:
            gitee_save(final_items)
        state.mark_source_crawled("gitee")
        print(f"  [Scheduler] Gitee: {gitee_new}/{len(final_items)} 篇入库 "
              f"(官方 {len(official)} 直通 + 社区 {len(gated)} 过门, 拦截 {len(community) - len(gated)})")
    except Exception as e:
        print(f"  [Scheduler] Gitee 爬取失败: {e}")

    # ── GitHub Issues + Docs（纯 API 拉取，备用）──
    try:
        from crawlers.github_issues import fetch_all_github_sources, save_to_markdown
        gh_issues, gh_docs = fetch_all_github_sources()
        gh_items = gh_issues + gh_docs
        gh_new = 0
        for item in gh_items:
            url = item.get("html_url", "")
            if not url or state.is_duplicate(url):
                continue
            state.mark_crawled({
                "url": url,
                "title": item.get("title", ""),
                "source": item.get("source", "github"),
            })
            new_articles.append({
                "title": item.get("title", ""),
                "url": url,
                "source": item.get("source", "github"),
            })
            gh_new += 1
        if gh_items:
            save_to_markdown(gh_items)
        state.mark_source_crawled("github_issues")
        print(f"  [Scheduler] GitHub Issues+Docs: {gh_new}/{len(gh_items)} 篇新文章 (总拉取 {len(gh_items)} 条)")
    except Exception as e:
        print(f"  [Scheduler] GitHub 爬取失败: {e}")

    # ── GitHub Trending（保留作为趋势参考，低权重来源）──
    try:
        from crawlers.multi_source import fetch_github_trending
        repos = fetch_github_trending(limit=5)
        gh_trend_new = 0
        for a in repos:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            state.mark_crawled({"url": url, "title": a['title'], "source": "GitHub Trending"})
            new_articles.append({"title": a['title'], "url": url, "source": "GitHub Trending"})
            gh_trend_new += 1
        state.mark_source_crawled("github")
        print(f"  [Scheduler] GitHub Trending: {gh_trend_new} 篇新文章")
    except Exception as e:
        print(f"  [Scheduler] GitHub Trending 爬取失败: {e}")

    # ── HackerNews ──
    try:
        from crawlers.multi_source import fetch_hackernews
        items = fetch_hackernews(limit=5)
        new_count = 0
        for a in items:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            state.mark_crawled({"url": url, "title": a['title'], "source": "HackerNews"})
            new_articles.append({"title": a['title'], "url": url, "source": "HackerNews"})
            new_count += 1
        state.mark_source_crawled("hackernews")
        print(f"  [Scheduler] HackerNews: {new_count} 篇新文章")
    except Exception as e:
        print(f"  [Scheduler] HackerNews 爬取失败: {e}")

    # ── OSChina ──
    try:
        from crawlers.multi_source import fetch_oschina
        items = fetch_oschina(limit=5)
        new_count = 0
        for a in items:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            state.mark_crawled({"url": url, "title": a['title'], "source": "OSChina"})
            new_articles.append({"title": a['title'], "url": url, "source": "OSChina"})
            new_count += 1
        state.mark_source_crawled("oschina")
        print(f"  [Scheduler] OSChina: {new_count} 篇新文章")
    except Exception as e:
        print(f"  [Scheduler] OSChina 爬取失败: {e}")

    state.mark_full_crawl_done()

    # ── 增量更新向量库 ──
    vs_updated = False
    if new_articles:
        try:
            from rag.vector_store import VectorStore
            vs = VectorStore()
            vs.load_articles()  # 重新加载（包括新文章）
            vs_updated = True
            print(f"  [Scheduler] 向量库已更新: {len(vs.all_chunks)} chunks")
        except Exception as e:
            print(f"  [Scheduler] 向量库更新失败: {e}")

    # ── 增量更新 KG ──
    kg_updated = False
    if new_articles:
        try:
            from rag.knowledge_graph import get_kg
            kg = get_kg()
            if kg.is_built:
                # KG 重建（轻量，全量重建约 1-2 秒）
                from rag.vector_store import VectorStore
                vs_tmp = VectorStore()
                vs_tmp.load_articles()
                kg.build(vs_tmp.all_chunks)
                kg_updated = True
                print(f"  [Scheduler] KG 已更新: {kg.entity_count} 实体")
        except Exception as e:
            print(f"  [Scheduler] KG 更新失败: {e}")

    total = len(new_articles)
    print(f"[Scheduler] [OK] 增量爬取完成 — {total} 篇新文章 | 向量库={'已更新' if vs_updated else '跳过'} | KG={'已更新' if kg_updated else '跳过'}")

    return {
        "success": True,
        "new_articles": total,
        "sources": list(set(a["source"] for a in new_articles)),
        "message": f"增量爬取完成: {total} 篇新文章" if total else "没有发现新文章",
    }


def _daily_digest_job() -> dict:
    """每日摘要邮件任务"""
    _ensure_project_on_path()

    from services.crawl_state import get_crawl_state
    from services.mailer import crawl_all_for_digest, generate_digest_html, send_email

    state = get_crawl_state()
    config = state.get_digest_config()

    if not config.get("enabled"):
        print(f"[Scheduler] 每日摘要已禁用，跳过发送")
        return {"success": False, "message": "每日摘要已禁用", "article_count": 0}

    to_email = config.get("email", "")
    if not to_email:
        print(f"[Scheduler] 未配置收件人邮箱，跳过发送")
        return {"success": False, "message": "未配置收件人邮箱", "article_count": 0}

    print(f"\n[Scheduler] [CRON] 每日摘要生成 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 先增量爬取
    print(f"[Scheduler] 先执行增量爬取...")
    crawl_result = _incremental_crawl_job()
    print(f"[Scheduler] 爬取结果: {crawl_result['message']}")

    # 爬取并生成摘要
    articles = crawl_all_for_digest()

    if not articles:
        # 尝试用近期文章
        recent = state.get_recent_articles(hours=24)
        if recent:
            articles = recent
        else:
            msg = "今日没有新的技术文章"
            print(f"[Scheduler] {msg}")
            return {"success": True, "message": msg, "article_count": 0}

    # 生成 HTML 并发送
    html = generate_digest_html(articles)
    result = send_email(html, to_email)

    if result["success"]:
        print(f"[Scheduler] [OK] 每日摘要已发送至 {to_email} — {len(articles)} 篇文章")
    else:
        print(f"[Scheduler] [ERR] 邮件发送失败: {result['message']}")

    return {
        "success": result["success"],
        "article_count": len(articles),
        "message": result["message"],
    }
