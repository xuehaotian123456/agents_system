"""
每日技术摘要邮件服务
=====================
提供:
1. 生成 HTML 日报（多源技术文章 + 热词趋势）
2. SMTP 发送邮件（单次 + 定时）
3. Harness 可通过 A2A 工具调用 send_digest_email

依赖:
- CrawlState: 爬取状态（去重、时间戳）
- crawlers: 掘金/博客园/GitHub/OSChina 爬虫
- trends: 热词趋势分析
"""

from __future__ import annotations

import os
import re
import random
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ==================== SMTP 智能检测 ====================

# 常见邮箱 → SMTP 服务器/端口/授权码帮助链接
SMTP_PROVIDERS = {
    "qq.com": {
        "host": "smtp.qq.com", "port": 465,
        "name": "QQ邮箱",
        "auth_help": "https://service.mail.qq.com/detail/0/75",
        "auth_steps": "设置 → 账户 → POP3/IMAP/SMTP → 开启SMTP → 生成授权码",
    },
    "foxmail.com": {
        "host": "smtp.qq.com", "port": 465,
        "name": "Foxmail",
        "auth_help": "https://service.mail.qq.com/detail/0/75",
        "auth_steps": "同QQ邮箱，在设置 → 账户中生成授权码",
    },
    "163.com": {
        "host": "smtp.163.com", "port": 465,
        "name": "网易163邮箱",
        "auth_help": "https://help.mail.163.com/faqDetail.do?code=d7a5dc8471cd0c0e8b4b8f4f8e49998b",
        "auth_steps": "设置 → POP3/SMTP/IMAP → 开启SMTP → 新增授权码",
    },
    "126.com": {
        "host": "smtp.126.com", "port": 465,
        "name": "网易126邮箱",
        "auth_help": "https://help.mail.126.com/faqDetail.do?code=d7a5dc8471cd0c0e8b4b8f4f8e49998b",
        "auth_steps": "设置 → POP3/SMTP/IMAP → 开启SMTP → 新增授权码",
    },
    "yeah.net": {
        "host": "smtp.yeah.net", "port": 465,
        "name": "网易Yeah邮箱",
        "auth_help": "https://help.mail.163.com/faqDetail.do?code=d7a5dc8471cd0c0e8b4b8f4f8e49998b",
        "auth_steps": "设置 → POP3/SMTP/IMAP → 开启SMTP → 新增授权码",
    },
    "gmail.com": {
        "host": "smtp.gmail.com", "port": 587,
        "name": "Gmail",
        "auth_help": "https://support.google.com/accounts/answer/185833",
        "auth_steps": "Google账户 → 安全性 → 两步验证 → 应用专用密码",
    },
    "outlook.com": {
        "host": "smtp-mail.outlook.com", "port": 587,
        "name": "Outlook",
        "auth_help": "https://support.microsoft.com/zh-cn/office/",
        "auth_steps": "Microsoft账户 → 安全性 → 高级安全选项 → 应用密码",
    },
    "hotmail.com": {
        "host": "smtp-mail.outlook.com", "port": 587,
        "name": "Hotmail",
        "auth_help": "https://support.microsoft.com/zh-cn/office/",
        "auth_steps": "同Outlook，Microsoft账户中生成应用密码",
    },
    "live.com": {
        "host": "smtp-mail.outlook.com", "port": 587,
        "name": "Outlook Live",
        "auth_help": "https://support.microsoft.com/zh-cn/office/",
        "auth_steps": "Microsoft账户 → 安全性 → 应用密码",
    },
    "sina.com": {
        "host": "smtp.sina.com", "port": 465,
        "name": "新浪邮箱",
        "auth_help": "https://mail.sina.com.cn/",
        "auth_steps": "设置 → 客户端POP/IMAP/SMTP → 开启SMTP",
    },
    "sohu.com": {
        "host": "smtp.sohu.com", "port": 465,
        "name": "搜狐邮箱",
        "auth_help": "https://mail.sohu.com/",
        "auth_steps": "设置 → 客户端协议 → 开启SMTP",
    },
    "aliyun.com": {
        "host": "smtp.aliyun.com", "port": 465,
        "name": "阿里云邮箱",
        "auth_help": "https://help.aliyun.com/document_detail/29444.html",
        "auth_steps": "阿里云邮箱设置 → 客户端密码 → 生成新密码",
    },
    "189.cn": {
        "host": "smtp.189.cn", "port": 465,
        "name": "电信189邮箱",
        "auth_help": "https://mail.189.cn/",
        "auth_steps": "设置 → 客户端协议 → 开启SMTP",
    },
}


def _detect_smtp(email: str) -> dict | None:
    """
    根据邮箱地址自动检测 SMTP 配置

    Args:
        email: 邮箱地址，如 "user@qq.com"

    Returns:
        {"host": ..., "port": ..., "name": ..., "auth_help": ..., "auth_steps": ...} 或 None
    """
    domain = email.rsplit("@", 1)[-1].lower() if "@" in email else ""
    return SMTP_PROVIDERS.get(domain)


def _get_auth_code_url(email: str) -> str | None:
    """获取某邮箱的授权码生成页面链接"""
    info = _detect_smtp(email)
    return info["auth_help"] if info else None


# ==================== 邮箱校验 ====================

def _validate_email(email: str) -> bool:
    """基本邮箱格式校验"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def _load_smtp_config() -> dict | None:
    """
    加载 SMTP 配置 — 两级优先级:
    1. 用户通过 Harness 对话配置 (crawl_state.json) — 优先
    2. 服务端 .env 环境变量 — 兜底

    这样 Web 用户不需要碰服务器文件系统，直接在对话中配置即可。
    """
    # 第一优先级: 用户对话配置
    try:
        from services.crawl_state import get_crawl_state
        state = get_crawl_state()
        user_cfg = state.get_smtp_config()
        if user_cfg.get("host") and user_cfg.get("user") and user_cfg.get("password"):
            return {
                "host": user_cfg["host"],
                "port": int(user_cfg.get("port", 465)),
                "user": user_cfg["user"],
                "password": user_cfg["password"],
                "from_addr": user_cfg.get("from_addr") or user_cfg["user"],
            }
    except Exception:
        pass

    # 第二优先级: .env 环境变量（服务器部署管理员设置）
    cfg = {
        "host": os.getenv("SMTP_HOST", ""),
        "port": int(os.getenv("SMTP_PORT", "465")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "from_addr": os.getenv("SMTP_FROM", ""),
    }
    if cfg["host"] and cfg["user"] and cfg["password"]:
        return cfg
    return None


def send_email(html: str, to_email: str, subject: str = "") -> dict:
    """
    发送 HTML 邮件

    Args:
        html: HTML 邮件正文
        to_email: 收件人邮箱
        subject: 邮件标题（可选，默认自动生成日期标题）

    Returns:
        {"success": bool, "message": str}
    """
    # 1. 校验邮箱
    if not _validate_email(to_email):
        return {"success": False, "message": f"邮箱格式无效: {to_email}"}

    # 2. 校验 SMTP 配置
    smtp = _load_smtp_config()
    if not smtp:
        return {
            "success": False,
            "message": (
                "SMTP 未配置。请在 .env 中设置 SMTP_HOST / SMTP_PORT / "
                "SMTP_USER / SMTP_PASSWORD / SMTP_FROM。"
                "\n这是一项需要人工操作的安全配置——邮件服务器密码不应由 Agent 处理。"
            ),
        }

    # 3. 构建邮件
    if not subject:
        subject = f"📰 DevPilot 每日技术摘要 - {datetime.now().strftime('%Y-%m-%d')}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp["from_addr"] or smtp["user"]
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html", "utf-8"))

    # 4. 发送
    try:
        port = smtp["port"]
        with smtplib.SMTP_SSL(smtp["host"], port, timeout=15) as server:
            server.login(smtp["user"], smtp["password"])
            server.sendmail(msg["From"], [to_email], msg.as_string())
        return {"success": True, "message": f"✅ 邮件已发送至 {to_email}"}
    except smtplib.SMTPAuthenticationError:
        return {"success": False, "message": "SMTP 认证失败，请检查邮箱账号/密码/授权码"}
    except smtplib.SMTPConnectError:
        return {"success": False, "message": f"无法连接 SMTP 服务器 {smtp['host']}:{port}"}
    except Exception as e:
        return {"success": False, "message": f"邮件发送异常: {type(e).__name__}: {e}"}


def generate_digest_html(articles: list[dict], include_header: bool = True) -> str:
    """
    生成每日技术摘要 HTML

    Args:
        articles: 文章列表 [{"title": str, "url": str, "brief": str, "source": str, "crawled_at": str, "images": list}, ...]
        include_header: 是否包含完整 header/footer（邮件模式=True，嵌入模式=False）

    Returns:
        HTML 字符串
    """
    today = datetime.now().strftime("%Y年%m月%d日 %A")
    total = len(articles)

    # ── 热词趋势 ──
    kw_html = ""
    try:
        from services.trends import analyze_trends
        trends = analyze_trends()
        top_kw = trends.get("keywords", [])[:15]
        if top_kw:
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
            random.seed(42)
            max_freq = max(top_kw[0][1], 1)
            for w, c in top_kw:
                color = random.choice(colors)
                size = max(14, min(28, 14 + (c / max_freq) * 14))
                kw_html += (
                    f'<span style="font-size:{size}px;color:{color};'
                    f'margin:4px;display:inline-block">{w}</span> '
                )
    except Exception:
        kw_html = "暂无趋势数据"

    # ── 文章卡片 ──
    source_colors = {
        "掘金": "#1e80ff",
        "Juejin": "#1e80ff",
        "博客园": "#2ecc71",
        "Cnblogs": "#2ecc71",
        "GitHub Trending": "#333",
        "GitHub": "#333",
        "HackerNews": "#ff6600",
        "OSChina": "#27ae60",
    }

    cards = ""
    for a in articles:
        src = a.get("source", "未知")
        color = source_colors.get(src, "#999")

        # 图片
        img_tag = ""
        images = a.get("images", []) or []
        if images and images[0]:
            img_tag = (
                f'<img src="{images[0]}" '
                f'style="width:100%;max-height:240px;object-fit:cover;border-radius:8px;margin-bottom:8px" '
                f'onerror="this.style.display=\'none\'">'
            )

        # 时间
        time_str = a.get("crawled_at", "") or a.get("date", "")
        if time_str and len(time_str) > 10:
            time_str = time_str[:10]

        cards += f"""
        <div style="margin-bottom:24px;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
            {img_tag}
            <div style="padding:16px">
                <span style="background:{color};color:#fff;padding:2px 10px;border-radius:4px;font-size:12px">{src}</span>
                <span style="color:#999;font-size:12px;margin-left:8px">{time_str}</span>
                <h3 style="margin:10px 0 6px;font-size:18px;line-height:1.4">
                    <a href="{a.get('url', '#')}" style="color:#2c3e50;text-decoration:none" target="_blank">
                        {a.get('title', '')[:80]}
                    </a>
                </h3>
                <p style="color:#666;font-size:14px;line-height:1.6;margin:0">
                    {a.get('brief', '')[:200]}
                </p>
                <a href="{a.get('url', '#')}"
                   style="display:inline-block;margin-top:10px;color:{color};font-size:13px;text-decoration:none"
                   target="_blank">阅读全文 →</a>
            </div>
        </div>"""

    # ── 来源统计 ──
    source_counts = {}
    for a in articles:
        src = a.get("source", "未知")
        source_counts[src] = source_counts.get(src, 0) + 1

    if not include_header:
        # 嵌入模式：只返回卡片 + 词云
        return f"""
        <div class="cloud" style="background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;text-align:center;line-height:2.2">
            <div style="font-size:14px;color:#999;margin-bottom:10px">🔥 技术热词趋势</div>
            {kw_html}
        </div>
        {cards}
        """

    # ── 完整 HTML ──
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{margin:0;padding:0;background:#f5f6fa;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.container{{max-width:680px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#2c3e50,#3498db);color:#fff;padding:30px 20px;text-align:center;border-radius:12px;margin-bottom:20px}}
.header h1{{margin:0;font-size:26px}}
.header p{{margin:8px 0 0;opacity:0.85;font-size:14px}}
.cloud{{background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;text-align:center;line-height:2.2}}
.cloud-title{{font-size:14px;color:#999;margin-bottom:10px}}
.stat-bar{{display:flex;gap:12px;margin-bottom:20px}}
.stat{{flex:1;background:#fff;border-radius:12px;padding:16px;text-align:center}}
.stat-num{{font-size:28px;font-weight:bold;color:#2c3e50}}
.stat-label{{font-size:12px;color:#999;margin-top:4px}}
.footer{{text-align:center;padding:20px;color:#999;font-size:12px}}
.footer a{{color:#3498db}}
</style></head>
<body>
<div class="container">

<div class="header">
    <h1>📰 DevPilot 每日技术摘要</h1>
    <p>{today} · 自动从多源技术社区采集</p>
</div>

<div class="stat-bar">
    <div class="stat"><div class="stat-num">{total}</div><div class="stat-label">📄 今日文章</div></div>
    <div class="stat"><div class="stat-num">{len(source_counts)}</div><div class="stat-label">🌐 数据源</div></div>
    <div class="stat"><div class="stat-num">{len([a for a in articles if a.get('images')])}</div><div class="stat-label">🖼️ 含图文章</div></div>
</div>

<div class="cloud">
    <div class="cloud-title">🔥 技术热词趋势</div>
    {kw_html}
</div>

<h2 style="color:#2c3e50;margin:20px 0">📖 今日精选</h2>
{cards}

<div class="footer">
    <p>由 <a href="#">DevPilot</a> 自动生成 · 数据来源: {', '.join(source_counts.keys())}</p>
    <p>如需调整推送频率或退订，请通过 Harness 对话配置</p>
</div>

</div></body></html>"""
    return html


def crawl_all_for_digest() -> list[dict]:
    """
    爬取所有数据源，返回带完整元数据的文章列表。
    自动去重——已爬过的 URL 跳过。

    Returns:
        [{title, url, brief, images, source, crawled_at}, ...]
    """
    from services.crawl_state import get_crawl_state

    state = get_crawl_state()
    articles = []
    data_dir = PROJECT_ROOT / "data" / "articles"
    os.makedirs(data_dir, exist_ok=True)

    def _save_article(title: str, content: str, prefix: str):
        """保存文章到本地，跳过已存在的"""
        safe = ''.join(c for c in title[:40] if c.isalnum() or c in (' ', '-', '_')).strip()
        fpath = data_dir / f'{prefix}_{safe}.md'
        if not fpath.exists():
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)

    # ── 掘金 ──
    try:
        from crawlers.juejin import fetch_juejin_hot, fetch_juejin_article
        ids = fetch_juejin_hot(limit=6)
        for a in ids:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            detail = fetch_juejin_article(a['id'])
            if detail and detail.get('title'):
                _save_article(detail['title'],
                              f"# {detail['title']}\n\n> 来源: 掘金 | {url}\n\n{detail['content']}",
                              "digest_juejin")
                crawled_at = datetime.now().isoformat()
                state.mark_crawled({
                    "url": url, "title": detail['title'],
                    "source": "掘金", "content": detail.get('content', ''),
                })
                articles.append({
                    "title": detail['title'], "url": url,
                    "brief": detail.get('content', '')[:200].replace('\n', ' '),
                    "images": detail.get('images', [])[:3],
                    "source": "掘金", "crawled_at": crawled_at,
                })
    except Exception as e:
        print(f"  [Digest] 掘金爬取失败: {e}")

    # ── 博客园 ──
    try:
        from crawlers.cnblogs import fetch_cnblogs_rss, fetch_cnblogs_article
        rss = fetch_cnblogs_rss(limit=6)
        for a in rss:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            content = fetch_cnblogs_article(url)
            if content:
                _save_article(a['title'], content, "digest_cnblogs")
            crawled_at = datetime.now().isoformat()
            state.mark_crawled({
                "url": url, "title": a['title'], "source": "博客园",
            })
            articles.append({
                "title": a['title'], "url": url,
                "brief": a.get('brief', '')[:200],
                "images": [],
                "source": "博客园", "crawled_at": crawled_at,
                "date": a.get('published', '')[:10] or datetime.now().strftime("%m-%d"),
            })
    except Exception as e:
        print(f"  [Digest] 博客园爬取失败: {e}")

    # ── GitHub Trending ──
    try:
        from crawlers.multi_source import fetch_github_trending
        repos = fetch_github_trending(limit=6)
        for a in repos:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            crawled_at = datetime.now().isoformat()
            state.mark_crawled({"url": url, "title": a['title'], "source": "GitHub Trending"})
            articles.append({
                "title": a['title'], "url": url,
                "brief": a.get('brief', '')[:200],
                "images": [],
                "source": "GitHub Trending", "crawled_at": crawled_at,
            })
    except Exception as e:
        print(f"  [Digest] GitHub Trending 爬取失败: {e}")

    # ── HackerNews ──
    try:
        from crawlers.multi_source import fetch_hackernews
        items = fetch_hackernews(limit=5)
        for a in items:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            crawled_at = datetime.now().isoformat()
            state.mark_crawled({"url": url, "title": a['title'], "source": "HackerNews"})
            articles.append({
                "title": a['title'], "url": url,
                "brief": a.get('brief', '')[:200],
                "images": [],
                "source": "HackerNews", "crawled_at": crawled_at,
            })
    except Exception as e:
        print(f"  [Digest] HackerNews 爬取失败: {e}")

    # ── OSChina ──
    try:
        from crawlers.multi_source import fetch_oschina
        items = fetch_oschina(limit=5)
        for a in items:
            url = a.get("url", "")
            if state.is_duplicate(url):
                continue
            crawled_at = datetime.now().isoformat()
            state.mark_crawled({"url": url, "title": a['title'], "source": "OSChina"})
            articles.append({
                "title": a['title'], "url": url,
                "brief": a.get('brief', '')[:200],
                "images": [],
                "source": "OSChina", "crawled_at": crawled_at,
            })
    except Exception as e:
        print(f"  [Digest] OSChina 爬取失败: {e}")

    # 更新爬取时间
    state.mark_full_crawl_done()
    return articles


def send_daily_digest(to_email: str) -> dict:
    """
    生成并发送每日技术摘要邮件。
    Harness 可通过 A2A 调用此函数。

    Args:
        to_email: 收件人邮箱

    Returns:
        {"success": bool, "message": str, "article_count": int}
    """
    # 1. 校验邮箱
    if not _validate_email(to_email):
        return {"success": False, "message": f"邮箱格式无效: {to_email}", "article_count": 0}

    # 2. 检查 SMTP 配置
    smtp = _load_smtp_config()
    if not smtp:
        return {
            "success": False,
            "message": "📧 需要配置邮件发送。请告诉我你的**邮箱地址**，我会自动识别SMTP服务器。"
                       "\n\n例如: \"我的邮箱是 xxx@qq.com\"",
            "article_count": 0,
            "need_smtp": True,
        }

    # 3. 爬取最新文章
    print(f"[Mailer] 开始爬取最新文章...")
    articles = crawl_all_for_digest()
    print(f"[Mailer] 爬取完成: {len(articles)} 篇新文章")

    # 4. 若无新文章，用近期已爬取的文章（来自 crawl_state 记录）
    if not articles:
        from services.crawl_state import get_crawl_state
        state = get_crawl_state()
        recent = state.get_recent_articles(hours=24)
        if recent:
            articles = [
                {"title": r.get("title", ""), "url": r.get("url", ""), "brief": "",
                 "source": r.get("source", "未知"), "crawled_at": r.get("crawled_at", ""),
                 "images": []}
                for r in recent
            ]
            print(f"[Mailer] 无新文章，使用近期 {len(articles)} 篇已爬取文章")
        else:
            return {"success": True, "message": "当前没有技术文章可推送。请稍后再试或手动触发 force_update。", "article_count": 0}

    # 5. 生成 HTML
    html = generate_digest_html(articles)

    # 6. 发送
    result = send_email(html, to_email)
    result["article_count"] = len(articles)
    return result
