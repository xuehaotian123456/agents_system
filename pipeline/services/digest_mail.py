"""每日技术摘要 — 信息流邮件 + HTML 报告"""
import os, sys, re, argparse
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def crawl_all_sources() -> list[dict]:
    """爬取所有数据源，返回带元数据的文章列表 [{title,url,images,brief,source,date}]"""
    articles = []
    data_dir = PROJECT_ROOT / "data" / "articles"
    os.makedirs(data_dir, exist_ok=True)

    # ── 掘金 ──
    try:
        from crawlers.juejin import fetch_juejin_hot, fetch_juejin_article
        ids = fetch_juejin_hot(limit=6)
        for a in ids:
            detail = fetch_juejin_article(a['id'])
            if detail and detail.get('title'):
                safe = ''.join(c for c in detail['title'][:40] if c.isalnum() or c in (' ','-','_')).strip()
                fpath = data_dir / f'digest_juejin_{safe}.md'
                content = f"# {detail['title']}\n\n> 来源: 掘金\n\n{detail['content']}"
                if not fpath.exists():
                    with open(fpath, 'w', encoding='utf-8') as f: f.write(content)
                articles.append({
                    "title": detail['title'], "url": detail['url'],
                    "brief": detail['content'][:200].replace('\n',' '),
                    "images": detail.get('images', [])[:3],
                    "source": "掘金", "date": datetime.now().strftime("%m-%d")
                })
    except Exception as e: print(f"  掘金: {e}")

    # ── 博客园 ──
    try:
        from crawlers.cnblogs import fetch_cnblogs_rss, fetch_cnblogs_article
        rss = fetch_cnblogs_rss(limit=6)
        for a in rss:
            content = fetch_cnblogs_article(a['url'])
            if content:
                safe = ''.join(c for c in a['title'][:40] if c.isalnum() or c in (' ','-','_')).strip()
                fpath = data_dir / f'digest_cnblogs_{safe}.md'
                if not fpath.exists():
                    with open(fpath, 'w', encoding='utf-8') as f: f.write(content)
            articles.append({
                "title": a['title'], "url": a['url'],
                "brief": a.get('brief','')[:200],
                "images": [],
                "source": "博客园", "date": a.get('published','')[:10] or datetime.now().strftime("%m-%d")
            })
    except Exception as e: print(f"  博客园: {e}")

    # ── GitHub Trending ──
    try:
        from crawlers.multi_source import fetch_github_trending
        repos = fetch_github_trending(limit=6)
        for a in repos:
            safe = ''.join(c for c in a['title'][:40] if c.isalnum() or c in (' ','-','_')).strip()
            fpath = data_dir / f'digest_github_{safe}.md'
            if not fpath.exists():
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(f"# {a['title']}\n\n> GitHub Trending\n\n{a.get('brief','')}\n{a['url']}")
            articles.append({
                "title": a['title'], "url": a['url'],
                "brief": a.get('brief','')[:200],
                "images": [],
                "source": "GitHub Trending", "date": datetime.now().strftime("%m-%d")
            })
    except Exception as e: print(f"  GitHub: {e}")

    # ── OSChina ──
    try:
        from crawlers.multi_source import fetch_oschina
        items = fetch_oschina(limit=5)
        for a in items:
            articles.append({
                "title": a['title'], "url": a['url'],
                "brief": a.get('brief','')[:200],
                "images": [],
                "source": "OSChina", "date": datetime.now().strftime("%m-%d")
            })
    except Exception as e: print(f"  OSChina: {e}")

    return articles

def generate_digest(articles: list[dict]) -> str:
    """生成信息流 HTML 邮件"""
    today = datetime.now().strftime("%Y年%m月%d日 %A")
    total = len(articles)

    # ── 文章卡片 ──
    cards = ""
    for i, a in enumerate(articles):
        img_tag = ""
        if a.get("images"):
            img_tag = f'<img src="{a["images"][0]}" style="width:100%;max-height:240px;object-fit:cover;border-radius:8px;margin-bottom:8px" onerror="this.style.display=\'none\'">'

        source_color = {"掘金": "#1e80ff", "博客园": "#2ecc71", "GitHub Trending": "#333", "OSChina": "#27ae60"}.get(a["source"], "#999")

        cards += f"""
        <div style="margin-bottom:24px;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
            {img_tag}
            <div style="padding:16px">
                <span style="background:{source_color};color:#fff;padding:2px 10px;border-radius:4px;font-size:12px">{a['source']}</span>
                <span style="color:#999;font-size:12px;margin-left:8px">{a.get('date','')}</span>
                <h3 style="margin:10px 0 6px;font-size:18px;line-height:1.4">
                    <a href="{a['url']}" style="color:#2c3e50;text-decoration:none" target="_blank">{a['title'][:80]}</a>
                </h3>
                <p style="color:#666;font-size:14px;line-height:1.6;margin:0">{a.get('brief','')[:200]}</p>
                <a href="{a['url']}" style="display:inline-block;margin-top:10px;color:{source_color};font-size:13px;text-decoration:none" target="_blank">阅读全文 →</a>
            </div>
        </div>"""

    # ── 趋势热词 ──
    kw_html = ""
    try:
        from services.trends import analyze_trends
        trends = analyze_trends()
        top_kw = trends["keywords"][:15]
        colors = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c"]
        import random; random.seed(42)
        for w, c in top_kw:
            color = random.choice(colors)
            size = max(14, min(28, 14 + (c / max(1, top_kw[0][1])) * 14))
            kw_html += f'<span style="font-size:{size}px;color:{color};margin:4px;display:inline-block">{w}</span> '
    except Exception:
        kw_html = "趋势数据暂不可用"

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
    <div class="stat"><div class="stat-num">{len(set(a['source'] for a in articles))}</div><div class="stat-label">🌐 数据源</div></div>
    <div class="stat"><div class="stat-num">{len([a for a in articles if a.get('images')])}</div><div class="stat-label">🖼️ 含图文章</div></div>
</div>

<div class="cloud">
    <div class="cloud-title">🔥 技术热词趋势</div>
    {kw_html}
</div>

<h2 style="color:#2c3e50;margin:20px 0">📖 今日精选</h2>
{cards}

<div class="footer">
    <p>由 <a href="#">DevPilot</a> 自动生成 · 每日 8:00 推送</p>
    <p>如需调整推送频率或退订，请联系管理员</p>
</div>

</div></body></html>"""
    return html

def load_smtp() -> dict | None:
    c = {k: os.getenv(k, "") for k in ["SMTP_HOST","SMTP_PORT","SMTP_USER","SMTP_PASSWORD","SMTP_FROM"]}
    return c if c["SMTP_HOST"] and c["SMTP_USER"] else None

def send_email(html: str, to: str, smtp: dict):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📰 DevPilot 每日技术摘要 - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = smtp.get("SMTP_FROM") or smtp["SMTP_USER"]
    msg["To"] = to
    msg.attach(MIMEText(html, "html", "utf-8"))
    port = int(smtp.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(smtp["SMTP_HOST"], port, timeout=15) as s:
        s.login(smtp["SMTP_USER"], smtp["SMTP_PASSWORD"])
        s.sendmail(msg["From"], [to], msg.as_string())
    print(f"✅ 邮件已发送至 {to}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DevPilot 每日技术摘要")
    parser.add_argument("--send", action="store_true", help="发送邮件")
    parser.add_argument("--crawl", action="store_true", help="先爬取再生成")
    parser.add_argument("--to", type=str, help="收件人邮箱")
    parser.add_argument("--output", type=str, help="输出路径")
    args = parser.parse_args()

    articles = []
    if args.crawl:
        print("📥 爬取最新文章...")
        articles = crawl_all_sources()
        print(f"   共获取 {len(articles)} 篇")
    else:
        # 仅从已有文件生成
        print("📊 从已有数据生成摘要...")

    html = generate_digest(articles or crawl_all_sources())

    out = args.output or str(PROJECT_ROOT / "data" / f"digest_{datetime.now().strftime('%Y%m%d')}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 已保存至 {out}")

    if args.send:
        smtp = load_smtp()
        if not smtp:
            print("❌ SMTP 未配置: set SMTP_HOST / SMTP_USER / SMTP_PASSWORD")
            sys.exit(1)
        send_email(html, args.to or os.getenv("DIGEST_EMAIL_TO",""), smtp)
