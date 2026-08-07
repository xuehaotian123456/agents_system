"""热词趋势分析服务 — 从已爬取文章中提取技术热词和趋势"""
import os
import re
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
import jieba
import jieba.posseg as pseg
from utils.logger_handler import logger

DATA_DIR = Path(__file__).parent.parent / "data"
ARTICLES_DIR = DATA_DIR / "articles"

# 技术关键词白名单 — 保持这些词即使低频也保留
TECH_WHITELIST = {
    "AI", "LLM", "GPT", "RAG", "Agent", "LangGraph", "LangChain",
    "React", "Vue", "Next.js", "Nuxt", "TypeScript", "JavaScript",
    "Python", "Rust", "Go", "Kubernetes", "Docker", "Redis",
    "MySQL", "PostgreSQL", "MongoDB", "GraphQL", "REST",
    "微服务", "Serverless", "DevOps", "CI/CD", "Git",
    "机器学习", "深度学习", "大模型", "Transformer",
    "前端", "后端", "全栈", "架构", "性能优化",
    "Codex", "Cursor", "Copilot", "VSCode", "WebAssembly",
}

# 停用词
_STOP = {"的", "了", "在", "是", "有", "和", "就", "不", "人", "都",
         "一", "很", "去", "能", "到", "说", "要", "会", "也", "着",
         "可以", "使用", "需要", "一个", "这个", "那个", "什么", "怎么",
         "如何", "目前", "现在", "通过", "包括", "没有", "已经", "比较",
         "非常", "大家", "以上", "以下", "相关", "主要", "所有", "不同",
         "各种", "关于", "对于", "说明", "点击", "查看", "链接", "阅读",
         "文章", "作者", "来源", "掘金", "博客园", "内容", "可以", "如果",
         "我们", "他们", "因为", "所以", "但是", "而且", "然后",
         "提供", "支持", "实现", "采用", "应用", "开发", "系统", "技术",
         # 分析噪声
         "数据", "代码", "问题", "方案", "项目", "方式", "效果",
         "情况", "选择", "类型", "性能", "质量", "体验", "数量",
}

def extract_keywords(text: str, top_k: int = 20) -> list[tuple[str, int]]:
    """从文本中提取技术关键词及频次"""
    words = []
    # 先匹配白名单词
    for kw in TECH_WHITELIST:
        if kw.lower() in text.lower():
            count = text.lower().count(kw.lower())
            words.extend([kw] * min(count, 5))

    # jieba 分词补充
    for pair in pseg.cut(text[:3000]):
        word, flag = pair.word.strip(), pair.flag
        if len(word) < 2 or word in _STOP:
            continue
        if flag in {'n', 'nr', 'ns', 'nz', 'eng', 'x'} or flag.startswith('n'):
            words.append(word)

    counter = Counter(words)
    return counter.most_common(top_k)

def analyze_trends() -> dict:
    """分析所有已爬取文章的全局热词趋势"""
    if not ARTICLES_DIR.exists():
        return {"keywords": [], "timeline": [], "article_count": 0}

    all_keywords = Counter()
    timeline = []  # [{date: ..., keywords: [...], article: ...}]

    files = sorted(ARTICLES_DIR.glob("*.md"), key=os.path.getmtime, reverse=True)
    for fpath in files:
        try:
            mtime = os.path.getmtime(fpath)
            date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            # 提取标题
            title = ""
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()[:60]
                    break

            # 提取关键词
            keywords = extract_keywords(content, top_k=10)

            # 如果文件里有日期标记，用那个
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', content[:500])
            if date_match:
                date = date_match.group(1)

            for kw, freq in keywords:
                all_keywords[kw] += freq

            timeline.append({
                "date": date,
                "title": title or fpath.stem[:40],
                "keywords": [{"word": w, "freq": f} for w, f in keywords[:5]],
                "file": fpath.name,
            })
        except Exception as e:
            logger.warning(f"分析文件失败: {fpath.name}: {e}")

    # 合并同一天
    daily_trends = {}
    for item in timeline:
        date = item["date"]
        if date not in daily_trends:
            daily_trends[date] = {"date": date, "articles": [], "keywords": Counter()}
        daily_trends[date]["articles"].append(item["title"])
        for kw in item["keywords"]:
            daily_trends[date]["keywords"][kw["word"]] += kw["freq"]

    daily_list = []
    for date in sorted(daily_trends.keys(), reverse=True):
        d = daily_trends[date]
        daily_list.append({
            "date": date,
            "articles": d["articles"][:5],
            "keywords": d["keywords"].most_common(8),
        })

    return {
        "keywords": all_keywords.most_common(30),
        "timeline": timeline[:30],
        "daily_trends": daily_list[:14],
        "article_count": len(files),
    }

def detect_trend_direction() -> dict:
    """检测技术热词的趋势方向（上升/下降/稳定）"""
    trends = analyze_trends()
    daily = trends.get("daily_trends", [])

    if len(daily) < 2:
        return {"rising": [], "falling": [], "stable": [], "summary": "数据不足，需要至少2天的数据。"}

    today_kw = {w: c for w, c in daily[0]["keywords"][:15]}
    yesterday_kw = {w: c for w, c in daily[1]["keywords"][:15]} if len(daily) > 1 else {}

    rising, falling, stable = [], [], []

    all_words = set(list(today_kw.keys()) + list(yesterday_kw.keys()))
    for word in all_words:
        t = today_kw.get(word, 0)
        y = yesterday_kw.get(word, 0)
        if t == 0 and y == 0:
            continue
        if y == 0:
            change_pct = 100
        else:
            change_pct = round((t - y) / y * 100, 1)

        if change_pct >= 30:
            rising.append({"word": word, "today": t, "yesterday": y, "change": f"+{change_pct}%"})
        elif change_pct <= -30:
            falling.append({"word": word, "today": t, "yesterday": y, "change": f"{change_pct}%"})
        else:
            stable.append({"word": word, "today": t, "yesterday": y, "change": f"{change_pct:+}%"})

    rising.sort(key=lambda x: x["today"], reverse=True)
    falling.sort(key=lambda x: x["today"], reverse=True)
    stable.sort(key=lambda x: x["today"], reverse=True)

    today_date = daily[0]["date"] if daily else ""
    yesterday_date = daily[1]["date"] if len(daily) > 1 else ""

    return {
        "rising": rising[:8],
        "falling": falling[:8],
        "stable": stable[:5],
        "today_date": today_date,
        "yesterday_date": yesterday_date,
        "summary": f"{today_date} vs {yesterday_date}: {len(rising)}升 {len(falling)}降 {len(stable)}稳"
    }

def generate_trend_html() -> str | None:
    """生成热词词云 + 时间轴 HTML，供 Agent 回答内嵌渲染"""
    trends = analyze_trends()
    if not trends["keywords"]:
        return None

    top_words = trends["keywords"][:20]
    max_freq = top_words[0][1] if top_words else 1
    import random; random.seed(42)

    # === 词云 ===
    colors = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c","#e67e22","#2980b9"]
    word_spans = []
    for word, freq in top_words:
        size = max(12, min(36, int(14 + (freq / max_freq) * 22)))
        color = random.choice(colors)
        word_spans.append(
            f'<span style="font-size:{size}px;color:{color};margin:6px;display:inline-block;'
            f'cursor:default" title="频次: {freq}">{word}</span>'
        )

    # === 时间轴 ===
    daily = trends.get("daily_trends", [])[:7]
    timeline_rows = []
    for d in daily:
        kw_str = ", ".join([f'<span style="background:#f0f0f0;padding:2px 6px;border-radius:4px;margin:2px">{w}</span>' for w, c in d["keywords"][:5]])
        articles_str = "<br>".join([f'&nbsp;&nbsp;📄 {a[:50]}' for a in d.get("articles", [])[:3]])
        timeline_rows.append(f"""
        <div style="margin-bottom:12px;padding:8px;border-left:3px solid #3498db;background:#fafafa">
            <strong>📅 {d['date']}</strong>
            <div style="margin:4px 0">{kw_str}</div>
            <div style="font-size:12px;color:#888">{articles_str}</div>
        </div>""")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
        body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:12px}}
        .cloud{{text-align:center;padding:16px;background:#f8f9fa;border-radius:12px;margin-bottom:16px;line-height:2.2}}
        .timeline{{max-height:400px;overflow-y:auto}}
    </style></head><body>
    <div class="cloud">{''.join(word_spans)}</div>
    <div class="timeline">{''.join(timeline_rows)}</div>
    </body></html>"""

    # 保存
    import os
    from pathlib import Path
    out_dir = Path(__file__).parent.parent / "data" / "graphs"
    os.makedirs(out_dir, exist_ok=True)
    fpath = out_dir / "trend_viz.html"
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)
    return str(fpath)

def get_trend_summary() -> str:
    """生成热词趋势摘要，供 Agent 使用"""
    trends = analyze_trends()
    if not trends["keywords"]:
        return "暂无趋势数据。"

    top_kw = trends["keywords"][:15]
    kw_lines = [f"{i+1}. {w}({c}次)" for i, (w, c) in enumerate(top_kw)]

    daily = trends["daily_trends"][:3]
    daily_lines = []
    for d in daily:
        dk = ", ".join([f"{w}({c})" for w, c in d["keywords"][:5]])
        daily_lines.append(f"  {d['date']}: {dk}")

    return f"""## 📊 技术热词趋势

**总文章数**: {trends['article_count']} 篇

**全局热词 Top 15**:
{chr(10).join(kw_lines)}

**最近 3 天趋势**:
{chr(10).join(daily_lines)}
"""
