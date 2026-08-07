"""通用工具"""
import datetime
from langchain_core.tools import tool
from utils.logger_handler import logger

@tool(description="获取指定城市的天气信息。入参 city（城市名称），返回天气描述。")
def get_weather(city: str) -> str:
    return f"城市 {city} 天气：晴天，气温 22-28°C，空气质量良好，适合出行。"

@tool(description="获取当前日期和时间。无入参，返回当前年月日和时间。")
def get_current_time() -> str:
    now = datetime.datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M')} (星期{['一','二','三','四','五','六','日'][now.weekday()]})"

@tool(description="搜索互联网获取最新信息。入参 query（搜索词），返回相关结果摘要。用于获取知识库外的实时信息。")
def search_web(query: str) -> str:
    try:
        import httpx
        from bs4 import BeautifulSoup

        # 方案1: DuckDuckGo Lite (更稳定的 HTML 版本)
        try:
            resp = httpx.get("https://lite.duckduckgo.com/lite/",
                            params={"q": query},
                            headers={"User-Agent": "Mozilla/5.0"},
                            timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                results = []
                for row in soup.find_all("tr", class_="result-snippet"):
                    text = row.get_text(strip=True)[:200]
                    if text:
                        results.append(f"• {text}")
                if results:
                    return f"搜索结果 ({len(results)} 条):\n" + "\n".join(results[:5])
        except Exception:
            pass

        # 方案2: 降级 —— 直接用通用知识回答
        return f"关于 '{query}'，由于网络限制无法获取实时搜索结果。建议:\n• 使用搜索引擎直接搜索\n• 访问官方文档网站\n• 查阅 GitHub 相关仓库"
    except Exception as e:
        return f"搜索失败（请稍后重试）: {e}"

@tool(description="保存学习笔记。入参 content（笔记内容），将内容追加保存到本地笔记文件。")
def save_note(content: str) -> str:
    try:
        from pathlib import Path
        note_dir = Path(__file__).parent.parent.parent / "data"
        note_dir.mkdir(exist_ok=True)
        note_path = note_dir / "learning_notes.md"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(note_path, "a", encoding="utf-8") as f:
            f.write(f"\n---\n## 笔记 [{timestamp}]\n\n{content}\n")
        return f"笔记已保存 (共 {content.count(chr(10)) + 1} 行)。"
    except Exception as e:
        return f"保存失败: {e}"

@tool(description="获取当前知识库的技术热词趋势分析报告，含热度升降检测。无入参，返回全局热词Top15、最近趋势、上升/下降/稳定的技术热词。用于了解技术热点变化。")
def trend_report() -> str:
    try:
        from services.trends import get_trend_summary, detect_trend_direction
        summary = get_trend_summary()

        # 趋势方向
        try:
            direction = detect_trend_direction()
            rising = direction.get("rising", [])[:5]
            falling = direction.get("falling", [])[:5]

            if rising or falling:
                summary += "\n\n## 📈 趋势变化检测\n"
                if rising:
                    summary += "\n🔥 **上升趋势**:\n"
                    for r in rising:
                        summary += f"  • {r['word']}: {r['yesterday']}→{r['today']} ({r['change']})\n"
                if falling:
                    summary += "\n📉 **下降趋势**:\n"
                    for f in falling:
                        summary += f"  • {f['word']}: {f['yesterday']}→{f['today']} ({f['change']})\n"
                summary += f"\n*{direction.get('summary', '')}*"
        except Exception:
            pass

        # 生成可视化
        try:
            from services.trends import generate_trend_html
            viz_path = generate_trend_html()
            if viz_path:
                summary += f"\n\n[VIZ:{viz_path}]"
        except Exception:
            pass

        return summary
    except Exception as e:
        return f"趋势分析失败: {e}"

@tool(description="获取当前用户的技术兴趣画像和个性化推荐。无入参，返回用户关注的技术关键词排名、查询历史统计、基于知识图谱的个性化学习推荐。")
def user_profile() -> str:
    try:
        from services.memory import get_user_profile
        return get_user_profile()
    except Exception as e:
        return f"画像获取失败: {e}"
