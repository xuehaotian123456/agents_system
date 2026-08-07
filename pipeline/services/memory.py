"""Agent 记忆系统 — 基于 KG 的用户兴趣追踪 + 个性化推荐"""
import re
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from utils.logger_handler import logger

DATA_DIR = Path(__file__).parent.parent / "data"
MEMORY_FILE = DATA_DIR / "user_memory.json"

def _load_memory() -> dict:
    """加载用户记忆"""
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 恢复 Counter 对象（JSON序列化后变成普通dict）
        for sid in data.get("interests", {}):
            kw = data["interests"][sid].get("keywords", {})
            if isinstance(kw, dict):
                data["interests"][sid]["keywords"] = Counter(kw)
        return data
    return {"interests": {}, "sessions": [], "topics": []}

def _save_memory(memory: dict):
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def remember_query(query: str, session_id: str = "default"):
    """记录用户查询，提取兴趣关键词并存入记忆"""
    try:
        import jieba.posseg as pseg

        memory = _load_memory()
        user = memory.setdefault("interests", {}).setdefault(session_id, {
            "keywords": Counter(), "last_seen": "", "query_count": 0
        })

        # 提取技术关键词
        tech_words = []
        for word, flag in pseg.cut(query):
            if flag in ('n', 'nr', 'nz', 'eng', 'x') and len(word) >= 2:
                tech_words.append(word)

        for w in tech_words:
            user["keywords"][w] += 1

        user["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        user["query_count"] += 1

        # 记录会话
        memory["sessions"].append({
            "session": session_id,
            "query": query[:100],
            "keywords": tech_words[:5],
            "time": user["last_seen"]
        })

        # 只保留最近 50 条
        if len(memory["sessions"]) > 50:
            memory["sessions"] = memory["sessions"][-50:]

        # 更新全局话题榜
        all_keywords = Counter()
        for sid, data in memory.get("interests", {}).items():
            all_keywords.update(data["keywords"])
        memory["topics"] = all_keywords.most_common(20)

        _save_memory(memory)

        # 同步到 KG（作为用户实体）
        try:
            from rag.knowledge_graph import get_kg
            kg = get_kg()
            if kg.is_built:
                for w in tech_words[:3]:
                    user_key = f"user:{w}"
                    kg.entity_to_chunks[user_key].append(-1)
                    kg.entity_freq[user_key] += 1
                    kg.co_occurrence[user_key][w] += 1
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"[Memory] 记录失败: {e}")

def get_user_profile(session_id: str = "default") -> str:
    """获取用户兴趣画像"""
    memory = _load_memory()
    user = memory.get("interests", {}).get(session_id)

    if not user or not user["keywords"]:
        return "暂无用户画像数据。"

    top = user["keywords"].most_common(8)
    lines = [
        f"## 👤 用户技术画像",
        f"查询次数: {user['query_count']}",
        f"最近活跃: {user['last_seen']}",
        f"\n**兴趣关键词**:",
    ]
    for i, (w, c) in enumerate(top, 1):
        lines.append(f"  {i}. {w} (关注 {c} 次)")

    # 基于兴趣推荐
    lines.append(f"\n## 💡 个性化推荐")
    for w, c in top[:3]:
        try:
            from rag.knowledge_graph import get_kg
            kg = get_kg()
            if kg.is_built:
                info = kg.get_entity(w)
                if info:
                    related = [r['entity'] for r in info.get('related', [])[:3]]
                    lines.append(f"  • 关注 **{w}** → 推荐了解: {', '.join(related)}")
                else:
                    lines.append(f"  • 关注 **{w}** → 搜索知识库获取相关文章")
        except Exception:
            lines.append(f"  • 关注 **{w}** → 尝试知识库搜索")

    return "\n".join(lines)

def get_context_for_query(query: str, session_id: str = "default") -> str:
    """根据用户历史兴趣，为当前查询提供个性化上下文"""
    memory = _load_memory()
    user = memory.get("interests", {}).get(session_id)

    if not user or not user["keywords"]:
        return ""

    top = user["keywords"].most_common(5)
    interest_words = [w for w, _ in top]
    return f"用户历史兴趣: {', '.join(interest_words)}"
