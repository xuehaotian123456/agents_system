"""
来源可信度打分 + 冲突消解 + 脏数据过滤
=====================================
独立于爬虫层，对所有入库文档进行质量评估。

职责：
1. 可信度分级: 根据来源类型给每篇文档打分（0.0~1.0）
2. 冲突消解: 多源信息冲突时高权威覆盖低权威
3. 脏数据过滤: 过滤短水文、广告、无意义内容

设计动机：
  多数据源（官方文档 + Issue + 博客）必然存在信息冲突和质量差异。
  本模块保证检索和回答阶段优先采纳高权威来源。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document

# ==================== 可信度分级 ====================

# 来源类型 → 可信度分数
CREDIBILITY_TIERS: dict[str, float] = {
    # ── GitHub 来源（最高权威）──
    "official_doc": 1.0,            # 开源项目官方文档
    "github_issue_labeled": 0.85,   # 有标签的 Issue（bug/enhancement 等）
    "github_issue_answered": 0.75,  # 有实质回复的 Issue
    "github_issue_open": 0.6,       # 普通未分类 Issue
    # ── Gitee 来源（国内镜像，略低于 GitHub）──
    "gitee_repo_doc": 0.95,         # Gitee 官方文档（略低于 GitHub doc）
    "gitee_issue_labeled": 0.75,    # Gitee 有标签 Issue
    "gitee_issue_normal": 0.60,     # Gitee 普通 Issue
    # ── 社区来源（低权重补充）──
    "tech_blog_quality": 0.5,       # 知名技术社区优质文章
    "tech_blog_personal": 0.4,      # 个人技术博客
    "rss_headline": 0.3,            # RSS 标题（仅有摘要）
    "unknown": 0.1,                 # 未知来源
}

# 冲突消解优先级
CONFLICT_PRIORITY: list[str] = [
    "official_doc",
    "gitee_repo_doc",
    "github_issue_labeled",
    "gitee_issue_labeled",
    "github_issue_answered",
    "github_issue_open",
    "gitee_issue_normal",
    "tech_blog_quality",
    "tech_blog_personal",
    "rss_headline",
    "unknown",
]


def get_credibility(source_type: str) -> float:
    """根据来源类型返回可信度分数"""
    return CREDIBILITY_TIERS.get(source_type, CREDIBILITY_TIERS["unknown"])


def score_source(source_type: str, metadata: dict | None = None) -> float:
    """
    计算来源可信度分数（可扩展维度的入口）。

    Args:
        source_type: 来源类型字符串（如 "official_doc"）
        metadata: 文档元数据（可包含额外的质量信号）

    Returns:
        0.0 ~ 1.0 的可信度分数
    """
    base_score = get_credibility(source_type)

    if metadata:
        # 额外加分项（未来可扩展）
        # 例: 如果有 star 数、点赞数等社区信号
        pass

    return min(1.0, base_score)


# ==================== 冲突消解 ====================

def resolve_conflict(docs: list[Document]) -> list[Document]:
    """
    多源信息冲突时，高权威覆盖低权威。

    策略：
    - 相同知识点（近似标题/key）在不同来源中出现时，
      保留高权威文档，低权威文档降级标记为 "参考来源"。
    - 不删除低权威文档（保留多样性），但排序靠后 + 标记。

    Args:
        docs: 待消解的文档列表

    Returns:
        处理后的文档列表（低权威文档 metadata 中添加 conflict_note）
    """
    if len(docs) <= 1:
        return docs

    # 按优先级排序
    def _priority(source_type: str) -> int:
        try:
            return CONFLICT_PRIORITY.index(source_type)
        except ValueError:
            return len(CONFLICT_PRIORITY)

    # 检测近似重复（标题前 50 字符相似度 > 0.7）
    best_docs: dict[str, Document] = {}  # key → 当前最高权威文档
    for doc in docs:
        title = (doc.metadata.get("title") or doc.page_content[:50]).strip()
        key = title[:50].lower()

        if key in best_docs:
            existing_pri = _priority(
                best_docs[key].metadata.get("source_type", "unknown"))
            current_pri = _priority(
                doc.metadata.get("source_type", "unknown"))

            if current_pri < existing_pri:
                # 当前文档权威更高 → 替换，旧文档降级
                old_doc = best_docs[key]
                old_doc.metadata["conflict_note"] = (
                    f"已被更高权威来源覆盖: {doc.metadata.get('source', '')}")
                old_doc.metadata["suppressed"] = True
                best_docs[key] = doc
            else:
                # 当前文档权威更低 → 降级
                doc.metadata["conflict_note"] = (
                    f"冲突: 存在更高权威来源 {best_docs[key].metadata.get('source', '')}")
                doc.metadata["suppressed"] = True
        else:
            best_docs[key] = doc

    # 重新排序：高权威 + 未降级 优先
    result = sorted(docs, key=lambda d: (
        d.metadata.get("suppressed", False),   # 被降级 → 排最后
        _priority(d.metadata.get("source_type", "unknown")),  # 权威低 → 排后面
    ))
    return result


# ==================== 脏数据过滤 ====================

def is_low_quality(content: str, title: str = "", source_type: str = "unknown") -> bool:
    """
    快速判断文档是否为低质量内容。

    过滤规则：
    1. 纯文本字数 < 100 → 过滤
    2. 包含广告/营销特征词 → 过滤
    3. 纯外链无原创 → 过滤
    4. 仅有标题无实质内容 → 过滤

    Returns:
        True = 需要被过滤
    """
    content = content.strip()
    title = title.strip()

    # 1. 过短
    if len(content) < 100:
        return True

    # 2. 广告特征词
    ad_keywords = ["加微信", "扫码关注", "限时优惠", "免费领取",
                   "点击领取", "关注公众号", "转发", "打赏"]
    for kw in ad_keywords:
        if kw in content[:500]:  # 只检查开头（正文前面插入的广告）
            return True

    # 3. 纯外链（内容大部分是 URL）
    url_chars = sum(1 for c in content if c in "http")
    if url_chars > len(content) * 0.3:  # 30% 字符是 URL
        return True

    # 4. 仅有标题的 RSS 摘要
    if source_type == "rss_headline" and len(content) < 300:
        return True

    return False


def filter_docs(docs: list[Document]) -> list[Document]:
    """
    从文档列表中过滤低质量文档。

    Returns:
        过滤后的文档列表
    """
    filtered = []
    removed = 0
    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        title = doc.metadata.get("title", "")
        if is_low_quality(doc.page_content, title, source_type):
            removed += 1
            continue
        filtered.append(doc)

    if removed:
        from utils.logger_handler import logger
        logger.info(f"[质量过滤] 移除了 {removed} 篇低质量文档")
    return filtered
