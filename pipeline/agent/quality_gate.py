"""
文档质量门控 — LangGraph 条件路由流水线
========================================
数据质量治理的第二层 (LLM 语义判定)，第一层是规则过滤 (source_credibility)。

图结构:
    CRAWL (候选文档)
       ↓
    RULE_FILTER (免费规则层: 广告词/长度/纯外链 → 直接 SKIP)
       ↓
    LLM_GATE (语义判定: 技术相关? 质量如何?)
       ↓ 条件边
    ┌──────────┬────────────┬──────────┐
    ↓          ↓            ↓
  INGEST    DEMOTE        SKIP
 (原可信度) (可信度×0.4  (丢弃, 理由留痕)
            + 降权标记)

设计要点:
1. 两级互补: 规则层免费抓格式垃圾, LLM 层抓语义垃圾 (八卦/软文/无关内容)
2. 三路而非二元: judge 会误判, demote 通道"宁可降权不误杀"
3. 判定留痕: 每个文档的 verdict + reason 写入 state, 可审计
4. 成本控制: 批量判定 (每批 5 篇) + 内容 hash 缓存 (跨轮次复用)
5. 故障开放: LLM 不可用时默认 INGEST, 不阻塞入库

使用 (调度器集成):
    from agent.quality_gate import run_quality_gate
    result = run_quality_gate(docs)   # docs: [{title, content, source, credibility}]
    # result.ingest / result.demote / result.skip / result.trace
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, TypedDict

# 独立运行时的路径引导
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END

from utils.logger_handler import logger

# ==================== 状态定义 ====================


class QualityGateState(TypedDict, total=False):
    """质量门控流水线状态"""
    docs: list[dict]                 # 候选文档 [{title, content, source, credibility}]
    rule_skipped: list[dict]         # 规则层拦截 [{doc, reason}]
    llm_inputs: list[dict]           # 通过规则层、待 LLM 判定的文档
    verdicts: dict[str, str]         # content_hash -> "ingest"/"demote"/"skip"
    verdict_reasons: dict[str, str]  # content_hash -> 判定理由
    ingest: list[dict]               # 正常入库
    demote: list[dict]               # 降权入库
    skip: list[dict]                 # 丢弃
    trace: list[dict]                # 判定留痕 [{title, verdict, reason}]
    stats: dict                      # 统计


VERDICT_CACHE_PATH = Path(__file__).parent.parent / "data" / "quality_verdicts.json"

JUDGE_PROMPT = """你是技术知识库的内容质量审核员。判断以下文档是否适合收录进技术知识库。

判定标准:
- ingest (入库): 技术相关内容, 有实质信息 (教程/原理/报错分析/框架介绍等), 质量正常
- demote (降权): 技术沾边但质量差 (软文推广/八卦夹杂/信息量极低/标题党), 或对技术知识库价值有限
- skip (丢弃): 与技术完全无关 (生活八卦/广告/招聘软文/新闻), 或纯营销内容

文档列表:
{docs_text}

请以 JSON 数组格式输出判定, 不要输出其他内容:
[{{"index": 0, "verdict": "ingest", "reason": "一句话理由"}}, ...]"""


# ==================== 工具函数 ====================


def _content_hash(doc: dict) -> str:
    """内容 hash (用于判定缓存)"""
    content = (doc.get("content") or doc.get("brief") or "").strip()
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]


def _load_verdict_cache() -> dict:
    """加载判定缓存 (跨轮次复用, 内容没变不重复花 LLM 钱)"""
    if VERDICT_CACHE_PATH.exists():
        try:
            return json.loads(VERDICT_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_verdict_cache(cache: dict):
    try:
        VERDICT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 限制缓存大小 (保留最近 2000 条)
        if len(cache) > 2000:
            keys = list(cache.keys())[-2000:]
            cache = {k: cache[k] for k in keys}
        VERDICT_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[QualityGate] 缓存保存失败: {e}")


# ==================== 图节点 ====================


def rule_filter_node(state: QualityGateState) -> QualityGateState:
    """
    第一层: 规则过滤 (免费)。
    复用 source_credibility 的格式级过滤: 广告词/过短/纯外链。
    """
    from crawlers.source_credibility import is_low_quality

    docs = state.get("docs", [])
    rule_skipped = []
    passed = []

    for doc in docs:
        content = doc.get("content") or doc.get("brief") or ""
        title = doc.get("title", "")
        source_type = doc.get("source_type", "unknown")

        if is_low_quality(content, title, source_type):
            rule_skipped.append({
                "doc": doc,
                "reason": f"规则层拦截 (格式垃圾: {source_type})",
            })
            continue
        passed.append(doc)

    state["rule_skipped"] = rule_skipped
    state["llm_inputs"] = passed
    return state


def llm_gate_node(state: QualityGateState) -> QualityGateState:
    """
    第二层: LLM 语义判定 (批量 + 缓存)。

    - 先查内容 hash 缓存, 命中直接复用判定
    - 未命中按 5 篇一批调用 LLM
    - LLM 失败 → 默认 ingest (故障开放, 不阻塞入库)
    """
    docs = state.get("llm_inputs", [])
    verdicts: dict[str, str] = {}
    reasons: dict[str, str] = {}

    cache = _load_verdict_cache()

    # ── 缓存命中 ──
    uncached = []
    for doc in docs:
        h = _content_hash(doc)
        if h in cache:
            entry = cache[h]
            verdicts[h] = entry.get("verdict", "ingest")
            reasons[h] = entry.get("reason", "缓存命中")
        else:
            uncached.append(doc)

    # ── 批量 LLM 判定 ──
    if uncached:
        try:
            from model.factory import robust_llm_call

            batch_size = 5
            for i in range(0, len(uncached), batch_size):
                batch = uncached[i:i + batch_size]
                docs_text = "\n\n".join(
                    f"[文档{idx}]\n标题: {b.get('title', '')[:100]}\n"
                    f"内容: {(b.get('content') or b.get('brief') or '')[:600]}"
                    for idx, b in enumerate(batch)
                )

                resp = robust_llm_call(JUDGE_PROMPT.format(docs_text=docs_text))
                raw = resp.content if hasattr(resp, 'content') else str(resp)

                # 解析 JSON 数组 (多层容错)
                parsed = _parse_judge_output(raw, len(batch))

                # 解析失败 → 重试一次 (严格指令)
                if len(parsed) < len(batch):
                    retry_prompt = (JUDGE_PROMPT +
                                    "\n\n★ 严格要求: 只输出 JSON 数组本身, "
                                    "不要任何解释文字, 不要 markdown 代码块。")
                    resp = robust_llm_call(retry_prompt.format(docs_text=docs_text))
                    raw = resp.content if hasattr(resp, 'content') else str(resp)
                    parsed = _parse_judge_output(raw, len(batch))

                for idx, b in enumerate(batch):
                    h = _content_hash(b)
                    item = parsed.get(idx, {"verdict": "ingest", "reason": "解析失败, 默认入库"})
                    v = item.get("verdict", "ingest")
                    if v not in ("ingest", "demote", "skip"):
                        v = "ingest"
                    verdicts[h] = v
                    reasons[h] = item.get("reason", "")
                    # 写入缓存
                    cache[h] = {"verdict": v, "reason": reasons[h]}
        except Exception as e:
            # 故障开放: LLM 不可用 → 全部 ingest
            logger.warning(f"[QualityGate] LLM 判定失败, 全部默认入库: {e}")
            for doc in uncached:
                h = _content_hash(doc)
                verdicts[h] = "ingest"
                reasons[h] = f"LLM不可用, 默认入库: {e}"

        _save_verdict_cache(cache)

    state["verdicts"] = verdicts
    state["verdict_reasons"] = reasons
    return state


def _parse_judge_output(raw: str, n: int) -> dict[int, dict]:
    """解析 LLM 判定输出 (JSON 数组, 多层容错)"""
    import re
    data = None

    # 1. 直接解析
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, AttributeError):
        pass

    # 2. 代码块提取 (```json ... ```)
    if data is None:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
        if m:
            try:
                data = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                data = None

    # 3. 首尾括号提取
    if data is None:
        m = re.search(r'\[[\s\S]*\]', raw)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None

    # 4. 逐项正则兜底 (LLM 输出"verdict: xxx"格式时)
    if data is None:
        items = []
        for m in re.finditer(
                r'index["\']?\s*[:：]\s*(\d+)[\s\S]{0,80}?'
                r'verdict["\']?\s*[:：]\s*["\']?(ingest|demote|skip)["\']?', raw):
            items.append({"index": int(m.group(1)), "verdict": m.group(2), "reason": "正则兜底解析"})
        if items:
            data = items

    if not isinstance(data, list):
        return {}

    out = {}
    for item in data:
        if isinstance(item, dict) and "index" in item:
            try:
                out[int(item["index"])] = item
            except (ValueError, TypeError):
                continue
    return out


# ==================== 条件路由 + 三路分支节点 ====================


def _next_branch(state: QualityGateState) -> str:
    """
    条件边路由函数: 按 verdict 决定下一分支。
    优先级: demote > skip > ingest (降权先处理, 丢弃次之, 剩余全入库)。
    每个分支处理后从 llm_inputs 移除对应文档, 循环收敛 ≤3 轮。
    """
    verdicts = state.get("verdicts", {})
    llm_inputs = state.get("llm_inputs", [])

    for doc in llm_inputs:
        if verdicts.get(_content_hash(doc), "ingest") == "demote":
            return "demote_node"
    for doc in llm_inputs:
        if verdicts.get(_content_hash(doc), "ingest") == "skip":
            return "skip_node"
    return "ingest_node"


def demote_node(state: QualityGateState) -> QualityGateState:
    """DEMOTE 通道: 提取 demote 文档, 可信度 ×0.4 + 标记, 判定留痕"""
    verdicts = state.get("verdicts", {})
    reasons = state.get("verdict_reasons", {})
    remaining = []
    demoted = list(state.get("demote", []))

    for doc in state.get("llm_inputs", []):
        h = _content_hash(doc)
        if verdicts.get(h, "ingest") == "demote":
            doc["credibility"] = round(doc.get("credibility", 0.5) * 0.4, 2)
            doc["quality_demoted"] = True
            demoted.append(doc)
            state["trace"].append({
                "title": doc.get("title", "")[:80], "verdict": "demote",
                "reason": reasons.get(h, "")})
        else:
            remaining.append(doc)

    state["llm_inputs"] = remaining
    state["demote"] = demoted
    return state


def skip_node(state: QualityGateState) -> QualityGateState:
    """SKIP 通道: 提取 skip 文档, 理由留痕"""
    verdicts = state.get("verdicts", {})
    reasons = state.get("verdict_reasons", {})
    remaining = []
    skipped = list(state.get("skip", []))

    for doc in state.get("llm_inputs", []):
        h = _content_hash(doc)
        if verdicts.get(h, "ingest") == "skip":
            skipped.append(doc)
            state["trace"].append({
                "title": doc.get("title", "")[:80], "verdict": "skip",
                "reason": reasons.get(h, "")})
        else:
            remaining.append(doc)

    state["llm_inputs"] = remaining
    state["skip"] = skipped
    return state


def ingest_node(state: QualityGateState) -> QualityGateState:
    """INGEST 通道 (终点): 剩余文档全部入库 + 规则层拦截归入 skip + 统计"""
    ingest = list(state.get("ingest", []))

    for doc in state.get("llm_inputs", []):
        ingest.append(doc)
        state["trace"].append({
            "title": doc.get("title", "")[:80], "verdict": "ingest", "reason": ""})
    state["ingest"] = ingest
    state["llm_inputs"] = []

    # 规则层拦截的也归入 skip
    skipped = list(state.get("skip", []))
    for item in state.get("rule_skipped", []):
        skipped.append(item["doc"])
        state["trace"].append({
            "title": item["doc"].get("title", "")[:80], "verdict": "skip",
            "reason": item["reason"]})
    state["skip"] = skipped

    state["stats"] = {
        "total": len(state.get("docs", [])),
        "ingest": len(ingest),
        "demote": len(state.get("demote", [])),
        "skip": len(skipped),
    }
    # 判定留痕日志
    for t in state["trace"]:
        logger.info(f"[QualityGate] {t['verdict'].upper():>6}: {t['title']} | {t['reason'][:50]}")
    return state


# ==================== 图组装 ====================


def build_quality_graph():
    """
    质量门控图 (真正的条件路由):
        rule_filter → llm_gate ──条件边──┬→ demote_node ─┐
                                        ├→ skip_node  ──┼→ (循环收敛) → ingest_node → END
                                        └→ ingest_node ─┘
    """
    builder = StateGraph(QualityGateState)
    builder.add_node("rule_filter", rule_filter_node)
    builder.add_node("llm_gate", llm_gate_node)
    builder.add_node("demote_node", demote_node)
    builder.add_node("skip_node", skip_node)
    builder.add_node("ingest_node", ingest_node)

    builder.set_entry_point("rule_filter")
    builder.add_edge("rule_filter", "llm_gate")

    # ★ 条件边: 三路分支 + 循环收敛
    builder.add_conditional_edges(
        "llm_gate", _next_branch,
        {"demote_node": "demote_node", "skip_node": "skip_node", "ingest_node": "ingest_node"})
    builder.add_conditional_edges(
        "demote_node", _next_branch,
        {"demote_node": "demote_node", "skip_node": "skip_node", "ingest_node": "ingest_node"})
    builder.add_conditional_edges(
        "skip_node", _next_branch,
        {"demote_node": "demote_node", "skip_node": "skip_node", "ingest_node": "ingest_node"})

    builder.add_edge("ingest_node", END)

    return builder.compile()


# 全局图实例
quality_graph = build_quality_graph()


# ==================== 便捷入口 ====================


def run_quality_gate(docs: list[dict], enabled: bool = True) -> dict:
    """
    运行质量门控。

    Args:
        docs: 候选文档 [{title, content, source, credibility, source_type}]
        enabled: False 时全部直接 ingest (总开关)

    Returns:
        {"ingest": [...], "demote": [...], "skip": [...], "trace": [...], "stats": {...}}
    """
    if not enabled or not docs:
        return {"ingest": docs, "demote": [], "skip": [], "trace": [],
                "stats": {"total": len(docs), "ingest": len(docs), "demote": 0, "skip": 0,
                          "gate_disabled": True}}

    state: QualityGateState = {
        "docs": docs, "trace": [], "ingest": [], "demote": [], "skip": [],
    }
    result = quality_graph.invoke(state)

    return {
        "ingest": result.get("ingest", []),
        "demote": result.get("demote", []),
        "skip": result.get("skip", []),
        "trace": result.get("trace", []),
        "stats": result.get("stats", {}),
    }


if __name__ == "__main__":
    # 独立测试: 3 篇语义垃圾 (规则层抓不住)
    test_docs = [
        {"title": "携程员工月薪3万没涨薪", "content": "携程员工在携程4年月薪3万左右基本没涨薪了不过挺满足的" * 5,
         "source": "掘金", "credibility": 0.5, "source_type": "tech_blog_quality"},
        {"title": "React 18 Suspense 原理解析", "content": "React 18 的 Suspense 用于处理异步组件加载..." * 10,
         "source": "掘金", "credibility": 0.5, "source_type": "tech_blog_quality"},
        {"title": "限时优惠加微信", "content": "加微信领取免费课程，限时优惠，扫码关注公众号领取" * 5,
         "source": "博客园", "credibility": 0.4, "source_type": "tech_blog_personal"},
    ]
    result = run_quality_gate(test_docs)
    print(f"\n统计: {result['stats']}")
    for t in result["trace"]:
        print(f"  {t['verdict']:>6}: {t['title'][:40]} | {t['reason'][:40]}")
