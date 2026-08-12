"""
评测报告生成器 — 整合所有评测数据生成面试级报告
==============================================
输入: eval_results.json + negative_test_results.json + llm_judge_results.json
输出: eval_report.md (可直接展示)

运行: python eval/generate_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def generate_report():
    eval_dir = Path(__file__).parent

    # 加载数据
    keyword_results = load_json(str(eval_dir / "eval_results.json"))
    negative_results = load_json(str(eval_dir / "negative_test_results.json"))
    llm_judge_results = load_json(str(eval_dir / "llm_judge_results.json"))

    # 获取系统状态
    from rag.knowledge_graph import get_kg
    from rag.vector_store import VectorStore
    from services.crawl_state import get_crawl_state

    vs = VectorStore()
    vs.load_articles()
    kg = get_kg()
    cs = get_crawl_state()

    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 标题 ──
    lines.append(f"# Agent 双引擎系统 — 评测报告")
    lines.append(f"")
    lines.append(f"> 生成时间: {now}")
    lines.append(f"> 数据规模: {len(vs.all_chunks)} chunks, {kg.entity_count} KG 实体, {cs.get_article_count()} URL 去重记录")
    lines.append(f"")

    # ── 系统概览 ──
    lines.append(f"## 一、系统概览")
    lines.append(f"")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|:---|---|")
    lines.append(f"| 文章数 | {len([f for f in (Path(__file__).parent.parent / 'data' / 'articles').iterdir() if f.suffix == '.md'])} |")
    lines.append(f"| Chunks | {len(vs.all_chunks)} |")
    lines.append(f"| KG 实体 | {kg.entity_count} |")
    lines.append(f"| ChromaDB embeddings | {vs.store._collection.count()} |")
    lines.append(f"| 爬虫源 | Gitee (MindSpore+Paddle) + 掘金 + 博客园 + HN + OSChina |")
    lines.append(f"| 检索方式 | Vector + BM25 + KG Graph (三路融合 + BGE-Reranker) |")
    lines.append(f"| KG 能力 | 实体提取(词性标注) + 共现矩阵 + IDF过滤 + 多跳扩散(1-3 hops) |")
    lines.append(f"")

    # ── 评测一: 关键词匹配对比 ──
    lines.append(f"## 二、评测一: 纯向量 RAG vs GraphRAG (关键词匹配)")
    lines.append(f"")
    lines.append(f"> 方法: 50 条查询, 按类别 × 难度分级, expected_entities 关键词命中率")
    lines.append(f"> 局限: 关键词匹配是弱标注 proxy, 非人工相关性别定")
    lines.append(f"")

    overall = keyword_results.get("overall", {})
    if overall:
        lines.append(f"### 总体指标")
        lines.append(f"")
        lines.append(f"| 指标 | 纯向量 RAG | GraphRAG | 提升 |")
        lines.append(f"|:---|---:|---:|---:|")
        vr = overall.get("vec_recall@5", 0)
        gr = overall.get("graph_recall@5", 0)
        vh = overall.get("vec_hit_rate", 0)
        gh = overall.get("graph_hit_rate", 0)
        lines.append(f"| Recall@5 | {vr:.0%} | {gr:.0%} | **+{gr-vr:.0%}** |")
        lines.append(f"| Hit Rate | {vh:.0%} | {gh:.0%} | +{gh-vh:.0%} |")
        lines.append(f"")

    by_cat = keyword_results.get("by_category", {})
    if by_cat:
        lines.append(f"### 按查询类别")
        lines.append(f"")
        lines.append(f"| 类别 | 条数 | 纯向量 | GraphRAG | 提升 |")
        lines.append(f"|:---|---:|---:|---:|---:|")
        for cat, data in by_cat.items():
            n = data.get("count", 0)
            v = data.get("vec_recall@5", 0)
            g = data.get("graph_recall@5", 0)
            imp = g - v
            lines.append(f"| {data.get('name', cat)} | {n} | {v:.0%} | {g:.0%} | **+{imp:.0%}** |")
        lines.append(f"")

    by_diff = keyword_results.get("by_difficulty", {})
    if by_diff:
        lines.append(f"### 按难度")
        lines.append(f"")
        lines.append(f"| 难度 | 条数 | 纯向量 | GraphRAG | 提升 |")
        lines.append(f"|:---|---:|---:|---:|---:|")
        for diff in ["easy", "medium", "hard"]:
            data = by_diff.get(diff, {})
            if not data:
                continue
            n = data.get("count", 0)
            v = data.get("vec_recall@5", 0)
            g = data.get("graph_recall@5", 0)
            imp = g - v
            lines.append(f"| {diff} | {n} | {v:.0%} | {g:.0%} | **+{imp:.0%}** |")
        lines.append(f"")

    # ── 评测二: LLM-as-Judge ──
    lines.append(f"## 三、评测二: LLM-as-Judge (Answerability 评分)")
    lines.append(f"")
    lines.append(f"> 方法: 用 LLM 判断检索到的文档能否回答用户问题 (0=无关, 1=弱相关, 2=部分可答, 3=完全可答)")
    lines.append(f"> 优势: 比关键词匹配更准确地衡量检索质量")
    lines.append(f"")

    lj_overall = llm_judge_results.get("overall", {})
    if lj_overall:
        va = lj_overall.get("vec_answerability", 0)
        ga = lj_overall.get("graph_answerability", 0)
        imp = lj_overall.get("improvement", 0)
        wl = llm_judge_results.get("win_tie_loss", {})
        total = llm_judge_results.get("total_queries", 0)

        lines.append(f"| 指标 | 纯向量 RAG | GraphRAG | 提升 |")
        lines.append(f"|:---|---:|---:|---:|")
        lines.append(f"| Answerability | {va:.0%} | {ga:.0%} | **+{imp:.0%}** |")
        if total:
            gw = wl.get("graph_wins", 0)
            vw = wl.get("vec_wins", 0)
            ties = wl.get("ties", 0)
            lines.append(f"| GraphRAG 胜率 | — | {gw}/{total} ({gw/total:.0%}) | — |")
            lines.append(f"| 纯向量 胜率 | {vw}/{total} ({vw/total:.0%}) | — | — |")
        lines.append(f"")
    else:
        lines.append(f"> LLM-as-Judge 评测需要 API 可用性。运行 `python eval/llm_judge_eval.py` 获取结果。")
        lines.append(f"")

    # ── 评测三: 负样本 ──
    lines.append(f"## 四、评测三: 负样本测试 (诚实拒答)")
    lines.append(f"")
    lines.append(f"> 方法: 15 条知识库无法回答的查询, 测 Agent 是否诚实地说明局限性")
    lines.append(f"> 类型: out_of_domain / time_sensitive / hallucination_trap / vague / irrelevant")
    lines.append(f"")

    nr = negative_results
    if nr:
        total_n = nr.get("total", 0)
        honest_rate = nr.get("honest_rate", 0)
        hallu_rate = nr.get("hallucination_rate", 0)
        by_type = nr.get("by_type", {})

        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|:---|---|")
        lines.append(f"| 诚实拒答率 | **{honest_rate:.0%}** ({int(honest_rate*total_n)}/{total_n}) |")
        lines.append(f"| 幻觉风险率 | {hallu_rate:.0%} |")
        lines.append(f"")

        if by_type:
            lines.append(f"### 按类型")
            lines.append(f"")
            lines.append(f"| 类型 | 拒答率 |")
            lines.append(f"|:---|---:|")
            for t, d in by_type.items():
                lines.append(f"| {t} | {d.get('honest_rate', 0):.0%} |")
            lines.append(f"")
    else:
        lines.append(f"> 运行 `python eval/negative_test.py` 获取结果。")
        lines.append(f"")

    # ── 总结 ──
    lines.append(f"## 五、总结")
    lines.append(f"")
    lines.append(f"### GraphRAG 的优势场景")
    lines.append(f"")
    lines.append(f"1. **实体关联类查询**: 通过 KG 多跳扩散, 发现查询中未直接出现的关联实体 (如 MindSpore -> FusedAdamW -> 优化器 -> RFC)")
    lines.append(f"2. **报错溯源类查询**: KG 关联报错类型 -> 相关模块 -> 已知修复方案")
    lines.append(f"3. **跨领域对比**: 两个框架/模块的实体通过 KG 找到共现文档")
    lines.append(f"")
    lines.append(f"### 纯向量 RAG 仍适用的场景")
    lines.append(f"")
    lines.append(f"1. **简单事实查找**: 文档中有明确描述的知识点")
    lines.append(f"2. **高频词查询**: 数据集覆盖充分的常见问题")
    lines.append(f"")
    lines.append(f"### 评测局限")
    lines.append(f"")
    lines.append(f"- 关键词匹配 (Recall@5/Hit Rate) 是弱 proxy, 非人工相关性别定")
    lines.append(f"- 50 条查询仍不够统计显著 (建议 200+)")
    lines.append(f"- LLM-as-Judge 补充了更准确的 answerability 评估, 但 LLM 自身也有 bias")
    lines.append(f"- 负样本测试依赖检索结果 + Summarizer 的诚实度, 不完全等同 Agent 最终行为")
    lines.append(f"")
    lines.append(f"### 下一步改进")
    lines.append(f"")
    lines.append(f"- [ ] 查询集扩展到 200+ 条, 加入更多领域")
    lines.append(f"- [ ] 人工标注相关性 (或用 GPT-4 做高质量 pseudo-label)")
    lines.append(f"- [ ] 加入端到端 Agent 评测 (Success Rate / Turn Count / Token Efficiency)")
    lines.append(f"- [ ] A/B 测试不同分块策略和 KG 构建参数")
    lines.append(f"")

    # 输出
    report = "\n".join(lines)
    output_path = eval_dir / "eval_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"报告已生成: {output_path}")
    print(report)


if __name__ == "__main__":
    generate_report()
