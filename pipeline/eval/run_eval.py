"""
对比评测脚本 - 纯向量 RAG vs Graph 增强 RAG
===========================================
运行方式:
    cd E:/agent-system/pipeline
    python eval/run_eval.py

指标:
    - Recall@5: 检索到相关文档的比例
    - MRR: 第一个相关文档的倒数排名
    - 分类对比: 实体关联/报错溯源/事实查找/趋势查询 各自表现
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Windows GBK 编码修复
if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# 确保项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"


def load_queries(path: str = "") -> list[dict]:
    """加载测试查询"""
    if not path:
        path = str(Path(__file__).parent / "test_queries.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_vector_store(graph_enabled: bool = True):
    """
    初始化向量库。

    Args:
        graph_enabled: True = Graph 增强模式, False = 纯向量模式

    Returns:
        (retriever_fn, all_chunks, all_docs, mode, vs_instance)
        retriever_fn(query, k) -> list[Document]
    """
    from rag.vector_store import VectorStore

    vs = VectorStore()
    vs.load_articles()

    if graph_enabled and vs.hybrid_retriever:
        # HybridRetriever.search 只接受 query，包装为统一接口
        def graph_fn(q, k=5):
            return vs.hybrid_retriever.search(q)
        retriever_fn = graph_fn
        mode = "Graph-RAG (三路融合)"
    else:
        # 纯向量检索（仅 ChromaDB 相似度，不走混合检索）
        def vec_fn(q, k=5):
            return vs.store.similarity_search(q, k=k)
        retriever_fn = vec_fn
        mode = "纯向量 RAG (仅向量)"

    return retriever_fn, vs.all_chunks, vs.all_docs, mode, vs


def compute_recall_at_k(
    query: str,
    retriever_fn,  # callable: (query, k) -> list[Document]
    all_chunks: list[str],
    all_docs: list,
    expected_entities: list[str],
    k: int = 5,
) -> dict:
    """
    计算 Recall@k 和 MRR。

    简化版：使用 expected_entities 中的关键词在检索文档中出现的比例
    作为 recall 的代理指标（无需人工标注每个文档相关性）。
    """
    # 检索
    docs = retriever_fn(query, k) if callable(retriever_fn) else []

    # 确保返回的是 Document 列表
    if docs is None:
        docs = []

    retrieved_texts = [d.page_content.lower() if hasattr(d, 'page_content') else str(d).lower() for d in docs[:k]]

    # 计算命中
    hits = 0
    first_hit_rank = float('inf')
    for entity in expected_entities:
        entity_lower = entity.lower()
        for rank, text in enumerate(retrieved_texts):
            if entity_lower in text:
                hits += 1
                first_hit_rank = min(first_hit_rank, rank + 1)
                break

    total = len(expected_entities)
    recall = hits / total if total > 0 else 0
    mrr = 1.0 / first_hit_rank if first_hit_rank != float('inf') else 0
    # Hit Rate: 至少命中 1 个期望实体即为成功（比 Recall 更宽松，适合数据稀疏场景）
    hit_rate = 1.0 if hits > 0 else 0.0

    return {
        "recall@k": round(recall, 3),
        "hit_rate": round(hit_rate, 3),   # ★ 新增
        "mrr": round(mrr, 3),
        "hits": hits,
        "total_expected": total,
    }


def run_evaluation():
    """运行完整对比评测"""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Agent 双引擎系统 — RAG 对比评测{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    queries = load_queries()
    print(f"测试查询: {len(queries)} 条\n")

    # ── 初始化 ──
    print(f"{CYAN}[...] 初始化向量库 + KG...{RESET}")
    t0 = time.time()

    # Graph 增强模式
    graph_fn, chunks, docs, graph_mode, vs_graph = init_vector_store(graph_enabled=True)
    t1 = time.time()

    # 纯向量模式
    vec_fn, _, _, vec_mode, vs_vec = init_vector_store(graph_enabled=False)

    print(f"   Graph-RAG 模式: {graph_mode} ({len(chunks)} chunks)")
    print(f"   纯向量模式: {vec_mode}")
    print(f"   初始化耗时: {(t1 - t0):.1f}s\n")

    # ── 统计语料库词频（用于区分 difficulty）──
    all_text_lower = " ".join(chunks).lower()

    # ── 逐类对比 ──
    categories = {
        "entity_association": "实体关联类（测 Graph-RAG 优势）",
        "fact_lookup": "事实查找类（测基础 RAG）",
        "error_tracing": "报错溯源类（测图谱关联能力）",
        "trending": "时效查询类（测爬虫链路）",
        "compare": "对比分析类（测多工具编排）",
        "multi_hop": "多跳推理类（测多步关联能力）",
    }

    difficulty_levels = {}

    results = {cat: {"vec": [], "graph": [], "queries": []} for cat in categories}

    entity_freq_warnings = []

    for q in queries:
        cat = q.get("category", "unknown")
        question = q.get("question", "")
        entities = q.get("expected_entities", [])
        difficulty = q.get("difficulty", "medium")

        # 检查 expected_entities 质量：太常见或不存在都记录
        for e in entities:
            freq = all_text_lower.count(e.lower())
            if freq == 0:
                entity_freq_warnings.append(f"  [WARN] q{q['id']}: '{e}' 在语料库中不存在")
            elif freq > 100:
                entity_freq_warnings.append(f"  [INFO] q{q['id']}: '{e}' 出现 {freq} 次 (区分度低)")

        # 纯向量检索
        vec_result = compute_recall_at_k(
            question, vec_fn, chunks, docs, entities, k=5)

        # Graph 增强检索
        graph_result = compute_recall_at_k(
            question, graph_fn, chunks, docs, entities, k=5)

        if cat in results:
            results[cat]["vec"].append(vec_result)
            results[cat]["graph"].append(graph_result)
            results[cat]["queries"].append(q["id"])

        difficulty_levels.setdefault(difficulty, {"vec": [], "graph": []})
        difficulty_levels[difficulty]["vec"].append(vec_result)
        difficulty_levels[difficulty]["graph"].append(graph_result)

    # ── Entity 质量警告 ──
    if entity_freq_warnings:
        print(f"{YELLOW}  Entity 质量报告:{RESET}")
        for w in entity_freq_warnings[:8]:
            print(w)
        if len(entity_freq_warnings) > 8:
            print(f"  ... 共 {len(entity_freq_warnings)} 条")
        print()

    # ── 输出报告 ──
    print(f"{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  评测结果汇总 (按查询类别){RESET}")
    print(f"{BOLD}{'─'*60}{RESET}\n")

    print(f"{'类别':<30} {'条数':>4} {'纯向量R':>8} {'GraphR':>8} {'提升':>8}")
    print(f"{'─'*30} {'─'*4} {'─'*8} {'─'*8} {'─'*8}")

    all_vec_recall = []
    all_graph_recall = []
    all_vec_hit = []
    all_graph_hit = []
    all_vec_mrr = []
    all_graph_mrr = []

    for cat_key, cat_name in categories.items():
        cat_data = results[cat_key]
        if not cat_data["vec"]:
            continue

        n = len(cat_data["vec"])
        vec_avg_r = sum(r["recall@k"] for r in cat_data["vec"]) / n
        graph_avg_r = sum(r["recall@k"] for r in cat_data["graph"]) / n
        vec_avg_h = sum(r.get("hit_rate", 0) for r in cat_data["vec"]) / n
        graph_avg_h = sum(r.get("hit_rate", 0) for r in cat_data["graph"]) / n
        vec_avg_m = sum(r["mrr"] for r in cat_data["vec"]) / n
        graph_avg_m = sum(r["mrr"] for r in cat_data["graph"]) / n

        improvement_r = graph_avg_r - vec_avg_r

        all_vec_recall.extend([r["recall@k"] for r in cat_data["vec"]])
        all_graph_recall.extend([r["recall@k"] for r in cat_data["graph"]])
        all_vec_hit.extend([r.get("hit_rate", 0) for r in cat_data["vec"]])
        all_graph_hit.extend([r.get("hit_rate", 0) for r in cat_data["graph"]])
        all_vec_mrr.extend([r["mrr"] for r in cat_data["vec"]])
        all_graph_mrr.extend([r["mrr"] for r in cat_data["graph"]])

        arrow = f"{GREEN}+{improvement_r:.0%}{RESET}" if improvement_r > 0 else f"{RED}{improvement_r:.0%}{RESET}"
        print(f"{cat_name:<30} {n:>4} {vec_avg_r:>7.0%} {graph_avg_r:>7.0%} {arrow:>8}")

    # ── 按难度 ──
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  按难度分级{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}\n")
    print(f"{'难度':<10} {'条数':>4} {'纯向量R':>8} {'GraphR':>8} {'提升':>8}")
    print(f"{'─'*10} {'─'*4} {'─'*8} {'─'*8} {'─'*8}")

    for diff in ["easy", "medium", "hard"]:
        data = difficulty_levels.get(diff, {"vec": [], "graph": []})
        if not data["vec"]:
            continue
        n = len(data["vec"])
        vec_r = sum(r["recall@k"] for r in data["vec"]) / n
        graph_r = sum(r["recall@k"] for r in data["graph"]) / n
        imp = graph_r - vec_r
        arrow = f"{GREEN}+{imp:.0%}{RESET}" if imp > 0 else f"{RED}{imp:.0%}{RESET}"
        print(f"{diff:<10} {n:>4} {vec_r:>7.0%} {graph_r:>7.0%} {arrow:>8}")

    # ── 总体 ──
    total_vec_r = sum(all_vec_recall) / max(len(all_vec_recall), 1)
    total_graph_r = sum(all_graph_recall) / max(len(all_graph_recall), 1)
    total_vec_h = sum(all_vec_hit) / max(len(all_vec_hit), 1)
    total_graph_h = sum(all_graph_hit) / max(len(all_graph_hit), 1)
    total_vec_m = sum(all_vec_mrr) / max(len(all_vec_mrr), 1)
    total_graph_m = sum(all_graph_mrr) / max(len(all_graph_mrr), 1)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  总体汇总 ({len(queries)} 条查询){RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    print(f"  Recall@5:   {total_vec_r:.0%}  ->  {GREEN}{total_graph_r:.0%}{RESET}  ({GREEN}+{total_graph_r - total_vec_r:.0%}{RESET})")
    print(f"  Hit Rate:   {total_vec_h:.0%}  ->  {GREEN}{total_graph_h:.0%}{RESET}  ({GREEN}+{total_graph_h - total_vec_h:.0%}{RESET})")
    print(f"  MRR:        {total_vec_m:.3f}  ->  {GREEN}{total_graph_m:.3f}{RESET}  (+{total_graph_m - total_vec_m:.3f})")

    # 查询成功率
    total_queries = len(queries)
    successful = sum(
        1 for cat_data in results.values()
        for r in cat_data["graph"]
        if r["recall@k"] > 0 or r["hits"] > 0
    )
    degradation_rate = successful / max(total_queries, 1)
    print(f"  成功率:     {degradation_rate:.0%} ({successful}/{total_queries})")

    # 评测注意事项
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{YELLOW}  评测说明:{RESET}")
    print(f"  - 数据规模: {len(chunks)} chunks, {len(queries)} 条查询")
    print(f"  - 指标说明: Recall@5 = 期望实体在top5文档中的命中比例")
    print(f"  - Hit Rate = 至少命中1个期望实体 (比Recall更宽松)")
    print(f"  - MRR = 第一个相关文档排名的倒数均值")
    print(f"{YELLOW}  局限性:{RESET}")
    print(f"  - expected_entities 关键词匹配是弱标注，并非人工相关性判断")
    print(f"  - 50 条查询仍不够统计显著，建议 200+ 条")
    print(f"  - 建议引入 LLM-as-judge 做更准确的相关性评估")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # 保存结果
    output_path = Path(__file__).parent / "eval_results.json"
    summary = {
        "total_queries": total_queries,
        "total_chunks": len(chunks),
        "evaluation_note": "expected_entities关键词匹配是弱标注proxy，非人工相关性别定。建议引入LLM-as-judge补充。",
        "overall": {
            "vec_recall@5": round(total_vec_r, 3),
            "graph_recall@5": round(total_graph_r, 3),
            "vec_hit_rate": round(total_vec_h, 3),
            "graph_hit_rate": round(total_graph_h, 3),
            "vec_mrr": round(total_vec_m, 3),
            "graph_mrr": round(total_graph_m, 3),
            "improvement_recall": round(total_graph_r - total_vec_r, 3),
            "improvement_hit_rate": round(total_graph_h - total_vec_h, 3),
            "degradation_success_rate": round(degradation_rate, 3),
        },
        "by_category": {
            cat: {
                "name": cat_name,
                "count": len(data["vec"]),
                "vec_recall@5": round(sum(r["recall@k"] for r in data["vec"]) / max(len(data["vec"]), 1), 3),
                "graph_recall@5": round(sum(r["recall@k"] for r in data["graph"]) / max(len(data["graph"]), 1), 3),
            }
            for cat, data in results.items() if data["vec"]
        },
        "by_difficulty": {
            diff: {
                "count": len(data["vec"]),
                "vec_recall@5": round(sum(r["recall@k"] for r in data["vec"]) / max(len(data["vec"]), 1), 3),
                "graph_recall@5": round(sum(r["recall@k"] for r in data["graph"]) / max(len(data["graph"]), 1), 3),
            }
            for diff, data in difficulty_levels.items() if data["vec"]
        },
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    run_evaluation()
