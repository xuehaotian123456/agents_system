"""
端到端 Agent 演示 — GraphRAG 增强技术知识助手
============================================
运行方式:
    cd E:/agent-system/pipeline
    python eval/e2e_demo.py

输出:
    1. Agent 决策链 (规划→检索→反思→回答)
    2. GraphRAG 多跳推理链路
    3. 检索来源 + 可信度
    4. 对比: 纯向量 RAG vs GraphRAG

面试时直接跑这个脚本，输出就是最好的项目展示。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows 编码修复
if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

SEP = "=" * 70
SEP2 = "-" * 70


def print_header():
    print(f"""
{SEP}
  Agent 双引擎系统 — GraphRAG 增强技术知识助手
  端到端演示 | {datetime.now().strftime('%Y-%m-%d %H:%M')}
{SEP}
""")


def demo_1_rag_search():
    """演示1: 基础 GraphRAG 检索 — 三路融合效果"""
    print(f"\n{SEP2}")
    print("  演示 1: GraphRAG 三路混合检索")
    print(f"{SEP2}\n")

    from rag.vector_store import VectorStore
    from rag.knowledge_graph import get_kg

    vs = VectorStore()
    vs.load_articles()

    kg = get_kg()

    queries = [
        "MindSpore 的 FusedAdamW 优化器怎么用",
        "PaddleNLP Trainer 支持哪些训练参数",
        "MindSpore BatchNormGradExt dtype 推导报错",
    ]

    for q in queries:
        print(f"  查询: {q}")

        # ── GraphRAG 多跳扩散 ──
        mh = kg.multi_hop_expand(q, max_hops=2, top_per_hop=3)
        chain = mh.get("chain_text", "(无)") if mh else "(KG未构建)"
        print(f"  推理链:\n    {chain.replace(chr(10), chr(10)+'    ')}")

        # ── 检索 ──
        docs = vs.search(q)
        print(f"  检索结果 (top 3):")
        for i, doc in enumerate(docs):
            src = doc.metadata.get("retrieval_source", "?")
            cred = doc.metadata.get("credibility", "?")
            title = doc.metadata.get("title", doc.metadata.get("source", "?"))
            rerank = doc.metadata.get("reranker_score", "?")
            print(
                f"    {i+1}. [{src}] {title[:60]} "
                f"(可信度={cred}, rerank={rerank})"
            )
        print()


def demo_2_kg_multi_hop():
    """演示2: KG 多跳推理 — 实体关联链路"""
    print(f"\n{SEP2}")
    print("  演示 2: 知识图谱多跳推理")
    print(f"{SEP2}\n")

    from rag.knowledge_graph import get_kg
    kg = get_kg()

    # 实体关联查询
    pairs = [
        ("MindSpore", "FusedAdamW"),
        ("PaddleNLP", "Taskflow"),
        ("HyperOffload", "MemorySaver"),
    ]

    for a, b in pairs:
        path = kg.find_path(a, b, max_hops=3)
        if path:
            chain = " -> ".join(path["path"])
            print(f"  {a} -> {b}: {chain} ({path['hops']} 跳)")
            for r in path["relations"]:
                print(f"    {r['from']} --({r['co_occur']})--> {r['to']}")
        else:
            print(f"  {a} -> {b}: 未找到路径 (实体不在同一子图中)")
        print()


def demo_3_agent_pipeline():
    """演示3: Agent 完整决策链"""
    print(f"\n{SEP2}")
    print("  演示 3: Agent 动态决策链")
    print(f"{SEP2}\n")

    import asyncio
    from agent.state import initial_state
    from agent.nodes import planner_node, retriever_node, reflector_node, summarizer_node
    from services.crawl_state import get_crawl_state

    state_obj = get_crawl_state()

    queries = [
        "MindSpore 社区有哪些开源实习任务可以参与",
        "PaddleNLP 量化训练时报 TypeError __init__ 怎么解决",
    ]

    for q in queries:
        print(f"  用户查询: {q}\n")

        state = initial_state(q, session_id=f"demo_{hash(q) % 10000}")

        # ── Step 1: Planner ──
        t0 = time.time()
        state = planner_node(state)
        plan = state.get("plan", {})
        print(f"  [*] Planner ({time.time()-t0:.1f}s)")
        print(f"      意图: {plan.get('intent', '?')[:80]}")
        print(f"      工具: {plan.get('suggested_tools', [])}")
        print(f"      实体: {plan.get('target_entity', '(无)')}")

        # ── Step 2: Retriever ──
        t0 = time.time()
        state = retriever_node(state)
        context = state.get("context", "")
        n_tools = len(state.get("tool_calls", []))
        degraded = state.get("degradation_triggered", False)
        errors = state.get("errors", [])
        print(f"  [*] Retriever ({time.time()-t0:.1f}s)")
        print(f"      工具调用: {n_tools} 次" + (" (全部失败, 触发降级)" if degraded else ""))
        if errors:
            print(f"      异常: {len(errors)} 次")
        print(f"      上下文长度: {len(context)} 字符")

        # ── Step 3: Reflector ──
        if not degraded:
            t0 = time.time()
            state = reflector_node(state)
            confidence = state.get("confidence", 0)
            retries = state.get("retry_count", 0)
            print(f"  [*] Reflector ({time.time()-t0:.1f}s)")
            print(f"      置信度: {confidence:.0%}")
            print(f"      重试: {retries}/{state.get('max_retries', 2)}")
        else:
            print(f"  [*] Reflector: 跳过 (降级模式)")

        # ── Step 4: Summarizer ──
        t0 = time.time()
        state = summarizer_node(state)
        answer = state.get("answer", "")
        print(f"  [*] Summarizer ({time.time()-t0:.1f}s)")
        print(f"      回答长度: {len(answer)} 字符")
        print(f"\n  最终回答:\n{answer[:500]}{'...' if len(answer) > 500 else ''}")
        print()


def demo_4_compare():
    """演示4: 纯向量 vs GraphRAG 对比"""
    print(f"\n{SEP2}")
    print("  演示 4: 纯向量 RAG vs GraphRAG 效果对比")
    print(f"{SEP2}\n")

    # 读取之前的评测结果
    result_path = Path(__file__).parent / "eval_results.json"
    if result_path.exists():
        results = json.loads(result_path.read_text(encoding="utf-8"))
        overall = results.get("overall", {})

        print("  ┌──────────────────────┬──────────┬──────────┬────────┐")
        print("  │ 指标                 │ 纯向量   │ GraphRAG │ 提升   │")
        print("  ├──────────────────────┼──────────┼──────────┼────────┤")
        vec_r = overall.get("vec_recall@5", 0)
        graph_r = overall.get("graph_recall@5", 0)
        vec_h = overall.get("vec_hit_rate", 0)
        graph_h = overall.get("graph_hit_rate", 0)
        imp_r = graph_r - vec_r
        imp_h = graph_h - vec_h
        print(f"  │ Recall@5             │ {vec_r:>6.0%}   │ {graph_r:>6.0%}   │ +{imp_r:.0%}   │")
        print(f"  │ Hit Rate             │ {vec_h:>6.0%}   │ {graph_h:>6.0%}   │ +{imp_h:.0%}   │")
        print("  └──────────────────────┴──────────┴──────────┴────────┘")
        print()

        # 按类别
        if "by_category" in results:
            print("  按查询类别:")
            print("  ┌────────────────────────┬──────────┬──────────┬────────┐")
            print("  │ 类别                   │ 纯向量   │ GraphRAG │ 提升   │")
            print("  ├────────────────────────┼──────────┼──────────┼────────┤")
            for cat, data in results["by_category"].items():
                name = data.get("name", cat)
                v = data.get("vec_recall@5", 0)
                g = data.get("graph_recall@5", 0)
                imp = g - v
                print(f"  │ {name:<22} │ {v:>6.0%}   │ {g:>6.0%}   │ +{imp:.0%}   │")
            print("  └────────────────────────┴──────────┴──────────┴────────┘")

    print()
    print("  GraphRAG 优势场景:")
    print("  - 实体关联类: 通过多跳扩散发现 MindSpore->FusedAdamW->优化器 等关联链")
    print("  - 报错溯源类: 从 TypeError->__init__->Trainer 追踪问题根因链")
    print("  - 知识图谱提供向量检索无法发现的隐含关联")


def demo_5_system_stats():
    """演示5: 系统数据统计"""
    print(f"\n{SEP2}")
    print("  演示 5: 系统规模与数据治理")
    print(f"{SEP2}\n")

    from rag.vector_store import VectorStore
    from rag.knowledge_graph import get_kg
    from services.crawl_state import get_crawl_state

    vs = VectorStore()
    vs.load_articles()

    kg = get_kg()
    cs = get_crawl_state()

    # 数据规模
    data_dir = Path(__file__).parent.parent / "data" / "articles"
    n_articles = len([f for f in os.listdir(data_dir) if f.endswith(".md")])

    # 来源分布
    from collections import Counter
    source_dist = Counter()
    for f in os.listdir(data_dir):
        if f.endswith(".md"):
            p = f.split("_")[0]
            source_dist[p] += 1

    print(f"  数据规模:")
    print(f"    文章总数:      {n_articles}")
    print(f"    Chunk 数:      {len(vs.all_chunks)}")
    print(f"    KG 实体数:     {kg.entity_count}")
    print(f"    ChromaDB:      {vs.store._collection.count()} embeddings")
    print(f"    URL 去重记录:  {cs.get_article_count()}")

    print(f"\n  数据来源:")
    for src, cnt in source_dist.most_common(8):
        label = {
            "gitee": "Gitee Issues+Docs (主数据源)",
            "sched": "定时爬取 (掘金+博客园)",
            "digest": "日报摘要",
            "batch": "批量导入",
            "cnblogs": "博客园 RSS",
            "juejin": "掘金热榜",
            "ai": "AI 参考文档",
        }.get(src, src)
        print(f"    {label:<35} {cnt} 篇")

    print(f"\n  数据质量治理:")
    print(f"    可信度分级:     官方文档(1.0) > Issue标签(0.85) > 社区(0.5) > RSS(0.3)")
    print(f"    冲突消解:       高权威覆盖低权威, 低权威降级标记")
    print(f"    脏数据过滤:     广告/纯外链/过短内容自动过滤")

    print(f"\n  爬取时效:")
    last_crawl = cs.get_last_crawl_time()
    if last_crawl:
        print(f"    最后全量爬取:   {last_crawl[:19]}")
    for src, ts in cs._data.get("last_crawl", {}).items():
        print(f"    {src:<15} {ts[:19]}")


def main():
    print_header()

    # 初始化
    print("  初始化向量库 + 知识图谱...")
    from rag.vector_store import VectorStore
    from rag.knowledge_graph import get_kg

    t0 = time.time()
    vs = VectorStore()
    vs.load_articles()
    kg = get_kg()
    print(f"  就绪: {len(vs.all_chunks)} chunks, {kg.entity_count} 实体 ({time.time()-t0:.1f}s)")

    # 按顺序执行所有演示
    demo_1_rag_search()
    demo_2_kg_multi_hop()
    demo_3_agent_pipeline()
    demo_4_compare()
    demo_5_system_stats()

    print(f"\n{SEP}")
    print(f"  演示完成。以上输出展示了 Agent 系统的全部核心能力:")
    print(f"  1. GraphRAG 三路混合检索 (向量 + BM25 + 知识图谱)")
    print(f"  2. KG 多跳推理 (实体关联链路发现)")
    print(f"  3. Agent 动态决策 (规划→检索→反思→回答)")
    print(f"  4. 纯向量 vs GraphRAG 效果对比 (基于 20 条查询的评测数据)")
    print(f"  5. 数据规模与质量治理")
    print(f"{SEP}")


if __name__ == "__main__":
    main()
