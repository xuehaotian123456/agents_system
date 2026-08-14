"""
社区全局检索评测 — 验证社区摘要索引的宏观问答能力
==================================================
测什么: global_search(query) 返回的社区摘要能否覆盖查询主题。
方法: 轻量关键词命中 (社区摘要中出现期望主题词即命中), 与主评测同口径的弱标注 proxy。

运行: python eval/community_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

GLOBAL_QUERIES = [
    {"query": "MindSpore 性能优化", "expected_topics": ["性能", "优化", "算子", "内存"],
     "note": "应命中性能优化相关社区"},
    {"query": "PaddleNLP 文本处理", "expected_topics": ["NLP", "文本", "模型", "数据"],
     "note": "应命中 NLP 相关社区"},
    {"query": "PaddleOCR 图像识别", "expected_topics": ["检测", "识别", "OCR", "图像"],
     "note": "应命中 OCR/检测相关社区"},
    {"query": "MindSpore 分布式训练", "expected_topics": ["分布式", "训练", "并行"],
     "note": "应命中训练相关社区"},
    {"query": "PaddleDetection 目标检测", "expected_topics": ["检测", "COCO", "模型", "PaddleDetection"],
     "note": "应命中检测相关社区"},
    {"query": "数据增强和数据集构建", "expected_topics": ["数据", "dataset", "增强"],
     "note": "应命中数据相关社区"},
    {"query": "模型部署和推理优化", "expected_topics": ["部署", "推理", "优化"],
     "note": "应命中部署相关社区"},
    {"query": "算子开发和 CUDA 编程", "expected_topics": ["算子", "CUDA", "GPU", "编译"],
     "note": "应命中底层开发相关社区"},
]


def run_community_eval():
    from rag.knowledge_graph import get_kg
    kg = get_kg()

    if not kg.community_summaries:
        print("社区摘要未构建。先运行: python scripts/build_communities.py")
        return

    print(f"\n{'='*60}")
    print(f"  社区全局检索评测 ({len(GLOBAL_QUERIES)} 条宏观查询)")
    print(f"  社区摘要数: {len(kg.community_summaries)}")
    print(f"{'='*60}\n")

    hits_total = 0
    for qi, item in enumerate(GLOBAL_QUERIES):
        q = item["query"]
        topics = item["expected_topics"]
        result = kg.global_search(q, top_k=3)
        communities = result.get("communities", [])
        all_text = " ".join(
            c.get("summary", "") + " " + " ".join(c.get("top_entities", []))
            for c in communities
        )
        hits = sum(1 for t in topics if t.lower() in all_text.lower())
        hit = hits > 0
        hits_total += 1 if hit else 0

        print(f"  [{qi+1}/{len(GLOBAL_QUERIES)}] {'✓' if hit else '✗'} {q}")
        for c in communities[:2]:
            print(f"      → 社区{c['id']} ({c['size']}实体): {c['summary'][:55]}")
        if not hit:
            print(f"      (期望主题 {topics} 均未命中)")

    rate = hits_total / len(GLOBAL_QUERIES)
    print(f"\n{'='*60}")
    print(f"  主题覆盖率: {rate:.0%} ({hits_total}/{len(GLOBAL_QUERIES)})")
    print(f"  说明: 弱标注 proxy (关键词命中), 与主评测同口径局限")
    print(f"{'='*60}")
    return {"total": len(GLOBAL_QUERIES), "hit": hits_total, "rate": rate}


if __name__ == "__main__":
    run_community_eval()
