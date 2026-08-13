"""
社区检测 + 社区摘要构建 (微软 GraphRAG 机制)
============================================
流程:
  1. 加载现有 KG (或重建)
  2. 强边子图 (P75 权重) 上做标签传播社区检测
  3. 每个社区 (>=5 实体) LLM 批量生成摘要
  4. 持久化到 kg_state.json (communities + community_summaries)

成本: 约 30-80 个社区, 每批 10 个 → 3-8 次 LLM 调用
运行: python scripts/build_communities.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)


def main():
    t0 = time.time()

    from rag.vector_store import VectorStore
    from rag.knowledge_graph import KnowledgeGraph

    # 1. 加载向量库 (获取 chunks)
    print("[1/3] 加载向量库...")
    vs = VectorStore()
    vs.load_articles()
    chunks = vs.all_chunks
    print(f"  chunks: {len(chunks)}")

    # 2. KG 构建/加载 + 社区检测
    print("[2/3] 社区检测 (强边子图标签传播)...")
    kg = KnowledgeGraph()
    kg.build(chunks)
    kg.detect_communities()

    # 3. 社区摘要
    print("[3/3] 社区摘要 (LLM 批量)...")
    summaries = kg.build_community_summaries(chunks=chunks)

    print(f"\n完成: {len(summaries)} 个社区摘要, 耗时 {time.time()-t0:.0f}s")
    print("\n示例社区:")
    for cid, info in list(summaries.items())[:5]:
        print(f"  社区{cid} ({info['size']} 实体): {info['summary'][:60]}")
        print(f"    代表实体: {', '.join(info['top_entities'][:6])}")

    # 4. 全局检索演示
    print("\n全局检索演示:")
    for q in ["MindSpore 性能优化", "PaddleNLP 文本处理"]:
        result = kg.global_search(q, top_k=2)
        print(f"\n  查询: {q}")
        for c in result["communities"]:
            print(f"    社区{c['id']} ({c['size']}实体, 命中{c['match_count']}): {c['summary'][:60]}")


if __name__ == "__main__":
    main()
