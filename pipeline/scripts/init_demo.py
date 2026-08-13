"""
一键初始化脚本 — clone 后 30 秒让系统可用
==========================================
运行: python scripts/init_demo.py

流程:
  1. 复制种子文章 → data/articles/
  2. 重建 ChromaDB 向量库 (force_rebuild)
  3. 重建知识图谱
  4. 验证检索可用

全量数据: 运行后执行 force_update 拉取全部 6 源数据
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows GBK 修复
if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

PROJECT_ROOT = Path(__file__).parent.parent
SEED_DIR = PROJECT_ROOT / "data" / "seed_articles"
ARTICLES_DIR = PROJECT_ROOT / "data" / "articles"


def step(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


def init_seed_articles():
    """复制种子文章"""
    if not SEED_DIR.exists() or not any(SEED_DIR.iterdir()):
        print("  [跳过] 无种子文章")
        return 0

    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in SEED_DIR.glob("*.md"):
        target = ARTICLES_DIR / f.name
        if not target.exists():
            shutil.copy2(f, target)
            copied += 1
    print(f"  [OK] 种子文章: {copied} 篇新复制 (已有 {len(list(ARTICLES_DIR.glob('*.md')))} 篇)")
    return copied


def init_vector_store():
    """重建向量库"""
    from rag.vector_store import VectorStore
    t0 = time.time()
    vs = VectorStore()
    vs.load_articles(force_rebuild=True)
    elapsed = time.time() - t0
    print(f"  [OK] 向量库: {vs.store._collection.count()} embeddings, {len(vs.all_chunks)} chunks ({elapsed:.1f}s)")
    return vs


def init_knowledge_graph(vs):
    """重建 KG"""
    from rag.knowledge_graph import get_kg
    t0 = time.time()
    kg = get_kg()
    kg.build(vs.all_chunks)
    elapsed = time.time() - t0
    print(f"  [OK] 知识图谱: {kg.entity_count} 实体, {len(kg.co_occurrence)} 共现边 ({elapsed:.1f}s)")
    return kg


def verify():
    """验证检索可用"""
    from rag.vector_store import VectorStore
    from rag.knowledge_graph import get_kg

    vs = VectorStore()
    vs.load_articles()
    kg = get_kg()

    print()
    print("  验证检索...")
    test_query = "MindSpore 优化器"
    docs = vs.search(test_query)
    if docs:
        print(f"  [OK] 检索正常: '{test_query}' → {len(docs)} 篇")
        for d in docs[:3]:
            src = d.metadata.get("retrieval_source", "?")
            title = d.metadata.get("source", "?")[:50]
            print(f"      [{src}] {title}")
    else:
        print("  [WARN] 检索返回空!")

    # KG 多跳
    mh = kg.multi_hop_expand(test_query, max_hops=2)
    if mh and mh.get("hops"):
        print(f"  [OK] KG 多跳: {mh['total_expanded']} 实体扩散")
    else:
        print("  [WARN] KG 多跳无结果!")


def main():
    print(f"\n{'='*60}")
    print(f"  Agent 双引擎系统 — 一键初始化")
    print(f"{'='*60}")

    t0 = time.time()

    step("1/4 种子数据")
    init_seed_articles()

    step("2/4 向量库重建")
    vs = init_vector_store()

    step("3/4 知识图谱重建")
    kg = init_knowledge_graph(vs)

    step("4/4 验证")
    verify()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  初始化完成! 耗时 {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"")
    print(f"  下一步:")
    print(f"    1. 启动 Pipeline:   uvicorn a2a_server:app --port 8010")
    print(f"    2. 启动 Interaction: uvicorn server.app:app --port 8020 (在 interaction/ 目录)")
    print(f"    3. 看效果:          python eval/e2e_demo.py")
    print(f"    4. 全量数据:        curl -X POST http://localhost:8010/tools/force_update -d '{{}}'")
    print(f"    (全量拉取 Gitee+掘金+博客园约需 10-15 分钟, 受 API 限流影响)")


if __name__ == "__main__":
    main()
