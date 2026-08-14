"""
全量评测数字一键复现
======================
clone 后数据为空, 本脚本按顺序完成:
  1. force_update 全量爬取 (约 10-20 分钟, 受 Gitee API 节流)
  2. 向量库 + KG 干净重建 (约 5 分钟)
  3. 关键词对比评测 run_eval (约 10 分钟)
  4. 负样本测试 negative_test (约 12 分钟)
  5. 汇总对比 (输出与仓库文档一致的数字)

运行: python scripts/reproduce_eval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)


def step(label: str):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")


def main():
    t0 = time.time()

    # ── 1. 全量爬取 ──
    step("1/5 全量爬取 (force_update, 10-20 分钟)")
    from services.scheduler import _incremental_crawl_job
    result = _incremental_crawl_job()
    print(f"  爬取结果: {result['message']}")

    # ── 2. 干净重建 ──
    step("2/5 向量库 + KG 干净重建")
    import os
    data_dir = Path(__file__).parent.parent / "data"
    chroma = data_dir / "chroma_db"
    md5 = data_dir / "md5.txt"
    if chroma.exists():
        import shutil
        shutil.rmtree(chroma)
    if md5.exists():
        os.remove(md5)

    from rag.vector_store import VectorStore
    from rag.knowledge_graph import KnowledgeGraph
    vs = VectorStore()
    vs.load_articles()
    assert len(vs.all_chunks) == vs.store._collection.count(), "重建后 1:1 校验失败"
    kg = KnowledgeGraph()
    kg.build(vs.all_chunks)
    print(f"  重建完成: {len(vs.all_chunks)} chunks, {kg.entity_count} 实体")

    # ── 3. 关键词评测 ──
    step("3/5 关键词对比评测 (run_eval, ~10 分钟)")
    from eval.run_eval import run_evaluation
    run_evaluation()

    # ── 4. 负样本测试 ──
    step("4/5 负样本测试 (negative_test, ~12 分钟)")
    from eval.negative_test import run_negative_test
    run_negative_test()

    # ── 5. 汇总 ──
    step("5/5 汇总")
    import json
    eval_path = Path(__file__).parent.parent / "eval" / "eval_results.json"
    neg_path = Path(__file__).parent.parent / "eval" / "negative_test_results.json"
    e = json.loads(eval_path.read_text(encoding="utf-8"))
    n = json.loads(neg_path.read_text(encoding="utf-8"))
    o = e["overall"]
    print(f"  关键词评测: 纯向量 {o['vec_recall@5']:.1%} → GraphRAG {o['graph_recall@5']:.1%} "
          f"(+{o['improvement_recall']:.1%})")
    print(f"  负样本: 拒答率 {n['honest_rate']:.0%} ({int(n['honest_rate']*n['total'])}/{n['total']})")
    print(f"\n  总耗时: {(time.time()-t0)/60:.0f} 分钟")
    print(f"  结果文件: eval/eval_results.json, eval/negative_test_results.json")


if __name__ == "__main__":
    main()
