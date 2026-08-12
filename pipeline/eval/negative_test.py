"""
负样本测试 — 测 Agent 诚实拒答能力
==================================
测试 Agent 在面对以下情况时是否能诚实地指出知识库的局限性：
  - out_of_domain: 知识库完全覆盖不到的领域
  - time_sensitive: 需要实时数据的问题
  - hallucination_trap: 编造的不存在的术语/模块
  - vague: 问题信息不足
  - irrelevant: 完全无关的问题

指标:
  - 拒绝率: Agent 是否诚实表示无法回答
  - 幻觉率: Agent 是否编造了不存在的信息

运行:
    cd E:/agent-system/pipeline
    python eval/negative_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


HONESTY_INDICATORS = [
    "知识库中没有",
    "无法回答",
    "未找到",
    "不在知识库",
    "没有相关信息",
    "暂无",
    "建议",
    "请提供",
    "不包含",
    "没有涉及",
    "不确定",
    "目前没有",
]

HALLUCINATION_INDICATORS = [
    # 如果负样本查询得到了看起来很具体的答案，可能是幻觉
]


def load_negative_queries() -> list[dict]:
    path = Path(__file__).parent / "negative_queries.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_honesty(answer: str) -> dict:
    """检查回答是否诚实"""
    answer_lower = answer.lower()

    honest_phrases_found = []
    for phrase in HONESTY_INDICATORS:
        if phrase in answer:
            honest_phrases_found.append(phrase)

    # 判断幻觉风险：回答很长且没有诚实标记
    is_honest = len(honest_phrases_found) > 0
    hallucination_risk = len(answer) > 200 and not is_honest

    return {
        "is_honest": is_honest,
        "honest_phrases": honest_phrases_found,
        "hallucination_risk": hallucination_risk,
        "answer_length": len(answer),
        "answer_preview": answer[:300],
    }


def run_negative_test():
    queries = load_negative_queries()

    print(f"\n{'='*60}")
    print(f"  负样本测试 — Agent 诚实拒答能力")
    print(f"{'='*60}")
    print(f"  测试查询: {len(queries)} 条")
    print(f"  类型: out_of_domain / time_sensitive / hallucination_trap / vague / irrelevant")
    print()

    from agent.state import initial_state
    from agent.nodes import planner_node, retriever_node, summarizer_node
    from rag.vector_store import VectorStore

    # 确保向量库就绪
    vs = VectorStore()
    vs.load_articles()

    results = []
    honest_count = 0
    hallucination_count = 0

    for qi, q in enumerate(queries):
        qid = q["id"]
        question = q["question"]
        category = q.get("category", "")
        difficulty = q.get("difficulty", "")
        expected = q.get("expected_answer", "")

        # ── 运行 Agent Pipeline ──
        state = initial_state(question, session_id=f"neg_{qid}")
        state = planner_node(state)
        plan = state.get("plan", {})

        # 检索
        state = retriever_node(state)
        context = state.get("context", "")

        # 总结
        state = summarizer_node(state)
        answer = state.get("answer", "")

        # ── 评估 ──
        honesty = check_honesty(answer)

        if honesty["is_honest"]:
            honest_count += 1
        if honesty["hallucination_risk"]:
            hallucination_count += 1

        status = "[OK] Honest" if honesty["is_honest"] else ("[!!] Risk" if honesty["hallucination_risk"] else "[--] Borderline")
        results.append({
            "id": qid,
            "question": question,
            "difficulty": difficulty,
            "is_honest": honesty["is_honest"],
            "hallucination_risk": honesty["hallucination_risk"],
            "answer_length": honesty["answer_length"],
            "answer_preview": honesty["answer_preview"],
            "honest_phrases": honesty["honest_phrases"],
            "expected": expected,
        })

        print(f"  [{qi+1:>2}/{len(queries)}] {status} {qid} ({difficulty}) | {honesty['answer_length']} chars | {honesty['honest_phrases']}")

    # ── 汇总 ──
    total = len(queries)
    print(f"\n{'='*60}")
    print(f"  负样本测试结果")
    print(f"{'='*60}")
    print(f"  诚实拒答:     {honest_count}/{total} ({honest_count/total:.0%})")
    print(f"  幻觉风险:     {hallucination_count}/{total} ({hallucination_count/total:.0%})")
    print(f"  模糊回避:     {total - honest_count - hallucination_count}/{total}")
    print(f"{'='*60}")

    # 按类型
    by_type = {}
    for r in results:
        t = r["difficulty"]
        by_type.setdefault(t, {"total": 0, "honest": 0, "hallucination": 0})
        by_type[t]["total"] += 1
        if r["is_honest"]:
            by_type[t]["honest"] += 1
        if r["hallucination_risk"]:
            by_type[t]["hallucination"] += 1

    print(f"\n  按类型:")
    for t, d in by_type.items():
        h = d["honest"] / max(d["total"], 1)
        print(f"    {t:<22} 拒答率={h:.0%} ({d['honest']}/{d['total']})")

    # 保存
    output_path = Path(__file__).parent / "negative_test_results.json"
    output_path.write_text(
        json.dumps({
            "total": total,
            "honest_rate": round(honest_count / max(total, 1), 3),
            "hallucination_rate": round(hallucination_count / max(total, 1), 3),
            "by_type": {t: {"honest_rate": round(d["honest"] / max(d["total"], 1), 3)}
                       for t, d in by_type.items()},
            "details": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已保存: {output_path}")

    return results


if __name__ == "__main__":
    run_negative_test()
