"""
LLM-as-Judge 评测 — 替代关键词匹配
==================================
用 LLM 判断检索到的文档能否回答用户问题，而非简单的关键词命中。
这是衡量 RAG 质量的黄金标准。

指标:
  - Answerability@5: top-5 文档能否回答用户问题 (0-1)
  - NDCG@5: 考虑排序的检索质量
  - Win Rate: GraphRAG > 纯向量 的比例

运行:
    cd E:/agent-system/pipeline
    python eval/llm_judge_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows 编码: 运行时设置 PYTHONIOENCODING=utf-8 或 chcp 65001


JUDGE_PROMPT = """你是一个检索质量评估专家。给你一个用户问题和一系列检索到的文档，请判断这些文档能否回答用户问题。

逐文档评分标准 (scores, 每篇 0-3):
  0 = 完全无关: 文档与问题毫无关系
  1 = 弱相关: 提到了相关术语但没有实际内容
  2 = 部分可答: 文档包含部分答案，但信息不完整
  3 = 完全可答: 文档包含足够信息来完整回答问题

整体判定标准 (overall_judgment, 文档集合作为整体能否支撑回答):
  fully_answerable = 集合中信息足以完整回答
  partially_answerable = 集合能回答一部分, 需补充
  not_answerable = 集合无法支撑任何有效回答

用户问题: {question}

检索到的文档:
{documents}

请以JSON格式输出你的判断，不要输出其他内容:
{{"scores": [3, 1, 0, 2, 1], "best_doc_index": 0, "overall_judgment": "partially_answerable", "reasoning": "简短理由(30字以内)"}}"""


def judge_answerability(question: str, docs: list[str], llm_model=None) -> dict:
    """
    用 LLM 判断文档能否回答问题。

    Returns:
        {scores: [0-3], best_doc: int, overall: str, reasoning: str}
    """
    # 如果没有 LLM 适配器，用模型工厂
    if llm_model is None:
        from model.factory import robust_llm_call
    else:
        robust_llm_call = llm_model

    # 格式化文档
    # ★ 历史 bug: 截断到 300 字符, judge 只能看到每篇的一小部分
    #   (chunk=800 时仅 37.5%, 答案在后半部分则永远看不到, 系统性低估)
    # ★ 另一偏差: 官方文档 chunk 头部是 "来源/URL/爬取时间" 样板文字,
    #   占掉 judge 的注意力
    # 修复: 展示完整文档 + 剥离样板头
    def _strip_boilerplate(text: str) -> str:
        lines = [ln for ln in text.split("\n")
                 if not ln.startswith(("> 来源", "> URL", "> 爬取时间",
                                       "> 可信度", "> 标签"))]
        return "\n".join(lines).strip()

    doc_texts = []
    for i, doc in enumerate(docs[:5]):
        content = _strip_boilerplate(doc).replace("\n", " ")
        doc_texts.append(f"[文档{i+1}]\n{content}")

    prompt = JUDGE_PROMPT.format(
        question=question,
        documents="\n\n".join(doc_texts),
    )

    try:
        resp = robust_llm_call(prompt)
        content = resp.content if hasattr(resp, 'content') else str(resp)

        if not content or len(content) < 5:
            raise ValueError("LLM 返回空响应")

        # 提取 JSON
        import re
        data = {}
        # 尝试直接解析
        try:
            data = json.loads(content.strip())
        except json.JSONDecodeError:
            # 尝试提取 ```json ... ``` 代码块
            m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            if not data:
                # 尝试提取 { ... }
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    try:
                        data = json.loads(content[start:end])
                    except json.JSONDecodeError:
                        pass

        # 安全提取 scores + 分制归一化
        # ★ LLM 输出非确定: 有时用 0-3 分制, 有时混入 0-100 分制 (曾致
        #   Answerability 14387% 的荒谬数字) → 统一钳制到 [0,3]
        raw_scores = data.get("scores", [0] * min(len(docs), 5))
        scores = []
        for s in raw_scores[:5]:
            try:
                v = float(s)
            except (ValueError, TypeError):
                v = 0.0
            if v > 3:
                # 疑似 100 分制 → 等比缩放到 0-3
                v = v / 100.0 if v <= 100 else 3.0
            scores.append(max(0.0, min(3.0, v)))

        # 补齐
        while len(scores) < min(len(docs), 5):
            scores.append(0)

        # 解析完全失败 (scores 全 0 且 raw 无法解析) → 重试一次
        if not data or all(s == 0 for s in scores):
            retry_prompt = prompt + "\n\n★ 严格要求: 只输出 JSON 对象本身, 不要解释文字, 评分用 0-3 整数。"
            resp = robust_llm_call(retry_prompt)
            content = resp.content if hasattr(resp, 'content') else str(resp)
            retry_data = {}
            try:
                retry_data = json.loads(content.strip())
            except json.JSONDecodeError:
                m = re.search(r'\{[\s\S]*\}', content)
                if m:
                    try:
                        retry_data = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        retry_data = {}
            if retry_data:
                data = retry_data
                raw_scores = data.get("scores", [0] * min(len(docs), 5))
                scores = []
                for s in raw_scores[:5]:
                    try:
                        v = float(s)
                    except (ValueError, TypeError):
                        v = 0.0
                    if v > 3:
                        v = v / 100.0 if v <= 100 else 3.0
                    scores.append(max(0.0, min(3.0, v)))
                while len(scores) < min(len(docs), 5):
                    scores.append(0)

        return {
            "scores": scores,
            "best_doc_index": int(data.get("best_doc_index", 0)),
            "overall_judgment": str(data.get("overall_judgment", "unknown")),
            "reasoning": str(data.get("reasoning", ""))[:100],
            "raw_response": content[:200],
        }
    except Exception as e:
        return {
            "scores": [0] * min(len(docs), 5),
            "best_doc_index": 0,
            "overall_judgment": "error",
            "reasoning": str(e)[:100],
            "raw_response": "",
        }


def run_llm_judge_eval(queries_path: str = "", max_queries: int = 0, silent: bool = False):
    """
    运行 LLM-as-Judge 评测。

    Args:
        queries_path: 查询文件路径
        max_queries: 限制评测数量 (0=全部)
        silent: 静默模式, 不打印进度
    """
    from eval.run_eval import load_queries

    if not queries_path:
        queries_path = str(Path(__file__).parent / "test_queries.json")

    queries = load_queries(queries_path)
    if max_queries > 0:
        queries = queries[:max_queries]

    if not silent:
        print(f"\n{'='*60}")
        print(f"  LLM-as-Judge 评测 — RAG 检索质量评估")
        print(f"{'='*60}")
        print(f"  查询数: {len(queries)}")
        print(f"  评分: 0=无关 1=弱相关 2=部分可答 3=完全可答")
        print()

    # ── 初始化检索器 ──
    from rag.vector_store import VectorStore

    vs_graph = VectorStore()
    vs_graph.load_articles()

    t0 = time.time()

    results = {
        "vec": [],
        "graph": [],
        "agentic": [],
        "vec_answerable": [],
        "graph_answerable": [],
        "agentic_answerable": [],
        "per_query": [],
        "win_tie_loss": {"graph_wins": 0, "vec_wins": 0, "ties": 0},
    }

    # Agentic RAG 检索器 (第三臂: 改写+HyDE+反思循环)
    agentic_retriever = None
    try:
        from rag.agentic_retriever import AgenticRetriever
        agentic_retriever = AgenticRetriever(vs_graph, top_k=5)
    except Exception:
        pass

    for qi, q in enumerate(queries):
        question = q.get("question", "")
        category = q.get("category", "unknown")
        difficulty = q.get("difficulty", "medium")
        qid = q.get("id", f"q{qi}")

        # 检索（基线口径与 run_eval 一致: 纯向量 = 仅 ChromaDB similarity）
        graph_docs = vs_graph.search(question, top_k=5)
        vec_docs = vs_graph.store.similarity_search(question, k=5)

        graph_texts = [d.page_content for d in graph_docs]
        vec_texts = [d.page_content for d in vec_docs]

        # Agentic 臂 (检索循环, 失败降级为 graph 结果)
        agentic_texts = graph_texts
        agentic_trace = []
        if agentic_retriever is not None:
            try:
                agentic_docs, agentic_trace = agentic_retriever.search(question)
                agentic_texts = [d.page_content for d in agentic_docs[:8]]
            except Exception:
                agentic_texts = graph_texts

        # LLM 打分
        graph_judge = judge_answerability(question, graph_texts)
        vec_judge = judge_answerability(question, vec_texts)
        agentic_judge = judge_answerability(question, agentic_texts)

        graph_score = sum(graph_judge["scores"][:5]) / max(len(graph_judge["scores"][:5]), 1) / 3.0
        vec_score = sum(vec_judge["scores"][:5]) / max(len(vec_judge["scores"][:5]), 1) / 3.0
        agentic_score = sum(agentic_judge["scores"][:5]) / max(len(agentic_judge["scores"][:5]), 1) / 3.0

        # 集合级可答性 (文档集合整体能否支撑回答 — RAG 评测标准口径)
        def _is_answerable(j: dict) -> bool:
            v = j.get("overall_judgment", "")
            return v in ("fully_answerable", "partially_answerable")
        graph_ans = _is_answerable(graph_judge)
        vec_ans = _is_answerable(vec_judge)
        agentic_ans = _is_answerable(agentic_judge)

        # Win/Tie/Loss (GraphRAG vs 纯向量)
        if graph_score > vec_score + 0.05:
            results["win_tie_loss"]["graph_wins"] += 1
            wl = "graph_win"
        elif vec_score > graph_score + 0.05:
            results["win_tie_loss"]["vec_wins"] += 1
            wl = "vec_win"
        else:
            results["win_tie_loss"]["ties"] += 1
            wl = "tie"

        results["vec"].append(vec_score)
        results["graph"].append(graph_score)
        results["agentic"].append(agentic_score)
        results["vec_answerable"].append(1 if vec_ans else 0)
        results["graph_answerable"].append(1 if graph_ans else 0)
        results["agentic_answerable"].append(1 if agentic_ans else 0)
        results["per_query"].append({
            "id": qid,
            "question": question[:80],
            "category": category,
            "difficulty": difficulty,
            "vec_score": round(vec_score, 3),
            "graph_score": round(graph_score, 3),
            "agentic_score": round(agentic_score, 3),
            "vec_answerable": vec_ans,
            "graph_answerable": graph_ans,
            "agentic_answerable": agentic_ans,
            "agentic_retried": any(t.get("action") == "retry" for t in agentic_trace),
            "graph_reasoning": graph_judge.get("reasoning", ""),
            "wl": wl,
        })

        if not silent:
            status = "G↑" if wl == "graph_win" else ("V↑" if wl == "vec_win" else "==")
            a_status = "R2" if any(t.get("action") == "retry" for t in agentic_trace) else "R1"
            print(f"  [{qi+1:>2}/{len(queries)}] {status} {qid} | vec={vec_score:.2f} "
                  f"graph={graph_score:.2f} agentic={agentic_score:.2f}({a_status}) | "
                  f"{graph_judge.get('reasoning', '')[:30]}")

    elapsed = time.time() - t0

    # ── 汇总 ──
    avg_vec = sum(results["vec"]) / max(len(results["vec"]), 1)
    avg_graph = sum(results["graph"]) / max(len(results["graph"]), 1)
    improvement = avg_graph - avg_vec
    total = len(queries)
    w = results["win_tie_loss"]

    if not silent:
        avg_agentic = sum(results["agentic"]) / max(len(results["agentic"]), 1)
        print(f"\n{'='*60}")
        print(f"  评测结果 (三臂: 纯向量 / GraphRAG / Agentic RAG)")
        print(f"{'='*60}")
        print(f"  逐文档均分 (严格口径):")
        print(f"    纯向量 RAG Answerability:  {avg_vec:.2%}")
        print(f"    GraphRAG Answerability:    {avg_graph:.2%}")
        print(f"    Agentic RAG Answerability: {avg_agentic:.2%}")
        print(f"    GraphRAG vs 向量:          {improvement:+.1%}")
        print(f"    Agentic vs GraphRAG:       {avg_agentic - avg_graph:+.1%}")
        ans_vec_rate = sum(results["vec_answerable"]) / max(len(results["vec_answerable"]), 1)
        ans_graph_rate = sum(results["graph_answerable"]) / max(len(results["graph_answerable"]), 1)
        ans_agentic_rate = sum(results["agentic_answerable"]) / max(len(results["agentic_answerable"]), 1)
        print(f"  集合级可答率 (RAG 标准口径, 文档集合整体能否支撑回答):")
        print(f"    纯向量:  {ans_vec_rate:.1%} ({sum(results['vec_answerable'])}/{len(results['vec_answerable'])})")
        print(f"    GraphRAG: {ans_graph_rate:.1%} ({sum(results['graph_answerable'])}/{len(results['graph_answerable'])})")
        print(f"    Agentic:  {ans_agentic_rate:.1%} ({sum(results['agentic_answerable'])}/{len(results['agentic_answerable'])})")
        n_retried = sum(1 for r in results["per_query"] if r.get("agentic_retried"))
        print(f"  Agentic 触发二轮检索: {n_retried}/{total} 查询")
        print(f"  GraphRAG 胜: {w['graph_wins']}/{total} ({w['graph_wins']/total:.0%})")
        print(f"  纯向量 胜:   {w['vec_wins']}/{total} ({w['vec_wins']/total:.0%})")
        print(f"  平局:        {w['ties']}/{total} ({w['ties']/total:.0%})")
        print(f"  耗时: {elapsed:.0f}s ({elapsed/total:.1f}s/query)")
        print(f"{'='*60}")

    # 按类别/难度汇总
    by_category = {}
    by_difficulty = {}
    for r in results["per_query"]:
        cat = r["category"]
        by_category.setdefault(cat, {"vec": [], "graph": [], "agentic": [], "count": 0})
        by_category[cat]["vec"].append(r["vec_score"])
        by_category[cat]["graph"].append(r["graph_score"])
        by_category[cat]["agentic"].append(r.get("agentic_score", r["graph_score"]))
        by_category[cat]["count"] += 1

        diff = r["difficulty"]
        by_difficulty.setdefault(diff, {"vec": [], "graph": [], "agentic": [], "count": 0})
        by_difficulty[diff]["vec"].append(r["vec_score"])
        by_difficulty[diff]["graph"].append(r["graph_score"])
        by_difficulty[diff]["agentic"].append(r.get("agentic_score", r["graph_score"]))
        by_difficulty[diff]["count"] += 1

    category_summary = {}
    for cat, data in by_category.items():
        category_summary[cat] = {
            "count": data["count"],
            "vec_answerability": round(sum(data["vec"]) / max(len(data["vec"]), 1), 3),
            "graph_answerability": round(sum(data["graph"]) / max(len(data["graph"]), 1), 3),
            "agentic_answerability": round(sum(data["agentic"]) / max(len(data["agentic"]), 1), 3),
            "improvement": round(
                sum(data["graph"]) / max(len(data["graph"]), 1) -
                sum(data["vec"]) / max(len(data["vec"]), 1), 3
            ),
        }

    difficulty_summary = {}
    for diff, data in by_difficulty.items():
        difficulty_summary[diff] = {
            "count": data["count"],
            "vec_answerability": round(sum(data["vec"]) / max(len(data["vec"]), 1), 3),
            "graph_answerability": round(sum(data["graph"]) / max(len(data["graph"]), 1), 3),
            "agentic_answerability": round(sum(data["agentic"]) / max(len(data["agentic"]), 1), 3),
            "improvement": round(
                sum(data["graph"]) / max(len(data["graph"]), 1) -
                sum(data["vec"]) / max(len(data["vec"]), 1), 3
            ),
        }

    return {
        "total_queries": total,
        "overall": {
            "vec_answerability": round(avg_vec, 3),
            "graph_answerability": round(avg_graph, 3),
            "agentic_answerability": round(
                sum(results["agentic"]) / max(len(results["agentic"]), 1), 3),
            "improvement": round(improvement, 3),
            "vec_set_answerable_rate": round(
                sum(results["vec_answerable"]) / max(len(results["vec_answerable"]), 1), 3),
            "graph_set_answerable_rate": round(
                sum(results["graph_answerable"]) / max(len(results["graph_answerable"]), 1), 3),
            "agentic_set_answerable_rate": round(
                sum(results["agentic_answerable"]) / max(len(results["agentic_answerable"]), 1), 3),
        },
        "win_tie_loss": w,
        "by_category": category_summary,
        "by_difficulty": difficulty_summary,
        "per_query": results["per_query"],
        "elapsed_seconds": round(elapsed, 0),
    }


if __name__ == "__main__":
    result = run_llm_judge_eval()

    # 保存
    output_path = Path(__file__).parent / "llm_judge_results.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已保存: {output_path}")
