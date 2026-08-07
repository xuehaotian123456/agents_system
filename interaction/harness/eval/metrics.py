"""
评测指标计算 — Eval Metrics

支持的指标:
    - MRR (Mean Reciprocal Rank): 第一个相关文档的倒数排名均值
    - Precision@K / Recall@K: 精确率和召回率
    - Hit Rate@K: Top-K 中是否有相关文档
    - Tool Call Accuracy: 工具选择的准确率
    - Avg Latency: 平均延迟
    - Success Rate: 任务完成率
    - Token Efficiency: Token 使用效率（越少 token 得到正确答案越好）
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MetricReport:
    """单次评测的指标报告"""
    # ── 检索指标 ──
    mrr: float = 0.0                     # MRR
    precision_at_k: float = 0.0           # Precision@K
    recall_at_k: float = 0.0              # Recall@K
    hit_rate_at_k: float = 0.0            # Hit Rate@K

    # ── Agent 指标 ──
    tool_accuracy: float = 0.0            # 工具选择准确率
    avg_turns: float = 0.0                # 平均推理轮次
    success_rate: float = 0.0             # 任务完成率

    # ── 效率指标 ──
    avg_latency_sec: float = 0.0          # 平均延迟（秒）
    p50_latency_sec: float = 0.0          # P50 延迟
    p95_latency_sec: float = 0.0          # P95 延迟
    avg_tokens_input: float = 0.0         # 平均输入 token
    avg_tokens_output: float = 0.0        # 平均输出 token

    # ── LLM Judge 指标 ──
    avg_judge_score: float = 0.0          # LLM Judge 平均分 (0-100)
    judge_accuracy: float = 0.0            # 答案准确性
    judge_completeness: float = 0.0        # 答案完整性
    judge_relevance: float = 0.0           # 答案相关性
    judge_hallucination_rate: float = 0.0  # 幻觉率（越高越差）

    # ── 元信息 ──
    total_questions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "retrieval": {
                "mrr": round(self.mrr, 4),
                "precision_at_k": round(self.precision_at_k, 4),
                "recall_at_k": round(self.recall_at_k, 4),
                "hit_rate_at_k": round(self.hit_rate_at_k, 4),
            },
            "agent": {
                "tool_accuracy": round(self.tool_accuracy, 4),
                "avg_turns": round(self.avg_turns, 2),
                "success_rate": round(self.success_rate, 4),
            },
            "efficiency": {
                "avg_latency_sec": round(self.avg_latency_sec, 2),
                "p50_latency_sec": round(self.p50_latency_sec, 2),
                "p95_latency_sec": round(self.p95_latency_sec, 2),
                "avg_tokens_input": round(self.avg_tokens_input, 0),
                "avg_tokens_output": round(self.avg_tokens_output, 0),
            },
            "llm_judge": {
                "avg_score": round(self.avg_judge_score, 1),
                "accuracy": round(self.judge_accuracy, 4),
                "completeness": round(self.judge_completeness, 4),
                "relevance": round(self.judge_relevance, 4),
                "hallucination_rate": round(self.judge_hallucination_rate, 4),
            },
            "total_questions": self.total_questions,
            "metadata": self.metadata,
        }

    def format_table(self) -> str:
        """格式化输出为可读的对比表格"""
        lines = [
            "┌──────────────────────────────────────────┐",
            "│            📊 Eval Report                │",
            "├──────────────────────────────────────────┤",
            f"│ Total Questions: {self.total_questions:<24} │",
            "├──────────────────────────────────────────┤",
            "│ 🎯 Retrieval                              │",
            f"│   MRR:          {self.mrr:.1%}                     │",
            f"│   Precision@K:  {self.precision_at_k:.1%}                     │",
            f"│   Recall@K:     {self.recall_at_k:.1%}                     │",
            f"│   Hit Rate@K:   {self.hit_rate_at_k:.1%}                     │",
            "├──────────────────────────────────────────┤",
            "│ 🤖 Agent Behavior                         │",
            f"│   Tool Accuracy: {self.tool_accuracy:.1%}                     │",
            f"│   Avg Turns:     {self.avg_turns:.1f}                      │",
            f"│   Success Rate:  {self.success_rate:.1%}                     │",
            "├──────────────────────────────────────────┤",
            "│ ⚡ Efficiency                             │",
            f"│   Avg Latency:   {self.avg_latency_sec:.1f}s                   │",
            f"│   P50 Latency:   {self.p50_latency_sec:.1f}s                   │",
            f"│   P95 Latency:   {self.p95_latency_sec:.1f}s                   │",
            f"│   Avg Input Tok: {self.avg_tokens_input:.0f}                  │",
            f"│   Avg Output Tok:{self.avg_tokens_output:.0f}                  │",
            "├──────────────────────────────────────────┤",
            "│ 🔍 LLM Judge                              │",
            f"│   Avg Score:     {self.avg_judge_score:.1f}/100                │",
            f"│   Accuracy:      {self.judge_accuracy:.3f}                    │",
            f"│   Completeness:  {self.judge_completeness:.3f}                    │",
            f"│   Relevance:     {self.judge_relevance:.3f}                    │",
            f"│   Hallucination: {self.judge_hallucination_rate:.1%}                     │",
            "└──────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


class EvalMetrics:
    """
    评测指标计算器

    使用方式:
        calc = EvalMetrics()
        calc.add_result(ground_truth="...", predicted="...", tools_called=["rag_search"], ...)
        report = calc.compute()
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._results: list[dict] = []

    def add_result(
        self,
        question_id: str,
        ground_truth: str,
        predicted_answer: str,
        expected_tools: list[str] | None = None,
        actual_tools: list[str] | None = None,
        retrieved_docs: list[str] | None = None,
        relevant_doc_indices: list[int] | None = None,
        turns: int = 0,
        latency_sec: float = 0.0,
        tokens_input: int = 0,
        tokens_output: int = 0,
        success: bool = True,
        judge_scores: dict[str, float] | None = None,
        error: str = "",
    ):
        """添加一条评测结果"""
        self._results.append({
            "question_id": question_id,
            "ground_truth": ground_truth,
            "predicted": predicted_answer,
            "expected_tools": expected_tools or [],
            "actual_tools": actual_tools or [],
            "retrieved_docs": retrieved_docs or [],
            "relevant_doc_indices": relevant_doc_indices or [],
            "turns": turns,
            "latency_sec": latency_sec,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "success": success,
            "judge_scores": judge_scores or {},
            "error": error,
        })

    def compute(self) -> MetricReport:
        """计算所有指标"""
        if not self._results:
            return MetricReport()

        report = MetricReport(total_questions=len(self._results))

        # ── 检索指标 ──
        rr_values = []  # reciprocal ranks
        precision_values = []
        recall_values = []
        hit_values = []

        for r in self._results:
            relevant = set(r.get("relevant_doc_indices", []))
            if relevant:
                # MRR: 找第一个相关文档的排名
                for i in range(min(self.k, len(r["retrieved_docs"]))):
                    if i in relevant:
                        rr_values.append(1.0 / (i + 1))
                        hit_values.append(1.0)
                        break
                else:
                    rr_values.append(0.0)
                    hit_values.append(0.0)

                # Precision & Recall
                retrieved_set = set(range(min(self.k, len(r["retrieved_docs"]))))
                tp = len(retrieved_set & relevant)
                precision_values.append(tp / max(len(retrieved_set), 1))
                recall_values.append(tp / max(len(relevant), 1))

        if rr_values:
            report.mrr = statistics.mean(rr_values)
            report.precision_at_k = statistics.mean(precision_values) if precision_values else 0
            report.recall_at_k = statistics.mean(recall_values) if recall_values else 0
            report.hit_rate_at_k = statistics.mean(hit_values) if hit_values else 0

        # ── Agent 指标 ──
        tool_matches = 0
        tool_total = 0
        for r in self._results:
            if r["expected_tools"]:
                tool_total += 1
                exp = set(r["expected_tools"])
                act = set(r["actual_tools"])
                if exp & act or (not exp and not act):
                    tool_matches += 1

        report.tool_accuracy = tool_matches / max(tool_total, 1)
        report.avg_turns = statistics.mean([r["turns"] for r in self._results])
        report.success_rate = sum(1 for r in self._results if r["success"]) / len(self._results)

        # ── 效率指标 ──
        latencies = [r["latency_sec"] for r in self._results if r["latency_sec"] > 0]
        if latencies:
            sorted_lat = sorted(latencies)
            report.avg_latency_sec = statistics.mean(latencies)
            report.p50_latency_sec = sorted_lat[len(sorted_lat) // 2]
            report.p95_latency_sec = sorted_lat[int(len(sorted_lat) * 0.95)]

        report.avg_tokens_input = statistics.mean([r["tokens_input"] for r in self._results])
        report.avg_tokens_output = statistics.mean([r["tokens_output"] for r in self._results])

        # ── LLM Judge 指标 ──
        scores = [r["judge_scores"] for r in self._results if r["judge_scores"]]
        if scores:
            report.avg_judge_score = statistics.mean([s.get("overall", 0) for s in scores])
            report.judge_accuracy = statistics.mean([s.get("accuracy", 0) for s in scores])
            report.judge_completeness = statistics.mean([s.get("completeness", 0) for s in scores])
            report.judge_relevance = statistics.mean([s.get("relevance", 0) for s in scores])
            report.judge_hallucination_rate = statistics.mean([
                s.get("hallucination_score", 0) for s in scores
            ])

        return report

    def compare(self, baseline: "EvalMetrics", label_baseline: str = "Baseline",
                label_current: str = "Current") -> str:
        """生成对比报告"""
        base = baseline.compute()
        curr = self.compute()

        def delta_str(new, old) -> str:
            if old == 0:
                return "N/A"
            change = (new - old) / abs(old) * 100
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            return f"{arrow} {abs(change):.1f}%"

        lines = [
            "┌──────────────────────────────────────────────────────┐",
            "│              📊 对比报告                              │",
            f"│  {label_baseline:<20} vs {label_current:<20} │",
            "├──────────┬─────────────────┬────────────────┬─────────┤",
            "│ 指标     │ Baseline        │ Current        │ 变化    │",
            "├──────────┼─────────────────┼────────────────┼─────────┤",
            f"│ MRR      │ {base.mrr:<15.4f} │ {curr.mrr:<14.4f} │ {delta_str(curr.mrr, base.mrr):>7} │",
            f"│ Prec@5   │ {base.precision_at_k:<15.4f} │ {curr.precision_at_k:<14.4f} │ {delta_str(curr.precision_at_k, base.precision_at_k):>7} │",
            f"│ Hit@5    │ {base.hit_rate_at_k:<15.4f} │ {curr.hit_rate_at_k:<14.4f} │ {delta_str(curr.hit_rate_at_k, base.hit_rate_at_k):>7} │",
            f"│ Tool Acc │ {base.tool_accuracy:<15.4f} │ {curr.tool_accuracy:<14.4f} │ {delta_str(curr.tool_accuracy, base.tool_accuracy):>7} │",
            f"│ Avg Turn │ {base.avg_turns:<15.2f} │ {curr.avg_turns:<14.2f} │ {delta_str(curr.avg_turns, base.avg_turns):>7} │",
            f"│ Latency  │ {base.avg_latency_sec:<15.2f}s│ {curr.avg_latency_sec:<13.2f}s│ {delta_str(curr.avg_latency_sec, base.avg_latency_sec):>7} │",
            f"│ Tokens   │ {base.avg_tokens_input:<15.0f} │ {curr.avg_tokens_input:<14.0f} │ {delta_str(curr.avg_tokens_input, base.avg_tokens_input):>7} │",
            f"│ Judge    │ {base.avg_judge_score:<15.1f} │ {curr.avg_judge_score:<14.1f} │ {delta_str(curr.avg_judge_score, base.avg_judge_score):>7} │",
            "└──────────┴─────────────────┴────────────────┴─────────┘",
        ]
        return "\n".join(lines)
