"""
CC-Harness Agent — 可靠性评测 (Reliability Testing)
======================================================
对标 τ-bench (Sierra, 2024) 和 MINT-Bench (ICLR 2024)。

核心概念（来自前沿论文）:

1. pass^k (τ-bench):
   单次成功率 pass^1=60% 不代表可靠。
   pass^8 衡量"连续 8 次独立运行全部成功"的概率。
   GPT-4o 在 τ-retail 上 pass^1=60% → pass^8=25% (60 个点的可靠性差距!)

2. Reliability Decay Curve (RDC):
   画出 pass^k 随 k 递增的衰减曲线。
   理想 Agent: 曲线平坦（每次都稳定）。
   脆弱 Agent: 曲线陡降（偶发成功，换个问法就挂）。

3. Multi-Turn Degradation (MINT-Bench):
   85% 单轮准确率 → 65% 十轮准确率。
   大多数 Agent 在多轮对话中逐步退化。

4. Variance Amplification Factor (VAF):
   任务越复杂，结果波动越大。
   VAF = Var(pass^k 结果) / Var(单轮结果)。

使用方式:
    from harness.eval import ReliabilityRunner

    runner = ReliabilityRunner(eval_runner, passes=8)
    report = await runner.run()
    # → pass^1=95%, pass^4=82%, pass^8=68%
    # → RDC 曲线显示可靠性衰减
    # → 识别出 3 个"偶发成功"的脆弱问题
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ==================== 可靠性指标 ====================

@dataclass
class ReliabilityScore:
    """pass^k 评分"""
    k: int                                    # 连续运行次数
    pass_rate: float = 0.0                    # pass^k 成功率
    mean_score: float = 0.0                   # 平均分
    std_score: float = 0.0                    # 标准差
    min_score: float = 0.0                    # 最差分
    max_score: float = 1.0                    # 最好分

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "pass_rate": round(self.pass_rate, 4),
            "mean_score": round(self.mean_score, 4),
            "std_score": round(self.std_score, 4),
            "min_score": round(self.min_score, 4),
            "max_score": round(self.max_score, 4),
        }


@dataclass
class ReliabilityDecayCurve:
    """可靠性衰减曲线 (RDC)"""
    question_id: str
    scores: list[float]                      # k 次运行的所有得分
    pass_k_values: list[ReliabilityScore]    # 不同 k 值的 pass^k

    @property
    def is_fragile(self) -> bool:
        """是否脆弱（pass^1 高但 pass^8 低）"""
        if len(self.pass_k_values) < 2:
            return False
        p1 = self.pass_k_values[0].pass_rate
        p_max = self.pass_k_values[-1].pass_rate
        return p1 > 0.7 and (p1 - p_max) > 0.25

    @property
    def decay_slope(self) -> float:
        """衰减斜率（越大越不稳定）"""
        if len(self.pass_k_values) < 2:
            return 0.0
        return self.pass_k_values[0].pass_rate - self.pass_k_values[-1].pass_rate


@dataclass
class ReliabilityReport:
    """可靠性评测完整报告"""
    suite_name: str
    total_questions: int
    passes: int                              # 每题重复运行次数

    # pass^k 汇总
    pass_1: ReliabilityScore
    pass_4: ReliabilityScore | None = None
    pass_8: ReliabilityScore | None = None

    # 衰减分析
    fragile_questions: list[str] = field(default_factory=list)  # 脆弱的题目
    avg_decay_slope: float = 0.0            # 平均衰减斜率
    variance_amplification: float = 0.0     # VAF (方差放大因子)

    # 多轮退化
    turn_degradation: dict[int, float] = field(default_factory=dict)

    # 详情
    curves: list[ReliabilityDecayCurve] = field(default_factory=list)
    duration_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "total_questions": self.total_questions,
            "passes": self.passes,
            "pass_k": {
                "pass_1": self.pass_1.to_dict(),
                "pass_4": self.pass_4.to_dict() if self.pass_4 else None,
                "pass_8": self.pass_8.to_dict() if self.pass_8 else None,
            },
            "fragile_questions": self.fragile_questions,
            "avg_decay_slope": round(self.avg_decay_slope, 4),
            "variance_amplification": round(self.variance_amplification, 4),
            "turn_degradation": self.turn_degradation,
            "duration_sec": round(self.duration_sec, 1),
        }

    def format(self) -> str:
        """人类可读报告"""
        lines = [
            "┌──────────────────────────────────────────────────┐",
            f"│  Reliability Report: {self.suite_name:<30} │",
            f"│  Questions: {self.total_questions}, Passes: {self.passes:<30} │",
            "├──────────────────────────────────────────────────┤",
            "│ pass^k Results                                   │",
            f"│   pass^1  = {self.pass_1.pass_rate:>6.1%}  (mean={self.pass_1.mean_score:.3f})              │",
        ]
        if self.pass_4:
            lines.append(
                f"│   pass^4  = {self.pass_4.pass_rate:>6.1%}  (mean={self.pass_4.mean_score:.3f})              │"
            )
        if self.pass_8:
            gap = self.pass_1.pass_rate - self.pass_8.pass_rate
            lines.append(
                f"│   pass^8  = {self.pass_8.pass_rate:>6.1%}  (gap: {gap:.1%})                    │"
            )
        lines.extend([
            "├──────────────────────────────────────────────────┤",
            "│ Stability Analysis                              │",
            f"│   Fragile questions: {len(self.fragile_questions):>2}/{self.total_questions}                         │",
            f"│   Avg decay slope:  {self.avg_decay_slope:>.3f}                          │",
            f"│   Variance Amp:     {self.variance_amplification:>.3f}                          │",
        ])
        if self.fragile_questions:
            lines.append("│   Fragile IDs:                                   │")
            for qid in self.fragile_questions[:5]:
                lines.append(f"│     - {qid:<42} │")
        lines.append("└──────────────────────────────────────────────────┘")
        return "\n".join(lines)


# ==================== 可靠性评测器 ====================

class ReliabilityRunner:
    """
    可靠性评测器

    使用方式:
        # 基础用法
        runner = ReliabilityRunner(eval_runner, passes=8)
        report = await runner.run()
        print(report.format())

        # 只跑前 N 题（快速验证）
        report = await runner.quick_check(n=5, passes=8)

        # 多轮退化测试
        report = await runner.multi_turn_test(questions, max_turns=10)
    """

    def __init__(self, eval_runner=None, passes: int = 8):
        """
        Args:
            eval_runner: EvalRunner 实例
            passes: 每题重复运行次数（默认 8，对齐 τ-bench）
        """
        self.eval_runner = eval_runner
        self.passes = passes

    async def run(self) -> ReliabilityReport:
        """完整可靠性评测"""
        if not self.eval_runner:
            raise ValueError("需要 EvalRunner 实例")

        suite = self.eval_runner.suite
        start = time.time()

        all_curves = []
        fragile = []

        for qa in suite.qa_pairs:
            # 对同一题跑 k 次
            scores = []
            for _ in range(self.passes):
                result = await self.eval_runner._run_single(qa)
                # 评分: 关键词命中率 + 成功标记
                score = self._score_result(result, qa)
                scores.append(score)

            # 计算 pass^k
            pass_k_values = self._compute_pass_k(scores)

            curve = ReliabilityDecayCurve(
                question_id=qa.id,
                scores=scores,
                pass_k_values=pass_k_values,
            )
            all_curves.append(curve)

            if curve.is_fragile:
                fragile.append(qa.id)

        # 汇总 pass^k
        pass_1 = self._aggregate_pass_k(all_curves, 1)
        pass_4 = self._aggregate_pass_k(all_curves, 4) if self.passes >= 4 else None
        pass_8 = self._aggregate_pass_k(all_curves, 8) if self.passes >= 8 else None

        # 衰减斜率
        avg_slope = statistics.mean([c.decay_slope for c in all_curves]) if all_curves else 0.0

        # VAF
        vaf = self._compute_vaf(all_curves)

        return ReliabilityReport(
            suite_name=suite.name,
            total_questions=len(suite.qa_pairs),
            passes=self.passes,
            pass_1=pass_1,
            pass_4=pass_4,
            pass_8=pass_8,
            fragile_questions=fragile,
            avg_decay_slope=avg_slope,
            variance_amplification=vaf,
            curves=all_curves,
            duration_sec=time.time() - start,
        )

    async def quick_check(self, n: int = 5, passes: int = 8) -> ReliabilityReport:
        """快速可靠性检查（只跑前 n 题）"""
        if not self.eval_runner:
            raise ValueError("需要 EvalRunner 实例")

        # 临时限制题目数量
        original = self.eval_runner.suite.qa_pairs
        self.eval_runner.suite.qa_pairs = original[:n]
        self.passes = passes

        try:
            return await self.run()
        finally:
            self.eval_runner.suite.qa_pairs = original

    # ==================== 多轮退化测试 ====================

    async def multi_turn_test(self, questions: list[dict],
                               max_turns: int = 10) -> dict[int, float]:
        """
        多轮对话退化测试 (MINT-Bench 风格)

        在一段长对话中插入测试问题，测量 Agent 在
        不同对话长度下的准确率变化。

        Returns:
            {turn: accuracy} — 每个轮次的准确率
        """
        results = {}

        for turn in range(1, max_turns + 1):
            # 构造一个长对话前缀（填充轮次）
            # 在指定轮次插入测试问题
            scores = []
            for q in questions:
                # 简化实现：用运行轮次模拟对话长度
                if self.eval_runner:
                    # 限制 AgentLoop 到指定轮次
                    score = 0.5  # placeholder
                    scores.append(score)

            if scores:
                results[turn] = statistics.mean(scores)

        return results

    # ==================== 内部方法 ====================

    def _score_result(self, result, qa) -> float:
        """计算单次运行得分 (0-1)"""
        score = 0.0

        # 成功 +0.5
        if result.success:
            score += 0.5

        # 关键词命中
        if hasattr(result, 'judge_score') and result.judge_score:
            score += result.judge_score.overall / 200  # 0-100 → 0-0.5
        else:
            # 看预期关键词
            expected = getattr(qa, 'keywords', [])
            if expected:
                hits = sum(1 for kw in expected if kw.lower() in result.predicted.lower())
                score += (hits / len(expected)) * 0.5

        return min(1.0, score)

    def _compute_pass_k(self, scores: list[float]) -> list[ReliabilityScore]:
        """计算不同 k 值的 pass^k"""
        results = []
        n = len(scores)

        for k in [1, 2, 4, 8]:
            if k > n:
                break

            # pass^k: 所有大小为 k 的滑动窗口中全部成功的比例
            all_pass_count = 0
            total_windows = 0

            for i in range(n - k + 1):
                window = scores[i:i + k]
                if all(s >= 0.5 for s in window):  # 所有都"通过"
                    all_pass_count += 1
                total_windows += 1

            pass_rate = all_pass_count / max(total_windows, 1)

            results.append(ReliabilityScore(
                k=k,
                pass_rate=pass_rate,
                mean_score=statistics.mean(scores),
                std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
                min_score=min(scores),
                max_score=max(scores),
            ))

        return results

    def _aggregate_pass_k(self, curves: list[ReliabilityDecayCurve],
                           k: int) -> ReliabilityScore:
        """汇总所有题目的 pass^k"""
        all_pass_rates = []
        all_means = []
        all_stds = []

        for curve in curves:
            for pk in curve.pass_k_values:
                if pk.k == k:
                    all_pass_rates.append(pk.pass_rate)
                    all_means.append(pk.mean_score)
                    all_stds.append(pk.std_score)
                    break

        if not all_pass_rates:
            return ReliabilityScore(k=k)

        return ReliabilityScore(
            k=k,
            pass_rate=statistics.mean(all_pass_rates),
            mean_score=statistics.mean(all_means),
            std_score=statistics.mean(all_stds),
            min_score=min(all_pass_rates),
            max_score=max(all_pass_rates),
        )

    def _compute_vaf(self, curves: list[ReliabilityDecayCurve]) -> float:
        """
        计算 Variance Amplification Factor (VAF)

        VAF = Var(pass^k) / Var(单轮)
        越高 → 任务复杂度对稳定性影响越大
        """
        if not curves:
            return 0.0

        single_var = statistics.mean([
            statistics.variance(c.scores) if len(c.scores) > 1 else 0
            for c in curves
        ])

        if single_var == 0:
            return 0.0

        pass8_vars = []
        for c in curves:
            for pk in c.pass_k_values:
                if pk.k == min(8, self.passes):
                    pass8_vars.append(pk.std_score ** 2)
                    break

        if not pass8_vars:
            return 0.0

        multi_var = statistics.mean(pass8_vars)
        return multi_var / single_var


# ==================== 便捷函数 ====================

def interpret_reliability(report: ReliabilityReport) -> str:
    """
    解读可靠性报告，给出人类可读的结论

    参考 τ-bench 论文的解读标准:
    - pass^8 > 80%: 生产就绪
    - pass^8 50-80%: 需要改进
    - pass^8 < 50%: 不可靠
    """
    p8 = report.pass_8.pass_rate if report.pass_8 else report.pass_1.pass_rate
    fragile_rate = len(report.fragile_questions) / max(report.total_questions, 1)

    if p8 > 0.8 and fragile_rate < 0.1:
        level = "🟢 生产就绪 (Production Ready)"
    elif p8 > 0.5 and fragile_rate < 0.3:
        level = "🟡 需要改进 (Needs Improvement)"
    else:
        level = "🔴 不可靠 (Not Reliable)"

    return f"""
可靠性评级: {level}
  pass^8 = {p8:.1%}
  脆弱题目比例 = {fragile_rate:.1%} ({len(report.fragile_questions)}/{report.total_questions})
  平均衰减斜率 = {report.avg_decay_slope:.3f} (越小越稳定)
  方差放大因子 = {report.variance_amplification:.2f} (>2 表示任务复杂度显著影响稳定性)

建议:
  {_get_recommendations(report)}
""".strip()


def _get_recommendations(report: ReliabilityReport) -> str:
    parts = []
    if len(report.fragile_questions) > 0:
        parts.append(f"- 关注 {len(report.fragile_questions)} 个脆弱题目，它们偶发成功但不可靠")
    if report.variance_amplification > 2.0:
        parts.append("- 任务复杂度显著影响稳定性 (VAF>2)，建议简化或拆分复杂任务")
    if report.avg_decay_slope > 0.1:
        parts.append("- 衰减斜率较大，Agent 在高负载下不稳定，建议增加重试和降级策略")
    if not parts:
        parts.append("- Agent 表现稳定，可以继续扩展测试覆盖面")
    return "\n".join(parts)
