"""
CC-Harness Agent — 评测框架 (Evaluation Framework)
===================================================
轻量级 Agent 评测体系，对标 LangSmith / RAGAS / tau-bench。

核心组件：
- BenchmarkSuite: 评测数据集定义
- EvalRunner: 批量执行器
- LLMJudge: LLM-as-Judge 自动评分
- EvalMetrics: 指标计算（MRR, Precision, Tool Accuracy, Latency）
- ReliabilityRunner: pass^k 可靠性评测（对标 tau-bench）
"""

from harness.eval.benchmark import BenchmarkSuite, QAPair, EvalScenario
from harness.eval.metrics import EvalMetrics, MetricReport
from harness.eval.judge import LLMJudge, JudgeScore
from harness.eval.runner import EvalRunner, EvalResult
from harness.eval.reliability import (
    ReliabilityRunner, ReliabilityReport, ReliabilityScore,
    ReliabilityDecayCurve, interpret_reliability,
)

__all__ = [
    "BenchmarkSuite",
    "QAPair",
    "EvalScenario",
    "EvalMetrics",
    "MetricReport",
    "LLMJudge",
    "JudgeScore",
    "EvalRunner",
    "EvalResult",
    "ReliabilityRunner",
    "ReliabilityReport",
    "ReliabilityScore",
    "ReliabilityDecayCurve",
    "interpret_reliability",
]
