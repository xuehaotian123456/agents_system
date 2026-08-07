"""
评测执行器 — Eval Runner

批量跑 Benchmark，自动收集结果、计算指标、生成报告。

使用方式:
    # 1. 准备 Benchmark
    suite = create_rag_benchmark()

    # 2. 定义 Agent 工厂函数
    def create_agent():
        llm = LLMAdapter(model="qwen-plus")
        registry = ToolRegistry()
        registry.register(RAGTool(...))
        return llm, registry

    # 3. 跑评测
    runner = EvalRunner(suite, create_agent)
    report = await runner.run()
    print(report.format_table())
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

from harness.eval.benchmark import BenchmarkSuite, QAPair
from harness.eval.metrics import EvalMetrics, MetricReport
from harness.eval.judge import LLMJudge, JudgeScore


# Agent 工厂类型: () -> (llm, tool_registry, prompt_engine) 或任意 callable
AgentFactory = Callable[[], Any]


@dataclass
class SingleResult:
    """单条问答的评测结果"""
    question_id: str
    question: str
    ground_truth: str
    predicted: str
    success: bool
    turns: int
    latency_sec: float
    tokens_input: int
    tokens_output: int
    actual_tools: list[str]
    judge_score: JudgeScore | None
    trace_json: str = ""      # Agent 完整轨迹 JSON
    error: str = ""


@dataclass
class EvalResult:
    """完整的评测结果"""
    suite_name: str
    total: int
    success_count: int
    metrics: MetricReport
    details: list[SingleResult]
    timestamp: str = ""
    duration_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "total": self.total,
            "success_count": self.success_count,
            "timestamp": self.timestamp,
            "duration_sec": round(self.duration_sec, 1),
            "metrics": self.metrics.to_dict(),
            "details": [
                {
                    "id": d.question_id,
                    "question": d.question,
                    "predicted": d.predicted[:300],
                    "success": d.success,
                    "turns": d.turns,
                    "latency": round(d.latency_sec, 2),
                    "tokens_input": d.tokens_input,
                    "tokens_output": d.tokens_output,
                    "actual_tools": d.actual_tools,
                    "judge_score": d.judge_score.to_dict() if d.judge_score else None,
                    "error": d.error,
                }
                for d in self.details
            ],
        }

    def save_json(self, filepath: str):
        """保存结果为 JSON"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class EvalRunner:
    """
    评测执行器

    使用方式:
        async def create_agent():
            llm = LLMAdapter(model="qwen-plus")
            registry = ToolRegistry()
            registry.register(RAGTool(...))
            session = Session(config=AgentConfig(max_turns=3))
            session.set_system_prompt(...)
            return session, llm, registry

        runner = EvalRunner(suite, create_agent, llm_judge=judge)
        result = await runner.run()
        print(result.metrics.format_table())
    """

    def __init__(
        self,
        suite: BenchmarkSuite,
        agent_factory: Callable[[], Coroutine[Any, Any, tuple]],
        llm_judge: LLMJudge | None = None,
        max_concurrent: int = 5,
        timeout_per_question: float = 60.0,
    ):
        """
        Args:
            suite: 评测数据集
            agent_factory: 创建 Agent 的异步工厂函数
                           async def f() -> (session, llm_adapter, tool_registry)
            llm_judge: LLM Judge（可选，不传则用简单规则评分）
            max_concurrent: 最大并发数
            timeout_per_question: 每题超时（秒）
        """
        self.suite = suite
        self.agent_factory = agent_factory
        self.judge = llm_judge
        self.max_concurrent = max_concurrent
        self.timeout = timeout_per_question

    async def run(self) -> EvalResult:
        """
        批量执行所有评测

        Returns:
            EvalResult: 完整评测结果
        """
        start_time = time.time()
        results: list[SingleResult] = []

        # 信号量控制并发
        sem = asyncio.Semaphore(self.max_concurrent)

        async def run_one(qa: QAPair) -> SingleResult:
            async with sem:
                return await self._run_single(qa)

        # 并发执行
        tasks = [run_one(qa) for qa in self.suite.qa_pairs]
        results = await asyncio.gather(*tasks)

        # 计算指标
        metrics_calc = EvalMetrics()
        for r in results:
            metrics_calc.add_result(
                question_id=r.question_id,
                ground_truth=r.ground_truth,
                predicted_answer=r.predicted,
                expected_tools=[],  # 从 suite 获取
                actual_tools=r.actual_tools,
                turns=r.turns,
                latency_sec=r.latency_sec,
                tokens_input=r.tokens_input,
                tokens_output=r.tokens_output,
                success=r.success,
                judge_scores=r.judge_score.to_dict() if r.judge_score else {},
                error=r.error,
            )

        metrics = metrics_calc.compute()
        duration = time.time() - start_time

        return EvalResult(
            suite_name=self.suite.name,
            total=len(results),
            success_count=sum(1 for r in results if r.success),
            metrics=metrics,
            details=results,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_sec=duration,
        )

    async def _run_single(self, qa: QAPair) -> SingleResult:
        """执行单条评测"""
        t_start = time.time()
        error = ""

        try:
            # 创建 Agent
            agent_components = await self.agent_factory()
            session, llm_adapter, tool_registry = agent_components

            # 设置问题
            session.append_user_message(qa.question)

            # 创建 AgentLoop 并运行（带超时）
            from harness.agent_loop import AgentLoop
            loop = AgentLoop(
                session=session,
                llm_adapter=llm_adapter,
                tool_registry=tool_registry,
                prompt_engine=None,  # 使用 session 自带的 system prompt
            )

            predicted = await asyncio.wait_for(
                loop.run(),
                timeout=self.timeout,
            )

            success = True
            turns = session.total_turns
            actual_tools = []  # 可从 tracer 提取

        except asyncio.TimeoutError:
            predicted = f"评测超时 (>{self.timeout}s)"
            success = False
            turns = 0
            actual_tools = []
            error = "timeout"
        except Exception as e:
            predicted = f"执行异常: {e}"
            success = False
            turns = 0
            actual_tools = []
            error = str(e)

        latency = time.time() - t_start

        # LLM Judge 评分
        judge_score = None
        if success and self.judge:
            try:
                judge_score = await self.judge.evaluate(
                    question=qa.question,
                    ground_truth=qa.ground_truth,
                    predicted=predicted,
                    keywords=qa.keywords,
                )
            except Exception:
                pass

        return SingleResult(
            question_id=qa.id,
            question=qa.question,
            ground_truth=qa.ground_truth,
            predicted=predicted,
            success=success,
            turns=turns,
            latency_sec=latency,
            tokens_input=getattr(session, '_total_tokens_input', 0),
            tokens_output=getattr(session, '_total_tokens_output', 0),
            actual_tools=actual_tools,
            judge_score=judge_score,
            error=error,
        )

    # ==================== 便捷方法 ====================

    async def run_and_save(self, output_path: str) -> EvalResult:
        """跑评测并保存结果"""
        result = await self.run()
        result.save_json(output_path)
        print(f"📁 评测结果已保存至: {output_path}")
        return result

    def quick_benchmark(self, agent_factory, num_samples: int = 5) -> str:
        """
        快速基准测试（只跑前 N 题，输出简述）

        用于 CI/CD 或快速验证，不跑全量。
        """
        pass  # 简化实现
