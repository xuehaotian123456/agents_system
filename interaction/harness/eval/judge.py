"""
LLM-as-Judge 自动评分器

用 LLM 做自动化质量评估，从三个维度评分:
1. Accuracy (准确性): 回答是否与标准答案一致
2. Completeness (完整性): 是否覆盖了所有关键信息
3. Relevance (相关性): 回答是否紧扣问题，有无冗余
4. Hallucination (幻觉): 是否包含标准答案中没有的编造内容

策略:
    默认用 DeepSeek/gpt-4o-mini 做 Judge，成本低且评判质量高。
    对简单问题可跳过 Judge（关键词匹配即可）。
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class JudgeScore:
    """LLM Judge 评分结果"""
    overall: float = 0.0             # 综合分 (0-100)
    accuracy: float = 0.0            # 准确性 (0-100)
    completeness: float = 0.0         # 完整性 (0-100)
    relevance: float = 0.0            # 相关性 (0-100)
    hallucination_score: float = 0.0  # 幻觉程度 (0-100, 越高越严重)
    reasoning: str = ""               # 评分理由
    is_valid: bool = True             # 答案是否有效

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "relevance": self.relevance,
            "hallucination_score": self.hallucination_score,
            "reasoning": self.reasoning,
            "is_valid": self.is_valid,
        }


JUDGE_PROMPT = """你是一个严格但公正的评测专家。请根据以下标准对 Agent 的回答进行评分。

## 用户问题
{question}

## 标准答案
{ground_truth}

## Agent 回答
{predicted}

## 评分标准

请从以下 4 个维度评分（每个维度 0-100 分）：

1. **准确性 (Accuracy)**: Agent 回答中的事实是否与标准答案一致？有无错误信息？
   - 100: 完全正确，无任何错误
   - 70-90: 大部分正确，有少量不精确
   - 40-70: 部分正确，有一些错误
   - 0-40: 大部分错误或完全错误

2. **完整性 (Completeness)**: Agent 是否覆盖了标准答案中的所有关键信息？
   - 100: 覆盖所有关键点
   - 60-90: 覆盖大部分关键点
   - 30-60: 遗漏较多
   - 0-30: 遗漏严重

3. **相关性 (Relevance)**: Agent 的回答是否紧扣用户问题？有无跑题或冗余？
   - 100: 完全扣题
   - 60-90: 基本扣题，少量冗余
   - 30-60: 部分跑题
   - 0-30: 严重跑题

4. **幻觉程度 (Hallucination)**: Agent 是否编造了标准答案中不存在的内容？
   - 0: 无幻觉（最好）
   - 10-30: 轻微夸大或过度概括
   - 40-70: 有明显编造
   - 80-100: 严重编造（最差）

## 输出格式（严格 JSON）
```json
{{
  "accuracy": 85,
  "completeness": 80,
  "relevance": 90,
  "hallucination_score": 10,
  "overall": 85,
  "reasoning": "评分理由（中文，100字以内）"
}}
```

只输出 JSON，不要输出其他内容。"""


class LLMJudge:
    """
    LLM-as-Judge 自动评分器

    使用方式:
        judge = LLMJudge(llm_adapter)
        score = await judge.evaluate(
            question="什么是 Agentic RAG？",
            ground_truth="Agentic RAG 是基于智能体的...",
            predicted="Agentic RAG 使用智能体来..."
        )
        print(f"综合分: {score.overall}/100")
    """

    def __init__(self, llm_adapter=None, model: str = "deepseek-chat"):
        """
        Args:
            llm_adapter: LLM 适配器（可选，可延迟注入）
            model: Judge 使用的模型（建议用便宜的模型）
        """
        self.llm = llm_adapter
        self.model = model
        self._scores: list[JudgeScore] = []

    async def evaluate(
        self,
        question: str,
        ground_truth: str,
        predicted: str,
        keywords: list[str] | None = None,
    ) -> JudgeScore:
        """
        评估单条回答

        Args:
            question: 用户问题
            ground_truth: 标准答案
            predicted: Agent 回答
            keywords: 答案应包含的关键词（快速校验）

        Returns:
            JudgeScore: 评分结果
        """
        # 空回答快速判定
        if not predicted or len(predicted) < 10:
            score = JudgeScore(
                overall=0, accuracy=0, completeness=0, relevance=0,
                hallucination_score=0, reasoning="回答为空或过短", is_valid=False,
            )
            self._scores.append(score)
            return score

        # 关键词快速校验（提升评分速度）
        keyword_hit_rate = 1.0
        if keywords:
            hits = sum(1 for kw in keywords if kw.lower() in predicted.lower())
            keyword_hit_rate = hits / len(keywords) if keywords else 1.0

            # 关键词全不命中 → 大概率幻觉，快速返回
            if keyword_hit_rate == 0 and len(predicted) > 50:
                score = JudgeScore(
                    overall=30, accuracy=20, completeness=20, relevance=40,
                    hallucination_score=60,
                    reasoning=f"关键词均未命中 (keywords: {keywords})",
                )
                self._scores.append(score)
                return score

        # 调用 LLM Judge
        if self.llm:
            return await self._llm_evaluate(question, ground_truth, predicted)
        else:
            # 无 LLM → 基于关键词的简单评分
            return self._simple_evaluate(question, ground_truth, predicted, keywords)

    async def evaluate_batch(
        self,
        qa_pairs: list[dict],  # [{"question":..., "ground_truth":..., "predicted":...}]
    ) -> list[JudgeScore]:
        """批量评估"""
        scores = []
        for qa in qa_pairs:
            score = await self.evaluate(
                question=qa["question"],
                ground_truth=qa["ground_truth"],
                predicted=qa["predicted"],
                keywords=qa.get("keywords"),
            )
            scores.append(score)
        return scores

    def aggregate(self, scores: list[JudgeScore] | None = None) -> dict:
        """汇总评分统计"""
        data = scores or self._scores
        if not data:
            return {}

        return {
            "avg_overall": statistics.mean([s.overall for s in data]),
            "avg_accuracy": statistics.mean([s.accuracy for s in data]),
            "avg_completeness": statistics.mean([s.completeness for s in data]),
            "avg_relevance": statistics.mean([s.relevance for s in data]),
            "avg_hallucination": statistics.mean([s.hallucination_score for s in data]),
            "valid_rate": sum(1 for s in data if s.is_valid) / len(data),
            "total_evaluated": len(data),
        }

    # ── 内部方法 ──

    async def _llm_evaluate(self, question: str, ground_truth: str, predicted: str) -> JudgeScore:
        """使用 LLM 评分"""
        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
        )

        try:
            resp = await self.llm.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
        except Exception:
            # LLM Judge 调用失败，降级为简单评分
            return self._simple_evaluate(question, ground_truth, predicted)

        score = JudgeScore(
            overall=float(data.get("overall", 0)),
            accuracy=float(data.get("accuracy", 0)),
            completeness=float(data.get("completeness", 0)),
            relevance=float(data.get("relevance", 0)),
            hallucination_score=float(data.get("hallucination_score", 0)),
            reasoning=data.get("reasoning", ""),
        )
        self._scores.append(score)
        return score

    def _simple_evaluate(self, question: str, ground_truth: str,
                          predicted: str, keywords: list[str] | None = None) -> JudgeScore:
        """简单规则评分（无 LLM 时的降级方案）"""
        # 基于关键词匹配的启发式评分
        if keywords:
            hits = sum(1 for kw in keywords if kw.lower() in predicted.lower())
            kw_rate = hits / len(keywords)
            score_val = min(100, int(kw_rate * 80 + 20))
        else:
            # 基于字符重叠率
            gt_words = set(ground_truth)
            pred_words = set(predicted)
            overlap = len(gt_words & pred_words) / max(len(gt_words), 1)
            score_val = min(100, int(overlap * 70 + 30))

        score = JudgeScore(
            overall=score_val,
            accuracy=score_val - 5,
            completeness=score_val - 10,
            relevance=score_val,
            hallucination_score=max(0, 100 - score_val),
            reasoning="规则评分（无 LLM Judge）",
        )
        self._scores.append(score)
        return score
