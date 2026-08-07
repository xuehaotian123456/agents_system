"""
Layer 3: 输出安全护栏 (Output Guard)

Agent 生成回答后的最终安全检查。

检测维度:
    1. 幻觉检测 — 回答是否包含知识库/标准答案中没有的编造内容
    2. 有害内容 — 暴力/色情/仇恨言论/自残建议
    3. 敏感信息泄露 — Agent 是否在回答中暴露了系统 prompt/内部数据
    4. 格式校验 — 回答是否完整、长度是否合理

使用方式:
    guard = OutputGuard(llm_adapter=llm)
    result = await guard.check(answer, context={"retrieved_docs": [...], "ground_truth": "..."})

    if result.hallucination_score > 0.5:
        answer = "抱歉，我无法确认这个回答的准确性。"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class OutputRisk(str, Enum):
    NONE = "none"
    HALLUCINATION = "hallucination"
    HARMFUL = "harmful"
    DATA_LEAK = "data_leak"
    INCOMPLETE = "incomplete"


@dataclass
class OutputCheckResult:
    """输出检查结果"""
    safe: bool = True
    hallucination_score: float = 0.0       # 0=无幻觉, 1=严重幻觉
    harmful_score: float = 0.0             # 0=无害, 1=严重有害
    leak_detected: bool = False
    risks: list[OutputRisk] = field(default_factory=list)
    reason: str = ""
    sanitized_answer: str = ""             # 净化后的回答
    recommended_action: str = "pass"       # pass / rewrite / block


class OutputGuard:
    """
    输出安全护栏

    使用方式:
        guard = OutputGuard(llm_adapter=llm)

        # 基础检查（规则，快速）
        result = guard.basic_check(answer)

        # 深度检查（LLM，更准但更慢）
        result = await guard.deep_check(answer, retrieved_docs=[...])
    """

    # ── 有害内容关键词 ──
    HARMFUL_PATTERNS = [
        # 暴力/自残
        r"(?:自杀|自残|如何.*杀|怎样.*毒|制作.*炸弹)",
        r"(?:how to (?:make|build|create).{0,15}(?:bomb|weapon|explosive))",
        r"(?:suicide|self[- ]harm|kill yourself)",
        # 色情
        r"(?:色情|淫秽|裸体|性行为)",
        r"(?:porn|explicit sexual|nude)",
        # 违法
        r"(?:黑客|破解.*密码|盗取|诈骗|洗钱|赌博)",
        r"(?:hack|steal password|fraud|money laundering)",
        # 歧视
        r"(?:种族.*歧视|性别.*歧视|地域.*歧视)",
        r"(?:racist|sexist|discriminat)",
    ]

    # ── 信息泄露特征 ──
    LEAK_PATTERNS = [
        r"(?:System\s*Prompt|系统提示词|system\s*instruction)",
        r"(?:API\s*[Kk]ey|Access\s*Token|Secret)",
        r"(?:sk-[A-Za-z0-9]{20,})",           # OpenAI API key 格式
        r"(?:DASHSCOPE_API_KEY|OPENAI_API_KEY)",
        r"(?:密码|password)\s*[是:：=]\s*\S+",
    ]

    # ── 幻觉信号 ──
    HALLUCINATION_SIGNALS = [
        r"(?:据我所知|据我了解|可能|大概|也许|不确定)",
        r"(?:根据.*?报道|根据.*?研究|根据.*?调查)",    # 模糊引用
        r"\b\d{4}年\d{1,2}月\d{1,2}日\b",             # 精确日期（大概率编造）
    ]

    def __init__(self, llm_adapter=None):
        self.llm = llm_adapter

    # ==================== 基础检查（规则，同步） ====================

    def basic_check(self, answer: str, context: dict | None = None) -> OutputCheckResult:
        """
        快速规则检查

        Args:
            answer: Agent 生成的回答
            context: 上下文信息（检索文档、标准答案等）

        Returns:
            OutputCheckResult
        """
        result = OutputCheckResult(safe=True)

        # 1. 空回答/过短检查
        if not answer or len(answer.strip()) < 5:
            result.safe = False
            result.risks.append(OutputRisk.INCOMPLETE)
            result.reason = "回答为空或过短"
            result.recommended_action = "block"
            return result

        # 2. 有害内容检查
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                result.safe = False
                result.risks.append(OutputRisk.HARMFUL)
                result.harmful_score = 0.8
                result.reason = "回答包含疑似有害内容"
                result.recommended_action = "block"

        # 3. 信息泄露检查
        for pattern in self.LEAK_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                result.safe = False
                result.leak_detected = True
                result.risks.append(OutputRisk.DATA_LEAK)
                result.reason = "回答疑似包含敏感信息泄露"
                result.sanitized_answer = self._sanitize_leak(answer)
                result.recommended_action = "rewrite"

        # 4. 幻觉信号检测
        hallucination_count = sum(
            1 for p in self.HALLUCINATION_SIGNALS if re.search(p, answer)
        )
        if hallucination_count >= 3:
            result.hallucination_score = 0.4
            result.risks.append(OutputRisk.HALLUCINATION)
            if not result.reason:
                result.reason = "回答包含较多不确定表述，可能存在幻觉"
                result.recommended_action = "rewrite"

        # 5. 与知识库一致性检查（如有检索文档）
        if context and context.get("retrieved_docs"):
            result.hallucination_score = self._check_grounding(answer, context["retrieved_docs"])
            if result.hallucination_score > 0.5:
                result.risks.append(OutputRisk.HALLUCINATION)
                result.reason = f"回答与知识库内容一致性低 (幻觉评分: {result.hallucination_score:.2f})"
                result.recommended_action = "rewrite"

        return result

    # ==================== 深度检查（LLM，异步） ====================

    async def deep_check(
        self,
        answer: str,
        question: str = "",
        retrieved_docs: list[str] | None = None,
        ground_truth: str = "",
    ) -> OutputCheckResult:
        """
        LLM 深度质量检查

        用 LLM 做多维度评分：准确性、有害内容、泄露、完整性
        """
        # 先做基础规则检查
        result = self.basic_check(answer)

        # 如果基础检查已拦截，不再浪费 LLM 调用
        if result.recommended_action == "block":
            return result

        # 没有 LLM 则返回基础结果
        if not self.llm:
            return result

        # 构建 LLM 评估 prompt
        docs_text = "\n---\n".join(retrieved_docs[:3]) if retrieved_docs else "无"

        prompt = f"""你是输出安全审查员。评估以下 Agent 回答的质量。

## 用户问题
{question[:500] if question else '无'}

## 参考知识（Agent 检索到的文档）
{docs_text[:2000]}

## Agent 回答
{answer[:2000]}

## 评分维度（每项 0-100，越高越差）
1. hallucination (幻觉): 回答是否编造了知识库中不存在的内容
2. harmfulness (有害): 回答是否包含暴力、色情、歧视等
3. data_leak (泄露): 回答是否暴露了系统信息
4. incompleteness (不完整): 回答是否明显截断或未完成

输出 JSON:
{{"hallucination": 0, "harmfulness": 0, "data_leak": 0, "incompleteness": 0, "verdict": "safe|rewrite|block", "reason": "..."}}"""

        import json
        try:
            resp = await self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())

            # 更新评分
            result.hallucination_score = max(result.hallucination_score, data.get("hallucination", 0) / 100)
            result.harmful_score = max(result.harmful_score, data.get("harmfulness", 0) / 100)

            # 更新判定
            verdict = data.get("verdict", "safe")
            if verdict == "block":
                result.safe = False
                result.reason = data.get("reason", result.reason)
                result.recommended_action = "block"
            elif verdict == "rewrite":
                result.reason = data.get("reason", result.reason)
                result.recommended_action = "rewrite"
            elif result.hallucination_score > 0.3:
                result.recommended_action = "rewrite"

        except Exception:
            pass  # LLM 检查失败，降级为规则结果

        return result

    # ==================== 内部方法 ====================

    def _check_grounding(self, answer: str, docs: list[str]) -> float:
        """
        检查回答是否"有据可查"（grounding check）

        简单实现：关键词重叠率。重叠越低，幻觉风险越高。
        """
        if not docs:
            return 0.5

        answer_words = set(answer)
        docs_words = set()
        for doc in docs[:3]:
            docs_words.update(doc)

        if not answer_words:
            return 1.0

        overlap = len(answer_words & docs_words) / len(answer_words)
        return round(1.0 - overlap, 2)

    def _sanitize_leak(self, text: str) -> str:
        """脱敏处理（替换疑似泄露信息）"""
        sanitized = text
        # 替换 API Key
        sanitized = re.sub(r'sk-[A-Za-z0-9]{20,}', '***API_KEY***', sanitized)
        # 替换密码
        sanitized = re.sub(r'(密码|password)\s*[是:：=]\s*\S+', r'\1=***', sanitized)
        return sanitized
