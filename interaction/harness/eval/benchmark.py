"""
评测数据集定义 — Benchmark Suite

设计:
    每个 Benchmark 包含一组 QAPair（问答对），每条有:
    - 问题 + 标准答案
    - 预期的工具调用（可选，用于评估工具选择准确率）
    - 分类标签（难度、领域、所需能力）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Capability(str, Enum):
    """Agent 能力维度"""
    FACTUAL_QA = "factual_qa"            # 事实问答
    MULTI_HOP = "multi_hop"              # 多跳推理
    TOOL_USE = "tool_use"                # 工具调用
    COMPARISON = "comparison"             # 对比分析
    COMPLEX_REASONING = "complex_reasoning"  # 复杂推理
    CHITCHAT = "chitchat"                # 闲聊（不需要检索）


@dataclass
class QAPair:
    """
    单条问答评测对

    Example:
        QAPair(
            id="q001",
            question="Agentic RAG 和普通 RAG 有什么区别？",
            ground_truth="Agentic RAG 基于智能体动态检索，支持多次迭代；普通 RAG 是固定流水线。",
            expected_tools=["rag_search"],
            difficulty=Difficulty.MEDIUM,
            capability=Capability.COMPARISON,
        )
    """
    id: str
    question: str
    ground_truth: str
    expected_tools: list[str] = field(default_factory=list)   # 预期会调用的工具
    difficulty: Difficulty = Difficulty.MEDIUM
    capability: Capability = Capability.FACTUAL_QA
    keywords: list[str] = field(default_factory=list)          # 答案应包含的关键词
    max_turns: int = 5                                          # 预期最大推理步数
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "expected_tools": self.expected_tools,
            "difficulty": self.difficulty.value,
            "capability": self.capability.value,
            "keywords": self.keywords,
            "max_turns": self.max_turns,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QAPair":
        return cls(
            id=d["id"],
            question=d["question"],
            ground_truth=d["ground_truth"],
            expected_tools=d.get("expected_tools", []),
            difficulty=Difficulty(d.get("difficulty", "medium")),
            capability=Capability(d.get("capability", "factual_qa")),
            keywords=d.get("keywords", []),
            max_turns=d.get("max_turns", 5),
            metadata=d.get("metadata", {}),
        )


@dataclass
class EvalScenario:
    """评测场景配置"""
    name: str
    description: str = ""
    qa_pairs: list[QAPair] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BenchmarkSuite:
    """
    基准测试集

    使用方式:
        suite = BenchmarkSuite("agent-bench")
        suite.add(QAPair(id="q1", question="...", ground_truth="..."))
        suite.add_from_jsonl("benchmark/questions.jsonl")
        print(suite.stats())
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.qa_pairs: list[QAPair] = []

    def add(self, qa: QAPair):
        """添加一条评测"""
        self.qa_pairs.append(qa)

    def add_batch(self, qas: list[QAPair]):
        """批量添加"""
        self.qa_pairs.extend(qas)

    def add_from_jsonl(self, filepath: str):
        """从 JSONL 文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.add(QAPair.from_dict(json.loads(line)))

    def save_jsonl(self, filepath: str):
        """保存为 JSONL 文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            for qa in self.qa_pairs:
                f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")

    def filter(self, difficulty: Difficulty | None = None,
               capability: Capability | None = None) -> list[QAPair]:
        """按条件筛选"""
        result = self.qa_pairs
        if difficulty:
            result = [q for q in result if q.difficulty == difficulty]
        if capability:
            result = [q for q in result if q.capability == capability]
        return result

    def stats(self) -> dict:
        """数据集统计"""
        if not self.qa_pairs:
            return {"total": 0}

        diffs = {}
        caps = {}
        for q in self.qa_pairs:
            diffs[q.difficulty.value] = diffs.get(q.difficulty.value, 0) + 1
            caps[q.capability.value] = caps.get(q.capability.value, 0) + 1

        return {
            "total": len(self.qa_pairs),
            "by_difficulty": diffs,
            "by_capability": caps,
            "with_expected_tools": sum(1 for q in self.qa_pairs if q.expected_tools),
        }

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def __iter__(self):
        return iter(self.qa_pairs)


# ==================== 内置 Benchmark ====================

def create_rag_benchmark() -> BenchmarkSuite:
    """创建 RAG 评测基准（通用）"""
    suite = BenchmarkSuite("rag-benchmark", "通用 RAG 问答评测")

    # ── 事实问答 (Easy) ──
    easy_questions = [
        ("什么是 Agentic RAG？",
         "Agentic RAG 是基于智能体实现动态检索的 RAG 系统，支持多次迭代查询和 LLM 自主决策"),
        ("LangGraph 的核心概念是什么？",
         "LangGraph 使用状态图 (StateGraph) 实现 Agent 工作流，通过节点 (Node) 和条件边 (Edge) 定义业务流程"),
    ]
    for i, (q, a) in enumerate(easy_questions):
        suite.add(QAPair(
            id=f"easy_{i+1}", question=q, ground_truth=a,
            difficulty=Difficulty.EASY, capability=Capability.FACTUAL_QA,
            expected_tools=["rag_search"],
        ))

    # ── 对比分析 (Medium) ──
    medium_questions = [
        ("Agentic RAG 和普通 RAG 有什么区别？",
         "Agentic RAG 基于智能体动态检索，支持多次迭代查询；普通 RAG 是固定流水线，只需一次检索。"
         "Agentic RAG 有文档评分和查询改写能力，普通 RAG 没有。"),
        ("Claude Code 的 AgentLoop 与 LangGraph 的 StateGraph 有什么本质区别？",
         "AgentLoop 是运行时动态决策，每轮由 LLM 自主决定下一步；StateGraph 是编译时静态编排，"
         "开发阶段预定义所有节点和边。前者适合开放式任务，后者适合固定流程。"),
    ]
    for i, (q, a) in enumerate(medium_questions):
        suite.add(QAPair(
            id=f"medium_{i+1}", question=q, ground_truth=a,
            difficulty=Difficulty.MEDIUM, capability=Capability.COMPARISON,
            expected_tools=["rag_search"],
        ))

    # ── 多跳推理 (Hard) ──
    suite.add(QAPair(
        id="hard_1",
        question="GraphRAG 和 Agentic RAG 各有什么优缺点？在什么场景下应该选择哪种方案？",
        ground_truth=(
            "GraphRAG 通过抽取实体关系构建知识图谱，擅长处理实体间关联查询，但构建成本高；"
            "Agentic RAG 通过 Agent 动态决策检索策略，灵活性强，适合复杂问答场景。"
            "GraphRAG 适合知识图谱密集型场景，Agentic RAG 适合需要多步推理的开放式任务。"
        ),
        difficulty=Difficulty.HARD,
        capability=Capability.COMPLEX_REASONING,
        expected_tools=["rag_search"],
        keywords=["知识图谱", "动态决策", "多步推理"],
    ))

    return suite


# 预置数据集注册表
BUILTIN_BENCHMARKS = {
    "rag": create_rag_benchmark,
}
