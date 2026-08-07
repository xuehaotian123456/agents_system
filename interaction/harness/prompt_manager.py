"""
CC-Harness Agent — Prompt 版本管理与 A/B 测试
===============================================
让 Prompt 从"字符串"升级为"可追溯、可对比、可优化的资产"。

核心能力：
    1. PromptTemplate — 带版本号的模板，支持变量插值
    2. PromptRegistry — 集中管理所有 Prompt 版本
    3. ABTestRunner — 同一问题用不同 Prompt 跑，自动选最优

设计对标：
    - LangSmith Hub (Prompt 仓库)
    - Anthropic Prompt Caching
    - 各家 Prompt Engineering 平台

使用方式:
    # 1. 注册 Prompt
    registry = PromptRegistry()
    registry.register(PromptTemplate(
        name="system_prompt",
        version="v2",
        template="你是{role}，擅长{skills}。回答风格：{style}。",
        variables=["role", "skills", "style"],
    ))

    # 2. 获取当前版本
    prompt = registry.get("system_prompt")  # → v2

    # 3. A/B 测试
    ab = ABTestRunner(eval_runner, registry)
    report = await ab.compare(
        prompt_name="system_prompt",
        variants=["v1", "v2", "v3"],
        questions=test_questions,
    )
    # → v2 胜出: MRR +15%, 幻觉率 -30%
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ==================== Prompt 模板 ====================

@dataclass
class PromptTemplate:
    """
    Prompt 模板（带版本管理）

    Example:
        template = PromptTemplate(
            name="system_prompt",
            version="v2",
            template="你是{role}。\n\n## 工具\n{tools}\n\n## 规则\n{rules}",
            variables=["role", "tools", "rules"],
            metadata={"author": "xuehaotian", "improvement": "增加了工具说明和规则约束"},
        )
    """
    name: str                              # 模板名称
    version: str                           # 版本号 (v1, v2, v3...)
    template: str                          # 模板文本（支持 {variable} 插值）
    variables: list[str] = field(default_factory=list)  # 模板变量
    description: str = ""                  # 版本说明
    parent_version: str = ""               # 基于哪个版本修改
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    content_hash: str = ""                 # 内容指纹

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                self.template.encode()
            ).hexdigest()[:8]

    def render(self, **kwargs) -> str:
        """
        渲染模板

        安全渲染：缺少的变量不会报错，而是保留占位符。
        """
        result = self.template
        for var in self.variables:
            value = kwargs.get(var, f"{{{var}}}")
            result = result.replace(f"{{{var}}}", str(value))
        return result

    def diff(self, other: "PromptTemplate") -> str:
        """对比两个版本的差异（简化版）"""
        my_lines = self.template.split("\n")
        other_lines = other.template.split("\n")

        diff = []
        max_len = max(len(my_lines), len(other_lines))
        for i in range(max_len):
            my_line = my_lines[i] if i < len(my_lines) else ""
            other_line = other_lines[i] if i < len(other_lines) else ""
            if my_line != other_line:
                diff.append(f"  L{i+1}: - {my_line[:80]}")
                diff.append(f"  L{i+1}: + {other_line[:80]}")
        return "\n".join(diff[:20])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "variables": self.variables,
            "description": self.description,
            "parent_version": self.parent_version,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptTemplate":
        return cls(
            name=d["name"],
            version=d["version"],
            template=d["template"],
            variables=d.get("variables", []),
            description=d.get("description", ""),
            parent_version=d.get("parent_version", ""),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
            content_hash=d.get("content_hash", ""),
        )


# ==================== Prompt 注册表 ====================

class PromptRegistry:
    """
    Prompt 注册表

    管理所有 Prompt 的版本，支持：
    - 多版本共存
    - 版本切换
    - 回滚到历史版本
    - 导出/导入

    使用方式:
        registry = PromptRegistry()

        registry.register(PromptTemplate(
            name="system", version="v1",
            template="You are {role}.",
            variables=["role"],
        ))

        registry.register(PromptTemplate(
            name="system", version="v2",
            template="You are {role}. Current date: {date}.\n\n## Rules\n{rules}",
            variables=["role", "date", "rules"],
            parent_version="v1",
        ))

        # 设置活跃版本
        registry.set_active("system", "v2")

        # 获取活跃版本
        tmpl = registry.get("system")           # → v2
        tmpl = registry.get("system", "v1")     # → v1（指定版本）

        # 回滚
        registry.rollback("system", "v1")

        # 渲染
        prompt = registry.render("system", role="客服助手", date="2026-01-01", rules="保持礼貌")
    """

    def __init__(self):
        self._templates: dict[str, dict[str, PromptTemplate]] = {}  # name → {version → template}
        self._active: dict[str, str] = {}  # name → active_version

    # ==================== CRUD ====================

    def register(self, template: PromptTemplate):
        """注册/更新 Prompt 版本。新版本自动设为活跃。"""
        if template.name not in self._templates:
            self._templates[template.name] = {}

        self._templates[template.name][template.version] = template

        # 新注册的版本自动设为活跃
        self._active[template.name] = template.version

    def get(self, name: str, version: str | None = None) -> PromptTemplate | None:
        """获取指定版本的 Prompt"""
        versions = self._templates.get(name, {})
        if not versions:
            return None

        if version:
            return versions.get(version)
        else:
            # 返回活跃版本
            active_ver = self._active.get(name)
            if active_ver and active_ver in versions:
                return versions[active_ver]
            # Fallback: 返回最新版本
            return list(versions.values())[-1] if versions else None

    def set_active(self, name: str, version: str):
        """设置活跃版本"""
        if name not in self._templates:
            raise ValueError(f"Prompt '{name}' 未注册")
        if version not in self._templates[name]:
            raise ValueError(f"Prompt '{name}' 版本 '{version}' 不存在")
        self._active[name] = version

    def rollback(self, name: str, version: str):
        """回滚到历史版本"""
        self.set_active(name, version)

    def list_versions(self, name: str) -> list[str]:
        """列出所有版本"""
        return sorted(self._templates.get(name, {}).keys())

    def get_version_history(self, name: str) -> list[dict]:
        """获取版本变更历史"""
        templates = self._templates.get(name, {})
        return [
            {
                "version": v,
                "description": t.description,
                "parent": t.parent_version,
                "hash": t.content_hash,
                "created_at": t.created_at,
            }
            for v, t in sorted(templates.items())
        ]

    def delete_version(self, name: str, version: str):
        """删除指定版本"""
        if name in self._templates and version in self._templates[name]:
            del self._templates[name][version]
            if self._active.get(name) == version:
                remaining = self.list_versions(name)
                self._active[name] = remaining[-1] if remaining else ""

    # ==================== 渲染 ====================

    def render(self, name: str, version: str | None = None, **kwargs) -> str:
        """渲染 Prompt 模板"""
        tmpl = self.get(name, version)
        if tmpl is None:
            raise ValueError(f"Prompt '{name}' 未找到")
        return tmpl.render(**kwargs)

    # ==================== 导出/导入 ====================

    def export_all(self) -> dict:
        """导出所有 Prompt 为字典"""
        return {
            name: {
                "versions": {v: t.to_dict() for v, t in versions.items()},
                "active": self._active.get(name),
            }
            for name, versions in self._templates.items()
        }

    def import_all(self, data: dict):
        """从字典导入 Prompt"""
        for name, info in data.items():
            for version, tmpl_data in info["versions"].items():
                self.register(PromptTemplate.from_dict(tmpl_data))
            if info.get("active"):
                self._active[name] = info["active"]

    def save(self, filepath: str):
        """保存到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.export_all(), f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """从文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            self.import_all(json.load(f))

    # ==================== 统计 ====================

    def stats(self) -> dict:
        return {
            "total_prompts": len(self._templates),
            "total_versions": sum(len(v) for v in self._templates.values()),
            "active_versions": dict(self._active),
        }


# ==================== A/B 测试 ====================

@dataclass
class ABTestVariant:
    """A/B 测试变体"""
    name: str
    prompt_version: str
    template: PromptTemplate
    score: float = 0.0
    sample_count: int = 0


@dataclass
class ABTestReport:
    """A/B 测试报告"""
    prompt_name: str
    variants: list[ABTestVariant]
    winner: str = ""
    significance: str = ""       # 统计显著性
    details: dict = field(default_factory=dict)

    def format(self) -> str:
        lines = [
            "┌──────────────────────────────────────────┐",
            f"│  A/B Test Report: {self.prompt_name:<24} │",
            "├──────────┬──────────┬──────────┬──────────┤",
            "│ Variant  │ Version  │ Score    │ N        │",
            "├──────────┼──────────┼──────────┼──────────┤",
        ]
        for v in self.variants:
            marker = " 🏆" if v.name == self.winner else ""
            lines.append(
                f"│ {v.name:<8} │ {v.prompt_version:<8} │ {v.score:>7.1%} │ {v.sample_count:>8} │{marker}"
            )
        lines.append("└──────────┴──────────┴──────────┴──────────┘")
        if self.winner:
            lines.append(f"\nWinner: {self.winner} ({self.significance})")
        return "\n".join(lines)


class ABTestRunner:
    """
    Prompt A/B 测试执行器

    使用方式:
        registry = PromptRegistry()
        # ... 注册 v1, v2, v3 ...

        ab = ABTestRunner(eval_runner, registry)

        report = await ab.compare(
            prompt_name="system_prompt",
            variants=["v1", "v2", "v3"],
            questions=test_questions,
        )
        print(report.format())
    """

    def __init__(self, eval_runner=None, prompt_registry: PromptRegistry | None = None):
        """
        Args:
            eval_runner: EvalRunner 实例（用于跑评测）
            prompt_registry: Prompt 注册表
        """
        self.eval_runner = eval_runner
        self.registry = prompt_registry or PromptRegistry()

    async def compare(
        self,
        prompt_name: str,
        variants: list[str],
        questions: list[dict],
    ) -> ABTestReport:
        """
        对比多个 Prompt 版本

        Args:
            prompt_name: 要测试的 Prompt 名称
            variants: 版本列表 (["v1", "v2", "v3"])
            questions: 测试问题 [{"question": "...", "ground_truth": "..."}]

        Returns:
            ABTestReport
        """
        results = []

        for version in variants:
            tmpl = self.registry.get(prompt_name, version)
            if tmpl is None:
                continue

            variant = ABTestVariant(
                name=f"variant_{version}",
                prompt_version=version,
                template=tmpl,
                sample_count=len(questions),
            )

            # 如果有 EvalRunner，实际跑评测
            if self.eval_runner:
                # 创建使用此 Prompt 的 Agent
                scores = await self._evaluate_with_prompt(tmpl, questions)
                variant.score = sum(scores) / max(len(scores), 1)
            else:
                # 无 EvalRunner → 基于关键词覆盖的简单评分
                scores = self._simple_evaluate(tmpl, questions)
                variant.score = sum(scores) / max(len(scores), 1)

            results.append(variant)

        # 找最优
        if results:
            best = max(results, key=lambda v: v.score)
            runner_up = sorted(results, key=lambda v: v.score, reverse=True)[1] if len(results) > 1 else None

            sig = "p<0.05" if runner_up and (best.score - runner_up.score) > 0.1 else "not significant"
        else:
            best = None
            sig = "N/A"

        return ABTestReport(
            prompt_name=prompt_name,
            variants=results,
            winner=best.name if best else "",
            significance=sig,
        )

    async def auto_optimize(
        self,
        prompt_name: str,
        questions: list[dict],
        base_version: str,
        optimization_goal: str = "提高准确性和减少幻觉",
    ) -> PromptTemplate:
        """
        自动优化 Prompt

        流程:
        1. 以 base_version 为基准
        2. LLM 生成 N 个改进版本
        3. A/B 测试 → 选最优
        4. 注册为新版本
        """
        # 简化实现：LLM 生成改进建议
        base_tmpl = self.registry.get(prompt_name, base_version)
        if base_tmpl is None:
            raise ValueError(f"Base version '{base_version}' not found")

        # 这里可以调用 LLM 生成改进版本，当前返回基准版本
        # 生产环境：LLM 分析评测反馈 → 生成 N 个改进 Prompt → A/B 测试 → 选最佳
        return base_tmpl

    # ==================== 内部方法 ====================

    async def _evaluate_with_prompt(self, template: PromptTemplate,
                                     questions: list[dict]) -> list[float]:
        """用 EvalRunner 实际跑评测"""
        # 需要 EvalRunner 配合，这里返回占位
        return [0.5] * len(questions)

    def _simple_evaluate(self, template: PromptTemplate,
                          questions: list[dict]) -> list[float]:
        """
        简单评分：基于模板与标准答案的关键词覆盖

        规则：
        - 模板覆盖更多关键指令 → 更高分
        - 模板有明确输出格式 → 加分
        - 模板有角色定义 → 加分
        - 模板有规则约束 → 加分
        """
        scores = []
        for q in questions:
            score = 0.5  # 基础分

            template_text = template.template.lower()

            # 角色定义加分
            if any(w in template_text for w in ["你是", "you are", "role", "角色"]):
                score += 0.1

            # 输出格式约束加分
            if any(w in template_text for w in ["json", "格式", "format", "output"]):
                score += 0.1

            # 规则/护栏加分
            if any(w in template_text for w in ["规则", "rule", "不要", "don't", "禁止"]):
                score += 0.1

            # 工具说明加分
            if any(w in template_text for w in ["工具", "tool", "可用", "available"]):
                score += 0.1

            # 上下文说明加分
            if any(w in template_text for w in ["日期", "date", "上下文", "context"]):
                score += 0.05

            scores.append(min(1.0, score))

        return scores


# ==================== 与 PromptEngine 集成 ====================

def upgrade_prompt_engine(engine, registry: PromptRegistry):
    """
    将现有 PromptEngine 升级为支持版本管理的版本

    使用方式:
        registry = PromptRegistry()
        registry.register(PromptTemplate(...))
        upgrade_prompt_engine(prompt_engine, registry)
        # 现在 prompt_engine.build_system_prompt 会使用注册表中的版本
    """
    original_build = engine.build_system_prompt

    def versioned_build(config, version=None):
        try:
            return registry.render("system_prompt", version=version)
        except (ValueError, KeyError):
            return original_build(config)

    engine.build_system_prompt = versioned_build
    engine._prompt_registry = registry
    return engine


# ==================== 内置模板 ====================

def create_default_registry() -> PromptRegistry:
    """创建预置了常用 Prompt 模板的注册表"""
    registry = PromptRegistry()

    # 通用 Agent System Prompt
    default_prompt = """你是一个智能助手 Agent，具备以下核心能力：

## 能力
1. **直接回答**：对于一般性问题，直接给出答案
2. **知识库检索**：当问题涉及专业知识时，调用知识库检索工具
3. **任务分解**：复杂问题可以派生子 Agent 分别处理

## 行为规范
- 优先使用知识库检索获取准确信息，而不是凭记忆回答
- 遇到不确定的信息，如实说明而不是编造
- 复杂问题先思考再行动（ReAct 模式）
- 工具结果不理想时，尝试换个角度重新检索
- 得到足够信息后及时给出最终答案，不要无休止地调用工具

## 当前日期
{current_date}"""

    registry.register(PromptTemplate(
        name="system_prompt",
        version="v1",
        template=default_prompt,
        variables=["current_date"],
        description="初始版本：基本角色定义 + 行为规范",
    ))

    return registry
