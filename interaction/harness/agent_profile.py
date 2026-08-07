"""
CC-Harness Agent — Agent 配置中心 & 构建工厂
==============================================
从"写代码构建 Agent"升级为"配置驱动构建 Agent"。

核心理念（对标 OpenAI GPTs / Claude Custom Styles）:
    每个 Agent 由一份 Profile 定义——人设、技能、工具、护栏、记忆……
    一份 YAML 配置文件 = 一个完整的、可直接运行的 Agent。

架构:
    AgentProfile (配置模型)
        ├── Persona        — 人设 / System Prompt
        ├── Skills         — 可复用技能模块（内置 + 自定义）
        ├── ToolConfig     — 本地工具 + MCP Server
        ├── GuardConfig    — 输入/输出/工具护栏
        ├── HITLConfig     — 人工审批策略
        ├── MemoryConfig   — 工作记忆 + 向量记忆
        ├── ModelConfig    — LLM 模型 + 降级链
        └── LoopConfig     — 循环控制参数

    AgentBuilder (构建工厂)
        └── Profile → 完整的 AgentLoop 实例

    SkillLibrary (技能库)
        └── 预置技能: web_researcher, code_reviewer, data_analyst...

使用方式:
    # 方式 1: YAML 配置文件
    agent = await AgentBuilder.from_yaml("profiles/customer_support.yaml")

    # 方式 2: 代码构建
    profile = AgentProfile(
        name="客服助手",
        persona="你是一个专业客服...",
        skills=["rag_search", "sentiment_analysis"],
        tools=["rag_search", "query_order"],
        mcp_servers=[MCPServerConfig(name="crm", ...)],
    )
    agent = await AgentBuilder.build(profile)

    # 方式 3: CLI
    python -m harness.agent_profile --profile profiles/customer_support.yaml
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml  # PyYAML

from harness.types import AgentConfig, Message, MessageRole
from harness.mcp.registry import MCPServerConfig


# ==================== Skill 系统 ====================

@dataclass
class SkillDefinition:
    """
    可复用技能模块

    一个 Skill 封装了：人设片段 + 推荐工具 + 可选 MCP Server + 可选护栏规则

    类似 Claude Code 的 Skill 概念：
    - 每个 Skill 是 Agent 的一个"能力单元"
    - 可以组合多个 Skill 构建复合 Agent
    """
    name: str                                    # 技能名 (唯一标识)
    description: str = ""                        # 技能描述
    prompt_snippet: str = ""                     # 注入 system prompt 的片段
    recommended_tools: list[str] = field(default_factory=list)  # 推荐工具
    recommended_mcp: list[str] = field(default_factory=list)    # 推荐 MCP Server 名
    guardrails: dict = field(default_factory=dict)              # 护栏配置覆盖
    icon: str = ""                               # 图标 emoji
    tags: list[str] = field(default_factory=list)


class SkillLibrary:
    """
    技能库

    管理所有可用 Skill，支持注册、发现、组合。

    使用方式:
        lib = SkillLibrary()
        lib.register(SkillDefinition(name="rag_search", ...))
        skills = lib.match(["rag_search", "code_review"])
        combined_prompt = lib.combine_prompts(skills)
    """

    def __init__(self):
        self._skills: dict[str, SkillDefinition] = {}
        self._register_builtins()

    def register(self, skill: SkillDefinition):
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def match(self, names: list[str]) -> list[SkillDefinition]:
        """按名称获取多个 Skill"""
        return [self._skills[n] for n in names if n in self._skills]

    def search(self, query: str) -> list[SkillDefinition]:
        """搜索匹配的 Skill（按名称/描述/标签）"""
        q = query.lower()
        results = []
        for s in self._skills.values():
            if q in s.name.lower() or q in s.description.lower() or any(q in t for t in s.tags):
                results.append(s)
        return results

    def combine_prompts(self, skills: list[SkillDefinition]) -> str:
        """合并多个 Skill 的 prompt 片段"""
        parts = []
        for s in skills:
            if s.prompt_snippet:
                parts.append(f"## {s.icon} {s.name}\n{s.prompt_snippet}")
        return "\n\n".join(parts)

    def list_all(self) -> list[str]:
        return list(self._skills.keys())

    # ── 内置技能 ──

    def _register_builtins(self):
        """注册内置技能库"""
        builtins = [
            SkillDefinition(
                name="web_researcher",
                description="网络调研：搜索网页、提取信息、总结报告",
                icon="🔍",
                prompt_snippet=(
                    "你擅长网络调研。对于任何问题，首先搜索多个来源获取信息，"
                    "交叉验证关键事实，最后给出结构化的调研报告并附上引用来源。"
                ),
                recommended_tools=["web_search", "fetch_url", "rag_search"],
                recommended_mcp=["brave-search"],
                tags=["research", "search", "web"],
            ),
            SkillDefinition(
                name="code_reviewer",
                description="代码审查：分析代码质量、安全漏洞、性能问题",
                icon="📋",
                prompt_snippet=(
                    "你是资深代码审查专家。审查时关注：正确性（逻辑错误）、"
                    "安全性（注入/SQL注入/XSS）、性能（算法复杂度/冗余计算）、"
                    "可维护性（命名/注释/模块化）。输出结构化的审查报告。"
                ),
                recommended_tools=["read_file", "grep_search", "rag_search"],
                tags=["code", "review", "security"],
            ),
            SkillDefinition(
                name="data_analyst",
                description="数据分析：处理CSV/JSON数据，生成统计报告和可视化建议",
                icon="📊",
                prompt_snippet=(
                    "你是数据分析专家。收到数据后先做探索性分析（行数/列/类型/缺失值），"
                    "再做统计分析（分布/相关性/趋势），最后给出业务洞察和可视化建议。"
                ),
                recommended_tools=["run_code", "read_file", "rag_search"],
                tags=["data", "analysis", "statistics"],
            ),
            SkillDefinition(
                name="customer_support",
                description="客服助手：礼貌、耐心、高效地解决用户问题",
                icon="🎧",
                prompt_snippet=(
                    "你是专业客服代表。原则：1) 先共情再解决问题 2) 用简单语言解释技术问题 "
                    "3) 提供分步骤的解决方案 4) 必要时主动升级给人工。语气：温暖、专业、有耐心。"
                ),
                recommended_tools=["rag_search", "query_knowledge_base", "create_ticket"],
                tags=["support", "customer", "service"],
            ),
            SkillDefinition(
                name="security_auditor",
                description="安全审计：检测系统配置、代码和架构中的安全隐患",
                icon="🔒",
                prompt_snippet=(
                    "你是安全审计专家。审计维度：认证授权、数据加密、输入验证、"
                    "依赖漏洞、配置安全。对每个发现给出 CVSS 评分和修复建议。"
                ),
                recommended_tools=["read_file", "grep_search", "run_code"],
                guardrails={"input": {"strict_mode": True}, "output": {"no_leak": True}},
                tags=["security", "audit", "compliance"],
            ),
            SkillDefinition(
                name="doc_writer",
                description="文档撰写：生成技术文档、API文档、用户手册",
                icon="📝",
                prompt_snippet=(
                    "你是技术文档撰写专家。文档要求：清晰的结构（概述→快速开始→详细说明→FAQ）、"
                    "准确的术语、丰富的代码示例、版本和日期标注。"
                ),
                recommended_tools=["rag_search", "read_file"],
                tags=["writing", "documentation", "technical"],
            ),
        ]

        for skill in builtins:
            self.register(skill)


# 全局技能库单例
skill_library = SkillLibrary()


# ==================== Agent 配置模型 ====================

@dataclass
class PersonaConfig:
    """人设配置"""
    name: str = "Assistant"                    # Agent 名称
    role: str = "通用助手"                      # 角色描述
    tone: str = "professional"                 # 语气: professional/casual/friendly/strict
    language: str = "zh"                       # 主要语言
    constraints: list[str] = field(default_factory=list)  # 行为约束
    examples: list[dict] = field(default_factory=list)    # Few-shot 示例


@dataclass
class ToolConfig:
    """工具配置"""
    local_tools: list[str] = field(default_factory=list)    # 本地工具名
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)  # MCP Server
    auto_discover_mcp: bool = True                           # 自动发现 MCP 工具


@dataclass
class GuardConfig:
    """护栏配置"""
    enable_input_guard: bool = True
    enable_tool_guard: bool = True
    enable_output_guard: bool = True
    block_on_high_risk: bool = True
    auto_approve_read: bool = True          # 只读工具自动批准
    pii_detection: bool = True              # PII 检测
    injection_detection: bool = True        # 注入检测
    harmful_content_filter: bool = True     # 有害内容过滤


@dataclass
class HITLConfig:
    """人机协同配置"""
    enabled: bool = True
    timeout_seconds: float = 120.0
    auto_approve_read: bool = True
    reject_on_timeout: bool = True
    approval_handlers: dict[str, str] = field(default_factory=dict)  # tool_name → handler


@dataclass
class MemoryConfig:
    """记忆配置"""
    working_memory_size: int = 7
    vector_memory_enabled: bool = False
    vector_memory_dir: str = "./memory_db"
    cross_session_memory: bool = False      # 是否跨会话共享记忆


@dataclass
class ModelConfig:
    """模型配置"""
    model: str = "qwen-plus"
    provider: str = "dashscope"
    temperature: float = 0.0
    max_tokens: int = 4096
    fallback_models: list[str] = field(default_factory=list)


@dataclass
class LoopConfig:
    """循环控制配置"""
    max_turns: int = 10
    max_context_tokens: int = 8000
    enable_subagents: bool = True
    stream_output: bool = True


@dataclass
class AgentProfile:
    """
    Agent 完整配置

    一份 Profile = 一个可直接构建的 Agent。

    使用方式:
        # 从 YAML 加载
        profile = AgentProfile.from_yaml("profiles/customer_support.yaml")

        # 代码构建
        profile = AgentProfile(
            persona=PersonaConfig(name="客服小助手", role="电商客服"),
            skills=["customer_support", "web_researcher"],
            tools=ToolConfig(local_tools=["rag_search", "query_order"]),
        )
        agent = await AgentBuilder.build(profile)
        answer = await agent.run()
    """
    # 基本信息
    name: str = "default"
    version: str = "1.0"
    description: str = ""

    # 核心配置
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    skills: list[str] = field(default_factory=list)        # 技能名列表
    tools: ToolConfig = field(default_factory=ToolConfig)
    guardrails: GuardConfig = field(default_factory=GuardConfig)
    hitl: HITLConfig = field(default_factory=HITLConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── 序列化 ──

    def to_dict(self) -> dict:
        """导出为字典（可保存为 YAML/JSON）"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "persona": {
                "name": self.persona.name,
                "role": self.persona.role,
                "tone": self.persona.tone,
                "language": self.persona.language,
                "constraints": self.persona.constraints,
                "examples": self.persona.examples,
            },
            "skills": self.skills,
            "tools": {
                "local_tools": self.tools.local_tools,
                "mcp_servers": [
                    {
                        "name": s.name,
                        "transport": s.transport,
                        "command": s.command,
                        "args": s.args,
                        "url": s.url,
                    }
                    for s in self.tools.mcp_servers
                ],
            },
            "guardrails": {
                "enable_input_guard": self.guardrails.enable_input_guard,
                "enable_tool_guard": self.guardrails.enable_tool_guard,
                "enable_output_guard": self.guardrails.enable_output_guard,
                "block_on_high_risk": self.guardrails.block_on_high_risk,
                "auto_approve_read": self.guardrails.auto_approve_read,
            },
            "hitl": {
                "enabled": self.hitl.enabled,
                "timeout_seconds": self.hitl.timeout_seconds,
                "auto_approve_read": self.hitl.auto_approve_read,
                "reject_on_timeout": self.hitl.reject_on_timeout,
            },
            "memory": {
                "working_memory_size": self.memory.working_memory_size,
                "vector_memory_enabled": self.memory.vector_memory_enabled,
                "vector_memory_dir": self.memory.vector_memory_dir,
                "cross_session_memory": self.memory.cross_session_memory,
            },
            "model": {
                "model": self.model.model,
                "provider": self.model.provider,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "fallback_models": self.model.fallback_models,
            },
            "loop": {
                "max_turns": self.loop.max_turns,
                "max_context_tokens": self.loop.max_context_tokens,
                "enable_subagents": self.loop.enable_subagents,
                "stream_output": self.loop.stream_output,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentProfile":
        """从字典加载"""
        p = d.get("persona", {})
        t = d.get("tools", {})
        g = d.get("guardrails", {})
        h = d.get("hitl", {})
        m = d.get("memory", {})
        md = d.get("model", {})
        l = d.get("loop", {})

        # MCP servers
        mcp_servers = []
        for s in (t.get("mcp_servers") or []):
            mcp_servers.append(MCPServerConfig(
                name=s.get("name", ""),
                transport=s.get("transport", "stdio"),
                command=s.get("command", ""),
                args=s.get("args", []),
                url=s.get("url", ""),
            ))

        return cls(
            name=d.get("name", "default"),
            version=d.get("version", "1.0"),
            description=d.get("description", ""),
            persona=PersonaConfig(
                name=p.get("name", "Assistant"),
                role=p.get("role", "通用助手"),
                tone=p.get("tone", "professional"),
                language=p.get("language", "zh"),
                constraints=p.get("constraints", []),
                examples=p.get("examples", []),
            ),
            skills=d.get("skills", []),
            tools=ToolConfig(
                local_tools=t.get("local_tools", []),
                mcp_servers=mcp_servers,
                auto_discover_mcp=t.get("auto_discover_mcp", True),
            ),
            guardrails=GuardConfig(
                enable_input_guard=g.get("enable_input_guard", True),
                enable_tool_guard=g.get("enable_tool_guard", True),
                enable_output_guard=g.get("enable_output_guard", True),
                block_on_high_risk=g.get("block_on_high_risk", True),
                auto_approve_read=g.get("auto_approve_read", True),
            ),
            hitl=HITLConfig(
                enabled=h.get("enabled", True),
                timeout_seconds=h.get("timeout_seconds", 120.0),
                auto_approve_read=h.get("auto_approve_read", True),
                reject_on_timeout=h.get("reject_on_timeout", True),
            ),
            memory=MemoryConfig(
                working_memory_size=m.get("working_memory_size", 7),
                vector_memory_enabled=m.get("vector_memory_enabled", False),
                vector_memory_dir=m.get("vector_memory_dir", "./memory_db"),
                cross_session_memory=m.get("cross_session_memory", False),
            ),
            model=ModelConfig(
                model=md.get("model", "qwen-plus"),
                provider=md.get("provider", "dashscope"),
                temperature=md.get("temperature", 0.0),
                max_tokens=md.get("max_tokens", 4096),
                fallback_models=md.get("fallback_models", []),
            ),
            loop=LoopConfig(
                max_turns=l.get("max_turns", 10),
                max_context_tokens=l.get("max_context_tokens", 8000),
                enable_subagents=l.get("enable_subagents", True),
                stream_output=l.get("stream_output", True),
            ),
            metadata=d.get("metadata", {}),
        )

    # ── 文件操作 ──

    @classmethod
    def from_yaml(cls, filepath: str) -> "AgentProfile":
        """从 YAML 文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, filepath: str) -> "AgentProfile":
        """从 JSON 文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_yaml(self, filepath: str):
        """保存为 YAML"""
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def to_json(self, filepath: str):
        """保存为 JSON"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


# ==================== Agent 构建工厂 ====================

class AgentBuilder:
    """
    Agent 构建工厂

    将 AgentProfile 组装为完整可运行的 AgentLoop 实例。

    构建流程:
    1. 创建 LLMAdapter（模型 + 降级链）
    2. 注册本地工具 + 连接 MCP Server
    3. 装配护栏（InputGuard / ToolGuard / OutputGuard）
    4. 初始化 HITL 审批管理器
    5. 初始化记忆系统
    6. 生成 System Prompt（人设 + 技能 + 工具说明）
    7. 创建 Session + AgentLoop
    8. 返回可用的 Agent 实例

    使用方式:
        profile = AgentProfile.from_yaml("profiles/researcher.yaml")
        agent = await AgentBuilder.build(profile)
        answer = await agent.run("什么是 MCP 协议？")
    """

    @classmethod
    async def build(cls, profile: AgentProfile) -> "ConfiguredAgent":
        """
        从 Profile 构建完整 Agent

        Returns:
            ConfiguredAgent: 包含 AgentLoop + 所有基础设施的完整 Agent 对象
        """
        # 1. LLM Adapter
        from harness.llm_adapter import create_llm_adapter, Provider
        try:
            provider = Provider(profile.model.provider)
        except ValueError:
            provider = None

        llm = create_llm_adapter(
            model=profile.model.model,
            provider=provider,
            fallback_models=profile.model.fallback_models,
        )

        # 2. 工具系统
        from tools import ToolRegistry, RAGTool
        tool_registry = ToolRegistry()

        # 注册本地 RAG 工具
        if "rag_search" in profile.tools.local_tools:
            rag = RAGTool(collection_name=f"agent_{profile.name}", k=3)
            tool_registry.register(rag)

        # MCP Server 连接
        mcp_registry = None
        if profile.tools.mcp_servers:
            from harness.mcp import MCPServerRegistry, register_mcp_tools
            mcp_registry = MCPServerRegistry()
            for server_cfg in profile.tools.mcp_servers:
                mcp_registry.register(server_cfg)
            await mcp_registry.connect_all()
            await register_mcp_tools(tool_registry, mcp_registry)

        # 3. 护栏系统
        from harness.guardrails import InputGuard, ToolGuard, OutputGuard

        input_guard = InputGuard(
            enable_injection_check=profile.guardrails.injection_detection,
            enable_pii_check=profile.guardrails.pii_detection,
            block_on_high_risk=profile.guardrails.block_on_high_risk,
        ) if profile.guardrails.enable_input_guard else None

        tool_guard = ToolGuard() if profile.guardrails.enable_tool_guard else None

        output_guard = OutputGuard(llm_adapter=llm) if profile.guardrails.enable_output_guard else None

        # 4. HITL
        from harness.hitl import HumanInTheLoop
        hitl = HumanInTheLoop(
            timeout_seconds=profile.hitl.timeout_seconds,
            auto_approve_read=profile.hitl.auto_approve_read,
            default_timeout_action="reject" if profile.hitl.reject_on_timeout else "approve",
        ) if profile.hitl.enabled else None

        # 5. 记忆系统
        from harness.memory import WorkingMemory
        working_memory = WorkingMemory(max_items=profile.memory.working_memory_size)
        vector_memory = None
        if profile.memory.vector_memory_enabled:
            from harness.memory import VectorMemory
            vector_memory = VectorMemory(persist_dir=profile.memory.vector_memory_dir)

        # 6. System Prompt 生成
        system_prompt = cls._build_system_prompt(profile)

        # 7. Session + AgentLoop
        from harness.session import Session
        from harness.agent_loop import AgentLoop
        from harness.prompt_engine import PromptEngine
        from harness.tracer import AgentTracer

        config = AgentConfig(
            max_turns=profile.loop.max_turns,
            model=profile.model.model,
            temperature=profile.model.temperature,
            max_context_tokens=profile.loop.max_context_tokens,
            enable_subagents=profile.loop.enable_subagents,
        )

        session = Session(config=config)
        session.set_system_prompt(system_prompt)

        prompt_engine = PromptEngine(tool_registry)
        tracer = AgentTracer(verbose=True)

        loop = AgentLoop(
            session=session,
            llm_adapter=llm,
            tool_registry=tool_registry,
            prompt_engine=prompt_engine,
            tracer=tracer,
        )

        # 8. 组装返回
        return ConfiguredAgent(
            profile=profile,
            loop=loop,
            session=session,
            llm=llm,
            tool_registry=tool_registry,
            mcp_registry=mcp_registry,
            input_guard=input_guard,
            tool_guard=tool_guard,
            output_guard=output_guard,
            hitl=hitl,
            working_memory=working_memory,
            vector_memory=vector_memory,
            tracer=tracer,
        )

    @classmethod
    def _build_system_prompt(cls, profile: AgentProfile) -> str:
        """根据 Profile 生成完整的 System Prompt"""
        parts = []

        # 1. 角色定义
        parts.append(f"# {profile.persona.name}")
        parts.append(f"你是{profile.persona.role}。")

        # 2. 人设语气
        tone_map = {
            "professional": "请保持专业、严谨的语气。",
            "casual": "请保持轻松、友好的语气。",
            "friendly": "请保持温暖、友善的语气。",
            "strict": "请严格遵循规则，不进行任何越权的操作。",
        }
        parts.append(tone_map.get(profile.persona.tone, ""))

        # 3. 技能注入
        if profile.skills:
            skills = skill_library.match(profile.skills)
            skills_prompt = skill_library.combine_prompts(skills)
            if skills_prompt:
                parts.append("\n## 专业技能")
                parts.append(skills_prompt)

        # 4. 行为约束
        if profile.persona.constraints:
            parts.append("\n## 行为约束")
            for c in profile.persona.constraints:
                parts.append(f"- {c}")

        # 5. Few-shot 示例
        if profile.persona.examples:
            parts.append("\n## 对话示例")
            for ex in profile.persona.examples[:3]:
                parts.append(f"用户: {ex.get('user', '')}")
                parts.append(f"助手: {ex.get('assistant', '')}")

        # 6. 当前日期
        from datetime import date
        parts.append(f"\n当前日期: {date.today().isoformat()}")

        return "\n\n".join(parts)

    @classmethod
    async def from_yaml(cls, filepath: str) -> "ConfiguredAgent":
        """从 YAML 配置文件一键构建 Agent"""
        profile = AgentProfile.from_yaml(filepath)
        return await cls.build(profile)

    @classmethod
    async def from_json(cls, filepath: str) -> "ConfiguredAgent":
        """从 JSON 配置文件一键构建 Agent"""
        profile = AgentProfile.from_json(filepath)
        return await cls.build(profile)


# ==================== 构建产物 ====================

@dataclass
class ConfiguredAgent:
    """
    构建完成的 Agent 实例

    包含 AgentLoop 及其所有依赖的完整引用，
    可直接调用 run() 或通过 API 暴露。

    使用方式:
        agent = await AgentBuilder.build(profile)
        answer = await agent.run("用户问题")
        # 或获取内部状态
        print(agent.status())
    """
    profile: AgentProfile
    loop: Any                                    # AgentLoop
    session: Any                                 # Session
    llm: Any                                     # LLMAdapter
    tool_registry: Any                           # ToolRegistry
    mcp_registry: Any = None                     # MCPServerRegistry
    input_guard: Any = None                      # InputGuard
    tool_guard: Any = None                       # ToolGuard
    output_guard: Any = None                     # OutputGuard
    hitl: Any = None                             # HumanInTheLoop
    working_memory: Any = None                   # WorkingMemory
    vector_memory: Any = None                    # VectorMemory
    tracer: Any = None                           # AgentTracer

    async def run(self, question: str) -> str:
        """执行一次问答"""
        # 输入护栏
        if self.input_guard:
            result = self.input_guard.check(question)
            if not result.safe and result.risk_level in ("high", "critical"):
                return f"抱歉，您的输入被安全护栏拦截：{result.reason}"
            question = result.sanitized_input or question

        # 添加到工作记忆
        if self.working_memory:
            self.working_memory.add(question, category="fact", importance=0.3)

        # 注入工作记忆到 system prompt
        if self.working_memory and self.working_memory.size > 0:
            memory_text = self.working_memory.to_prompt_text()
            current_prompt = self.session._system_prompt
            if memory_text not in current_prompt:
                self.session.set_system_prompt(current_prompt + "\n\n" + memory_text)

        # 执行 AgentLoop
        self.session.append_user_message(question)
        answer = await self.loop.run()

        # 输出护栏
        if self.output_guard:
            check = self.output_guard.basic_check(answer)
            if check.recommended_action == "block":
                return "抱歉，生成的回答未通过安全审查。"
            if check.recommended_action == "rewrite":
                # 简单改写：追加免责声明
                answer += "\n\n*以上回答可能存在不确定信息，请以实际情况为准。*"

        return answer

    def status(self) -> dict:
        """获取 Agent 状态摘要"""
        return {
            "name": self.profile.name,
            "model": self.profile.model.model,
            "skills": self.profile.skills,
            "tools_count": len(self.tool_registry.list_tool_names()) if self.tool_registry else 0,
            "mcp_servers": self.mcp_registry.connected_count if self.mcp_registry else 0,
            "guards": {
                "input": self.input_guard is not None,
                "tool": self.tool_guard is not None,
                "output": self.output_guard is not None,
            },
            "hitl_enabled": self.hitl is not None,
            "working_memory_size": self.working_memory.size if self.working_memory else 0,
            "session_turns": self.session.total_turns if self.session else 0,
            "llm_stats": self.llm.get_usage_summary() if self.llm else {},
        }


# ==================== CLI ====================

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python -m harness.agent_profile <profile.yaml>")
            print("\nAvailable skills:")
            for name in skill_library.list_all():
                s = skill_library.get(name)
                print(f"  {s.icon} {name} — {s.description}")
            return

        profile_path = sys.argv[1]
        print(f"Loading profile: {profile_path}")
        agent = await AgentBuilder.from_yaml(profile_path)
        print(f"Agent ready: {agent.status()}")

        # 交互式对话
        print("\nAgent ready. Type your question (or /exit):")
        while True:
            try:
                q = input("> ").strip()
                if q.lower() in ("/exit", "/quit"):
                    break
                if not q:
                    continue
                answer = await agent.run(q)
                print(f"\n{answer}\n")
            except (EOFError, KeyboardInterrupt):
                break

    asyncio.run(main())
