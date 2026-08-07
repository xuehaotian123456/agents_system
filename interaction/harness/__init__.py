"""
CC-Harness Agent - 智能体驾驭层

基于 Claude Code Harness 架构思想的轻量级 Agent 框架。
无 LangChain / LangGraph 依赖，纯异步原生实现。
"""

from harness.agent_loop import AgentLoop
from harness.session import Session
from harness.llm_adapter import (
    LLMAdapter,
    Provider,
    ModelInfo,
    StreamCallbacks,
    estimate_tokens,
    create_llm_adapter,
    BUILTIN_MODELS,
)
from harness.prompt_engine import PromptEngine
from harness.context_engine import ContextEngine, TokenBudget, CompressionResult
from harness.tracer import AgentTracer, TraceEvent, TraceEventType, create_tracer
from harness.multi_agent import (
    DebateOrchestrator,
    MapReduceOrchestrator,
    HierarchyOrchestrator,
    CollaborationResult,
    SubResult,
    create_collaboration,
)
from harness.types import AgentConfig, AgentAction, Message, MessageRole

# guardrails
from harness.guardrails import InputGuard, ToolGuard, ToolPermission, OutputGuard
# human-in-the-loop
from harness.hitl import (
    HumanInTheLoop, ApprovalRequest, ApprovalResult, ApprovalStatus,
)
# prompt version management
from harness.prompt_manager import (
    PromptTemplate, PromptRegistry, ABTestRunner, ABTestReport,
)
# memory
from harness.memory import WorkingMemory, MemoryItem, VectorMemory
# agent profile
from harness.agent_profile import (
    AgentProfile, AgentBuilder, ConfiguredAgent,
    PersonaConfig, ToolConfig, GuardConfig, HITLConfig, MemoryConfig,
    ModelConfig, LoopConfig, SkillDefinition, SkillLibrary, skill_library,
)

__all__ = [
    # Core
    "AgentLoop",
    "Session",
    "LLMAdapter",
    "Provider",
    "ModelInfo",
    "StreamCallbacks",
    "estimate_tokens",
    "create_llm_adapter",
    "BUILTIN_MODELS",
    "PromptEngine",
    "ContextEngine",
    "TokenBudget",
    "CompressionResult",
    "AgentTracer",
    "TraceEvent",
    "TraceEventType",
    "create_tracer",
    # Multi-Agent
    "DebateOrchestrator",
    "MapReduceOrchestrator",
    "HierarchyOrchestrator",
    "CollaborationResult",
    "SubResult",
    "create_collaboration",
    # Guardrails
    "InputGuard",
    "ToolGuard",
    "ToolPermission",
    "OutputGuard",
    # HITL
    "HumanInTheLoop",
    "ApprovalRequest",
    "ApprovalResult",
    "ApprovalStatus",
    "ApprovalTimeout",
    "ApprovalRejected",
    # Prompt Management
    "PromptTemplate",
    "PromptRegistry",
    "ABTestRunner",
    "ABTestReport",
    "create_default_registry",
    "upgrade_prompt_engine",
    # Memory
    "WorkingMemory",
    "MemoryItem",
    "VectorMemory",
    # Agent Profile
    "AgentProfile",
    "AgentBuilder",
    "ConfiguredAgent",
    "PersonaConfig",
    "ToolConfig",
    "GuardConfig",
    "HITLConfig",
    "MemoryConfig",
    "ModelConfig",
    "LoopConfig",
    "SkillDefinition",
    "SkillLibrary",
    "skill_library",
    # Types
    "AgentConfig",
    "AgentAction",
    "Message",
    "MessageRole",
]
