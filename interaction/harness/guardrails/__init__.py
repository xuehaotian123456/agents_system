"""
CC-Harness Agent — 安全护栏 (Guardrails)
==========================================
三层防护体系，覆盖 Agent 的输入、工具调用、输出全链路。

层级架构:
    ┌──────────────────────────────────────────────┐
    │  Layer 1: InputGuard                          │
    │  - 注入攻击检测 (Prompt Injection)              │
    │  - 越狱检测 (Jailbreak)                        │
    │  - PII/敏感信息识别                             │
    │  - 输入长度/复杂度限制                           │
    ├──────────────────────────────────────────────┤
    │  Layer 2: ToolGuard                           │
    │  - 工具权限分级 (READ/WRITE/ADMIN)              │
    │  - 参数白名单/黑名单                             │
    │  - 敏感操作需人工确认                            │
    │  - 调用频率限制                                 │
    ├──────────────────────────────────────────────┤
    │  Layer 3: OutputGuard                         │
    │  - 幻觉检测 (事实一致性)                         │
    │  - 有害内容过滤                                 │
    │  - 敏感信息泄露检测                              │
    │  - 回答格式/长度校验                             │
    └──────────────────────────────────────────────┘

参考:
    - Anthropic Constitutional AI
    - OpenAI Moderation API
    - Nvidia NeMo Guardrails
    - Guardrails AI (guardrails-ai)
"""

from harness.guardrails.input_filter import InputGuard, InputCheckResult
from harness.guardrails.tool_guard import ToolGuard, ToolPermission, ToolCheckResult
from harness.guardrails.output_validator import OutputGuard, OutputCheckResult

__all__ = [
    "InputGuard",
    "InputCheckResult",
    "ToolGuard",
    "ToolPermission",
    "ToolCheckResult",
    "OutputGuard",
    "OutputCheckResult",
]
