"""
Layer 2: 工具安全护栏 (Tool Guard)

Agent 调用工具前的最后一道防线。

核心设计:
    1. 工具分级 — READ(只读) / WRITE(写入) / ADMIN(管理)
    2. 权限检查 — 当前 Session 是否有权调用此工具
    3. 参数净化 — 过滤危险参数（如 rm -rf / SQL 注入）
    4. 频率限制 — 防止短时间大量调用同一工具
    5. 敏感操作确认 — WRITE/ADMIN 操作需 Human-in-the-Loop 审批

使用方式:
    guard = ToolGuard()
    guard.register_policy("delete_record", ToolPermission.ADMIN, requires_approval=True)

    result = guard.check("delete_record", {"id": "123"}, session)
    if not result.allowed:
        return f"工具调用被拦截: {result.reason}"
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ToolPermission(str, Enum):
    READ = "read"         # 只读：search, query, get
    WRITE = "write"       # 写入：create, update, send
    ADMIN = "admin"       # 管理：delete, config, execute


@dataclass
class ToolCheckResult:
    """工具调用检查结果"""
    allowed: bool = True
    reason: str = ""
    permission: ToolPermission = ToolPermission.READ
    requires_approval: bool = False     # 是否需要人工确认
    sanitized_args: dict = field(default_factory=dict)  # 净化后的参数
    blocked_patterns: list[str] = field(default_factory=list)


@dataclass
class ToolPolicy:
    """工具安全策略"""
    tool_name: str
    permission: ToolPermission = ToolPermission.READ
    requires_approval: bool = False       # 是否需要人工确认
    rate_limit_per_minute: int = 30       # 每分钟最大调用次数
    allowed_args: list[str] | None = None  # 允许的参数名（白名单）
    blocked_args: list[str] | None = None  # 禁止的参数名（黑名单）
    max_arg_length: int = 5000            # 参数值最大长度
    dangerous_patterns: list[str] = field(default_factory=list)  # 危险参数模式


class ToolGuard:
    """
    工具安全护栏

    使用方式:
        guard = ToolGuard()

        # 注册策略
        guard.register_policy(ToolPolicy(
            tool_name="execute_shell",
            permission=ToolPermission.ADMIN,
            requires_approval=True,
            rate_limit_per_minute=1,
            dangerous_patterns=[r"rm\s+-rf", r"DROP\s+TABLE"],
        ))

        # 检查
        result = guard.check("execute_shell", {"command": "ls -la"})
        if result.allowed:
            result = tool_registry.execute(name, result.sanitized_args)
    """

    # 内置敏感参数模式
    BUILTIN_DANGEROUS_PATTERNS = [
        r"(?:rm\s+-rf|sudo\s+rm)",            # 删除命令
        r"(?:DROP\s+TABLE|DROP\s+DATABASE)",   # SQL 删库
        r"(?:DELETE\s+FROM\s+\w+\s*$)",        # SQL 全表删除
        r"(?:/etc/passwd|/etc/shadow)",        # 系统敏感文件
        r"(?:\$\{.*\}|\$\(.*\))",              # Shell 变量注入
        r"(?:;.*rm\s|&&.*rm\s|\|.*rm\s)",      # Shell 命令链注入
        r"<script.*>",                          # XSS
        r"\.\./\.\./",                          # 路径穿越
    ]

    def __init__(self):
        self._policies: dict[str, ToolPolicy] = {}
        self._call_history: dict[str, list[float]] = defaultdict(list)
        self._approval_callbacks: dict[str, Any] = {}  # 审批回调注册

    # ==================== 策略管理 ====================

    def register_policy(self, policy: ToolPolicy):
        """注册工具安全策略"""
        self._policies[policy.tool_name] = policy

    def auto_classify(self, tool_name: str, tool_description: str = "") -> ToolPolicy:
        """
        根据工具名和描述自动推断安全策略

        启发式规则：
        - search/query/get/list → READ
        - create/update/send/write → WRITE (requires_approval)
        - delete/execute/admin/config → ADMIN (requires_approval)
        """
        name_lower = tool_name.lower() + " " + tool_description.lower()

        # ADMIN 操作
        admin_keywords = ["delete", "remove", "execute", "exec", "admin",
                          "config", "configure", "sudo", "root", "grant"]
        if any(kw in name_lower for kw in admin_keywords):
            return ToolPolicy(
                tool_name=tool_name,
                permission=ToolPermission.ADMIN,
                requires_approval=True,
                rate_limit_per_minute=5,
            )

        # WRITE 操作
        write_keywords = ["create", "update", "write", "send", "post",
                          "put", "patch", "insert", "modify", "change",
                          "deploy", "publish", "upload"]
        if any(kw in name_lower for kw in write_keywords):
            return ToolPolicy(
                tool_name=tool_name,
                permission=ToolPermission.WRITE,
                requires_approval=True,
                rate_limit_per_minute=20,
            )

        # 默认 READ
        return ToolPolicy(
            tool_name=tool_name,
            permission=ToolPermission.READ,
            requires_approval=False,
            rate_limit_per_minute=60,
        )

    # ==================== 安全检查 ====================

    def check(self, tool_name: str, args: dict, session=None) -> ToolCheckResult:
        """
        检查工具调用是否允许

        Args:
            tool_name: 工具名
            args: 调用参数
            session: 当前会话（可选，用于权限上下文）

        Returns:
            ToolCheckResult: 检查结果
        """
        # 获取策略（如果没有注册，自动推断）
        policy = self._policies.get(tool_name)
        if policy is None:
            policy = self.auto_classify(tool_name)
            self._policies[tool_name] = policy

        # 1. 频率限制检查
        freq_check = self._check_rate_limit(tool_name, policy)
        if not freq_check:
            return ToolCheckResult(
                allowed=False,
                permission=policy.permission,
                reason=f"工具 '{tool_name}' 调用频率超限 (限制 {policy.rate_limit_per_minute}/分钟)",
            )

        # 2. 参数安全检查
        sanitized_args, blocked = self._sanitize_args(args, policy)
        if blocked:
            return ToolCheckResult(
                allowed=False,
                permission=policy.permission,
                reason=f"工具 '{tool_name}' 参数包含危险内容: {', '.join(blocked[:3])}",
                blocked_patterns=blocked,
            )

        # 3. 敏感操作需要确认
        result = ToolCheckResult(
            allowed=True,
            permission=policy.permission,
            requires_approval=policy.requires_approval,
            sanitized_args=sanitized_args,
        )

        return result

    # ==================== 内部方法 ====================

    def _check_rate_limit(self, tool_name: str, policy: ToolPolicy) -> bool:
        """频率限制检查（滑动窗口）"""
        now = time.time()
        window_start = now - 60  # 1分钟窗口

        # 清理过期记录
        self._call_history[tool_name] = [
            t for t in self._call_history[tool_name] if t > window_start
        ]

        # 检查是否超限
        if len(self._call_history[tool_name]) >= policy.rate_limit_per_minute:
            return False

        # 记录本次调用
        self._call_history[tool_name].append(now)
        return True

    def _sanitize_args(self, args: dict, policy: ToolPolicy) -> tuple[dict, list[str]]:
        """
        参数净化和安全检查

        Returns:
            (净化后的参数, 被拦截的危险模式列表)
        """
        sanitized = {}
        blocked = []

        for key, value in args.items():
            # 参数白名单检查
            if policy.allowed_args and key not in policy.allowed_args:
                blocked.append(f"参数 '{key}' 不在白名单中")
                continue

            # 参数黑名单检查
            if policy.blocked_args and key in policy.blocked_args:
                blocked.append(f"参数 '{key}' 在黑名单中")
                continue

            # 参数长度检查
            str_value = str(value)
            if len(str_value) > policy.max_arg_length:
                sanitized[key] = str_value[:policy.max_arg_length]
                continue

            # 危险模式检查（对所有参数值）
            all_patterns = self.BUILTIN_DANGEROUS_PATTERNS + policy.dangerous_patterns
            for pattern in all_patterns:
                if re.search(pattern, str_value, re.IGNORECASE):
                    blocked.append(f"参数 '{key}' 匹配危险模式: {pattern[:40]}")
                    break

            if key not in sanitized:
                sanitized[key] = value

        return sanitized, blocked

    def record_call(self, tool_name: str):
        """记录工具调用（在成功调用后调用）"""
        self._call_history[tool_name].append(time.time())

    # ==================== 审批回调 ====================

    def register_approval_handler(self, tool_name: str, handler):
        """
        注册审批处理器

        handler: async def my_handler(tool_name, args, session) -> bool
        返回 True = 批准, False = 拒绝
        """
        self._approval_callbacks[tool_name] = handler

    async def request_approval(self, tool_name: str, args: dict, session=None) -> bool:
        """请求人工审批"""
        if tool_name in self._approval_callbacks:
            return await self._approval_callbacks[tool_name](tool_name, args, session)

        # 默认：拒绝未配置审批处理器的高危操作
        return False

    # ==================== 策略查询 ====================

    def get_policy(self, tool_name: str) -> ToolPolicy | None:
        return self._policies.get(tool_name)

    def get_all_policies(self) -> dict[str, ToolPolicy]:
        return dict(self._policies)

    def get_stats(self) -> dict:
        """获取工具调用统计"""
        now = time.time()
        return {
            name: {
                "calls_last_minute": len([t for t in times if t > now - 60]),
                "policy": self._policies.get(name, None).permission if name in self._policies else "unset",
                "requires_approval": self._policies.get(name, None).requires_approval if name in self._policies else False,
            }
            for name, times in self._call_history.items()
        }
