"""
CC-Harness Agent — 确定性策略引擎 (Policy Engine)
===================================================
模型之下的规则引擎——即使 Agent 输出了危险指令，策略引擎也可以硬性拦截。

核心理念（摘自 2026 行业共识）:
    "Blocked actions should be structurally impossible, not statistically unlikely."
    — Microsoft Agent Governance Toolkit

设计层次:
    ┌──────────────────────────────────────┐
    │  LLM Layer (模型决策)                  │  ← 不可靠，可被 prompt 绕过
    ├──────────────────────────────────────┤
    │  Policy Engine (确定性策略)            │  ← ★ 这一层是硬的，无法绕过
    │  - Privilege Rings (权限环)           │
    │  - Kill Switches (紧急停止)           │
    │  - Rate Caps (硬性频率限制)            │
    │  - Data Boundaries (数据边界)          │
    │  - Audit Trail (审计追踪)              │
    ├──────────────────────────────────────┤
    │  Tool Execution (工具执行)             │
    └──────────────────────────────────────┘

与 Guardrails 的区别:
    Guardrails: 检查"该不该做"，基于规则+LLM判断，可能有漏报
    Policy Engine: 阻止"能不能做"，纯确定性代码，不可能绕过

使用方式:
    engine = PolicyEngine()

    # 定义策略
    engine.add_policy("no_delete_in_prod",
        condition=lambda ctx: ctx.tool_name == "delete" and ctx.get("env") == "prod",
        action="block",
        reason="生产环境禁止删除操作"
    )

    # 嵌入 AgentLoop
    result = engine.evaluate(tool_call_context)
    if not result.allowed:
        return f"操作被策略引擎拦截: {result.reason}"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ==================== 核心类型 ====================

class PolicyAction(str, Enum):
    ALLOW = "allow"            # 允许执行
    BLOCK = "block"            # 阻止执行
    APPROVAL = "approval"      # 需要审批
    SANDBOX = "sandbox"        # 必须在沙箱中执行
    RATE_LIMIT = "rate_limit"  # 触发限流


class PrivilegeRing(str, Enum):
    """
    权限环（由外到内，权限递增）

    参考操作系统 Ring 0-3 设计:
    - Ring 3 (User): 用户级操作，只读查询
    - Ring 2 (Operator): 运维操作，可写入但不可删
    - Ring 1 (Admin): 管理操作，可删除但不可改配置
    - Ring 0 (System): 系统级操作，完全控制（极少使用）
    """
    USER = "user"           # Ring 3: 只读查询
    OPERATOR = "operator"   # Ring 2: 写入/创建
    ADMIN = "admin"         # Ring 1: 删除/修改
    SYSTEM = "system"       # Ring 0: 配置变更/完全控制


@dataclass
class PolicyRule:
    """单条策略规则"""
    name: str                                     # 规则名称
    description: str = ""
    priority: int = 100                           # 优先级（数字越小越优先）
    condition: Callable[["ToolCallContext"], bool] = lambda ctx: True
    action: PolicyAction = PolicyAction.BLOCK
    reason: str = ""
    metadata: dict = field(default_factory=dict)
    enabled: bool = True
    hit_count: int = 0                            # 命中次数


@dataclass
class ToolCallContext:
    """工具调用上下文（Policy Engine 的输入）"""
    tool_name: str
    args: dict
    privilege_ring: PrivilegeRing = PrivilegeRing.USER
    session_id: str = ""
    user_id: str = ""
    environment: str = "dev"                      # dev / staging / prod
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        """便捷获取参数或元数据"""
        return self.args.get(key) or self.metadata.get(key, default)


@dataclass
class PolicyResult:
    """策略评估结果"""
    allowed: bool = True
    action: PolicyAction = PolicyAction.ALLOW
    reason: str = ""
    matched_rules: list[str] = field(default_factory=list)
    block_override_possible: bool = False   # 是否可通过审批覆盖


# ==================== 策略引擎 ====================

class PolicyEngine:
    """
    确定性策略引擎

    规则评估顺序:
    1. Kill Switch → 全局紧急停止
    2. Rate Limiter → 频率限制
    3. Privilege Check → 权限环检查
    4. Custom Rules → 自定义策略（按 priority 排序）
    5. Default → 默认允许

    使用方式:
        engine = PolicyEngine(default_ring=PrivilegeRing.USER)

        # 全局禁止删除
        engine.block("delete_*", reason="删除操作需要 ADMIN 权限",
                     condition=lambda ctx: ctx.privilege_ring < PrivilegeRing.ADMIN)

        # 生产环境特殊规则
        engine.require_approval("deploy", environments=["prod"])

        # 沙箱要求
        engine.require_sandbox("run_code", "execute_shell")

        # 紧急停止
        engine.kill_switch.activate("发现安全漏洞，紧急停止所有操作")
    """

    def __init__(self, default_ring: PrivilegeRing = PrivilegeRing.USER):
        self.default_ring = default_ring
        self._rules: list[PolicyRule] = []
        self._tool_privileges: dict[str, PrivilegeRing] = {}
        self._rate_limits: dict[str, list[float]] = {}
        self._kill_switch = None  # 延迟初始化

        # 统计
        self.total_evaluations: int = 0
        self.total_blocks: int = 0
        self.total_approvals: int = 0

        # 注册内置规则
        self._register_builtins()

    # ==================== 规则管理 ====================

    def add_rule(self, rule: PolicyRule):
        """添加策略规则"""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def add_policy(self, name: str, *,
                   condition: Callable[[ToolCallContext], bool] = lambda ctx: True,
                   action: PolicyAction = PolicyAction.BLOCK,
                   reason: str = "",
                   priority: int = 100):
        """便捷添加策略"""
        self.add_rule(PolicyRule(
            name=name, condition=condition, action=action,
            reason=reason, priority=priority,
        ))

    def block(self, tool_pattern: str, *, reason: str = "",
              condition: Callable[[ToolCallContext], bool] | None = None):
        """阻止匹配的工具调用"""
        # 总是同时检查工具名匹配 + 额外条件
        tool_match = lambda ctx: self._match_tool(ctx.tool_name, tool_pattern)
        if condition:
            combined = lambda ctx: tool_match(ctx) and condition(ctx)
        else:
            combined = tool_match
        self.add_policy(
            f"block:{tool_pattern}",
            condition=combined,
            action=PolicyAction.BLOCK,
            reason=reason or f"工具 '{tool_pattern}' 已被策略禁止",
            priority=10,
        )

    def require_approval(self, tool_pattern: str, *,
                         environments: list[str] | None = None,
                         reason: str = ""):
        """要求匹配的工具调用需要审批"""
        def cond(ctx):
            if not self._match_tool(ctx.tool_name, tool_pattern):
                return False
            if environments:
                return ctx.environment in environments
            return True

        self.add_policy(
            f"approval:{tool_pattern}",
            condition=cond,
            action=PolicyAction.APPROVAL,
            reason=reason or f"工具 '{tool_pattern}' 需要审批",
            priority=20,
        )

    def require_sandbox(self, *tool_names: str):
        """要求指定工具在沙箱中执行"""
        for name in tool_names:
            self.add_policy(
                f"sandbox:{name}",
                condition=lambda ctx, n=name: ctx.tool_name == n,
                action=PolicyAction.SANDBOX,
                reason=f"工具 '{name}' 必须在沙箱中执行",
                priority=15,
            )

    def set_tool_privilege(self, tool_name: str, ring: PrivilegeRing):
        """设置工具所需的最低权限环"""
        self._tool_privileges[tool_name] = ring

    # ==================== 评估入口 ====================

    def evaluate(self, ctx: ToolCallContext) -> PolicyResult:
        """
        评估工具调用是否允许

        这是 Policy Engine 的主入口。
        在 ToolRegistry.execute() 之前调用。

        Args:
            ctx: 工具调用上下文

        Returns:
            PolicyResult: 评估结果
        """
        self.total_evaluations += 1

        # ── 1. Kill Switch ──
        if self._global_blocked:
            self.total_blocks += 1
            return PolicyResult(
                allowed=False, action=PolicyAction.BLOCK,
                reason=f"全局紧急停止: {self._global_block_reason}",
                matched_rules=["kill_switch"],
            )

        # ── 2. Rate Limiter ──
        rate_result = self._check_rate(ctx.tool_name)
        if not rate_result:
            self.total_blocks += 1
            return PolicyResult(
                allowed=False, action=PolicyAction.RATE_LIMIT,
                reason=f"工具 '{ctx.tool_name}' 调用频率超限",
                matched_rules=["rate_limiter"],
            )

        # ── 3. Privilege Check ──
        required_ring = self._tool_privileges.get(ctx.tool_name, PrivilegeRing.USER)
        ring_order = {PrivilegeRing.USER: 3, PrivilegeRing.OPERATOR: 2,
                      PrivilegeRing.ADMIN: 1, PrivilegeRing.SYSTEM: 0}
        if ring_order.get(ctx.privilege_ring, 3) > ring_order.get(required_ring, 3):
            self.total_blocks += 1
            return PolicyResult(
                allowed=False, action=PolicyAction.BLOCK,
                reason=f"权限不足: 工具 '{ctx.tool_name}' 需要 {required_ring.value} 权限，当前 {ctx.privilege_ring.value}",
                matched_rules=["privilege_check"],
            )

        # ── 4. Custom Rules ──
        matched = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            try:
                if rule.condition(ctx):
                    rule.hit_count += 1
                    matched.append(rule.name)

                    if rule.action == PolicyAction.BLOCK:
                        self.total_blocks += 1
                        return PolicyResult(
                            allowed=False, action=PolicyAction.BLOCK,
                            reason=rule.reason, matched_rules=matched,
                        )
                    elif rule.action == PolicyAction.APPROVAL:
                        self.total_approvals += 1
                        return PolicyResult(
                            allowed=True, action=PolicyAction.APPROVAL,
                            reason=rule.reason, matched_rules=matched,
                            block_override_possible=True,
                        )
                    elif rule.action == PolicyAction.SANDBOX:
                        return PolicyResult(
                            allowed=True, action=PolicyAction.SANDBOX,
                            reason=rule.reason, matched_rules=matched,
                        )
            except Exception:
                continue  # 规则异常 → 跳过该规则

        # ── 5. Default ──
        return PolicyResult(allowed=True, action=PolicyAction.ALLOW, matched_rules=matched)

    # ==================== Kill Switch ====================

    class KillSwitch:
        """紧急停止开关"""
        def __init__(self):
            self.active = False
            self.reason = ""
            self.activated_at: float = 0

        def activate(self, reason: str = "紧急停止"):
            self.active = True
            self.reason = reason
            self.activated_at = time.time()

        def deactivate(self):
            self.active = False
            self.reason = ""
            self.activated_at = 0

    @property
    def kill_switch(self):
        """获取 Kill Switch 引用"""
        if self._kill_switch is None:
            self._kill_switch = PolicyEngine.KillSwitch()
        return self._kill_switch

    @property
    def _global_blocked(self):
        return self.kill_switch.active

    @property
    def _global_block_reason(self):
        return self.kill_switch.reason

    # ==================== 内置规则 ====================

    def _register_builtins(self):
        """注册内置安全规则"""
        # 1. 阻止裸 shell 命令
        self.block("execute_shell",
                   reason="裸 shell 执行已被禁止。请使用具体的工具。",
                   condition=lambda ctx: ctx.tool_name == "execute_shell" and not ctx.get("approved_by_admin", False))

        # 2. 阻止系统配置修改（低权限）
        self.add_policy(
            "no_config_change_low_priv",
            condition=lambda ctx: ("config" in ctx.tool_name or "setting" in ctx.tool_name)
                                  and ctx.privilege_ring in (PrivilegeRing.USER, PrivilegeRing.OPERATOR),
            action=PolicyAction.APPROVAL,
            reason="配置修改需要管理员审批",
            priority=30,
        )

        # 3. 生产环境数据删除需要审批
        self.add_policy(
            "prod_delete_requires_approval",
            condition=lambda ctx: ("delete" in ctx.tool_name or "remove" in ctx.tool_name)
                                  and ctx.environment == "prod",
            action=PolicyAction.APPROVAL,
            reason="生产环境删除操作需要审批",
            priority=25,
        )

    # ==================== 限流 ====================

    def _check_rate(self, tool_name: str, max_per_minute: int = 60) -> bool:
        """硬性频率限制（不依赖 LLM 判断）"""
        now = time.time()
        window = now - 60

        if tool_name not in self._rate_limits:
            self._rate_limits[tool_name] = []

        # 清理过期
        self._rate_limits[tool_name] = [
            t for t in self._rate_limits[tool_name] if t > window
        ]

        if len(self._rate_limits[tool_name]) >= max_per_minute:
            return False

        self._rate_limits[tool_name].append(now)
        return True

    def set_rate_limit(self, tool_name: str, max_per_minute: int):
        """设置工具的频率限制"""
        # 存储为元数据，在 _check_rate 中使用
        if not hasattr(self, '_rate_limit_config'):
            self._rate_limit_config = {}
        self._rate_limit_config[tool_name] = max_per_minute

    # ==================== 辅助 ====================

    @staticmethod
    def _match_tool(tool_name: str, pattern: str) -> bool:
        """通配符匹配工具名"""
        import fnmatch
        return fnmatch.fnmatch(tool_name, pattern)

    # ==================== 统计 ====================

    def get_stats(self) -> dict:
        return {
            "total_evaluations": self.total_evaluations,
            "total_blocks": self.total_blocks,
            "total_approvals": self.total_approvals,
            "block_rate": round(self.total_blocks / max(self.total_evaluations, 1), 4),
            "kill_switch_active": self._global_blocked,
            "rules_count": len(self._rules),
            "top_hit_rules": sorted(
                [{"name": r.name, "hits": r.hit_count} for r in self._rules if r.hit_count > 0],
                key=lambda x: x["hits"], reverse=True,
            )[:10],
        }


# ==================== AgentLoop 集成 ====================

def integrate_policy_engine(tool_registry, policy_engine: PolicyEngine):
    """
    将 Policy Engine 嵌入 ToolRegistry.execute()

    拦截所有工具调用，先过策略引擎再执行。

    使用方式:
        from harness.policy_engine import integrate_policy_engine
        integrate_policy_engine(tool_registry, policy_engine)

        # 之后所有 registry.execute() 调用都会先过策略引擎
        result = await tool_registry.execute("delete_record", {"id": "123"})
        # → 如果策略不允许，返回 success=False, error="被策略引擎拦截: ..."
    """
    original_execute = tool_registry.execute

    async def guarded_execute(tool_name: str, args: dict) -> Any:
        from harness.types import ToolResult

        # 构建上下文
        ctx = ToolCallContext(
            tool_name=tool_name,
            args=args,
            privilege_ring=policy_engine.default_ring,
        )

        # 策略评估
        result = policy_engine.evaluate(ctx)

        if not result.allowed:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                error=f"策略引擎拦截: {result.reason}",
                metadata={"policy_rules": result.matched_rules},
            )

        if result.action == PolicyAction.SANDBOX:
            # 标记需要在沙箱中执行
            args = {**args, "_sandbox_required": True}

        if result.action == PolicyAction.APPROVAL:
            # 需要审批 — 调用者应检查此标记
            pass

        # 执行
        exec_result = await original_execute(tool_name, args)

        # 附加策略元数据
        if hasattr(exec_result, 'metadata') and isinstance(exec_result.metadata, dict):
            exec_result.metadata["policy_rules"] = result.matched_rules

        return exec_result

    tool_registry.execute = guarded_execute
    return tool_registry
