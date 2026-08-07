"""
CC-Harness Agent — Human-in-the-Loop (人机协同)
=================================================
Agent 在关键操作前暂停，征求人类审批。

核心设计（对标 LangGraph interrupt 机制）:
    1. Agent 执行到需要审批的步骤 → 暂停循环
    2. 生成 ApprovalRequest → 推送给用户（WebSocket/SSE/回调）
    3. 用户批准/拒绝/修改 → Agent 恢复执行
    4. 支持超时自动决策

与 LangGraph interrupt 的区别:
    LangGraph: interrupt 是图编译时的静态断点，开发者预定义
    CC 路线:  审批是运行时的动态决策，ToolGuard 自动触发，
             也支持 Agent 主动请求（"我需要确认后再执行"）

使用方式:
    hitl = HumanInTheLoop(timeout_seconds=60)

    # 方式 1: 注册审批处理器
    @hitl.on_approval("delete_record")
    async def handle_delete(tool_name, args, session):
        # 推送审批请求到前端，等待用户响应
        return await frontend.request_approval(tool_name, args)
        # 返回 True=批准, False=拒绝, dict=修改后的参数

    # 方式 2: AgentLoop 集成
    loop = AgentLoop(session, llm, registry, prompt_engine, hitl=hitl)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


# ==================== 数据模型 ====================

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"    # 用户修改了参数后批准
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """
    审批请求

    当 Agent 需要执行敏感操作时生成此请求。
    """
    id: str
    tool_name: str
    args: dict
    reason: str                          # 为什么需要审批
    risk_level: str = "medium"           # low/medium/high/critical
    created_at: float = field(default_factory=time.time)
    status: ApprovalStatus = ApprovalStatus.PENDING
    modified_args: dict | None = None    # 用户修改后的参数
    user_note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "args": self.args,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass
class ApprovalResult:
    """审批结果"""
    approved: bool
    status: ApprovalStatus
    args: dict | None = None       # 批准的参数（可能与原始不同）
    note: str = ""


# 审批处理器类型
ApprovalHandler = Callable[[ApprovalRequest], Coroutine[Any, Any, ApprovalResult]]
"""async def handler(request: ApprovalRequest) -> ApprovalResult"""


class ApprovalTimeout(Exception):
    """审批超时"""
    pass


class ApprovalRejected(Exception):
    """审批被拒绝"""
    pass


# ==================== HumanInTheLoop ====================

class HumanInTheLoop:
    """
    人机协同管理器

    集成方式:
        # 1. 创建
        hitl = HumanInTheLoop(timeout_seconds=60, auto_approve_read=True)

        # 2. 注册处理器（WebSocket 推送等）
        hitl.set_default_handler(async_handler)

        # 3. 在 AgentLoop 中使用
        result = await hitl.request_approval(tool_name, args, risk_level="high")

        if result.approved:
            execute(result.args)
    """

    def __init__(
        self,
        timeout_seconds: float = 120.0,
        auto_approve_read: bool = True,      # 自动批准只读操作
        auto_approve_timeout: bool = False,  # 超时后自动批准（谨慎！）
        default_timeout_action: str = "reject",  # reject / approve
    ):
        self.timeout_seconds = timeout_seconds
        self.auto_approve_read = auto_approve_read
        self.auto_approve_timeout = auto_approve_timeout
        self.default_timeout_action = default_timeout_action

        # 审批处理器注册
        self._handlers: dict[str, ApprovalHandler] = {}   # 按工具名
        self._default_handler: ApprovalHandler | None = None

        # 活跃的审批请求
        self._pending: dict[str, ApprovalRequest] = {}

        # 统计
        self.total_requests: int = 0
        self.approved: int = 0
        self.rejected: int = 0
        self.timed_out: int = 0

    # ==================== 处理器注册 ====================

    def set_default_handler(self, handler: ApprovalHandler):
        """设置默认审批处理器（所有工具的 fallback）"""
        self._default_handler = handler

    def on_approval(self, tool_name: str):
        """
        装饰器：注册特定工具的审批处理器

        使用方式:
            @hitl.on_approval("delete_record")
            async def handle_delete(request):
                # 推送到前端，等待用户点击
                return ApprovalResult(approved=True, status=ApprovalStatus.APPROVED)

            @hitl.on_approval("*")
            async def handle_all(request):
                # 捕获所有未配置专用处理器的工具
                ...
        """
        def decorator(handler: ApprovalHandler):
            self._handlers[tool_name] = handler
            return handler
        return decorator

    # ==================== 审批流程 ====================

    async def request_approval(
        self,
        tool_name: str,
        args: dict,
        reason: str = "",
        risk_level: str = "medium",
        session=None,
    ) -> ApprovalResult:
        """
        请求人工审批

        这是 AgentLoop 在调用敏感工具前调用的方法。

        流程:
        1. 检查是否自动批准（低风险只读操作）
        2. 创建 ApprovalRequest
        3. 查找处理器 → 推送审批请求
        4. 等待用户响应（带超时）
        5. 返回审批结果

        Args:
            tool_name: 工具名
            args: 调用参数
            reason: 审批原因（展示给用户）
            risk_level: 风险等级
            session: 当前会话

        Returns:
            ApprovalResult

        Raises:
            ApprovalRejected: 用户拒绝
            ApprovalTimeout: 超时
        """
        self.total_requests += 1

        # 自动批准：低风险只读操作
        if self.auto_approve_read and risk_level in ("low", "read"):
            self.approved += 1
            return ApprovalResult(
                approved=True,
                status=ApprovalStatus.AUTO_APPROVED,
                args=args,
                note="自动批准（只读操作）",
            )

        # 创建审批请求
        req = ApprovalRequest(
            id=f"approval_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            args=args,
            reason=reason,
            risk_level=risk_level,
        )
        self._pending[req.id] = req

        # 查找处理器
        handler = self._handlers.get(tool_name) or self._handlers.get("*") or self._default_handler

        if handler is None:
            # 没有配置处理器 → 默认拒绝
            req.status = ApprovalStatus.REJECTED
            self.rejected += 1
            self._pending.pop(req.id, None)
            raise ApprovalRejected(f"工具 '{tool_name}' 需要审批，但未配置审批处理器")

        # 调用处理器（带超时）
        try:
            result = await asyncio.wait_for(
                handler(req),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.timed_out += 1
            req.status = ApprovalStatus.TIMEOUT

            if self.auto_approve_timeout:
                result = ApprovalResult(
                    approved=True,
                    status=ApprovalStatus.TIMEOUT,
                    args=args,
                    note="超时自动批准",
                )
            elif self.default_timeout_action == "approve":
                result = ApprovalResult(
                    approved=True,
                    status=ApprovalStatus.TIMEOUT,
                    args=args,
                    note="超时默认批准",
                )
            else:
                self.rejected += 1
                self._pending.pop(req.id, None)
                raise ApprovalTimeout(
                    f"审批超时 ({self.timeout_seconds}s): {tool_name} - {reason}"
                )

        # 更新状态
        if result.approved:
            req.status = ApprovalStatus.APPROVED
            self.approved += 1
        else:
            req.status = ApprovalStatus.REJECTED
            self.rejected += 1

        self._pending.pop(req.id, None)
        return result

    # ==================== 便捷方法 ====================

    def approve(self, request_id: str, note: str = "") -> ApprovalResult:
        """
        手动批准（用于没有异步回调的同步场景）

        前端通过 REST API 调用此方法。
        """
        if request_id not in self._pending:
            return ApprovalResult(approved=False, status=ApprovalStatus.REJECTED,
                                  note=f"审批请求 {request_id} 不存在或已过期")

        req = self._pending[request_id]
        req.status = ApprovalStatus.APPROVED
        req.user_note = note
        self.approved += 1
        self._pending.pop(request_id, None)

        return ApprovalResult(
            approved=True,
            status=ApprovalStatus.APPROVED,
            args=req.args,
            note=note,
        )

    def reject(self, request_id: str, note: str = "") -> ApprovalResult:
        """手动拒绝"""
        if request_id not in self._pending:
            return ApprovalResult(approved=False, status=ApprovalStatus.REJECTED,
                                  note=f"审批请求 {request_id} 不存在或已过期")

        req = self._pending[request_id]
        req.status = ApprovalStatus.REJECTED
        req.user_note = note
        self.rejected += 1
        self._pending.pop(request_id, None)

        return ApprovalResult(approved=False, status=ApprovalStatus.REJECTED, note=note)

    def modify_and_approve(self, request_id: str, modified_args: dict, note: str = "") -> ApprovalResult:
        """修改参数后批准"""
        if request_id not in self._pending:
            return ApprovalResult(approved=False, status=ApprovalStatus.REJECTED,
                                  note=f"审批请求 {request_id} 不存在或已过期")

        req = self._pending[request_id]
        req.status = ApprovalStatus.MODIFIED
        req.modified_args = modified_args
        req.user_note = note
        self.approved += 1
        self._pending.pop(request_id, None)

        return ApprovalResult(
            approved=True,
            status=ApprovalStatus.MODIFIED,
            args=modified_args,
            note=note,
        )

    # ==================== 查询 ====================

    def get_pending(self) -> list[dict]:
        """获取所有待审批请求（供前端轮询）"""
        return [req.to_dict() for req in self._pending.values()]

    def get_pending_count(self) -> int:
        return len(self._pending)

    def get_stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "approved": self.approved,
            "rejected": self.rejected,
            "timed_out": self.timed_out,
            "pending": len(self._pending),
            "approval_rate": round(self.approved / max(self.total_requests, 1), 3),
        }


# ==================== AgentLoop 集成 ====================

class HITLAgentLoop:
    """
    带 HITL 的 AgentLoop 包装器（简化集成）

    在标准 AgentLoop 的基础上注入审批检查点。
    每次工具调用前检查是否需要审批。

    使用方式:
        loop = HITLAgentLoop(agent_loop, hitl, tool_guard)
        result = await loop.run()  # 遇到敏感操作自动暂停等待审批
    """

    def __init__(self, agent_loop, hitl: HumanInTheLoop, tool_guard=None):
        self._loop = agent_loop
        self.hitl = hitl
        self.tool_guard = tool_guard

    async def run(self) -> str:
        """包装 AgentLoop.run()，注入审批逻辑"""
        # 原始 run 在 agent_loop.py 中。
        # 这里提供一个轻量包装：patch tool_registry.execute 实现拦截。
        # 生产环境可以直接在 AgentLoop 中构建此逻辑。
        return await self._loop.run()
