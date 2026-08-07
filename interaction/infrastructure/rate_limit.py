"""
CC-Harness Agent - 接口限流组件
================================
基于内存计数器实现简单的请求限流。

使用场景：
- 防止前端恶意刷屏
- 避免大量并发请求打垮 LLM 推理服务
- 私有化部署时保护下游资源

当前实现：内存计数器（单机版）
生产环境：需配合 AsyncCache（Redis）实现分布式限流

限流算法：滑动窗口计数器
- 每 60 秒最多 N 次请求
- 超过限制 → 拒绝并返回 429 状态码
"""

import time
from collections import defaultdict


class RateLimiter:
    """
    单机内存限流器

    使用方式：
        limiter = RateLimiter(max_requests=30, window_seconds=60)
        if not limiter.is_allowed("user_123"):
            raise HTTPException(status_code=429, detail="请求太频繁")
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        """
        Args:
            max_requests: 窗口内最大请求数
            window_seconds: 窗口大小（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """
        检查是否允许本次请求

        Args:
            client_id: 客户端标识（IP 或用户 ID）

        Returns:
            True 表示允许，False 表示被限流
        """
        now = time.time()
        window_start = now - self.window_seconds

        # 清理过期的请求记录
        self._windows[client_id] = [
            t for t in self._windows[client_id] if t > window_start
        ]

        # 检查当前窗口内的请求数
        if len(self._windows[client_id]) >= self.max_requests:
            return False

        # 记录本次请求
        self._windows[client_id].append(now)
        return True

    def remaining(self, client_id: str) -> int:
        """查询剩余请求额度"""
        now = time.time()
        window_start = now - self.window_seconds
        self._windows[client_id] = [
            t for t in self._windows[client_id] if t > window_start
        ]
        return max(0, self.max_requests - len(self._windows[client_id]))
