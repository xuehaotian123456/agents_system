"""
CC-Harness Agent - 异步重试组件
================================
基于 tenacity 库实现 LLM 调用和工具执行的自动重试。

使用场景：
- LLM 调用网络超时 → 自动重试 2 次（指数退避）
- vLLM / 百炼 API 临时过载（HTTP 429/503）→ 等待后重试
- 向量数据库连接临时断开 → 重试连接

设计原则：
- 不要无条件无限重试
- 指数退避（exponential backoff）避免雪崩
- 对可重试异常和不可重试异常区别对待
"""

import functools
from typing import Callable, Type

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

import logging

logger = logging.getLogger(__name__)


def async_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    retryable_exceptions: tuple[Type[BaseException], ...] = (Exception,),
):
    """
    异步函数重试装饰器

    Args:
        max_attempts: 最大重试次数（含首次调用）
        min_wait: 最小等待时间（秒），第一次重试等待 min_wait
        max_wait: 最大等待时间（秒），等待时间不会超过此值
        retryable_exceptions: 可重试的异常类型

    使用示例：
        @async_retry(max_attempts=3, retryable_exceptions=(httpx.TimeoutException,))
        async def call_llm(prompt):
            ...

    指数退避：第1次等1s，第2次等2s，第3次等4s...直到 max_wait 封顶。
    这是防止雪崩的关键：不会在短时间内大量重试压垮下游服务。
    """
    def decorator(func: Callable):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=min_wait, max=max_wait),
            retry=retry_if_exception_type(retryable_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,  # 超过最大重试次数后抛出原始异常
        )
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator

