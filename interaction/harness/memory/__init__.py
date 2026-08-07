"""
CC-Harness Agent — 记忆系统
============================
双层记忆架构：

1. WorkingMemory (工作记忆)
   - 会话内重要事实的暂存
   - 类似人脑的"短期记忆"，容量有限（~7±2 条）
   - 自动提取用户偏好、对话中的重要实体

2. VectorMemory (向量记忆)
   - 跨会话长期记忆
   - 基于 ChromaDB 存储，语义检索
   - 类似人脑的"长期记忆"，容量近乎无限
"""

from harness.memory.working import WorkingMemory, MemoryItem
from harness.memory.vector import VectorMemory, VectorMemoryItem

__all__ = [
    "WorkingMemory",
    "MemoryItem",
    "VectorMemory",
    "VectorMemoryItem",
]
