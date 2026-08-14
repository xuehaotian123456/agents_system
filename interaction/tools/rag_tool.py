"""
CC-Harness Agent - RAG 检索工具 (DEPRECATED)
============================================
⚠️ 架构修正 (2026-08-14): 本工具已从服务主链路退役。

旧设计问题: interaction 层用朴素分段重建 pipeline 语料的第二份
embedding (影子库), 且本地 rag_search 同名抢占 A2A 注册,
导致 pipeline 的完整检索栈 (RRF 融合 + KG + Reranker +
Agentic 循环) 从未在 interaction 层生效 — 数据双源 + 质量割裂。

现行架构: 检索唯一来源 = Pipeline A2A rag_search
(pipeline/rag/agentic_retriever.py 内含 Agentic 循环)。

本文件仅保留用于独立 demo (cli.py / benchmarks) 的轻量检索演示。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Sequence

import httpx
from chromadb import PersistentClient, EmbeddingFunction
from openai import AsyncOpenAI

from harness.types import ToolResult
from tools.base import BaseTool


class _DashScopeEmbedding(EmbeddingFunction):
    """
    百炼 API 的 Chroma EmbeddingFunction 包装

    ChromaDB 要求 embedding_function 是同步的（EmbeddingFunction 协议），
    但百炼 API 是异步的。这里使用内部事件循环的方式适配。

    实际上这个类不会被真正调用（因为我们传 embedding 向量给 query），
    存在只是为了满足 ChromaDB 的接口要求，避免下载默认 ONNX 模型。
    """

    def __init__(self, embedding_client: AsyncOpenAI, model: str):
        self._client = embedding_client
        self._model = model

    def __call__(self, input: Sequence[str]) -> Sequence[Sequence[float]]:
        """同步包装异步调用（仅用于 ChromaDB 的 add 操作）"""
        import asyncio
        async def _embed():
            resp = await self._client.embeddings.create(model=self._model, input=list(input))
            return [d.embedding for d in resp.data]
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_embed())
        # 已有运行中的事件循环，使用线程池
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _embed())
            return future.result()


class RAGTool(BaseTool):
    """
    知识库 RAG 检索工具

    Agent 使用方式：
    1. LLM 决定需要查知识库 → 输出 tool_call: {"tool_name": "rag_search", "args": {"query": "..."}}
    2. AgentLoop 调用 registry.execute("rag_search", {"query": "..."})
    3. 工具内部执行 Agentic RAG 循环（检索→评分→可能的改写→重新检索）
    4. 返回最佳检索结果文本

    设计要点：
    - 工具内部有自己的"小循环"（retry loop），但不暴露给外部 Agent
    - 外部 Agent 只看到最终的检索结果
    - 这避免了 Agent 主循环被检索的细节淹没
    """

    name = "rag_search"
    description = "知识库检索工具。查询企业内部知识库，获取相关文档内容。适用于技术概念、产品信息、流程文档的查询。"
    parameters = {
        "query": "检索查询语句（建议使用关键词，而非完整句子）",
    }

    def __init__(
        self,
        collection_name: str = "cc_harness_knowledge_base",
        persist_dir: str = "./chroma_db",
        k: int = 3,
        max_rewrites: int = 1,
    ):
        """
        Args:
            collection_name: Chroma 集合名
            persist_dir: 持久化目录
            k: 检索返回文档数
            max_rewrites: 最大查询改写次数（Agentic RAG 的内部循环上限）
        """
        self.k = k
        self.max_rewrites = max_rewrites

        # 初始化 embedding 客户端（使用百炼 text-embedding-v1）
        embedding_base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        embedding_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        embedding_http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

        self._embedding_client = AsyncOpenAI(
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            http_client=embedding_http_client,
        )
        self._embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")

        # 自定义 embedding function（避免 ChromaDB 下载默认 ONNX 模型）
        emb_fn = _DashScopeEmbedding(self._embedding_client, self._embedding_model)

        # 初始化 Chroma（使用自定义 embedding function）
        self._chroma_client = PersistentClient(path=persist_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,  # ★ 关键：传入自定义 embedding，阻止下载 ONNX 模型
        )

    # ==================== 公开接口 ====================

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        执行 RAG 检索

        Args:
            query: 检索查询语句

        Returns:
            ToolResult: 检索结果文本
        """
        query = str(kwargs.get("query", ""))

        if not query.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="",
                error="查询语句不能为空",
            )

        try:
            result_text = await self._agentic_rag_search(query)
            return ToolResult(
                tool_name=self.name,
                success=True,
                content=result_text,
                metadata={"query": query},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="",
                error=f"RAG检索异常：{type(e).__name__}: {e}",
            )

    # ==================== Agentic RAG 内部循环 ====================

    async def _agentic_rag_search(self, query: str) -> str:
        """
        RAG 检索（简化版：跳过 LLM 评分，直接返回所有检索结果）

        原版 Agentic RAG 流程（检索→LLM评分→改写→重检索），
        但 LLM 评分环节每篇文档都要调一次 LLM，耗时太长。
        简化策略：向量检索直接返回 top-k 结果，让外部 AgentLoop 的 LLM 做最终筛选。

        这个闭环对外部 Agent 是透明的——Agent 只看到最终结果。
        """
        # 向量检索
        docs = await self._vector_search(query)

        if not docs:
            return "知识库中未找到相关文档。请尝试换个关键词。"

        # 跳过 LLM 评分，直接返回所有检索结果（外部 Agent 的 LLM 会自己判断相关性）
        return self._format_results(docs)

    # ==================== 内部方法 ====================

    async def _vector_search(self, query: str) -> list[str]:
        """
        向量检索

        将查询转为 embedding 向量，在 Chroma 中搜索最相似的 top-k 文档。
        使用 asyncio.to_thread 包装同步 Chroma 调用，避免阻塞事件循环。
        """
        # 获取 query 的 embedding
        emb_resp = await self._embedding_client.embeddings.create(
            model=self._embedding_model,
            input=query,
        )
        query_embedding = emb_resp.data[0].embedding

        # Chroma 同步查询 → 线程池执行
        def _search():
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=self.k,
                include=["documents"],
            )
            docs = results.get("documents", [[]])[0]
            return docs

        docs = await asyncio.to_thread(_search)
        return docs

    def _format_results(self, docs: list[str]) -> str:
        """格式化检索结果为 LLM 友好的文本"""
        parts = [f"以下是从知识库检索到的 {len(docs)} 篇相关文档：\n"]
        for i, doc in enumerate(docs):
            parts.append(f"[文档{i+1}] {doc}")
        return "\n\n".join(parts)

    # ==================== 知识库管理 ====================

    def add_documents(self, texts: list[str]):
        """
        向知识库添加文档（同步方法，初始化时使用）

        Args:
            texts: 文档文本列表
        """
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 检查是否已有文档，避免重复添加
        existing = self._collection.get(ids=ids)
        if existing["ids"]:
            return

        self._collection.add(
            ids=ids,
            documents=texts,
        )

    def clear(self):
        """清空知识库"""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
