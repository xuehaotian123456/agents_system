"""
CC-Harness Agent - RAG 检索工具
================================
将 RAG 能力封装为一个 Tool，而非顶层工作流节点。

这是 CC 路线与 LangGraph 路线最核心的架构差异：
  LangGraph:  RAG 是 Graph 中的多个节点（检索、评分、改写、生成）
              → RAG 流程是框架级别的，与 Agent 耦合
  CC路线:     RAG 是 Tool 注册表中的一个普通工具
              → Agent 按需调用，RAG 工具内部封装完整 Agentic RAG 循环
              → Agent 可以同时使用 RAG、Web Search、数据库查询等工具

RAG 工具内部同样运行一个简化的 Agentic RAG 循环：
  检索 → 文档评分 → (不相关? → 改写查询 → 重新检索) → 返回结果
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

        # 初始化 LLM（用于文档评分和查询改写）
        self._llm_client = AsyncOpenAI(
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            http_client=httpx.AsyncClient(timeout=httpx.Timeout(120.0)),
        )
        self._llm_model = os.getenv("LLM_MODEL", "qwen3.5-flash")

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

    async def _grade_documents(self, query: str, docs: list[str]) -> list[str]:
        """
        LLM 评分文档相关性

        让 LLM 判断每篇文档是否与查询相关，只保留相关的文档。
        这是 Agentic RAG 的关键步骤：不是所有检索结果都有用。
        """
        relevant = []

        for i, doc in enumerate(docs):
            prompt = (
                f"判断以下文档是否与用户问题相关。\n"
                f"用户问题：{query}\n"
                f"文档内容：{doc[:800]}\n"
                f"只输出 RELEVANT 或 IRRELEVANT："
            )

            resp = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )

            verdict = resp.choices[0].message.content.strip().upper() if resp.choices[0].message.content else ""

            if "RELEVANT" in verdict:
                relevant.append(doc)

        return relevant

    async def _rewrite_query(self, original_query: str, reason: str) -> str:
        """
        LLM 改写查询

        当原始查询检索结果不理想时，让 LLM 优化查询语句。
        改写策略：更具体的关键词、去掉冗余词、换个角度表述。
        """
        prompt = (
            f"原始查询未获得有效结果（原因：{reason}），请优化以下查询语句，"
            f"使其更适合向量检索。输出优化后的查询语句，不要输出其他内容。\n"
            f"原始查询：{original_query}"
        )

        resp = await self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        return resp.choices[0].message.content.strip() if resp.choices[0].message.content else original_query

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
