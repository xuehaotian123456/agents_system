"""
CC-Harness Agent LLM 适配层 (v2)
=================================
统一封装多模型调用，屏蔽底层差异。

v2 新增:
- 多 Provider 支持 (DashScope / DeepSeek / OpenAI / 任意兼容)
- 模型注册表 ModelRegistry — 按名称自动路由
- 增强结构化输出 — 原生 JSON mode + Prompt fallback 双重保障
- 流式回调 — on_token / on_complete 回调，AgentLoop 实时感知
- Token 估算增强 — 中文/英文/代码分别估算
- 模型降级链 — 主模型失败 → 自动切换备选模型

设计哲学：
    不依赖 LangChain ChatOpenAI，直接使用 openai 库。
    轻量、透明、可控 —— 每一行代码你都知道在做什么。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional

import httpx
from openai import AsyncOpenAI

from harness.types import AgentAction, Message


# ==================== Provider 定义 ====================

class Provider(str, Enum):
    """模型提供商"""
    DASHSCOPE = "dashscope"       # 阿里百炼 (通义千问系列)
    DEEPSEEK = "deepseek"         # DeepSeek
    OPENAI = "openai"             # OpenAI / 任意兼容端点
    CUSTOM = "custom"             # 自定义兼容端点


@dataclass
class ModelInfo:
    """模型元信息"""
    name: str                          # 模型名 (如 qwen-plus, deepseek-chat)
    provider: Provider                 # 提供商
    max_tokens: int = 131072           # 上下文窗口大小
    supports_json_mode: bool = True    # 是否支持原生 JSON mode
    supports_streaming: bool = True    # 是否支持流式输出
    supports_tool_calls: bool = True   # 是否支持原生 function calling
    cost_per_1k_input: float = 0.0     # 输入价格 ($/1k tokens)
    cost_per_1k_output: float = 0.0    # 输出价格 ($/1k tokens)


# 内置模型注册表
BUILTIN_MODELS: dict[str, ModelInfo] = {
    # ── 通义千问系列 (百炼) ──
    "qwen-plus": ModelInfo(
        name="qwen-plus", provider=Provider.DASHSCOPE,
        max_tokens=131072, cost_per_1k_input=0.0008, cost_per_1k_output=0.002,
    ),
    "qwen-turbo": ModelInfo(
        name="qwen-turbo", provider=Provider.DASHSCOPE,
        max_tokens=131072, cost_per_1k_input=0.0003, cost_per_1k_output=0.0006,
    ),
    "qwen-max": ModelInfo(
        name="qwen-max", provider=Provider.DASHSCOPE,
        max_tokens=32768, cost_per_1k_input=0.02, cost_per_1k_output=0.06,
    ),
    "qwen3-235b-a22b": ModelInfo(
        name="qwen3-235b-a22b", provider=Provider.DASHSCOPE,
        max_tokens=131072, supports_json_mode=True,
    ),

    # ── DeepSeek 系列 ──
    "deepseek-chat": ModelInfo(
        name="deepseek-chat", provider=Provider.DEEPSEEK,
        max_tokens=65536, cost_per_1k_input=0.00027, cost_per_1k_output=0.0011,
    ),
    "deepseek-reasoner": ModelInfo(
        name="deepseek-reasoner", provider=Provider.DEEPSEEK,
        max_tokens=65536, supports_json_mode=False,  # 推理模型不适合 JSON mode
    ),

    # ── OpenAI 系列 ──
    "gpt-4o": ModelInfo(
        name="gpt-4o", provider=Provider.OPENAI,
        max_tokens=128000, cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini", provider=Provider.OPENAI,
        max_tokens=128000, cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
    ),
}


# Provider → 默认 base_url 映射
PROVIDER_BASE_URLS = {
    Provider.DASHSCOPE: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    Provider.DEEPSEEK: "https://api.deepseek.com/v1",
    Provider.OPENAI: "https://api.openai.com/v1",
}


# ==================== Token 估算 ====================

def estimate_tokens(text: str, model: str = "") -> int:
    """
    粗略 token 估算（不依赖 tiktoken，离线可用）

    策略：
    - 中文字符: ~1.5 字符/token
    - 英文/数字: ~4 字符/token
    - 代码块: ~3 字符/token
    """
    if not text:
        return 0

    chinese = 0
    english = 0
    other = 0

    for c in text:
        if '一' <= c <= '鿿' or '　' <= c <= '〿':
            chinese += 1
        elif c.isascii():
            english += 1
        else:
            other += 1

    return int(chinese / 1.3 + english / 3.5 + other / 2.5)


# ==================== 回调类型 ====================

@dataclass
class StreamCallbacks:
    """流式回调集合"""
    on_token: Optional[Callable[[str], None]] = None       # 每收到一个 token
    on_thinking: Optional[Callable[[str], None]] = None    # 思考过程（DeepSeek-R1等推理模型）
    on_tool_call: Optional[Callable[[dict], None]] = None  # 工具调用
    on_complete: Optional[Callable[[str], None]] = None    # 完整文本生成完毕
    on_error: Optional[Callable[[str], None]] = None       # 出错


# ==================== LLMAdapter v2 ====================

class LLMAdapter:
    """
    LLM 统一适配器 (v2)

    职责：
    1. 封装 OpenAI 兼容 API 调用
    2. 支持多 Provider 自动路由
    3. 结构化输出（JSON mode + prompt fallback 双重保障）
    4. 流式生成 + 回调
    5. 模型降级链

    使用方式：
        # 方式 1: 自动检测（从环境变量）
        adapter = LLMAdapter(model="qwen-plus")

        # 方式 2: 指定 Provider
        adapter = LLMAdapter(model="deepseek-chat", provider=Provider.DEEPSEEK)

        # 方式 3: 自定义端点
        adapter = LLMAdapter(
            model="my-model",
            base_url="http://localhost:8000/v1",
            api_key="sk-xxx",
        )

        # 带降级链
        adapter = LLMAdapter(
            model="qwen-plus",
            fallback_models=["deepseek-chat", "qwen-turbo"],
        )

        # 结构化输出
        action = await adapter.generate_structured(messages, AgentAction)

        # 流式输出
        async for token in adapter.stream(messages):
            print(token, end="")
    """

    def __init__(
        self,
        model: str | None = None,
        provider: Provider | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        fallback_models: list[str] | None = None,
        callbacks: StreamCallbacks | None = None,
    ):
        """
        初始化 LLM 适配器

        Args:
            model: 模型名 (默认从 env LLM_MODEL 读取，再 fallback 到 qwen-plus)
            provider: Provider 枚举 (默认根据模型名自动推断)
            base_url: API 端点 (默认根据 Provider 自动设置)
            api_key: API Key (默认从环境变量读取)
            timeout: 请求超时（秒）
            fallback_models: 主模型失败时依次尝试的备选模型列表
            callbacks: 流式回调
        """
        self.model = model or os.getenv("LLM_MODEL", "qwen-plus")
        self.provider = provider or self._infer_provider(self.model)
        self.base_url = base_url or self._infer_base_url()
        self.api_key = api_key or self._infer_api_key()
        self.timeout = timeout
        self.fallback_models = fallback_models or []
        self.callbacks = callbacks

        # 查找模型元信息
        self.model_info = BUILTIN_MODELS.get(self.model)

        # 创建异步 HTTP 客户端
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=10),
        )

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=http_client,
        )

        # 统计
        self.total_calls: int = 0
        self.total_tokens_input: int = 0
        self.total_tokens_output: int = 0
        self.total_cost: float = 0.0
        self.last_latency_ms: float = 0.0

    # ==================== Provider 推断 ====================

    def _infer_provider(self, model: str) -> Provider:
        """根据模型名推断 Provider"""
        info = BUILTIN_MODELS.get(model)
        if info:
            return info.provider

        # 启发式推断
        if "qwen" in model.lower():
            return Provider.DASHSCOPE
        if "deepseek" in model.lower():
            return Provider.DEEPSEEK
        if "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
            return Provider.OPENAI

        # 检查环境变量
        base_url = os.getenv("DASHSCOPE_BASE_URL", "")
        if "dashscope" in base_url:
            return Provider.DASHSCOPE
        if "deepseek" in base_url:
            return Provider.DEEPSEEK

        return Provider.CUSTOM

    def _infer_base_url(self) -> str:
        """推断 API 端点"""
        # 优先使用环境变量
        env_url = os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", "")
        if env_url:
            return env_url

        # 根据 Provider 使用默认端点
        return PROVIDER_BASE_URLS.get(self.provider, "https://api.openai.com/v1")

    def _infer_api_key(self) -> str:
        """推断 API Key"""
        key_map = {
            Provider.DASHSCOPE: "DASHSCOPE_API_KEY",
            Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
            Provider.OPENAI: "OPENAI_API_KEY",
        }
        env_var = key_map.get(self.provider, "OPENAI_API_KEY")
        return os.getenv(env_var, os.getenv("DASHSCOPE_API_KEY", ""))

    # ==================== 文本生成 ====================

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> str:
        """
        普通文本生成（非流式）

        Args:
            messages: 对话历史（CC 自定义 Message）
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            model: 覆盖默认模型

        Returns:
            LLM 生成的文本
        """
        use_model = model or self.model
        openai_messages = [m.to_openai_format() for m in messages]

        start = time.time()
        self.total_calls += 1

        try:
            response = await self.client.chat.completions.create(
                model=use_model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            # 尝试降级
            if self.fallback_models:
                return await self._try_fallback(
                    messages, temperature, max_tokens, str(e)
                )
            raise

        self.last_latency_ms = (time.time() - start) * 1000
        self._update_usage(response)
        return response.choices[0].message.content or ""

    # ==================== 结构化输出 ====================

    async def generate_structured(
        self,
        messages: list[Message],
        output_schema: type[AgentAction],
        temperature: float = 0.0,
        model: str | None = None,
    ) -> AgentAction:
        """
        结构化输出生成 (v2 增强版)

        策略（双重保障）：
        1. 尝试原生 JSON mode（如果模型支持）
        2. Fallback: prompt 约束 + JSON 提取

        Args:
            messages: 对话历史
            output_schema: 输出的 Pydantic 模型类
            temperature: 温度参数
            model: 覆盖默认模型

        Returns:
            解析后的 Pydantic 模型实例
        """
        use_model = model or self.model
        schema_json = output_schema.model_json_schema()

        start = time.time()
        self.total_calls += 1

        # 准备消息（追加格式指令）
        openai_messages = [m.to_openai_format() for m in messages]
        format_instruction = self._build_format_instruction(output_schema)

        # 注入格式指令到 system message
        self._inject_format_instruction(openai_messages, format_instruction)

        # 调用 LLM
        try:
            # 尝试 JSON mode
            kwargs: dict[str, Any] = {
                "model": use_model,
                "messages": openai_messages,
                "temperature": max(0.0, min(temperature, 1.0)),
                "max_tokens": 4096,
            }

            # 原生 JSON mode（qwen-plus 等都支持）
            if self.model_info is None or self.model_info.supports_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**kwargs)

        except Exception as e:
            if self.fallback_models:
                try:
                    return await self._structured_fallback(
                        messages, output_schema, temperature, str(e)
                    )
                except Exception:
                    pass
            # 重试不带 json_mode
            try:
                kwargs.pop("response_format", None)
                response = await self.client.chat.completions.create(**kwargs)
            except Exception as e2:
                return self._error_action(f"LLM调用失败: {e2}")

        self.last_latency_ms = (time.time() - start) * 1000
        self._update_usage(response)

        raw_text = response.choices[0].message.content or "{}"

        # 解析 JSON
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            data = self._extract_json(raw_text)

        # 校验与修复
        try:
            return output_schema.model_validate(data)
        except Exception as ve:
            # 尝试修复常见问题
            repaired = self._repair_action(data, raw_text)
            try:
                return output_schema.model_validate(repaired)
            except Exception:
                return self._error_action(f"结构化输出解析失败: {ve}")

    # ==================== 流式生成 ====================

    async def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        model: str | None = None,
        callbacks: StreamCallbacks | None = None,
    ) -> AsyncIterator[str]:
        """
        流式文本生成 (v2 增强版)

        支持两种消费方式：
        1. async for token in adapter.stream(messages): ...  （迭代器）
        2. callbacks.on_token / on_complete / on_error          （回调）

        回调优先级高于迭代器——如果传了 callbacks，token 也会通过回调推送。

        Args:
            messages: 对话历史
            temperature: 温度参数
            max_tokens: 最大输出 token
            model: 覆盖默认模型
            callbacks: 流式回调（可选，与构造时传入的合并）

        Yields:
            每次 yield 一个 token 片段
        """
        use_model = model or self.model
        openai_messages = [m.to_openai_format() for m in messages]
        merged_callbacks = self._merge_callbacks(callbacks)

        start = time.time()
        self.total_calls += 1
        full_text = ""

        try:
            stream_resp = await self.client.chat.completions.create(
                model=use_model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream_resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_text += token

                    # 推送到回调
                    if merged_callbacks.on_token:
                        merged_callbacks.on_token(token)

                    yield token

                # 推理模型的思考过程（DeepSeek-R1）
                if hasattr(chunk.choices[0].delta, 'reasoning_content'):
                    rc = chunk.choices[0].delta.reasoning_content
                    if rc and merged_callbacks.on_thinking:
                        merged_callbacks.on_thinking(rc)

        except Exception as e:
            if merged_callbacks.on_error:
                merged_callbacks.on_error(str(e))
            raise

        self.last_latency_ms = (time.time() - start) * 1000

        # 完成回调
        if merged_callbacks.on_complete:
            merged_callbacks.on_complete(full_text)

    # ==================== 流式 + 结构化（用于 AgentLoop） ====================

    async def stream_structured(
        self,
        messages: list[Message],
        output_schema: type[AgentAction],
        temperature: float = 0.0,
        model: str | None = None,
        callbacks: StreamCallbacks | None = None,
    ) -> AgentAction:
        """
        流式输出 + 最终返回结构化 Action

        先将所有 token 流式推送（前端实时展示），最后解析为 AgentAction。
        用于 AgentLoop 中让用户看到 Agent 的实时思考过程。

        Returns:
            解析后的 AgentAction
        """
        merged_callbacks = self._merge_callbacks(callbacks)
        full_text = ""

        async for token in self.stream(messages, temperature, 4096, model, callbacks):
            full_text += token

        # 从完整文本中解析结构化输出
        try:
            data = json.loads(full_text)
        except json.JSONDecodeError:
            data = self._extract_json(full_text)

        try:
            return output_schema.model_validate(data)
        except Exception:
            return self._error_action(f"流式输出解析失败: {full_text[:200]}")

    # ==================== Embedding ====================

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """
        文本向量化

        Args:
            texts: 待向量化的文本列表
            model: embedding 模型名（默认 text-embedding-v1）

        Returns:
            向量列表
        """
        emb_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v1")

        resp = await self.client.embeddings.create(
            model=emb_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    # ==================== 辅助方法 ====================

    def _build_format_instruction(self, output_schema: type) -> str:
        """构建结构化输出格式指令"""
        schema_json = output_schema.model_json_schema()
        return (
            f"\n\n你必须以 JSON 格式输出，严格遵守以下 Schema：\n"
            f"```json\n{json.dumps(schema_json, ensure_ascii=False, indent=2)}\n```\n"
            f"只输出 JSON，不要输出任何其他内容。"
        )

    def _inject_format_instruction(self, openai_messages: list[dict], instruction: str):
        """将格式指令注入 system message"""
        for msg in openai_messages:
            if msg["role"] == "system":
                msg["content"] += instruction
                return
        # 没有 system message，插入一条
        openai_messages.insert(0, {"role": "system", "content": instruction.strip()})

    def _extract_json(self, text: str) -> dict[str, Any]:
        """
        从文本中提取 JSON（LLM 输出不严格时的降级策略）

        依次尝试：
        1. ```json ... ``` 代码块
        2. { ... } 首尾花括号
        3. 错误 fallback
        """
        # 尝试提取代码块
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 找第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # 完全无法解析
        return {
            "action_type": "final_answer",
            "thought": "JSON解析失败",
            "answer": f"内部错误：无法解析LLM输出。原始输出：{text[:200]}",
        }

    def _repair_action(self, data: dict, raw_text: str) -> dict:
        """修复常见的结构化输出问题"""
        # action_type 常见错误修复
        if "action_type" in data:
            at = str(data["action_type"]).lower()
            if at in ("final_answer", "tool_call", "spawn_subagent"):
                pass  # 正确值，无需修复
            elif "answer" in at or "final" in at:
                data["action_type"] = "final_answer"
            elif "tool" in at or "sub" in at or "spawn" in at:
                data["action_type"] = "tool_call" if "tool" in at else "spawn_subagent"
            else:
                # action_type 是一个未知值（很可能是 LLM 把工具名填到了这里）
                # 修复：如果存在 tool_call 字段 → 改成 tool_call
                #       如果存在 answer 字段 → 改成 final_answer
                #       否则 → 尝试当作工具名，自动构建 tool_call
                if data.get("tool_call") or data.get("tool_name"):
                    data["action_type"] = "tool_call"
                elif data.get("answer"):
                    data["action_type"] = "final_answer"
                else:
                    # LLM 把工具名填到了 action_type，自动构建 tool_call
                    data["action_type"] = "tool_call"
                    if not data.get("tool_call"):
                        data["tool_call"] = {
                            "tool_name": str(data.get("action_type", "")).strip() or at,
                            "args": data.get("args", data.get("query", {})),
                        }
                    # 如果 args 是字符串，包成 {"query": args}
                    tc = data.get("tool_call", {})
                    if isinstance(tc.get("args"), str):
                        tc["args"] = {"query": tc["args"]}
                    data["tool_call"] = tc

        # 如果 action_type=final_answer 但没有 answer 字段
        if data.get("action_type") == "final_answer" and "answer" not in data:
            data["answer"] = data.get("thought", raw_text[:500])

        # 确保 thought 字段存在
        if "thought" not in data:
            data["thought"] = ""

        return data

    def _error_action(self, error: str) -> AgentAction:
        """生成错误标记的 AgentAction"""
        return AgentAction(
            action_type="final_answer",  # type: ignore
            thought=f"系统错误: {error}",
            answer=f"抱歉，AI 服务处理您的请求时遇到问题。错误：{error}",
        )

    async def _try_fallback(
        self, messages: list[Message], temperature: float, max_tokens: int, original_error: str
    ) -> str:
        """依次尝试降级模型"""
        for fb_model in self.fallback_models:
            try:
                print(f"   ⚠️ 主模型失败 ({original_error[:60]}...)，降级到 {fb_model}")
                return await self.generate(messages, temperature, max_tokens, model=fb_model)
            except Exception:
                continue
        raise RuntimeError(f"所有模型均失败（主模型+{len(self.fallback_models)}个备选）: {original_error}")

    async def _structured_fallback(
        self, messages: list[Message], output_schema: type, temperature: float, original_error: str
    ) -> AgentAction:
        """结构化输出的降级尝试"""
        for fb_model in self.fallback_models:
            try:
                print(f"   ⚠️ 主模型结构化输出失败，降级到 {fb_model}")
                return await self.generate_structured(messages, output_schema, temperature, model=fb_model)
            except Exception:
                continue
        raise RuntimeError(f"结构化输出所有模型均失败: {original_error}")

    def _merge_callbacks(self, callbacks: StreamCallbacks | None) -> StreamCallbacks:
        """合并构造时回调和调用时回调"""
        if callbacks is None:
            return self.callbacks or StreamCallbacks()
        if self.callbacks is None:
            return callbacks

        return StreamCallbacks(
            on_token=lambda t: (self.callbacks.on_token and self.callbacks.on_token(t))
                                or (callbacks.on_token and callbacks.on_token(t)),
            on_thinking=callbacks.on_thinking or self.callbacks.on_thinking,
            on_complete=callbacks.on_complete or self.callbacks.on_complete,
            on_error=callbacks.on_error or self.callbacks.on_error,
        )

    def _update_usage(self, response):
        """更新用量统计"""
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens_input += response.usage.prompt_tokens
            self.total_tokens_output += response.usage.completion_tokens
            if self.model_info:
                self.total_cost += (
                    response.usage.prompt_tokens * self.model_info.cost_per_1k_input / 1000
                    + response.usage.completion_tokens * self.model_info.cost_per_1k_output / 1000
                )

    def get_usage_summary(self) -> dict:
        """获取用量统计摘要"""
        return {
            "model": self.model,
            "provider": self.provider.value,
            "total_calls": self.total_calls,
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_cost_usd": round(self.total_cost, 6),
            "last_latency_ms": round(self.last_latency_ms, 1),
        }

    def __repr__(self) -> str:
        return f"LLMAdapter(model={self.model!r}, provider={self.provider.value}, calls={self.total_calls})"


# ==================== 工厂函数 ====================

def create_llm_adapter(
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> LLMAdapter:
    """
    便捷工厂函数

    使用方式：
        # DeepSeek
        llm = create_llm_adapter("deepseek-chat")

        # 通义千问
        llm = create_llm_adapter("qwen-plus")

        # 自定义
        llm = create_llm_adapter("my-model", base_url="http://localhost:8080/v1")
    """
    prov = None
    if provider:
        try:
            prov = Provider(provider)
        except ValueError:
            prov = Provider.CUSTOM

    return LLMAdapter(
        model=model,
        provider=prov,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )
