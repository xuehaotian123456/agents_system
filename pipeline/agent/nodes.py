"""LangGraph 四节点: Planner → Retriever → Reflector → Summarizer"""
import json
from datetime import datetime
from model.factory import robust_llm_call
from agent.state import AgentState
from agent.tools import ALL_TOOLS
from utils.logger_handler import logger

# ========== Planner ==========

PLANNER_PROMPT = """你是一个技术助手规划器。精选 2-3 个最相关的工具，不要堆砌。

用户问题: {query}

可用工具: {tools}

选工具规则:
- 知识/编程问题 → rag_search + code_example
- 技术对比 → compare_tech + kg_lookup
- 热榜/趋势 → trending_list 或 daily_digest 或 trend_report
- 实体查询 → kg_lookup
- 联网搜索 → search_web（仅知识库不足时）
- 工具创建 → create_tool

请以 JSON 格式输出:
{{
    "intent": "用户意图",
    "reasoning": "为什么选这些工具",
    "requires_search": true/false,
    "suggested_tools": ["工具1", "工具2"],
    "target_entity": "kg_lookup要查的实体名",
    "tool_params": {{"create_tool时必填: name, description, keywords(逗号分隔)"}},
    "sub_queries": ["子问题"]
}}"""

def _rule_based_plan(query: str) -> dict:
    """
    规则兜底：当 Planner LLM JSON 解析失败时，根据 query 关键词智能选择工具。
    不依赖 LLM 输出质量，保证系统在最坏情况下也能运行。
    """
    q = query.lower()
    # 对比类
    if any(kw in q for kw in ["对比", "vs", "区别", "比较", "哪个好", "优缺点"]):
        return {"intent": query, "requires_search": True,
                "suggested_tools": ["compare_tech", "rag_search"], "target_entity": ""}
    # 热榜/趋势类
    if any(kw in q for kw in ["热榜", "热门", "趋势", "最新", "最近", "今天"]):
        return {"intent": query, "requires_search": True,
                "suggested_tools": ["trending_list"], "target_entity": ""}
    # 实体关联/报错类 → 优先 KG
    if any(kw in q for kw in ["关联", "相关", "关系", "依赖", "报错", "错误", "异常"]):
        return {"intent": query, "requires_search": True,
                "suggested_tools": ["kg_lookup", "rag_search"], "target_entity": ""}
    # 邮件/配置类
    if any(kw in q for kw in ["邮件", "发送", "推送", "订阅", "摘要"]):
        return {"intent": query, "requires_search": True,
                "suggested_tools": ["send_digest_email"], "target_entity": ""}
    # 默认: RAG 检索
    return {"intent": query, "requires_search": True,
            "suggested_tools": ["rag_search"], "target_entity": ""}

def planner_node(state: AgentState) -> AgentState:
    query = state["query"]
    tool_names = [t.name for t in ALL_TOOLS]
    prompt = PLANNER_PROMPT.format(query=query, tools=", ".join(tool_names))

    try:
        resp = robust_llm_call(prompt)
        content = resp.content if hasattr(resp, 'content') else str(resp)
        # 提取 JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            plan = json.loads(content[start:end])
        else:
            plan = _rule_based_plan(query)
            state.setdefault("errors", []).append({
                "node": "planner", "error": "JSON提取失败",
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        plan = _rule_based_plan(query)
        state.setdefault("errors", []).append({
            "node": "planner", "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        logger.warning(f"[Planner] LLM 解析失败，使用规则兜底: {e}")

    state["plan"] = plan
    logger.info(f"[Planner] intent={plan.get('intent', '')[:60]}")
    return state

# ========== Retriever ==========

def retriever_node(state: AgentState) -> AgentState:
    """执行工具调用 — 按 Planner 建议调用工具，获取实际数据"""
    plan = state.get("plan", {})
    query = state["query"]
    suggested = plan.get("suggested_tools", ["rag_search"])
    context_parts = []
    tools_map = {t.name: t for t in ALL_TOOLS}

    # 如果已有实时数据工具（trending_list/search_web/fetch_article），跳过 rag_search 避免噪音
    has_realtime_tools = any(t in suggested for t in ["trending_list", "search_web", "fetch_article"])

    # 记录用户查询到记忆系统
    try:
        from services.memory import remember_query
        remember_query(query)
    except Exception:
        pass

    # KG 多跳实体扩散：为所有检索类工具提供扩展词
    kg_expanded_terms = []
    kg_chain_text = ""
    try:
        from rag.knowledge_graph import get_kg
        kg = get_kg()
        if kg.is_built:
            mh = kg.multi_hop_expand(query, max_hops=2, top_per_hop=3)
            if mh and mh.get("expanded_entities"):
                kg_expanded_terms = mh["expanded_entities"][:10]
                kg_chain_text = mh.get("chain_text", "")
                logger.info(f"[KG多跳扩散] {len(kg_expanded_terms)} 实体, 链: {kg_chain_text[:100]}")
    except Exception:
        pass

    # 限制工具数 ≤ 3 避免冗余调用
    suggested = suggested[:3]

    # 执行 Planner 建议的每个工具
    failed_tools = []
    for tool_name in suggested:
        # 有实时工具时跳过 rag_search
        if has_realtime_tools and tool_name == "rag_search":
            continue
        tool = tools_map.get(tool_name)
        if not tool:
            continue

        skip_no_arg = {"save_article", "custom_search"}  # create_tool 参数由 Planner 提供

        if tool_name in skip_no_arg:
            continue

        try:
            # 通用调用：根据工具签名自动路由
            if tool_name == "rag_search":
                # 注入 KG 多跳扩散词，提升 RAG 召回
                enriched_query = query
                if kg_expanded_terms:
                    enriched_query = query + " " + " ".join(kg_expanded_terms[:5])
                result = tool.invoke({"query": enriched_query})
            elif tool_name == "trending_list":
                result = tool.invoke({"source": "juejin"})
            elif tool_name == "fetch_article":
                # 从 query 或 context 中提取 URL
                import re
                urls = re.findall(r'https?://[^\s]+', state.get("context", "") + " " + query)
                if urls:
                    result = tool.invoke({"url": urls[0]})
                else:
                    continue
            elif tool_name == "kg_lookup":
                # 用 Planner 提取的实体名，否则从 query 智能提取
                target = plan.get("target_entity", "")
                if not target:
                    import jieba
                    words = [w for w, flag in jieba.posseg(query) if flag in ('n','nr','nz','eng','x') and len(w)>=2]
                    target = words[0] if words else query[:20]
                result = tool.invoke({"entity_name": target})
            elif tool_name == "search_web":
                result = tool.invoke({"query": query})
            elif tool_name == "trend_report":
                result = tool.invoke({})
            elif tool_name == "daily_digest":
                result = tool.invoke({})
            elif tool_name == "compare_tech":
                # 用 target_entity 或 jieba 提取两个技术名词
                target = plan.get("target_entity", "")
                if target and "和" in target:
                    parts = target.replace(" 和 ", "|").replace(" vs ", "|").split("|")
                elif target and "与" in target:
                    parts = target.replace(" 与 ", "|").split("|")
                else:
                    # 从 query 用 jieba 提取名词
                    import jieba.posseg as pseg
                    nouns = [w for w, f in pseg.cut(query) if f in ('n','nr','nz','eng') and len(w)>=2]
                    parts = nouns[:2]
                techs = [p.strip() for p in parts if len(p.strip()) >= 2][:2]
                if len(techs) >= 2:
                    result = tool.invoke({"tech_a": techs[0], "tech_b": techs[1]})
                else:
                    continue
            elif tool_name == "code_example":
                kw = plan.get("target_entity", query[:20])
                result = tool.invoke({"keyword": kw})
            elif tool_name == "create_tool":
                tp = plan.get("tool_params", {})
                result = tool.invoke({
                    "name": tp.get("name", "custom_tool"),
                    "description": tp.get("description", "用户创建的自定义工具"),
                    "keywords": tp.get("keywords", query)
                })
            elif tool_name == "custom_search":
                tp = plan.get("tool_params", {})
                result = tool.invoke({
                    "tool_name": tp.get("name", "custom_tool"),
                    "query": query
                })
            elif tool_name == "user_profile":
                result = tool.invoke({})
            elif tool_name == "get_current_time":
                result = tool.invoke({})
            else:
                result = tool.invoke({"query": query})

            result_str = str(result.content) if hasattr(result, 'content') else str(result)
            context_parts.append(f"[{tool_name}]: {result_str}")
            state["tool_calls"].append({"tool": tool_name, "result": result_str[:800]})
            logger.info(f"[Retriever] {tool_name} 完成: {len(result_str)} 字符")
        except Exception as e:
            failed_tools.append(tool_name)
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            logger.warning(f"[Retriever] {tool_name} 失败: {error_msg}")
            context_parts.append(f"[{tool_name}]: 执行异常 - {error_msg}")
            state.setdefault("errors", []).append({
                "node": "retriever", "tool": tool_name, "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })

    # 降级：所有工具均失败 → 返回明确错误信息，让 LLM 诚实告知用户
    if not context_parts and failed_tools:
        logger.warning(f"[Retriever] 所有工具失败 ({failed_tools})，触发降级")
        state["degradation_triggered"] = True
        context_parts.append(
            "当前所有检索工具均暂时不可用。请诚实告知用户这一情况，"
            "建议用户稍后重试或换个方式提问。不要编造信息。"
        )

    # ── 注入 KG 多跳推理链到上下文 ──
    if kg_chain_text:
        context_parts.append(
            "[KG 推理链] (GraphRAG 多跳扩散):\n" + kg_chain_text)

    state["context"] = "\n\n---\n".join(context_parts) if context_parts else "无可用信息"
    state["retrieval_rounds"] += 1
    return state

# ========== Reflector ==========

REFLECTOR_PROMPT = """评估当前工具返回的信息是否足以回答用户问题。严格判定！

用户问题: {query}
已获取信息: {context}

判定规则:
- 如果信息包含实际数据（文章标题/内容/搜索结果），即使不完美也输出 pass（工具返回的是实时数据，不需要质疑时效性）
- 只有明确空数据、异常信息才输出 retry
- 热榜类工具返回的即是实时数据，直接判 pass

请以 JSON 格式输出:
{{"verdict": "pass" 或 "retry", "confidence": 0.0-1.0, "reason": "判断理由", "rewritten_query": "改写后的查询（仅retry时）"}}"""

def reflector_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = state.get("context", "")

    if state["retry_count"] >= state.get("max_retries", 2):
        state["confidence"] = 0.5
        logger.info(f"[Reflector] 已达最大重试次数，强制通过")
        return state

    if not context or any(kw in context for kw in ["无可用信息", "暂无数据", "加载中", "信息不足"]):
        state["confidence"] = 0.3
        if state["retry_count"] < state.get("max_retries", 2):
            state["retry_count"] += 1
            state["query"] = f"换个角度搜索: {query}"
            # 追加 tool 建议
            plan = state.get("plan", {})
            plan["suggested_tools"] = plan.get("suggested_tools", []) + ["search_web"]
            state["plan"] = plan
        return state

    try:
        prompt = REFLECTOR_PROMPT.format(query=query, context=context[:2000])
        resp = robust_llm_call(prompt)
        content = resp.content if hasattr(resp, 'content') else str(resp)
        start = content.find("{")
        end = content.rfind("}") + 1
        reflection = json.loads(content[start:end]) if start >= 0 and end > start else {"verdict": "pass", "confidence": 0.6}
    except Exception as e:
        # LLM 调用失败 → 跳过反思，直接进入 Summarizer
        logger.warning(f"[Reflector] LLM 调用失败，跳过反思: {e}")
        state.setdefault("errors", []).append({
            "node": "reflector", "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        reflection = {"verdict": "pass", "confidence": 0.5}

    state["confidence"] = reflection.get("confidence", 0.6)
    verdict = reflection.get("verdict", "pass")

    if verdict == "retry" and state["retry_count"] < state.get("max_retries", 2):
        state["retry_count"] += 1
        state["query"] = reflection.get("rewritten_query", query)
        logger.info(f"[Reflector] RETRY: {state['query'][:80]}")
    else:
        logger.info(f"[Reflector] PASS: confidence={state['confidence']}")

    return state

# ========== Summarizer ==========

SUMMARIZER_PROMPT = """根据以下信息，回答用户问题。请用中文，简洁专业。

用户问题: {query}
参考资料: {context}

要求:
1. 如果信息充分，直接回答。工具返回的实时数据（热榜、搜索结果）是可信的，不需要质疑其时效性
2. 如果信息不足，诚实说明
3. 是技术问题则给出代码示例或操作步骤
4. 热榜类问题：直接列出文章并给出推荐理由"""

def summarizer_node(state: AgentState) -> AgentState:
    prompt = SUMMARIZER_PROMPT.format(query=state["original_query"], context=state.get("context", "")[:3000])
    try:
        resp = robust_llm_call(prompt)
        state["answer"] = resp.content if hasattr(resp, 'content') else str(resp)
    except Exception as e:
        state["answer"] = f"生成回答时出现错误: {e}"

    # 透传可视化标记（不被 LLM 吃掉）
    context = state.get("context", "")
    for marker in ["[VIZ:", "[GRAPH:"]:
        idx = context.find(marker)
        if idx >= 0:
            end = context.find("]", idx) + 1
            if end > idx:
                state["answer"] += "\n\n" + context[idx:end]

    return state
