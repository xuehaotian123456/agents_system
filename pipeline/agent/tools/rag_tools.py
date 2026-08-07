"""RAG + KG 工具"""
import os
from pathlib import Path
from langchain_core.tools import tool
from utils.logger_handler import logger

_rag_service = None

def _get_rag():
    global _rag_service
    if _rag_service is None:
        from rag.vector_store import VectorStore
        vs = VectorStore()
        vs.load_articles()
        _rag_service = vs
    return _rag_service

@tool(description="从已入库的技术文章中搜索相关知识。入参 query（搜索词），返回相关文章内容和摘要。内部使用KG实体扩散优化检索效率。用于技术学习、问题解答等场景。")
def rag_search(query: str) -> str:
    try:
        vs = _get_rag()
        from model.factory import robust_llm_call

        # === KG 驱动检索优化 ===
        expanded_query = query
        kg_tokens_saved = 0
        try:
            from rag.knowledge_graph import get_kg
            kg = get_kg()
            if kg.is_built:
                expanded_terms = kg.one_hop_expand(query, max_entities=3)
                if expanded_terms:
                    expanded_query = query + " " + " ".join(expanded_terms[:5])
                    # 估算 token 节省: 不用逐个调工具去搜扩展词
                    kg_tokens_saved = len(expanded_terms) * 200
                    logger.info(f"[rag_search] KG扩散: {expanded_terms[:5]} (约省{kg_tokens_saved} tokens)")
        except Exception:
            pass

        # 用扩展后的 query 检索
        docs = vs.search(expanded_query, top_k=5)
        if not docs:
            # 回退到原始 query
            docs = vs.search(query, top_k=5)
        if not docs:
            return "未找到相关文章。"

        context = "\n\n---\n".join([f"[来源{i+1}: {d.metadata.get('source', '知识库')}]\n{d.page_content[:600]}" for i, d in enumerate(docs)])
        context_tokens = len(context) // 2
        prompt = f"""你是一个技术助手。根据以下参考资料回答用户问题。**每条关键信息后必须标注来源编号**。

参考资料（{len(docs)} 篇）:
{context}

用户问题: {query}

请用中文回答:
1. 给出具体答案，每条关键信息标注来源编号如 [来源1]
2. 如果资料信息不足，诚实说明
3. 如果有代码示例，给出完整可运行的代码"""

        resp = robust_llm_call(prompt)
        answer = resp.content if hasattr(resp, 'content') else str(resp)

        # 附加效率报告
        if kg_tokens_saved > 0:
            answer += f"\n\n💡 *KG 驱动检索: 实体扩散发现 '{', '.join(expanded_terms[:3] if expanded_terms else [])}', 约节省 {kg_tokens_saved} tokens。*"

        # === RAG 缓存：问答结果持久化入向量库，下次同类问题直接命中 ===
        try:
            # 去重：检查近 20 条缓存中是否有相似问题
            existing = vs.store.get(where={"source": "rag_cache"}, include=["metadatas"])
            is_dup = False
            if existing and existing.get("metadatas"):
                for meta in existing["metadatas"][-20:]:
                    if meta.get("query", "") == query[:100]:
                        is_dup = True
                        break
            if not is_dup:
                cache_doc = f"Q: {query}\nA: {answer}"
                vs.store.add_texts(
                    texts=[cache_doc],
                    metadatas=[{"source": "rag_cache", "query": query[:100], "cached": "true"}]
                )
                vs.all_chunks.append(cache_doc)
                answer += f"\n\n📥 *已缓存至向量库（共 {len(vs.all_chunks)} chunks）*"
                logger.info(f"[rag_cache] 缓存: {query[:50]}... ({len(vs.all_chunks)} total chunks)")
        except Exception:
            pass

        return answer
    except Exception as e:
        logger.error(f"[rag_search] {e}")
        return f"搜索异常: {e}"

@tool(description="查询知识图谱中的技术关键词实体信息。入参 entity_name（技术名词），返回该实体的频次、关联实体和知识图谱可视化。")
def kg_lookup(entity_name: str) -> str:
    try:
        from rag.knowledge_graph import get_kg
        kg = get_kg()
        if not kg.is_built:
            # 懒加载：尝试从向量库构建
            try:
                vs = _get_rag()
                if vs.all_chunks:
                    kg.build(vs.all_chunks)
            except Exception:
                pass
            if not kg.is_built:
                return "知识图谱尚未构建。请先导入文章数据。"

        info = kg.get_entity(entity_name)
        if not info:
            related = kg.search(entity_name, limit=5)
            if related:
                return f"未找到'{entity_name}'，相关实体: " + ", ".join(r['name'] for r in related)
            return f"知识图谱中未找到'{entity_name}'。"

        # 生成交互式图谱
        graph_html = ""
        try:
            from services.graph_viz import build_entity_graph
            fpath = build_entity_graph(entity_name)
            if fpath:
                graph_html = f"\n\n[GRAPH:{fpath}]"
        except Exception:
            pass

        lines = [f"实体: {info['name']}", f"频次: {info['freq']}",
                 f"覆盖文档块: {info['chunks']}", "关联实体:"]
        for r in info['related']:
            lines.append(f"  • {r['entity']} (共现 {r['co_occur']} 次)")

        return "\n".join(lines) + graph_html
    except Exception as e:
        return f"KG查询异常: {e}"

@tool(description="将文章内容或技术笔记保存到本地知识库。入参 title（标题）和 content（Markdown格式内容）。保存后可供后续检索使用。" )
def save_article(title: str, content: str) -> str:
    try:
        data_dir = Path(__file__).parent.parent.parent / "data" / "articles"
        os.makedirs(data_dir, exist_ok=True)
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        fname = f"user_{safe_title}.md"
        fpath = data_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{content}")
        return f"文章已保存: {fname}"
    except Exception as e:
        return f"保存失败: {e}"
