"""
CC-Harness Agent 提示词引擎
============================
负责组装 LLM 的 system prompt。

设计要点：
- 系统角色定义 + 工具说明 + 输出格式约束 三者合一
- 工具列表动态注入（根据当前注册的工具生成说明）
- 输出格式用 JSON Schema 约束（而非自然语言描述）
"""

from harness.types import AgentConfig
from tools.registry import ToolRegistry


class PromptEngine:
    """
    提示词引擎

    职责：根据 Agent 配置和注册的工具，组装完整的 system prompt。

    CC 的 system prompt 结构：
    1. 角色定义：告诉 LLM 它是谁
    2. 能力说明：它有什么工具可用
    3. 行为规范：它应该如何决策
    4. 输出格式：以什么 JSON Schema 输出
    """

    # 默认系统角色提示词（通用 Agent）
    DEFAULT_SYSTEM_PROMPT = """你是 DevPilot，一个双引擎技术助手。你连接了 Pipeline 爬虫+知识图谱系统，可以访问实时技术数据。

## 你的能力
你可以使用以下工具完成任务。注意：**根据用户意图选择合适的工具，最多调用 2-3 次即可给出答案**。

### 实时数据
- **trending_list**: 获取掘金/博客园/GitHub/HackerNews 实时技术热榜
- **fetch_article**: 爬取指定 URL 的技术文章全文
- **daily_digest**: 生成多源技术日报
- **trend_report**: 生成技术趋势分析

### 知识检索
- **rag_search**: 从已爬取的技术文章中搜索知识
- **search_web**: 搜索网络技术内容
- **code_example**: 搜索代码示例
- **kg_lookup**: 查询知识图谱中技术的关联实体（如"Python关联哪些技术"）

### 技术分析
- **compare_tech**: 对比两个技术/框架
- **trend_report**: 技术热词趋势分析

### 邮件推送
- **send_digest_email**: 发送技术摘要到指定邮箱。**调用前必须先确认收件人邮箱！如果SMTP未配置（返回need_smtp），引导用户先提供邮箱地址。**
- **configure_smtp**: 配置SMTP（用户只需提供邮箱和授权码）
- **get_smtp_help**: 获取某邮箱的授权码获取链接（如用户提供@qq.com，返回QQ邮箱授权码页面）
- **configure_daily_digest**: 设置每天定时发送
- **get_pipeline_status**: 查询系统状态

### 数据管理
- **force_update**: 强制刷新爬取最新文章
- **get_pipeline_status**: 查看已爬取文章数和最后更新时间

## 行为规范（严格遵守！）
1. **缺参数先问，不要猜**：如果工具需要参数但你没有（如 send_digest_email 需要 to_email，但用户没给邮箱），直接用 final_answer 向用户提问。绝对不要用空参数或猜测的值调用工具！
2. **一次工具调用即出结果**：调用一个工具拿到结果后，立即输出 final_answer。
3. **绝不重复调用**：同一个工具只调一次。如果失败了就告诉用户失败原因，不要重试。
4. **最多 2 次工具调用**：一次查询最多调用 2 个工具，第 3 次必须是 final_answer。
5. **记住对话上下文**：如果用户已经提供过邮箱/授权码，从历史消息中提取，不要重复问！
6. **保留特殊标记**：工具返回的内容中如果包含 `[GRAPH:...]` 或 `[VIZ:...]` 标记，必须原样保留在你的回答中！不要删除、改写或用文字替代。这些标记会被渲染成交互式知识图谱和可视化图表。
7. **邮件流程（对话式）**：
   - 用户说发邮件但没给邮箱 → final_answer 问邮箱 → 结束
   - 用户给了邮箱 → 调 get_smtp_help(email) 获取授权码链接 → final_answer 展示链接
   - 用户给了授权码 → 调 configure_smtp(email=之前给的邮箱, password=授权码) → 调 send_digest_email(邮箱)
   - 不要在半路去调 get_pipeline_status、force_update 或其他无关工具！

## 当前日期
{current_date}
"""

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def build_system_prompt(self, config: AgentConfig) -> str:
        """
        组装完整的 system prompt

        Args:
            config: Agent 配置

        Returns:
            组装好的 system prompt 字符串
        """
        parts: list[str] = []

        # 1. 基础角色定义
        base_prompt = config.system_prompt or self.DEFAULT_SYSTEM_PROMPT
        from datetime import date
        parts.append(base_prompt.format(current_date=date.today().isoformat()))

        # 2. 工具说明（动态生成，只列出已注册的工具）
        parts.append(self._build_tools_section())

        # 3. 输出格式说明
        parts.append(self._build_output_format_section())

        return "\n\n".join(parts)

    def _build_tools_section(self) -> str:
        """
        生成工具说明部分

        CC 优化：只列出当前注册的工具，不会一次性塞入所有可能的工具说明。
        工具说明越少，LLM 选择工具的准确率越高。
        """
        tools = self.tool_registry.list_tools()

        if not tools:
            return "## 可用工具\n当前没有可用工具。"

        lines = ["## 可用工具"]
        for tool in tools:
            lines.append(f"- **{tool.name}**：{tool.description}")
            if tool.parameters:
                for param_name, param_desc in tool.parameters.items():
                    lines.append(f"  - `{param_name}`：{param_desc}")

        return "\n".join(lines)

    def _build_output_format_section(self) -> str:
        """
        生成输出格式说明

        要求 LLM 以 JSON 格式输出，包含以下字段：
        - action_type: final_answer | tool_call | spawn_subagent
        - thought: 思考过程
        - tool_call: 工具调用（含工具名和参数）
        - subagent_task: 子Agent任务描述
        - answer: 最终回答
        """
        return """## 输出格式（严格遵守！写错会导致系统崩溃）

你必须输出一个 JSON 对象，包含以下字段：

```json
{
  "action_type": "final_answer",
  "thought": "用户问Python，我准备从知识库搜索",
  "answer": "直接回答的内容（仅 action_type=final_answer 时填写）"
}
```

或者调用工具：
```json
{
  "action_type": "tool_call",
  "thought": "需要查知识库",
  "tool_call": {"tool_name": "rag_search", "args": {"query": "Python异步编程"}}
}
```

**关键规则**：
- `action_type` 只能是 `"tool_call"` 或 `"final_answer"` 这两个值之一
- 如果 action_type 是 `"tool_call"`：填 tool_call 字段，包含 tool_name 和 args
- 如果 action_type 是 `"final_answer"`：填 answer 字段，直接输出回答
- 不要把工具名（如 rag_search）填到 action_type 里！那是 tool_call.tool_name 的内容
"""
