# CC-Harness Agent

> 基于 Claude Code Harness 架构思想的自研轻量级 Agent 框架
>
> **零 LangChain / LangGraph 依赖 | 纯异步 asyncio | Pydantic v2**

---

## 一、为什么自研框架？

### 与 LangGraph 路线的核心差异

| 维度 | 本框架（CC 路线） | LangGraph 路线 |
|------|-------------------|----------------|
| **核心引擎** | 异步 `AgentLoop` 循环引擎 | 通用 `StateGraph` 状态图引擎 |
| **流程定义** | 运行时由 LLM **动态决策** | 开发阶段**静态定义** Node + Edge |
| **多 Agent** | 主 Agent `spawn` Subagent（层级派生） | 多个独立节点共享全局 State |
| **RAG 定位** | 普通 Tool，Agent 按需调用 | 图节点，属于顶层工作流 |
| **上下文管理** | Session 内置自动压缩 | 开发者自行实现截断/压缩 |
| **依赖体量** | 极简（openai + chromadb + pydantic） | 重度依赖 LangChain 生态 |
| **适用场景** | 开放式任务、自主规划 Agent | 固定流程、明确分支的流水线 |

### LangChain 的痛点（也是公司选择自研的原因）

1. **抽象过重**：`BaseMessage` → `HumanMessage`/`AIMessage`/`ToolMessage`/`SystemMessage` 继承链复杂
2. **依赖繁杂**：安装 LangChain 连带安装 50+ 个包
3. **私有化难裁剪**：Checkpoint、Callback、Memory 等模块耦合紧密
4. **版本破坏性变更**：0.x → 1.x 大量 API 重命名
5. **不适合政企合规**：状态持久化、权限管控不好定制

---

## 二、架构全景图

```
┌──────────────────────────────────────────────────────────┐
│                    接入层（待扩展）                         │
│          HTTP/WebSocket 会话管理、流式推送                  │
├──────────────────────────────────────────────────────────┤
│                    会话管理层 Session                       │
│  • 对话历史管理（消息追加、上下文压缩）                      │
│  • 工具调用结果注入                                        │
│  • 子Agent 摘要注入                                        │
│  • 循环终止判断（最大轮次/中断信号）                         │
├──────────────────────────────────────────────────────────┤
│               ★ 核心：Agent Harness 智能体层 ★              │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │              AgentLoop 主循环引擎                  │     │
│  │                                                  │     │
│  │   while can_continue():                          │     │
│  │      ① prompt 组装 ← PromptEngine                │     │
│  │      ② LLM 调用   ← LLMAdapter（结构化输出）      │     │
│  │      ③ 动作解析   → AgentAction (Pydantic)       │     │
│  │      ④ 执行调度   → Tool / SubAgent / Answer     │     │
│  │      ⑤ 观察注入   → Session.append_xxx()         │     │
│  │      ⑥ 回到 ①                                     │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ PromptEngine │  │  LLMAdapter  │  │SubAgentSpawner│   │
│  │ 提示词组装    │  │ 模型适配层   │  │ 子Agent调度器 │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
├──────────────────────────────────────────────────────────┤
│                    工具系统 Tools                          │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ BaseTool │  │ ToolRegistry│  │ RAGTool  │  ...         │
│  │ 工具基类  │  │ 注册/执行  │  │ RAG检索  │               │
│  └──────────┘  └──────────┘  └──────────┘               │
├──────────────────────────────────────────────────────────┤
│                  LLM 适配层 LLMAdapter                     │
│       统一封装 OpenAI/通义/DeepSeek，屏蔽模型差异           │
├──────────────────────────────────────────────────────────┤
│              基础设施 Infrastructure                       │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  Retry   │  │  Cache   │  │RateLimit │               │
│  │ 异步重试  │  │ Redis缓存│  │ 接口限流  │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└──────────────────────────────────────────────────────────┘
```

---

## 三、核心模块详解

### 3.1 AgentLoop —— 替代 LangGraph 的主循环

```python
# LangGraph 方式：预定义图结构
builder = StateGraph(State)
builder.add_node("judge", judge_func)
builder.add_node("retrieve", retrieve_func)
builder.add_conditional_edges("judge", lambda s: ..., {...})
graph = builder.compile()
result = graph.invoke(inputs)

# CC 方式：运行时动态决策
session = Session(config)
session.append_user_message(question)

loop = AgentLoop(session, llm, tools, prompt_engine)
answer = await loop.run()  # ← while 循环内部 LLM 自主决定每一步
```

**关键区别**：
- LangGraph 的流程在**编译时**确定（图结构固定）
- AgentLoop 的流程在**运行时**由 LLM 动态决定（每步自主选择）

### 3.2 Session —— 替代 LangGraph State + Checkpoint

```python
# LangGraph 方式
class State(TypedDict):
    messages: Annotated[list, operator.add]
    docs: list
    ...

# CC 方式：Session 对象封装所有状态
session = Session(config)
session.append_user_message("你好")
session.append_tool_result("rag_search", "检索结果...")
session.append_subagent_result("分析文档", "摘要...")

if session.should_compress():
    session.compress()  # 内置上下文压缩
```

### 3.3 RAG Tool —— RAG 是工具，不是工作流

```python
# LangGraph 方式：RAG 是图的一部分
builder.add_node("retrieve", retrieve_docs)
builder.add_node("grade", grade_documents)
builder.add_node("rewrite", rewrite_query)
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", ...)

# CC 方式：RAG 是一个 Tool，与搜索、计算器等工具平等
rag_tool = RAGTool(...)
tool_registry.register(rag_tool)

# Agent 按需调用：LLM 决定 "我需要查知识库" → 调用 rag_search 工具
```

RAG Tool 内部同样封装了 Agentic RAG 循环（检索→评分→改写→重检索），但这个循环对外部 Agent 是**透明的**——Agent 只看到工具调用和结果，不知道内部细节。

### 3.4 Subagent —— 层级式多 Agent

```
用户请求
    ↓
主 Agent（协调者）
    ├─ 判断需要子任务 → spawn SubAgent 1（RAG 检索 Agent）
    ├─ spawn SubAgent 2（文档分析 Agent）
    └─ 收集所有子 Agent 摘要
    ↓
主 Agent 汇总，生成最终答案
```

每个 SubAgent 拥有：
- **独立 Session**（上下文隔离，不污染主 Agent）
- **完整 AgentLoop**（可以继续调用工具、派生子 Agent）
- **摘要返回**（不返回完整对话历史，节省 token）

---

## 四、数据流全景

```
用户输入
    ↓
Session.append_user_message(question)
    ↓
┌─ AgentLoop.run() ──────────────────────────────────────┐
│                                                         │
│  while session.can_continue():                          │
│      │                                                  │
│      ├─ session.build_messages()                        │
│      │   → [system_prompt, ...history, user_question]   │
│      │                                                  │
│      ├─ llm.generate_structured(messages, AgentAction)  │
│      │   → AgentAction {                                │
│      │       action_type: "tool_call",                  │
│      │       thought: "需要查知识库...",                  │
│      │       tool_call: {name: "rag_search", args:...}  │
│      │     }                                            │
│      │                                                  │
│      ├─ if tool_call:                                   │
│      │   tool_registry.execute(name, args)              │
│      │   → ToolResult(success=True, content="...")      │
│      │   session.append_tool_result(...)                │
│      │                                                  │
│      ├─ if spawn_subagent:                              │
│      │   subagent_spawner.spawn_and_run(task)           │
│      │   session.append_subagent_result(...)            │
│      │                                                  │
│      └─ if final_answer:                                │
│          return answer → 推送前端                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 五、项目结构

```
cc-harness-agent/
├── main.py                        # 启动入口（演示完整流程）
├── .env                           # API Key 配置
├── requirements.txt               # 依赖清单（极简）
├── README.md                      # 本文档
│
├── harness/                       # ★ 智能体核心层
│   ├── __init__.py
│   ├── types.py                   # Pydantic v2 数据模型
│   │   ├── Message               # 统一消息模型
│   │   ├── AgentAction           # 结构化动作输出
│   │   ├── AgentConfig           # 运行时配置
│   │   └── ToolResult            # 工具返回结果
│   │
│   ├── agent_loop.py              # ★ 核心引擎：AgentLoop
│   │   └── while 循环 → LLM决策 → 工具/子Agent/回答
│   │
│   ├── session.py                 # 会话管理
│   │   ├── 消息历史管理
│   │   ├── 上下文压缩（token超限自动摘要）
│   │   └── 循环终止判断
│   │
│   ├── llm_adapter.py             # LLM 适配层
│   │   ├── generate()            # 普通生成
│   │   ├── generate_structured() # 结构化输出（JSON Schema）
│   │   └── stream()              # 流式生成
│   │
│   ├── prompt_engine.py           # 提示词引擎
│   │   └── 动态组装 system prompt + 工具说明
│   │
│   └── subagent.py                # 子Agent 调度器
│       ├── spawn_and_run()       # 派生并执行子Agent
│       └── _summarize()          # 结果摘要压缩
│
├── tools/                         # 工具系统
│   ├── __init__.py
│   ├── base.py                    # 工具基类 BaseTool
│   ├── registry.py                # 工具注册表 ToolRegistry
│   └── rag_tool.py                # ★ RAG 检索工具
│       └── 内部封装完整 Agentic RAG 循环
│
└── infrastructure/                # 基础设施
    ├── __init__.py
    ├── retry.py                   # 异步重试（tenacity）
    ├── cache.py                   # 缓存（redis.asyncio）
    └── rate_limit.py              # 限流（滑动窗口）
```

---

## 六、快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（已配置百炼 DashScope）
# .env 文件已包含 DASHSCOPE_API_KEY

# 3. 运行
python main.py
```

---

## 七、与同目录 agentic-rag-lab 的对比学习

| | `agentic-rag-lab` (LangGraph) | `cc-harness-agent` (自研) |
|---|---|---|
| **核心依赖** | LangChain + LangGraph + Chroma | openai + chromadb + pydantic |
| **流程控制** | StateGraph + Node + Edge | AgentLoop while 循环 |
| **RAG 位置** | Graph 的 3 个节点 | Tool 注册表中的一个工具 |
| **状态管理** | TypedDict + Annotated[operator.add] | Session 对象封装 |
| **消息模型** | HumanMessage/AIMessage/ToolMessage | 单一 Message + role 枚举 |
| **多 Agent** | 多个 Graph 共享 State | 主 Agent spawn Subagent |
| **代码量** | ~250 行 | ~1200 行（含完整注释） |

**建议学习顺序**：
1. 先跑通 `agentic-rag-lab` → 理解 LangGraph 状态图模式
2. 再跑 `cc-harness-agent` → 理解 AgentLoop 循环模式
3. 对比两个项目的 `RAG Tool` vs `RAG Nodes` → 理解架构决策的 trade-off

---

## 八、Agentic RAG 三层质量保障（在 RAGTool 内部实现）

```
用户查询
    │
    ▼
┌─────────────────────────────────────────────┐
│ 第一层：向量检索（embedding + Chroma top-k）   │
│ → 把查询转为向量，在知识库中搜索最相似文档       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ 第二层：LLM 文档评分（Document Grader）         │
│ → 逐篇判断相关性，过滤不相关文档               │
└─────────────────────────────────────────────┘
    │ 全部不相关 + 未超最大改写次数
    ▼
┌─────────────────────────────────────────────┐
│ 第三层：LLM 查询改写（Query Rewrite）          │
│ → 优化查询语句 → 回到第一层重新检索            │
│ → 超限后返回"未找到相关文档"                  │
└─────────────────────────────────────────────┘
```

这个循环在 `RAGTool._agentic_rag_search()` 中实现，对外部 AgentLoop 透明。

---

## 九、面试话术储备

**Q: 为什么公司选择自研 Agent 框架，而不是用 LangChain/LangGraph？**

> 我们参考 Claude Code 的 Harness 架构自研了 Agent 框架。LangGraph 是通用状态编排引擎，适合流程固定的业务流水线；我们业务以开放式知识库复杂问答为主，需要 Agent 自主规划执行步骤、动态派生子 Agent 处理子任务。自研框架以 AgentLoop 异步循环为核心，采用主从 Subagent 多智能体模型。同时私有化部署要求严格管控依赖、数据权限、会话上下文管理，自研架构更容易满足政企合规裁剪需求。RAG 作为内置工具被智能体按需调用，实现 Agentic RAG 能力，而不是硬编码在框架层面。

**Q: AgentLoop 和 LangGraph 的 StateGraph 本质区别是什么？**

> StateGraph 是**静态编排**：开发时定义好所有节点和边，运行时按图遍历。AgentLoop 是**动态自治**：while 循环内每轮由 LLM 决定下一步，流程不是预定义的。前者适合"已知所有分支"的业务，后者适合"无法预判步骤"的开放式任务。

**Q: Subagent 和 LangGraph 的子图有什么区别？**

> LangGraph 子图与父图共享 State，子图结果自动写入全局状态。我们的 Subagent 拥有独立 Session，上下文完全隔离，只返回摘要。这样做的优势是：(1) 子Agent 上下文不污染主Agent，(2) 节省 token（不保留完整子对话历史），(3) 天然支持并行派发。

---

## 十、后续扩展方向

- [ ] FastAPI + WebSocket 接入层（流式对话）
- [ ] Redis 分布式会话存储
- [ ] 结构化日志 + LangFuse 链路追踪
- [ ] Agent 执行中断机制（WebSocket 断开 → Abort Task）
- [ ] 工具权限分级（只读工具 / 写操作工具）
- [ ] Prompt 版本管理 + A/B 测试
- [ ] Docker Compose 一键部署
