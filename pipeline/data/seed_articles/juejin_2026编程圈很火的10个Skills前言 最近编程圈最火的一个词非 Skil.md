# 2026编程圈很火的10个Skills前言 最近编程圈最火的一个词，非 Skills 莫属。 不是指程序员的技术能力，而

> 来源: 掘金

它不是一个功能，而是一整套包含 20+ 个子模块的工作流框架——TDD、系统化调试、计划编写、代码评审、并行 Agent 分发、Git Worktree 管理……每个都是独立的、可以单独触发的 Skill。

核心理念

：

Process over Prompt（流程大于提示词）

。

最核心的几个子 Skill：

① brainstorming（头脑风暴）⭐ 最常用

这是入口 Skill，也是最关键的一个。它的核心逻辑是：

未获得用户明确批准前，绝不允许 AI 动手写一行代码

。AI 像苏格拉底一样，一次只问一个问题，逐步澄清需求，提出 2-3 种方案让你选择。

② subagent-driven-development（子代理驱动开发）⭐ 核心 Skill

为每个计划任务派发独立子代理，互相隔离上下文。

每个任务完成后经过两阶段审查：规格合规审查 + 代码质量审查。

审查不通过就打回去重做。

③ test-driven-development（TDD）

强制遵循“红灯-绿灯-重构”循环，先写测试再写代码。

④ requesting-code-review（代码审查）

AI 自动获取当前分支的 git diff，逐文件审查，关键问题阻止推进。

安装命令

：

npx skills add obra/superpowers

# 或通过 Claude Code 官方插件市场

/plugin install superpowers@claude-plugins-official

适用场景

：中大型功能开发、需要高质量交付的核心模块。

注意事项

：简单任务会多花 15 分钟走流程。

三、Top 2：Karpathy Guidelines

它让 AI 像顶级工程师一样思考。

GitHub Star：125,000+

Andrej Karpathy（OpenAI 联合创始人、前 Tesla AI 总监）总结的 AI 编码规则，被社区整理成了 Skill。

核心四条规则：

读代码前先思考

：不是直接开始 grep 或修改

修改要精准

：不是“重写这个文件”

优先简洁

：不是“堆功能”

始终核对原始需求

安装命令

：

/install forrestchang/andrej-karpathy-skills

适用场景

：新项目启动、任何需要 AI 先理解再动手的场景。

这个 Skill 的价值在于它改变的是

AI 的行为模式

，而不是提供某个具体工具。

新起一个项目的时候特别有效——Claude 会更倾向于先理解再动手，而不是信心满满地直接开始重构。

四、Top 3：Frontend Design

它告别千篇一律的“AI 风格”UI。

安装量：277,000+

这是 Anthropic 官方 Skills 仓库中的明星 Skill。如果你用过 AI 生成的前端代码，一定会发现它们长得都差不多——那种一眼就能看出来的“AI 风格”。

Frontend Design 专门解决这个问题。它让 AI 生成

有特色、有设计感

的 UI，而不是千篇一律的模板风格。

安装命令

：

npx skills add anthropics/skills --skill frontend-design

适用场景

：前端页面生成、UI 原型设计。

注意事项

：它只提供风格约束，不能替代专业设计稿。

五、Top 4：Webapp Testing

它让 AI 帮你做 UI 自动化测试。

来源

：Anthropic 官方 Skills 仓库

这是一个基于 Playwright 的 UI 自动化测试 Skill。你只需要告诉 AI“测试这个登录页面”，它就会自动：

启动浏览器

执行操作（点击、输入、导航）

验证预期结果

生成测试报告

安装命令

：

npx skills add anthropics/skills --skill webapp-testing

适用场景

：Web 应用回归测试、UI 自动化验证。

六、Top 5：Security Review

它能做代码安全漏洞自动扫描。

来源

：Claude Code 内置 Skill

Security Review 是 Claude Code 默认自带的 Skill，在推 PR 前自动运行一次语义漏洞扫描。

它能检测的安全问题包括：

SQL 注入风险

XSS 跨站脚本漏洞

敏感信息泄露

不安全的依赖版本

认证和授权缺陷

使用方式

：

/security-review

适用场景

：提交 PR 前的安全检查、代码安全审计。

七、Top 6：MCP Builder

它能像搭积木一样构建 MCP 服务器。

来源

：Anthropic 官方 Skills 仓库

MCP（Model Context Protocol）是 2026 年 AI 编程领域最火的基础设施之一。

MCP Builder 让你可以通过自然语言描述，自动生成一个完整的 MCP 服务器。

你只需要说：“帮我建一个 MCP 服务器，可以查询天气”，AI 就会自动生成：

服务器骨架代码

工具定义

配置文件和部署脚本

安装命令

：

npx skills add anthropics/skills --skill mcp-builder

适用场景

：构建 MCP 服务器、快速原型验证。

八、Top 7：Skill Creator

它能自己动手造 Skill。

来源

：Anthropic 官方 Skills 仓库

Skill Creator 是一个“元 Skill”——它的作用是帮你

创建新的 Skill

。

你只需要描述你想让 AI 做什么，Skill Creator 就会自动生成完整的

SKILL.md

文件，包含：

YAML frontmatter（name、description）

标准化的指令结构

可选的脚本和参考文件模板

使用方式

：

/skill-creator

然后按提示输入你想创建的 Skill 描述即可。

适用场景

：为团队定制专属 Skill、将重复性工作流程固化。

九、Top 8：GSD（Get Shit Done）

别废话，直接干。

GitHub Star：快速增长中

GSD 的理念和 Superpowers 完全相反。Superpowers 让你先想清楚再动手，GSD 让你

别想了，直接干

。

这个 Skill 适合那些你非常确定要做什么、不需要反复确认的场景。

它会跳过头脑风暴、跳过计划、跳过审查——直接执行。

安装命令

：

git

clone

https://github.com/gsd-build/get-shit-done ~/.claude/skills/gsd

适用场景

：重复性任务、明确的操作指令、紧急修复。

注意事项

：只在你 100% 确定要做什么的时候用，否则容易翻车。

十、Top 9：GStack

YC 合伙人推荐的编码工作流。

GitHub Star：93,000+

Garry Tan（YC 现任 CEO）推荐的编码工作流 Skill。它把 Y Combinator 推崇的“快速迭代、持续交付”理念，编码成了 AI 可执行的工作流。

安装命令

：

npx skills add garrytan/gstack

适用场景

：创业项目快速开发、需要频繁迭代的产品。

十一、Top 10：Composio

它能连接 1000+ 外部应用。

GitHub Star：65,000+

Composio 是一个连接器 Skill，让 Claude Code 能直接操作 Jira、Linear、Slack、GitHub 等 1000+ 个外部应用。

安装后，你可以让 AI 直接：

在 Jira 里创建任务

在 Slack 里发消息

在 GitHub 里创建 PR

在 Linear 里更新工单状态

安装命令

：

git

clone

https://github.com/ComposioHQ/awesome-claude-skills.git

cd

awesome-claude-skills

claude --plugin-dir

"

$PWD

/connect-apps-plugin"

适用场景

：需要 AI 与外部工具交互的自动化工作流。

十二、Skill 安装与使用速查

通用安装方式

方式一：通过 npx（最通用）

npx skills add <仓库名>

示例：

npx skills add obra/superpowers

方式二：通过 Claude Code 插件市场

/plugin install <skill名>@<市场名>

示例：

/plugin install superpowers@claude-plugins-official

方式三：手动 git clone

git

clone

<仓库地址> ~/.claude/skills/<skill名>

Skill 存放位置

位置

作用范围

是否纳入版本控制

~/.claude/skills/

个人全局

否

.claude/skills/

当前项目

是（推荐）

插件市场安装

由插件管理

否

最佳实践

：项目级的 Skill 放入

.claude/skills/

并提交到 Git，团队成员 clone 后自动获得相同的 Skill 能力。

更多项目实战在我的技术网站：susan.net.cn/project

总结

回到最初的问题：

2026 年编程圈最火的 10 个 Skills 是什么？

我用一张表帮你总结：

排名

Skill

核心价值

最适合的场景

1

Superpowers

强制工程纪律，减少走弯路

中大型功能开发

2

Karpathy Guidelines

顶级工程师的编码思维

新项目启动

3

Frontend Design

告别千篇一律的 AI 风格 UI

前端页面生成

4

Webapp Testing

Playwright UI 自动化测试

Web 应用回归测试

5

Security Review

代码安全漏洞自动扫描

PR 前安全检查

6

MCP Builder

自然语言生成 MCP 服务器

快速构建 MCP

7

Skill Creator

自己动手创建 Skill

定制团队专属 Skill

8

GSD

跳过流程，直接执行

明确指令的重复任务

9

GStack

YC 合伙人的编码工作流

创业项目快速迭代

10

Composio

连接 1000+ 外部应用

AI 与外部工具交互

几点使用建议：

不要贪多

。装 50 个 Skill 不如装 5 个精的。70% 的社区 Skill 质量不合格，选对比选多重要。

从 Superpowers 开始

。它是整个生态的基石，装上它你就拥有了完整的工程流程能力。

按需添加

。遇到重复性工作就想一想——“这个能不能做成一个 Skill？”Skill Creator 可以帮你快速把想法变成可复用的 Skill。

注意 Skill 质量

。好的 Skill 是精准的——description 读起来像路由规则，不是宣传语；核心 SKILL.md 精简，细节推到 references/ 目录里按需加载。

如果你还没试过 Skills，

建议今天就装一个 Superpowers 试试

。

不用全部装，从

brainstorming

和

test-driven-development

这两个最常用的开始。

体验一下“AI 先想清楚再动手”的感觉，你会发现——

编程可以如此清爽

。