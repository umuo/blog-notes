---
title: "Agent 源码 CASE-02｜Hermes Agent 怎样留下经验，从真实执行循环到待审写入"
description: "按固定版本源码追踪 Hermes 的 CLI、AIAgent、会话存储、后台回顾、记忆与技能。用同一张合成退款表以及本地固定回包，验证经验暂存、审批、下一会话加载、编辑拒绝和移除。"
keywords: [llm, agent, memory, source-code]
---

![跟踪一条对话怎样变成可审阅的经验。](../../assets/case-02-hermes-runtime-memory-cover.jpg)

> **生成工具**：Codex\
> **生成模型**：gpt-6.2-sol

## 先用大白话理解

同一个助手手上可能有三本本子。聊天记录写「刚才聊了什么」，常用小抄写「这个人的数据规则」，操作手册写「以后遇到同类任务怎么做」。Hermes Agent 把这几类材料放在不同入口里。一次回答算对，说明当前工单可以继续走。以后仍能处理新表，还要看经验放进哪本本子、是否获准、下一次会不会读它。

本篇用同一份退款表。 `4, NA, 6` 排除缺失应得到两笔有效值、均值 5。我们让固定模型接口发出真实 `read_file` 调用，并观察它返回的 CSV；计算报告是夹具规则。源码固定 `v0.23.3`，实验使用独立的 Hermes profile（个人工作资料与配置的目录范围）。

## 1. 从 CLI 跟到会话循环

CLI（命令行界面）和 Gateway（消息入口）都可以进入执行结构。源码从 `cli.py` 里的入口找交互对象，跟到 `run_agent.py` 中的 `AIAgent`。该类把初始化交给专门函数，调用一轮时经 `agent/turn_facade.py` 转入 `agent/conversation_loop.py`。这段负责构建或恢复上下文、发模型请求、执行工具，再处理工具结果与后续请求。实验直接走 `AIAgent.run_conversation`，CLI 和 Gateway 路由用源码确认。

| 组件 | 本次走过的行为 |
| --- | --- |
| 输入和工具定义 | 固定开启文件工具集合，并将本实验面向模型的文件接口缩为 `read_file` |
| 第一次模型请求 | 固定端点要求读 `refunds.csv` |
| 工具与第二次请求 | 调用真实文件工具，回包含四行（表头加三行数据） |
| 返回与会话存储 | 读最终文本，在隔离 profile 中有本次会话库 |

[CLI](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/cli.py) · [AIAgent](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/run_agent.py) · [一轮入口](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/agent/turn_facade.py) · [会话循环](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/agent/conversation_loop.py)

## 2. 对话记录、短记忆和 Skill 各存什么

对话记录通过会话存储管理，后来可以用面向会话历史的入口检索。`MEMORY.md` 和 `USER.md` 则是短记忆，适合小而稳定的规则与用户约定。Hermes 可以在新会话装配系统上下文时加载短记忆。Skill 则保存可复用操作的步骤和资料，通常先发现名称和描述，再在需要时读正文。把某次退款原始详情全部复制进记忆，会使来源模糊并消耗预算。这里只记录合成表的口径。

一次任务结束后，框架有后台回顾路径。它可以根据任务轨迹评估哪些经验值得保存，也可以按设置阶段性触发。读取 `agent/background_review.py`，找回顾提示、何时派发以及可使用的工具。回顾建议写记忆时还要经过自己的写入门槛。本次验证关闭了后台模型回顾，直接调用同版受管的记忆和 Skill 接口，避免把追加模型回复当成审批观察。 [后台回顾源码](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/agent/background_review.py) · [固定版记忆文档](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/website/docs/user-guide/features/memory.md)

## 3. 从待审走到下一会话

设置 `memory.write_approval: true` 和 `skills.write_approval: true`。这两个配置独立。测试在非交互环境调用原生 `memory_tool`、`skill_manage`，再走同一版本的斜杠命令处理函数执行审批或拒绝。

| 动作 | 实际观察 |
| --- | --- |
| 提交一条短记忆 | 生成待审记录，`MEMORY.md` 尚未落盘，新会话仍算 3 条 |
| 批准添加 | 规则写进 profile 的记忆文件，新会话的系统上下文带上它，报告改成 2 条 |
| 提交移除，再批准 | 待审时现有条目仍在，批准后才移除，下一会话回到 3 条 |
| 提交 Skill 并批准，再提编辑 | 创建获准后正文可读，编辑待审时仍能读旧版本，拒绝编辑后旧正文保留 |

本次还通过原生会话库读回持久化的工具消息。复用会话时，从该库读出历史再交给执行循环，并且确认用户纠正出现在后续请求中。这样记录了两条独立路径：恢复对话历史，以及从新会话加载短记忆。Skill 的创建与编辑走自己的审批，不能用记忆写入成功来推断技能也已发布。

在源码里可以沿 `tools/memory_tool.py` 的写入门槛找暂存决定，再读 `tools/write_approval.py` 中的待审记录。斜杠命令的审批分发在 `hermes_cli/write_approval_commands.py`。修改或移除已有记忆时，选定版源码还会为待审动作固定匹配到的原条目，支持复核时发现目标已变。直接修改磁盘文件绕开应用的写入门槛，因此仍需要操作系统权限控制。 [记忆写入](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/tools/memory_tool.py) · [写入审批](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/tools/write_approval.py) · [审批命令](https://github.com/NousResearch/hermes-agent/blob/v0.23.3/hermes_cli/write_approval_commands.py)

这篇的「下一会话加载」检查的是底层请求里的系统提示是否实际带上规则，并用相同接口回放结果。恢复旧会话、召回过去聊天和加载长驻短记忆是不同的加入路径。更多系统间差异见 [CASE-03](case-03-openclaw-hermes-comparison.md)。 [实验代码](../../experiments/framework-replay/replay.py) · [实际结果](../../experiments/framework-replay/hermes-results.json)
