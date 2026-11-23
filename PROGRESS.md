# 写作进度

## 当前接续点

整条路线规划 14 个专题、138 篇文章，目前已发布 88 篇，尚待发布 50 篇。第六部分「Agent、Context Engineering、Loop Engineering 与 Harness」25 篇已全部发布。按路线图行序，下一篇写作是 **AE-06《第一篇，如何验证 Agent 真的完成了任务》**。

| 已发布专题 | 篇数 | 阅读入口 |
| --- | ---: | --- |
| LLM 基础与推理 | 15 | [F-01《Tokenizer 与 Chat Template》](https://blog.lacknb.cn/articles/2026/11/19/1795068393122.html) |
| LLM API 与流式协议 | 10 | [API-01《从原始 HTTP 请求认识 LLM API》](https://blog.lacknb.cn/articles/2026/11/21/1795268752670.html) |
| 评测入门与 LLM 效果评测 | 10 | [LE-07《什么是评测，如何跑通第一条用例》](https://blog.lacknb.cn/articles/2026/11/21/1795270270931.html) |
| 模型预训练与后训练 | 12 | [TRN-01《预训练语料与 Token 数据管线》](https://blog.lacknb.cn/articles/2026/11/21/1795272443691.html) · [专题目录](llm-agent-learning-roadmap.md#4-模型预训练与后训练) |
| 推理引擎与模型部署 | 16 | [SERV-01《用 Qwen 0.5B 搭一个极简 vLLM Server》](https://blog.lacknb.cn/articles/2026/11/23/1795400440818.html) · [专题目录](llm-agent-learning-roadmap.md#5-推理引擎与模型部署vllm-和-sglang) |
| Agent、上下文工程、循环与 Harness | 25 | [AGT-01《从工具调用到 Agent Loop》](https://blog.lacknb.cn/articles/2026/11/23/1795402332959.html) · [专题目录](llm-agent-learning-roadmap.md#6-agentcontext-engineeringloop-engineering-与-harness) |

公开入口见[置顶学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)。详细范围、先修顺序和源稿见[仓库路线图](llm-agent-learning-roadmap.md)。写作和发布流程见 [AGENTS.md](AGENTS.md)。

## 里程碑

| 日期 | 进展 | 记录 |
| --- | --- | --- |
| 2026-11-23 | 精修第二部分「LLM API 与流式协议」10 篇并同步博客 | 补齐 ELI5 导读、按需深入分层、11 张兼容 Mermaid 9.1.7 的流程图及常见踩坑，全量线上 HTML 校验通过 |
| 2026-11-23 | 精修第一部分「LLM 基础与推理」15 篇并同步博客 | 补齐 ELI5 大白话导读、按需深入分层、23 张兼容 Mermaid 9.1.7 的流程图及易错点，全量线上 HTML 校验通过 |
| 2026-11-23 | 发布 AGT-01～AGT-25，完成第六部分 | 归档退款报告运行时、MCP、A2A、Temporal、Pi、技能发现与界面回放的离线实测，共用固定 CSV，未部署模型 |
| 2026-11-23 | 发布 SERV-01～SERV-16，完成第五部分 | 归档推理引擎、流式、模型注册、部署和批作业实测，以及标明口径的调度及容量演算。核对公开正文、封面与置顶目录 |
| 2026-11-23 | 发布 [TRN-05](https://blog.lacknb.cn/articles/2026/11/23/1795398317459.html)、[TRN-06](https://blog.lacknb.cn/articles/2026/11/23/1795398359078.html)、[TRN-07](https://blog.lacknb.cn/articles/2026/11/23/1795398360808.html)、[TRN-08](https://blog.lacknb.cn/articles/2026/11/23/1795398362308.html)、[TRN-11](https://blog.lacknb.cn/articles/2026/11/23/1795398363907.html)、[TRN-12](https://blog.lacknb.cn/articles/2026/11/23/1795398365508.html)、[TRN-09](https://blog.lacknb.cn/articles/2026/11/23/1795398367155.html)，完成第四部分 | 归档 Qwen LoRA、DPO 和 Unsloth 实测，以及强化学习、蒸馏和数据筛选的教学实验。逐篇核对公开正文、封面与置顶路线图 |
| 2026-11-22 | 发布 [TRN-04《SFT 怎么把对话变成监督信号》](https://blog.lacknb.cn/articles/2026/11/22/1795353665754.html) | 核对公开正文、封面与实测结果，已同步置顶路线图 |
| 2026-11-22 | 完成 [TRN-04 本地稿](trn-04-sft-supervision.md) | 实测 Qwen 多轮助手标签、填充与截断，在六组留出题上比较基座、提示词和 SFT |
| 2026-11-22 | 发布 [TRN-02](https://blog.lacknb.cn/articles/2026/11/22/1795342908073.html) 和 [TRN-10](https://blog.lacknb.cn/articles/2026/11/22/1795342910568.html) | 核对公开正文、封面和篇间链接，已同步置顶路线图 |
| 2026-11-22 | 发布 TRN-03《选继续预训练、SFT 还是偏好优化》 | 核对独立可执行示例、公开正文和封面，已同步置顶路线图 |
| 2026-11-22 | 完成 TRN-03 本地稿 | 从同一微型基座实测继续预训练、SFT 与 DPO，归档公共指标和留出题输出 |
| 2026-11-22 | 完成 TRN-10 本地稿 | 实测两进程 FSDP 的训练状态分片、同批更新和分片检查点恢复 |
| 2026-11-22 | 完成 TRN-02 本地稿 | 复用 TRN-01 分片实测 60 步训练，验证 30 步断点恢复与连续训练结果完全一致 |
| 2026-11-21 | 发布第四部分 TRN-01《预训练语料与 Token 数据管线》 | 源稿、封面、实验清单已归档，核对公开页面与署名并同步置顶路线图 |
| 2026-11-21 | 补齐置顶路线图第一部分的 15 篇文章链接 | 回读确认公开目录收录全部 35 篇已发布文章 |
| 2026-11-21 | 第三部分「评测入门与 LLM 效果评测」10 篇全部发布 | 核对公开正文、封面和篇间链接，同步置顶路线图 |
| 2026-11-21 | 完成第三部分「评测入门与 LLM 效果评测」10 篇本地稿 | 验证代码与预览，归档真实模型运行记录 |
| 2026-11-21 | 第二部分「LLM API 与流式协议」10 篇全部发布 | 核对公开页面、封面和上一篇／下一篇导航，并更新线上路线图 |
| 2026-11-21 | 完成第二部分 10 篇本地稿 | 按路线图行序核对源稿、封面和实验 |
| 2026-11-21 | 完成 API-02 本地草稿 | 源稿与封面已归档 |
| 2026-11-21 | 完成 API-01 本地草稿 | 源稿与封面已归档 |
| 2026-11-20 | 第一部分「LLM 基础与推理」15 篇全部发布 | 源稿和封面已归档，路线图逐篇记录公开链接 |
| 2026-11-20 | 建立 Git 源稿基线 | 仓库记录文章、封面、写作规范和项目进度 |

每次完成一篇或一次发布后，更新上面的接续点。单篇的草稿与发布状态以路线图为准。
