# Blog Notes

这里维护 Solo 博客的文章源稿、封面、学习路线图和工程笔记。每篇专题文章在路线图中有固定编号，发布后会补上博客地址。

## 从这里继续

- [LLM 与 Agent 专题写作路线图](llm-agent-learning-roadmap.md)记录阅读顺序、各篇范围和发布状态。前三部分共 35 篇均已发布，第四部分「模型预训练与后训练」已发布 5 篇，依次阅读 [TRN-01《预训练语料与 Token 数据管线》](https://blog.lacknb.cn/articles/2026/11/21/1795272443691.html)、[TRN-02](https://blog.lacknb.cn/articles/2026/11/22/1795342908073.html)、[TRN-10](https://blog.lacknb.cn/articles/2026/11/22/1795342910568.html)、[TRN-03《选继续预训练、SFT 还是偏好优化》](https://blog.lacknb.cn/articles/2026/11/22/1795327133650.html) 和 [TRN-04《SFT 怎么把对话变成监督信号》](https://blog.lacknb.cn/articles/2026/11/22/1795353665754.html)。
- [写作进度](PROGRESS.md)记录已经完成的阶段和下一篇的接续点。
- [专题写作与协作规范](AGENTS.md)说明文章深度、术语解释与发布校验要求。
- [Agent 与 Sandbox 面试题库](agent-sandbox-interview-bank.md)和[Agent 工程笔记](agent-engineering-notes.md)用于查阅相关机制和面试题。

## 写作与维护

1. 按路线图的行序选题，沿用篇号。Markdown 源稿放在仓库根目录，封面放在 `assets/`。
2. 用具体案例说明新术语与关键步骤，跑通可执行示例。更新路线图状态后，先交付本地预览。
3. 确认后再发布。修改已发布文章时，先回读 Solo 上的当前正文与元数据，合并线上改动后更新原文章。核对公开页面，再同步路线图中的状态和链接。
4. 按文章或一组相关改动提交 Git，便于查阅和回退。凭据、虚拟环境和可重新生成的预览文件存放在工作区外或由 `.gitignore` 排除。
