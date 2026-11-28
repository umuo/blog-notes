# Blog Notes

这里维护 Solo 博客的文章源稿、封面、学习路线图和工程笔记。每篇专题文章在路线图中有固定编号，发布后会补上博客地址。

## 阅读入口

- [LLM 与 Agent 专题写作路线图](llm-agent-learning-roadmap.md)记录阅读顺序、各篇范围和发布状态。14 个专题共 139 篇，均已发布。第十四部分「生产化专题」共 7 篇，从模型网关讲到在途任务兼容与慢请求排查。
- [写作进度](PROGRESS.md)记录各专题的发布入口与里程碑。
- [专题写作与协作规范](AGENTS.md)说明文章深度、术语解释与发布校验要求。
- [Agent 与 Sandbox 面试题库](agent-sandbox-interview-bank.md)和[Agent 工程笔记](agent-engineering-notes.md)用于查阅相关机制和面试题。
- [交互式知识图谱脑图](mindmap/roadmap-mindmap.html)（含 [XMind OPML 导入稿](mindmap/roadmap-mindmap.opml)与 [Markdown 大纲](mindmap/roadmap-mindmap-outline.md)）将 14 个阶段 139 篇专题按板块、先修与跨篇关系可视化，可运行 [scripts/build_roadmap_mindmap.py](scripts/build_roadmap_mindmap.py) 一键刷新。

## 目录结构

- `articles/01-llm-foundations/`～`articles/14-production/`：按 14 个学习阶段分目录存放全部专题源稿（含配套入门文章 `opentelemetry-tracing-primer.md`）。
- `assets/`：文章封面图（保持根目录固定路径）。
- `experiments/`：各专题配套的可运行实验代码与样本数据（保持根目录固定路径，确保线上已发布文章中的 GitHub 实验链接始终有效）。
- `mindmap/` & `scripts/`：交互式路线图脑图、大纲导出与自动化生成脚本。

## 写作与维护

1. 按路线图的行序选题，沿用篇号。专题 Markdown 源稿存放在对应阶段的 `articles/<阶段目录>/` 下，封面放在 `assets/`，实验代码放在 `experiments/`。
2. 用具体案例说明新术语与关键步骤，跑通可执行示例。更新路线图状态后，先交付本地预览。
3. 确认后再发布。修改已发布文章时，先回读 Solo 上的当前正文与元数据，合并线上改动后更新原文章。核对公开页面，再同步路线图中的状态和链接。
4. 按文章或一组相关改动提交 Git，便于查阅和回退。凭据、虚拟环境和可重新生成的预览文件存放在工作区外或由 `.gitignore` 排除。
