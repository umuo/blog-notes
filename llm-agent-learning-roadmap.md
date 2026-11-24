# LLM 与 Agent 专题写作路线图

主线从一条真实请求出发：**文本 → Token → 模型推理 → HTTP 与流式返回 → Agent 执行工具 → 检索证据 → 累积知识**。先搞清每层的输入、输出和状态归属，再进入框架和生产架构。

| 顺序 | 专题系列 | 先修 | 阶段产物 |
| --- | --- | --- | --- |
| 1 | LLM 基础与推理（F） | Python、矩阵和概率直觉 | 解释一条提示从 Token 到首个输出 |
| 2 | LLM API 与流式协议（API） | F-01～F-04、HTTP/JSON | 用原始 HTTP 跑通文本、SSE 和工具调用 |
| 3 | 评测入门与 LLM 效果评测（LE） | 读懂模型请求和输出 | 手写评分器，做可复现的模型或 Prompt 对照 |
| 4 | 模型预训练与后训练（TRN） | Tokenizer、因果语言模型、训练损失与 LE 的评测口径 | 跑通微型预训练和一次 Qwen 微调，产出可复现的模型版本 |
| 5 | 推理引擎与模型部署（SERV） | Prefill、KV Cache、Chat Completions 与 SSE | 用同一份请求和记录对照 vLLM 与 SGLang 的服务行为 |
| 6 | Agent、Context 与 Loop Engineering（AGT） | 文本和工具协议、上下文预算 | 写一个可终止、可恢复的 Agent |
| 7 | Agent 效果评测（AE） | Agent Loop、工具执行、LE | 从验证一次任务终态，进阶到故障归因和发布回归 |
| 8 | RAG 与 LLM Wiki（KB） | 上下文组装、LE 与 AE 的基本口径 | 对同一批资料交付可溯源问答与知识页 |
| 9 | LLM 与 Agent 安全护栏（SEC） | 模型、工具和检索的信任边界 | 用 OWASP 清单做威胁建模和控制映射 |
| 10 | Sandbox 分类、源码与实测（SBX） | SEC、进程与容器基础 | 按相同任务与威胁模型对照执行环境 |
| 11 | Agent 自进化与 RSI（EVO） | 记忆与 Skills、离线评测、基础遥测与发布护栏 | 用运行反馈验证并发布可回滚的改进 |
| 12 | OpenClaw 与 Hermes Agent 源码案例（CASE） | Harness、会话、Skills、隔离与自进化 | 对照同一任务在两个框架里的运行与经验沉淀 |
| 13 | Vibe Coding 与工程化交付（DEV） | Agent Harness、评测、护栏与 Sandbox | 对比原型与可维护版本的验收证据 |
| 14 | 生产化（OPS） | 服务、评测、隔离与任务执行链路 | 用运行指标说明上线、限流、隔离和拆分决策 |

本路线用 **TTFT**（Time to First Token）称呼首 Token 延迟。**GRPO**（Group Relative Policy Optimization）归入后训练。F-11 解释组内相对奖励，TRN-08 再把 GRPO 放进强化学习的采样、奖励与更新链路，GQA 和 GPTQ 分别留在注意力结构与量化。 [DeepSeekMath / GRPO 原论文](https://arxiv.org/abs/2402.03300) · [GQA 论文](https://arxiv.org/abs/2305.13245) · [GPTQ 论文](https://arxiv.org/abs/2210.17323)

每篇最后一列给出一个可检查的学习目标。遇到手算题先算出预期值，遇到操作题带着目标去找正文里的命令和结果。涉及真实模型或特定硬件的实验，也可以先用文中归档的数据完成判读。

## 1. LLM 基础与推理

建议先跑 Tokenizer，再钻进 Transformer。读到注意力时，已经知道输入序列和张量的含义。以 decoder-only 文本模型为主，图像、语音和后训练放到基础推理之后。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| F-01 | [Tokenizer 与 Chat Template](https://blog.lacknb.cn/articles/2026/11/19/1795068393122.html)（已发布，[源稿](articles/01-llm-foundations/f-01-tokenizer-chat-template.md)） | 对同一句话打印普通分词、应用聊天模板、加生成起始标记这三份 Token ID，找到序列第一次不同的位置。 |
| F-02 | [Transformer 如何从 Token 算出 logits](https://blog.lacknb.cn/articles/2026/11/20/1795135053358.html)（已发布，[源稿](articles/01-llm-foundations/f-02-transformer-logits.md)） | 画出一个 Token 经过 Embedding、注意力和输出层的张量形状，再只替换末尾 Token，验证因果遮罩是否保护前面的位置。 |
| F-15 | [Encoder-only、Decoder-only 与 Encoder-Decoder](https://blog.lacknb.cn/articles/2026/11/20/1795135557100.html)（已发布，[源稿](articles/01-llm-foundations/f-15-transformer-architectures.md)） | 对同一句话画出三种 Transformer 的可见关系，再给检索、重排与生成分别选择可验收的输出。延伸到 KB-02、KB-04、KB-05。 |
| F-03 | [MHA、MQA、GQA 与 RoPE](https://blog.lacknb.cn/articles/2026/11/20/1795154536776.html)（已发布，[源稿](articles/01-llm-foundations/f-03-attention-heads-rope.md)） | 给定 Q 头、KV 头和每头维数，画出三种共享关系并计算 KV 缓存大小，解释同一组的 Q 头为何仍有不同结果。 |
| F-13 | [MoE 如何选择专家与分配计算](https://blog.lacknb.cn/articles/2026/11/20/1795161347233.html)（已发布，[源稿](articles/01-llm-foundations/f-13-moe-routing.md)） | 从 Router 分数选 Top-k 专家并算出合并结果，再逐专家统计分派次数。专家并行接 SERV-10。 |
| F-04 | [自回归生成与采样](https://blog.lacknb.cn/articles/2026/11/20/1795161419910.html)（已发布，[源稿](articles/01-llm-foundations/f-04-autoregressive-sampling.md)） | 从一组 logits 手算温度和截断后的候选概率，并用同一个小词表复现 EOS、停止词和输出上限。 |
| F-05 | [Prefill 与 Decode](https://blog.lacknb.cn/articles/2026/11/20/1795161425780.html)（已发布，[源稿](articles/01-llm-foundations/f-05-prefill-and-decode.md)） | 给定 21 个输入 Token，标出首个输出何时被选中，以及缓存长度何时从 21 增为 22。 |
| F-06 | [KV Cache 如何加速 Decode](https://blog.lacknb.cn/articles/2026/11/20/1795161430009.html)（已发布，[源稿](articles/01-llm-foundations/f-06-kv-cache.md)） | 按层和 KV 头数计算增量缓存字节数，并判断改动一个输入 Token 后哪些前缀位置还能复用。 |
| F-14 | [FlashAttention 如何减少注意力的显存读写](https://blog.lacknb.cn/articles/2026/11/20/1795161433683.html)（已发布，[源稿](articles/01-llm-foundations/f-14-flashattention.md)） | 当新块提高最大分数时手算旧分母的重缩放，和整表 Softmax 比对；区分这个分块和 SERV-04 的 KV 缓存块。 |
| F-07 | [TTFT、TPOT、ITL、吞吐与端到端延迟](https://blog.lacknb.cn/articles/2026/11/20/1795161437100.html)（已发布，[源稿](articles/01-llm-foundations/f-07-inference-latency-metrics.md)） | 在同一条请求上标出引擎首 Token、服务首事件、浏览器首字三个时间点，判断慢发生在生成、传输还是展示。 |
| F-08 | [模型精度与量化](https://blog.lacknb.cn/articles/2026/11/20/1795161440718.html)（已发布，[源稿](articles/01-llm-foundations/f-08-model-precision-and-quantization.md)） | 把一组权重压成 4 位编码并算出误差，分别估算权重和 KV 缓存占用；用同一题检查质量影响。 |
| F-09 | [多模态请求如何进入模型](https://blog.lacknb.cn/articles/2026/11/20/1795161444330.html)（已发布，[源稿](articles/01-llm-foundations/f-09-multimodal-input.md)） | 从图片网格数出视觉占位，比较纯文本和图文的入模长度，并检查两张图片的占位是否按顺序对应。 |
| F-10 | [模型从训练到可用的过程](https://blog.lacknb.cn/articles/2026/11/20/1795161450059.html)（已发布，[源稿](articles/01-llm-foundations/f-10-model-training-stages.md)） | 把同一退款任务改写为预训练原文、监督示范、偏好数据或强化学习奖励，标出损失使用哪段数据。 |
| F-11 | [GRPO 如何用于推理模型后训练](https://blog.lacknb.cn/articles/2026/11/20/1795161454784.html)（已发布，[源稿](articles/01-llm-foundations/f-11-grpo.md)） | 用四份答案的奖励手算组内相对优势，解释全对或全错的组为何没有组内排序信号。可接着看 LE-02。 |
| F-12 | [推理时计算与 Reasoning 模型](https://blog.lacknb.cn/articles/2026/11/20/1795161460545.html)（已发布，[源稿](articles/01-llm-foundations/f-12-inference-time-reasoning.md)） | 比较两档推理预算，把正确率、计费 Token、延迟和未完成请求放进同一张表。 |

基础阅读：[Transformer 论文](https://arxiv.org/abs/1706.03762) · [Transformers Encoder-Decoder 文档](https://huggingface.co/docs/transformers/model_doc/encoder-decoder) · [Hugging Face Chat Templates](https://huggingface.co/docs/transformers/chat_templating) · [KV Cache 机制](https://huggingface.co/docs/transformers/cache_explanation) · [OpenAI Reasoning 模型指南](https://developers.openai.com/api/docs/guides/reasoning) · [Switch Transformers](https://arxiv.org/abs/2101.03961) · [FlashAttention](https://arxiv.org/abs/2205.14135)

## 2. LLM API 与流式协议

全章用同一条普通消息和同一个无副作用函数走通协议。先抓原始 HTTP 请求和事件，再用 SDK 复现。真实在线 API 的字段与模型能力按文章写作时的官方文档复核。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| API-01 | [从原始 HTTP 请求认识 LLM API](https://blog.lacknb.cn/articles/2026/11/21/1795268752670.html)（已发布，[源稿](articles/02-llm-api/api-01-raw-http-request.md)） | 组装一次原始 HTTP 请求，从响应读出答案、完成状态、用量；按 401、429 和输出不完整选择检查项。 |
| API-02 | [OpenAI Chat Completions 的消息协议](https://blog.lacknb.cn/articles/2026/11/21/1795268752027.html)（已发布，[源稿](articles/02-llm-api/api-02-chat-completions-messages.md)） | 手写两轮 Chat Completions 的 messages，删掉第一轮的一条规则重放，找出追问丢了哪段上下文。 |
| API-10 | [Prompt Engineering 的可复现对照实验](https://blog.lacknb.cn/articles/2026/11/21/1795268751247.html)（已发布，[源稿](articles/02-llm-api/api-10-prompt-comparison.md)） | 用相同样本比较三版提示，分别记录格式、数值和输入 Token，把一条反例交给 LE-04 的发布回归。 |
| API-03 | [从 Chat Completions、Assistants 到 Responses](https://blog.lacknb.cn/articles/2026/11/21/1795268750432.html)（已发布，[源稿](articles/02-llm-api/api-03-openai-api-evolution.md)） | 在三种接口的请求与回包里找到会话载体和工具调用标识，迁移一次工具回填，再用缺 ID 的样本检查。 |
| API-04 | [Function Calling 与 Structured Outputs](https://blog.lacknb.cn/articles/2026/11/21/1795268749640.html)（已发布，[源稿](articles/02-llm-api/api-04-function-calling-structured-outputs.md)） | 完成一次工具定义、参数校验与结果回填，再让报告漏字段或数值错误，确认结构化输出与业务断言分别拦截。 |
| API-09 | [约束解码如何生成符合 Schema 的结构化输出](https://blog.lacknb.cn/articles/2026/11/21/1795268748887.html)（已发布，[源稿](articles/02-llm-api/api-09-constrained-decoding.md)） | 用一个小 Schema 手算每步允许的 Token，再让格式正确但数字错误的报告通过结构检查并被业务测试拒绝。 |
| API-05 | [SSE 和流式工具调用](https://blog.lacknb.cn/articles/2026/11/21/1795268748102.html)（已发布，[源稿](articles/02-llm-api/api-05-sse-streaming-tools.md)） | 把同一份流按字节切碎后重组，按调用 ID 还原工具参数，区分收到完整调用和断流时的半成品。 |
| API-06 | [Anthropic Messages API 的内容块与事件](https://blog.lacknb.cn/articles/2026/11/21/1795268747439.html)（已发布，[源稿](articles/02-llm-api/api-06-anthropic-messages.md)） | 从一段 Anthropic 流按块索引还原工具输入，再用 tool_use_id 关联工具结果，并与 OpenAI 报文对照。 |
| API-07 | [模型协议适配与能力矩阵](https://blog.lacknb.cn/articles/2026/11/21/1795268746690.html)（已发布，[源稿](articles/02-llm-api/api-07-provider-adapters.md)） | 用同一份请求测试适配器的能力声明、错误与用量映射，对缺少结构化输出能力的组合给出可见处理。 |
| API-08 | [实时语音 Agent 的双向流](https://blog.lacknb.cn/articles/2026/11/21/1795268619097.html)（已发布，[源稿](articles/02-llm-api/api-08-realtime-voice-agent.md)） | 用已生成、已接收和已播放的时间画出一次语音打断，验证旧音频及时停止及迟到数据不会再播。可接着看 F-09、API-05。 |

演进锚点为 **2023-03 Chat Completions → 2023-06 Function Calling → Assistants 的 Thread/Run 模型 → 2025-03 Responses**。重点追踪 API 的输入输出、状态归属与自定义工具由谁执行。 [ChatGPT API 发布](https://openai.com/index/introducing-chatgpt-and-whisper-apis/) · [Function Calling 发布](https://openai.com/index/function-calling-and-other-api-updates/) · [Responses 发布](https://openai.com/index/new-tools-for-building-agents/) · [OpenAI 迁移指南](https://developers.openai.com/api/docs/guides/migrate-to-responses) · [Anthropic Messages 流式文档](https://platform.claude.com/docs/en/build-with-claude/streaming) · [OpenAI Realtime 会话](https://developers.openai.com/api/docs/guides/realtime-conversations) · [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) · [Prompt Engineering 指南](https://developers.openai.com/api/docs/guides/prompt-engineering)

## 3. 评测入门与 LLM 效果评测

**评测（Eval）**把「效果更好」变成可重复验证的问题。先定义任务和通过条件，再运行候选版本，用评分器对结果打分，最后汇总失败样本和各项指标。一条测试用例可运行多次，每次都是一次试次（trial）。评测运行器负责调度、记录和汇总，评分器负责判断某一项是否达标。

从三行 CSV 开始。数值列是 10、空值和 0，要求忽略空值并保留真实的 0。期望记录数为 2，均值为 5。模型输出后分别检查 JSON 能否解析、分母是否正确、均值是否正确。用同一个评分器跑两个 Prompt，保留失败发生在哪个断言。只判断输出格式会漏掉统计口径错误，这个用例后续还可以复用于 Agent 和 RAG 评测。

**学习顺序**　先手写一条用例（LE-07），再做数据集（LE-01）、程序评分器（LE-08）与人工校准的模型评分器（LE-03），最后学公平比较（LE-09）、持续回归和线上验证。掌握小实验后，可以用 LE-10 选公开数据集，用 LE-02 跑通基准，再扩展到 LE-06 的长上下文实验。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| LE-07 | [第一篇，什么是评测，如何跑通第一条用例](https://blog.lacknb.cn/articles/2026/11/21/1795270270931.html)（已发布，[源稿](articles/03-llm-eval/le-07-first-eval.md)） | 给三行退款表写一个能区分 JSON 语法错误和业务算术错误的评分器，保存每条失败理由。 |
| LE-01 | [从业务目标到数据集、基线和评分](https://blog.lacknb.cn/articles/2026/11/21/1795270270258.html)（已发布，[源稿](articles/03-llm-eval/le-01-dataset-baseline.md)） | 建一份带数据版本和分层标签的黄金集，识别训练与留出间的同源样本，并按失败类别汇总。 |
| LE-08 | [如何设计可测试的评分器与评分准则](https://blog.lacknb.cn/articles/2026/11/21/1795270269520.html)（已发布，[源稿](articles/03-llm-eval/le-08-testable-graders.md)） | 用已知对错的样本独立测试 Schema、公差和业务规则，再为开放解释写出人工判分标准。 |
| LE-10 | [常用 LLM 评测数据集测什么，应该怎么选](https://blog.lacknb.cn/articles/2026/11/21/1795270268728.html)（已发布，[源稿](articles/03-llm-eval/le-10-public-datasets.md)） | 为一个任务选择合适的公开基准类型，并说明应核对选项、最终数值、可运行测试还是其他交付证据。 |
| LE-02 | [公开基准实战，怎样复现并比较模型结果](https://blog.lacknb.cn/articles/2026/11/21/1795270267947.html)（已发布，[源稿](articles/03-llm-eval/le-02-reproduce-benchmark.md)） | 在固定基准子集上重放同一份生成，比较两套答案抽取口径，再计算单次通过率和 pass@k。 |
| LE-03 | [LLM-as-a-Judge 与人工校准](https://blog.lacknb.cn/articles/2026/11/21/1795270267279.html)（已发布，[源稿](articles/03-llm-eval/le-03-llm-judge-calibration.md)） | 给一对答案先做人工标注，再互换位置检查模型裁判，把原始位置和还原后的答案 ID 分开记。 |
| LE-09 | [怎样判断两版效果差异是否可信](https://blog.lacknb.cn/articles/2026/11/21/1795270266572.html)（已发布，[源稿](articles/03-llm-eval/le-09-paired-comparison.md)） | 按同一道题计算候选与基线的差值，再整题取样比较区间，解释为何同题的多次运行不能拆成独立样本。 |
| LE-04 | [把效果评测接入 Prompt 和模型回归](https://blog.lacknb.cn/articles/2026/11/21/1795270265847.html)（已发布，[源稿](articles/03-llm-eval/le-04-regression-gates.md)） | 把数据版本、正确率、成本、延迟及失败反例写入发布检查，分别跑出放行和阻止发布的结果。 |
| LE-06 | [长上下文的有效性如何评测](https://blog.lacknb.cn/articles/2026/11/21/1795270265166.html)（已发布，[源稿](articles/03-llm-eval/le-06-long-context-eval.md)） | 把同一证据放进长上下文的开头、中间与末尾，分开统计引用与算术正确性、首字时间和用量。结合 KB-05 对照 RAG。可接着看 AGT-04。 |
| LE-05 | [模型改了，线上效果真的变好了吗](https://blog.lacknb.cn/articles/2026/11/21/1795270264488.html)（已发布，[源稿](articles/03-llm-eval/le-05-online-evaluation.md)） | 设计只读影子实验和用户级灰度，先核对随机分流，再比较成功率、接管率和回滚条件。 |

### 常用模型数据集与打分方式

| 数据集 / 排行榜 | 样本与评测目标 | 常见打分方式 |
| --- | --- | --- |
| [MMLU](https://github.com/hendrycks/test) / [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | 多学科英文选择题（4 选 1 经典版 → 10 选 1 进阶推理版），缓解原榜饱和与蒙题概率 | 对照正确选项，按指定规则汇总准确率 |
| [C-Eval](https://github.com/hkust-nlp/ceval) | 中文多学科选择题，观察中文知识表现 | 比较选项准确率并查看科目分项 |
| [GSM8K](https://github.com/openai/grade-school-math) / MATH & AIME | 数学文字题与竞赛级数学推理 | 提取和归一化数值或符号表达式，再与标准答案匹配 |
| [GPQA Diamond](https://github.com/idavidrein/gpqa) / [HLE](https://lastexam.ai/) | 博士级「防谷歌搜索」科学与专家前沿难题，探测高阶推理天花板 | 严格客观选项 / 唯一答案比对（HLE 含开放短答与专家核对） |
| [ARC-AGI / ARC-AGI-2](https://arcprize.org/) | 抽象彩色网格规律归纳（流体智力），专治死记硬背 | 未知新规则网格输出像素矩阵精确匹配 |
| [HumanEval](https://github.com/openai/human-eval) / [BigCodeBench](https://github.com/bigcode-project/bigcodebench) | 函数补全与多库调用真实 Python 工程编码任务 | 在隔离环境运行随题单元测试，按采样口径计算 pass@k |
| [LiveBench](https://livebench.ai/) / [LiveCodeBench](https://livecodebench.github.io/) | 按月更新题目、按时间切分编程真题，防预训练数据污染 | 确定性客观规则或隔离代码测试（不依赖 LLM 当裁判） |
| [IFEval](https://huggingface.co/datasets/google/IFEval) / [SimpleQA](https://github.com/openai/simple-evals) | 可验证格式指令遵循，与简短事实幻觉/拒答校准 | 确定性代码规则检查长度/格式约束；事实精确度与校准统计 |
| [MMMU](https://mmmu-benchmark.github.io/) / [RULER](https://github.com/NVIDIA/RULER) | 大学级多模态图文推理，与多跳长上下文压力测试 | 跨学科客观题 / 长文本多跳信息精确检索率 |
| [LMArena](https://lmarena.ai/) / [Artificial Analysis](https://artificialanalysis.ai/) | 真人双盲对比偏好，与智商-价格-TTFT-吞吐性价比综合榜 | 人类匿名配对投票算 Elo 分；客观跑分联合延迟与成本口径 |

### 评测方式怎么选

- **确定性评分**适合选项、数值、结构化结果和可执行产物。常用准确率、答案匹配、允许误差的数值断言、Schema 校验、单元测试和状态断言。Schema 能检查格式，业务断言负责检查值。
- **人工评分和 LLM-as-a-Judge**适合报告解释、摘要等存在多个合理答案的任务。先写清分档示例，再选逐项打分或两版成对比较。模型评分先与人工标注交叉核对，保留平局和无法判定类别。
- **单次和多次试验回答不同问题。** 准确率或 pass@1 观察单次机会的表现，pass@k 衡量 k 个候选里至少一个通过的概率。对需要重复稳定的 Agent，可另报同一任务所有指定试次都成功的比例，并注明试次和预算。
- **公开基准和业务回归集配合使用。** 公开基准用于对照能力，业务留出集决定发布门槛。比较时固定数据版本、划分、提示模板、采样预算、工具环境与评分器版本，报告样本量和失败分项。

开源项目按用途选。模型公开基准先读 **lm-eval**，需要多维场景和可复查报告时读 **HELM**。应用级的 Prompt 与模型对照先用 **Promptfoo** 跑小回归集。项目能力以仓库文档为依据，实际版本和数据集许可在写作实验时确认。 [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) · [Stanford HELM](https://github.com/stanford-crfm/helm) · [Lost in the Middle](https://arxiv.org/abs/2307.03172) · [LongBench](https://arxiv.org/abs/2308.14508) · [Promptfoo](https://github.com/promptfoo/promptfoo)

## 4. 模型预训练与后训练

先用极小的 decoder-only 模型理解「数据怎么变成权重」，再用已有 Qwen 模型做适配。教学预训练用小语料和小模型演示原理，微调实验需固定可用显卡、框架版本和模型 revision。需要先做 Agent 或 vLLM 的读者可以用已有权重进入 SERV，再回到本节。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| TRN-01 | [预训练语料与 Token 数据管线](https://blog.lacknb.cn/articles/2026/11/21/1795272443691.html)（已发布，[源稿](articles/04-model-training/trn-01-pretraining-data-pipeline.md)） | 对一组语料做文档级切分、去重、分词和打包，检查近似文档未跨集合，并解释一批移位标签。 |
| TRN-02 | [从随机权重预训练一个微型语言模型](https://blog.lacknb.cn/articles/2026/11/22/1795342908073.html)（已发布，[源稿](articles/04-model-training/trn-02-tiny-language-model.md)） | 从一批 Token 跟到下一词损失和参数更新，再核对中途恢复是否与连续训练产生相同结果。 |
| TRN-10 | [分布式训练的显存账本，FSDP 与 ZeRO](https://blog.lacknb.cn/articles/2026/11/22/1795342910568.html)（已发布，[源稿](articles/04-model-training/trn-10-distributed-memory.md)） | 画出参数、梯度和优化器状态在不同分片阶段的归属，核对双进程恢复与连续训练的下一步结果。 |
| TRN-03 | [选继续预训练、SFT 还是偏好优化](https://blog.lacknb.cn/articles/2026/11/22/1795327133650.html)（已发布，[源稿](articles/04-model-training/trn-03-training-objectives.md)） | 为同一退款任务分别准备领域原文、示范回答和偏好对照，并根据留出失败样本选择训练目标。 |
| TRN-04 | [SFT 怎么把对话变成监督信号](https://blog.lacknb.cn/articles/2026/11/22/1795353665754.html)（已发布，[源稿](articles/04-model-training/trn-04-sft-supervision.md)） | 给两轮对话标出进入损失的助手 Token，验证截断和填充前后的标签，再查独立生成能否通过业务条件。 |
| TRN-05 | [LoRA 与参数高效微调](https://blog.lacknb.cn/articles/2026/11/23/1795398317459.html)（已发布，[源稿](articles/04-model-training/trn-05-lora.md)） | 用目标层和秩算出 LoRA 的可训练参数量，再用同一输入确认独立加载、关闭与合并后的行为。 |
| TRN-06 | [QLoRA、训练量化和部署量化](https://blog.lacknb.cn/articles/2026/11/23/1795398359078.html)（已发布，[源稿](articles/04-model-training/trn-06-qlora-quantization.md)） | 拆开低位权重、运算精度和可训练 Adapter 三笔账，比较显存与同题输出，再区分 F-08 的推理量化。 |
| TRN-07 | [偏好数据、奖励模型与 DPO](https://blog.lacknb.cn/articles/2026/11/23/1795398360808.html)（已发布，[源稿](articles/04-model-training/trn-07-dpo-preference-data.md)） | 构造同题优劣答案，手算一条 DPO 偏好得分，并分别检查候选排序与自由生成的通过率。 |
| TRN-08 | [LLM 强化学习，从 PPO 到 GRPO](https://blog.lacknb.cn/articles/2026/11/23/1795398362308.html)（已发布，[源稿](articles/04-model-training/trn-08-ppo-grpo.md)） | 用同一组可验证奖励比较 PPO 与 GRPO 的优势，再用业务断言验证高奖励轨迹是否真的答对。 |
| TRN-11 | [知识蒸馏与学生模型评测](https://blog.lacknb.cn/articles/2026/11/23/1795398363907.html)（已发布，[源稿](articles/04-model-training/trn-11-knowledge-distillation.md)） | 给同一个学生对照硬标签与教师软分布，算出损失需要哪些输入，再检查分布差距变小时业务结果有无改善。 |
| TRN-12 | [合成训练数据与质量控制](https://blog.lacknb.cn/articles/2026/11/23/1795398365508.html)（已发布，[源稿](articles/04-model-training/trn-12-synthetic-training-data.md)） | 为退款任务补几条合成边界样本，重算标签、语义去重，再在隔离的人工留出集验证收益。 |
| TRN-09 | [用 Unsloth 跑通一次微调、评测和导出](https://blog.lacknb.cn/articles/2026/11/23/1795398367155.html)（已发布，[源稿](articles/04-model-training/trn-09-unsloth-finetune-export.md)） | 按固定模板和基座比较 LoRA 或 QLoRA 的导出产物，让同一份留出集分别跑依赖 Adapter 和合并后的配置。 |

工具分工可用一条链路理解。Transformers 提供模型与 Tokenizer，PEFT 提供参数高效适配，TRL 提供 SFT、偏好与强化学习训练器，Unsloth 用于这一支线的训练实战、运行和导出。对照同一模型和任务时，先统一数据、模板和评测口径。 [nanoGPT](https://github.com/karpathy/nanoGPT) · [TRL 文档](https://huggingface.co/docs/trl/index) · [PEFT 量化指南](https://huggingface.co/docs/peft/main/developer_guides/quantization) · [Unsloth 文档](https://unsloth.ai/docs)

原理阅读：[LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [DPO](https://arxiv.org/abs/2305.18290) · [PPO](https://arxiv.org/abs/1707.06347) · [GRPO](https://arxiv.org/abs/2402.03300) · [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) · [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) · [知识蒸馏论文](https://arxiv.org/abs/1503.02531) · [Self-Instruct](https://arxiv.org/abs/2212.10560)

## 5. 推理引擎与模型部署，vLLM 和 SGLang

教学模型选 `Qwen/Qwen2.5-0.5B-Instruct`。正式动手时记录模型 revision、vLLM tag、硬件与后端。Linux + NVIDIA CUDA 可作为主实验环境，CPU 或 Apple Silicon 的安装路径、支持的数据类型和性能需按选定版本实测。首个实验验证普通文本与流式文本，工具调用能力按模型、模板和解析器另测。 [Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart/) · [vLLM CPU 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| SERV-01 | [用 Qwen 0.5B 搭一个极简 vLLM Server](https://blog.lacknb.cn/articles/2026/11/23/1795400440818.html)（已发布，[源稿](articles/05-inference-serving/serv-01-vllm-server.md)） | 拿同一个探针检查模型列表、整包响应和流式结束，分别记录引擎启动进展和业务 JSON 是否合格。 |
| SERV-02 | [一次请求从浏览器到引擎再回浏览器](https://blog.lacknb.cn/articles/2026/11/23/1795400442549.html)（已发布，[源稿](articles/05-inference-serving/serv-02-request-lifecycle.md)） | 按请求标识对齐服务和浏览器的时间戳，定位首字晚出现发生在调度、代理缓冲还是客户端渲染。 |
| SERV-03 | [读 vLLM V1 源码，从路由跟到模型前向](https://blog.lacknb.cn/articles/2026/11/23/1795400444175.html)（已发布，[源稿](articles/05-inference-serving/serv-03-vllm-v1-source.md)） | 沿固定版本的 HTTP 路由、模板处理、调度和 Worker 画出一次进程间交接，并标注哪些环节有运行证据。 |
| SERV-04 | [PagedAttention、Continuous Batching 和 Prefix Cache](https://blog.lacknb.cn/articles/2026/11/23/1795400445885.html)（已发布，[源稿](articles/05-inference-serving/serv-04-paged-attention-batching.md)） | 手画两次请求的 KV 块表，预测改动前缀后哪些块要重填，再看并发下缓存命中与排队是否同时变化。 |
| SERV-05 | [端到端流式调试和取消](https://blog.lacknb.cn/articles/2026/11/23/1795400447564.html)（已发布，[源稿](articles/05-inference-serving/serv-05-streaming-cancellation.md)） | 用一次客户端取消核对代理是否断开上游、引擎是否释放请求，按时间点分辨真正流式和缓冲后的伪流式。 |
| SERV-06 | [Chunked Prefill 与 Speculative Decoding](https://blog.lacknb.cn/articles/2026/11/23/1795400449358.html)（已发布，[源稿](articles/05-inference-serving/serv-06-prefill-speculative.md)） | 分别画出分块预填充与投机解码的执行步骤，算草稿接受率，并设计能测量并发收益的单变量对照。 |
| SERV-07 | [SGLang 源码与 vLLM 对照](https://blog.lacknb.cn/articles/2026/11/23/1795400451069.html)（已发布，[源稿](articles/05-inference-serving/serv-07-sglang-source.md)） | 沿固定版 SGLang 的 HTTP 入口跟到 Scheduler，用同前缀和改前缀请求检查复用，再与 SERV-03、SERV-04 对照。 |
| SERV-12 | [模型注册、产物血缘与版本发布](https://blog.lacknb.cn/articles/2026/11/23/1795400452835.html)（已发布，[源稿](articles/05-inference-serving/serv-12-model-registry.md)） | 用一次历史请求的发布号查回基座、分词器和适配器清单，再用别名切换核对升级与回滚的目标。 |
| SERV-08 | [从训练产物部署成单机推理服务](https://blog.lacknb.cn/articles/2026/11/23/1795400454476.html)（已发布，[源稿](articles/05-inference-serving/serv-08-single-node-deployment.md)） | 写一份含模型、模板和 Adapter 的部署清单，用切换前、切换后和回滚后三个请求核对发布号与连接排空。 |
| SERV-13 | [本地推理与模型格式，llama.cpp 和 MLX-LM](https://blog.lacknb.cn/articles/2026/11/23/1795400456136.html)（已发布，[源稿](articles/05-inference-serving/serv-13-llamacpp-mlx.md)） | 用同一基座对照 GGUF 与 MLX，检查输入 Token ID、加载内存、首字时间和第一处输出差异。 |
| SERV-09 | [部署优化，从显存预算到延迟与吞吐](https://blog.lacknb.cn/articles/2026/11/23/1795400457884.html)（已发布，[源稿](articles/05-inference-serving/serv-09-inference-optimization.md)） | 先手算显存，再用包含业务正确性与延迟目标的达标吞吐比较两组负载或配置。 |
| SERV-14 | [GPU 性能剖析，从请求时间线到内核](https://blog.lacknb.cn/articles/2026/11/23/1795400459614.html)（已发布，[源稿](articles/05-inference-serving/serv-14-gpu-profiling.md)） | 从一条 CPU、拷贝和 CUDA 内核时间线算出内核区间并集，判断耗时集中在计算还是等待。 |
| SERV-10 | [多 GPU 与分离式推理部署](https://blog.lacknb.cn/articles/2026/11/23/1795400461248.html)（已发布，[源稿](articles/05-inference-serving/serv-10-distributed-inference.md)） | 为给定拓扑比较数据、张量、流水线和专家并行，再按传输量和带宽算 Prefill/Decode 分离的最低传输耗时。 |
| SERV-11 | [两种引擎的生产基准与选型报告](https://blog.lacknb.cn/articles/2026/11/23/1795400462988.html)（已发布，[源稿](articles/05-inference-serving/serv-11-engine-benchmark.md)） | 为引擎对比冻结权重、输入和采样，先测协议与取消，再用可比较样本报告延迟、吞吐和每次成功成本。 |
| SERV-15 | [推理服务容量规划与弹性伸缩](https://blog.lacknb.cn/articles/2026/11/23/1795400464747.html)（已发布，[源稿](articles/05-inference-serving/serv-15-serving-capacity.md)） | 根据延迟目标和流量估算容量，用突发时间线说明何时触发扩容、新副本何时就绪、积压何时下降。 |
| SERV-16 | [离线批量推理与异步作业](https://blog.lacknb.cn/articles/2026/11/23/1795400466528.html)（已发布，[源稿](articles/05-inference-serving/serv-16-offline-batch.md)） | 按批次与条目 ID 归并乱序结果，从失败点重跑单个条目并确认无重复写入；对照 SERV-04 的引擎批处理。 |

源码文章以 **一条请求的跟踪记录** 为验收。至少展示实际请求体、模板结果或 Token IDs、调度与输出事件、浏览器可见结果。对客户端取消，实测引擎是否回收同一请求。OpenAI 兼容接口按实际覆盖范围写，示范用 Chat Completions，Responses 和 Anthropic 形态另做兼容性对照。 [vLLM V1 架构](https://docs.vllm.ai/en/stable/design/arch_overview/) · [vLLM Prefix Cache](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) · [PagedAttention 论文](https://arxiv.org/abs/2309.06180) · [vLLM 调度与调优](https://docs.vllm.ai/en/stable/configuration/optimization/) · [vLLM 投机解码实验](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/tutorials/throughput_interactivity/) · [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/) · [SGLang 文档](https://docs.sglang.io/) · [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [MLX-LM](https://github.com/ml-explore/mlx-lm) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) · [vLLM 指标](https://docs.vllm.ai/en/latest/design/metrics/) · [OpenAI Batch API](https://developers.openai.com/api/docs/guides/batch)

## 6. Agent、Context Engineering、Loop Engineering 与 Harness

用「上传 CSV → 查询数据 → 生成报告 → 确认后归档」贯穿。**Context Engineering** 设计每轮模型看见什么，**Loop Engineering** 在本路线中是反馈闭环的工程设计，关注任务如何推进、失败如何修复及怎样判定完成。**Harness** 把两者接到模型、工具、状态、权限和运行环境中。按表格顺序学习，新增专题沿用新编号。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| AGT-01 | [从工具调用到 Agent Loop](https://blog.lacknb.cn/articles/2026/11/23/1795402332959.html)（已发布，[源稿](articles/06-agent-runtime/agt-01-agent-loop.md)） | 把一个工具调用跑过动作校验、结果回填和终止条件，验证用户取消或预算耗尽后循环能停。 |
| AGT-17 | [Context Engineering 总览](https://blog.lacknb.cn/articles/2026/11/23/1795402335217.html)（已发布，[源稿](articles/06-agent-runtime/agt-17-context-engineering.md)） | 画出模型输入的数据流，保存阶段快照，比较上下文压缩前后是否保留了任务事实和关键限制。 |
| AGT-02 | [上下文组装与 Prompt 层级](https://blog.lacknb.cn/articles/2026/11/23/1795402336863.html)（已发布，[源稿](articles/06-agent-runtime/agt-02-prompt-assembly.md)） | 打印一轮包含规则、历史和工具观察的真实请求，按调用 ID 对照结果，并标出哪些文本由外部提供。 |
| AGT-03 | [超长工具输出怎么处理](https://blog.lacknb.cn/articles/2026/11/23/1795402338418.html)（已发布，[源稿](articles/06-agent-runtime/agt-03-large-tool-outputs.md)） | 从大日志只返回概要和证据引用，再按权限和原文位置读回位于中间的错误。 |
| AGT-04 | [上下文裁剪](https://blog.lacknb.cn/articles/2026/11/23/1795402340040.html)（已发布，[源稿](articles/06-agent-runtime/agt-04-context-trimming.md)） | 给出窗口、输出预留和工具项开销，手算可用预算，并裁剪历史而保留关键约束和调用配对。 |
| AGT-05 | [上下文压缩](https://blog.lacknb.cn/articles/2026/11/23/1795402341623.html)（已发布，[源稿](articles/06-agent-runtime/agt-05-context-compaction.md)） | 保存含来源、决定和待办的摘要，再注入摘要与持久状态冲突，验证恢复不会重复写入。 |
| AGT-06 | [短期记忆与 Checkpoint](https://blog.lacknb.cn/articles/2026/11/23/1795402343288.html)（已发布，[源稿](articles/06-agent-runtime/agt-06-short-term-checkpoints.md)） | 区分历史、模型输入与持久状态，模拟中断后恢复到同一任务进度，并拒绝旧快照提交。 |
| AGT-07 | [长期记忆的写入、检索与治理](https://blog.lacknb.cn/articles/2026/11/23/1795402344957.html)（已发布，[源稿](articles/06-agent-runtime/agt-07-long-term-memory.md)） | 为跨会话偏好记录租户、来源和版本，验证过期、删除或跨租户的索引命中无法进入答复。 |
| AGT-08 | [工具超时、取消、重试与幂等](https://blog.lacknb.cn/articles/2026/11/23/1795402346585.html)（已发布，[源稿](articles/06-agent-runtime/agt-08-tool-retries.md)） | 给参数错误、超时和丢回执分别选处理方式，用同一动作 ID 查证已提交的写入。 |
| AGT-09 | [工具并发与依赖调度](https://blog.lacknb.cn/articles/2026/11/23/1795402348200.html)（已发布，[源稿](articles/06-agent-runtime/agt-09-tool-scheduling.md)） | 画并行和串行两种依赖图，按调用 ID 归并乱序回包，并复跑字典失败时统计节点是否会启动。 |
| AGT-10 | [工具太多时如何做工具治理](https://blog.lacknb.cn/articles/2026/11/23/1795402349728.html)（已发布，[源稿](articles/06-agent-runtime/agt-10-tool-governance.md)） | 对一组工具请求分别测正确工具进候选的比例和最终选中率，判断错误在搜索还是选择。 |
| AGT-11 | [MCP、Tools 和 Skills 的职责](https://blog.lacknb.cn/articles/2026/11/23/1795402351437.html)（已发布，[源稿](articles/06-agent-runtime/agt-11-mcp-tools-skills.md)） | 画一轮退款工具调用，标出 Tool、MCP 和 Skill 分别在哪层生效，以及哪一层决定是否授权。 |
| AGT-19 | [MCP Server 原理与协议生命周期](https://blog.lacknb.cn/articles/2026/11/23/1795402353046.html)（已发布，[源稿](articles/06-agent-runtime/agt-19-mcp-lifecycle.md)） | 读一次 MCP 初始化到调用的报文，区分 Host、Client、Server，并核对对应协议版本和传输。 |
| AGT-20 | [从零实现一个最小 MCP Server](https://blog.lacknb.cn/articles/2026/11/23/1795402354859.html)（已发布，[源稿](articles/06-agent-runtime/agt-20-mcp-csv-server.md)） | 通过客户端调用只读 CSV MCP 工具，验证非法列名、超时和结构化结果的三个行为。 |
| AGT-21 | [Agent Skills 的设计、加载与测试](https://blog.lacknb.cn/articles/2026/11/23/1795402356468.html)（已发布，[源稿](articles/06-agent-runtime/agt-21-agent-skills.md)） | 将退款规则写成 Skill，检查宿主能否发现，并为应触发和不应触发的请求各写一条验收。 |
| AGT-18 | [Loop Engineering 与反馈闭环](https://blog.lacknb.cn/articles/2026/11/23/1795402358094.html)（已发布，[源稿](articles/06-agent-runtime/agt-18-feedback-loop.md)） | 让同一任务经历错误草稿、修复和再次验收，再验证连续无进展或用完预算时会退出。 |
| AGT-12 | [Agent Harness 源码阅读](https://blog.lacknb.cn/articles/2026/11/23/1795402359765.html)（已发布，[源稿](articles/06-agent-runtime/agt-12-agent-harness-source.md)） | 从固定版本的 Agent 源码追一条消息，标出会话、工具、事件和权限各在哪个边界交接。 |
| AGT-13 | [Workflow、Agent 与混合编排](https://blog.lacknb.cn/articles/2026/11/23/1795402361827.html)（已发布，[源稿](articles/06-agent-runtime/agt-13-workflow-orchestration.md)） | 把退款处理拆成确定性计算、模型决策和人工审批，比较顺序、路由与并行的适用阶段。 |
| AGT-14 | [多 Agent 协作的几种模式](https://blog.lacknb.cn/articles/2026/11/23/1795402363518.html)（已发布，[源稿](articles/06-agent-runtime/agt-14-multi-agent-patterns.md)） | 给生成者和评审者划分上下文和写权限，拿两份互相冲突的统计结果设计合并规则。 |
| AGT-22 | [A2A 协议与跨 Agent 委派](https://blog.lacknb.cn/articles/2026/11/23/1795402365108.html)（已发布，[源稿](articles/06-agent-runtime/agt-22-a2a-protocol.md)） | 从 Agent Card 找到服务能力，发出一条 A2A 委派，并用任务和产物 ID 明确责任归属。 |
| AGT-15 | [Agent 的长任务与人机协作](https://blog.lacknb.cn/articles/2026/11/23/1795402366758.html)（已发布，[源稿](articles/06-agent-runtime/agt-15-long-running-tasks.md)） | 在等待审批时关闭运行时，重开后恢复到正确任务状态，拒绝旧审批并确保只写一份回执。 |
| AGT-24 | [持久化工作流的事件重放与状态恢复](https://blog.lacknb.cn/articles/2026/11/23/1795402368648.html)（已发布，[源稿](articles/06-agent-runtime/agt-24-temporal-replay.md)） | 重放一次 Temporal 工作流中断，数出哪些 Activity 复用历史、哪些实际重试、收据最终有几条。 |
| AGT-23 | [A2A 实战，两个 Agent 如何交接长任务](https://blog.lacknb.cn/articles/2026/11/23/1795402370736.html)（已发布，[源稿](articles/06-agent-runtime/agt-23-a2a-long-task.md)） | 跟踪只读校验的同步与流式状态，断线后凭任务 ID 查回结果，并验证取消和失败状态。 |
| AGT-16 | [Agent 运行轨迹与调试](https://blog.lacknb.cn/articles/2026/11/23/1795402373955.html)（已发布，[源稿](articles/06-agent-runtime/agt-16-agent-tracing.md)） | 用任务、Span、调用和动作标识串起脱敏轨迹，说明哪一步失败以及该查的下一条记录。 |
| AGT-25 | [Agent 前端事件协议与生成式 UI](https://blog.lacknb.cn/articles/2026/11/23/1795402375556.html)（已发布，[源稿](articles/06-agent-runtime/agt-25-ag-ui-a2ui.md)） | 用 API-05 的流式知识驱动报告卡片，断线重连后恢复状态，并检查确认绑定的是当前报告版本。 |

延伸阅读：[Anthropic Agent 架构模式](https://www.anthropic.com/research/building-effective-agents) · [Anthropic 上下文工程](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) · [Anthropic 多智能体实录](https://www.anthropic.com/research/multi-agent-research-system) · [LangChain Memory](https://docs.langchain.com/oss/python/concepts/memory) · [Anthropic 长任务 Harness](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents)

协议实战共用一个案例。**MCP Server** 暴露列统计和数据字典，**Agent Skill** 说明报告口径与验收步骤，**A2A** 负责把整项复核工作交给另一个独立 Agent 并取回产物。A2A Agent Card 中的 skill 是能力描述，Agent Skills 规范中的 Skill 是以 SKILL.md 为入口的内容包。实验记录协议和 SDK 版本。 [MCP 规范](https://modelcontextprotocol.io/specification/2025-11-25) · [MCP TypeScript SDK](https://ts.sdk.modelcontextprotocol.io/v2/) · [Agent Skills 规范](https://agentskills.io/specification) · [A2A 核心概念](https://a2a-protocol.org/latest/topics/key-concepts/) · [AG-UI](https://docs.ag-ui.com/introduction) · [A2UI](https://a2ui.org/)

源码对照从 Pi 的 Agent Loop 入手，需要看独立运行环境与更完整的 Coding Agent 时再读 OpenHands。两者都按固定 tag 验证一条调用路径。 [Pi](https://github.com/earendil-works/pi) · [OpenHands](https://github.com/OpenHands/OpenHands) · [Temporal Workflow 与重放](https://docs.temporal.io/workflow-definition)

## 7. Agent 效果评测

从单轮输出的评分扩展到任务终态。先在可重置的工作区里让 Agent 完成一份报告，独立读取报告产物和归档状态，再看调用轨迹解释失败。随后固定数据、工具权限、初始环境和任务预算，对比单 Agent、编排策略或 Harness 改动。结论覆盖产物正确性、业务副作用和执行成本。

**学习顺序**　AE-06 跑通第一条端到端用例 → AE-01～02 建任务集和评分标准 → AE-03 用开源框架实现 → AE-07 做故障归因 → AE-04 看公开基准 → AE-08 做发布回归。多轮强化学习（AE-05）放在能稳定评测任务之后。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| AE-06 | [第一篇，如何验证 Agent 真的完成了任务](https://blog.lacknb.cn/articles/2026/11/23/1795403176399.html)（已发布，[源稿](articles/07-agent-eval/ae-06-first-agent-eval.md)） | 沿 LE-07 的退款样本读回报告文件和归档表，独立核对两条、均值 5、批准前零条和批准后唯一写入。 |
| AE-01 | [设计 Agent 任务集与环境夹具](https://blog.lacknb.cn/articles/2026/11/23/1795403178111.html)（已发布，[源稿](articles/07-agent-eval/ae-01-task-fixtures.md)） | 写一条含初始状态、允许工具和预期产物的用例，每次重置环境，分别用固定回包和真实工具执行。 |
| AE-02 | [结果评测、轨迹评测和随机性](https://blog.lacknb.cn/articles/2026/11/23/1795403179638.html)（已发布，[源稿](articles/07-agent-eval/ae-02-outcomes-traces.md)） | 同看两段执行轨迹，区分真正的终态失败、冗余工具调用和越权，并给出独立的指标。 |
| AE-03 | [使用开源框架评测自己的 Agent](https://blog.lacknb.cn/articles/2026/11/23/1795403181369.html)（已发布，[源稿](articles/07-agent-eval/ae-03-inspect-ai.md)） | 用 Inspect AI 组织退款样本、Agent 和评分器，验证只说「已归档」的回复会被实际证据拦住。 |
| AE-07 | [Agent 失败如何归因，模型、工具还是 Harness](https://blog.lacknb.cn/articles/2026/11/23/1795403183016.html)（已发布，[源稿](articles/07-agent-eval/ae-07-failure-attribution.md)） | 对两个都写成 3 条的报告逐段核对，用一次只替换一层的回放定位 `tool_output` 或 `artifact`。 |
| AE-04 | [用开源 Benchmark 对照 Agent 能力](https://blog.lacknb.cn/articles/2026/11/23/1795403184588.html)（已发布，[源稿](articles/07-agent-eval/ae-04-public-benchmarks.md)） | 选一个适合自己目标的 Agent 基准，说明它验收的是工具参数、业务终态、仓库补丁还是可用成品。 |
| AE-08 | [从离线 Agent 回归到线上灰度](https://blog.lacknb.cn/articles/2026/11/23/1795403186137.html)（已发布，[源稿](articles/07-agent-eval/ae-08-regression-rollout.md)） | 为新能力、旧行为和权限各设一道回归门槛，再用灰度数据判断保持、放量或回退。 |
| AE-05 | [进阶实验，Agent 的多轮强化学习](https://blog.lacknb.cn/articles/2026/11/23/1795403187704.html)（已发布，[源稿](articles/07-agent-eval/ae-05-multi-turn-rl.md)） | 用同一多轮任务比较终态奖励和步骤奖励，找出有奖励却没完成的轨迹，并用未见任务复测。延伸到 TRN-08、EVO-01。可接着看 AGT-01。 |

### 常用 Agent 基准与打分方式

| 基准 / 排行榜 | 评测目标 | 验收依据 |
| --- | --- | --- |
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | 函数调用，覆盖单轮、多轮并行调用与拒调判断 | 按子任务检查 AST 工具名/参数结构、拒调或执行结果 |
| [τ²-Bench](https://github.com/sierra-research/tau2-bench) | 模拟用户与 Agent 双控制环境下的多轮业务交互 | 按任务规则独立检查交互结果及环境终态 |
| [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) | 通用助手联网检索、读取多格式文件与多步计算 | 简短、唯一且无歧义的精确终值匹配 |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) / [Verified](https://www.swebench.com/) / [Pro](https://github.com/scaleapi/SWE-bench_Pro-os) | 对真实仓库历史 Issue 生成代码补丁（含人工复核 Verified 与高难防污染 Pro 版） | 在隔离仓库中同时运行 Fail-to-Pass 与 Pass-to-Pass 测试 |
| [Terminal-Bench](https://www.tbench.ai/) | 真实 Linux Docker 容器里的纯命令行、编译与运维排障任务 | 在容器内运行确定性状态验证脚本 |
| [KingBench 4](https://github.com/aicodeking)（[AICodeKing](https://www.youtube.com/@AICodeKing)） | 社区实战排行榜（模型 + Harness 组合）：限时零人工从 0 到 1 交付 6 套完整硬核应用 | 60 分制按交付原样实测、安全逃逸与反虚报检查，并披露 API 成本与耗时 |
| [OSWorld](https://osworld-v1.xlang.ai/) / [WebArena](https://webarena.dev/) | 真实桌面 GUI（Computer-Use）与自托管真实网页浏览器工作流 | 检查虚拟机操作系统终态或网页后端数据库状态 |

**Inspect AI** 用于搭建和运行自有任务。覆盖 CSV 报告正确性、审批和归档的用例由业务终态评分。需要把线上轨迹与质量评分关联时可调研 **Langfuse**，并验证采集字段、脱敏和留存策略。 [Anthropic Agent 评测指南](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) · [Inspect AI Agent 文档](https://inspect.aisi.org.uk/agents.html) · [SWE-bench](https://github.com/SWE-bench/SWE-bench) · [τ²-Bench](https://github.com/sierra-research/tau2-bench) · [Langfuse](https://github.com/langfuse/langfuse) · [TRL 环境训练](https://huggingface.co/docs/trl/main/openenv) · [verl Agentic RL](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)

## 8. RAG、Embedding、Rerank 与 LLM Wiki

用同一小批原始文档做三次迭代：先做可引用的基础 RAG，再加排序与评测，最后编译成可维护的 Wiki，并以原始资料补证。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| KB-11 | [复杂文档解析，OCR、版面、表格与引用](https://blog.lacknb.cn/articles/2026/11/23/1795409706230.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-11-document-layout.md)） | 从一份跨页扫描表还原阅读顺序、表头和单元格，沿页码验证答复引用的原始证据。可接着看 KB-05。 |
| KB-01 | [文档解析、切块与索引](https://blog.lacknb.cn/articles/2026/11/23/1795409707913.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-01-chunk-index.md)） | 为退款条款做两种切块，检验单位和规则号是否保留在同一块；更新文档后核对引用版本。 |
| KB-02 | [Embedding 与向量相似度](https://blog.lacknb.cn/articles/2026/11/23/1795409709612.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-02-embedding-similarity.md)） | 手算两条向量的余弦相似度，再用同义词、字段名和规则编号检查召回。 |
| KB-03 | [BM25、向量检索与混合召回](https://blog.lacknb.cn/articles/2026/11/23/1795409711351.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-03-hybrid-retrieval.md)） | 对同一查询生成词法和向量排名，手算融合顺序，再检查范围外的文档是否在召回前已被筛掉。 |
| KB-04 | [Rerank 怎样改善证据排序](https://blog.lacknb.cn/articles/2026/11/23/1795409713044.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-04-reranking.md)） | 对一组候选重排，用正确证据有没有进入候选，判断应该改初召回还是重排分数。 |
| KB-05 | [从候选证据到可引用的 RAG 回答](https://blog.lacknb.cn/articles/2026/11/23/1795409714758.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-05-cited-rag.md)） | 把检索结果组装成可点验的证据包，验证每条引用支持原句，遇到材料不足或冲突能显式反馈。 |
| KB-12 | [RAG 评测入门，先标问题和依据](https://blog.lacknb.cn/articles/2026/11/23/1795409716558.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-12-rag-gold-evidence.md)） | 标注单段、跨段和无依据三类题，手算召回从 1/2 到 2/2，判断失败来自材料不足还是回答阶段。 |
| KB-08 | [RAG 的端到端质量评测](https://blog.lacknb.cn/articles/2026/11/23/1795409718320.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-08-rag-evaluation.md)） | 对同一份 30 天政策分别注入错误数字和错误引用，从答案一致、忠实度和引用三项判出问题所在。 |
| KB-13 | [用 BEIR 和 HotpotQA 练习检索与证据问答评测](https://blog.lacknb.cn/articles/2026/11/23/1795409720070.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-13-beir-hotpotqa.md)） | 分别对 BEIR 检索和 HotpotQA 答案与支持句算分，再为自己的退款问答保留同样的证据指标。 |
| KB-09 | [多租户 RAG 的索引权限与删除传播](https://blog.lacknb.cn/articles/2026/11/23/1795409721751.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-09-tenant-retrieval.md)） | 用旧引用测试撤权时的缓存和索引，记录拒绝何时生效，以及重新授权后能打开哪个版本。 |
| KB-10 | [结构化数据 Agent，Text-to-SQL 与指标口径](https://blog.lacknb.cn/articles/2026/11/23/1795409723408.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-10-text-to-sql.md)） | 手算退款统计后用 SQLite 对照去重、空值和时间区间的 SQL，再独立检查只读权限与预算。 |
| KB-06 | [Agentic RAG 与多跳检索](https://blog.lacknb.cn/articles/2026/11/23/1795409725088.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-06-agentic-rag.md)） | 根据第一跳返回的部门名写第二次查询，并从调用次数检查证据齐全、重复搜索和预算耗尽三种停止方式。 |
| KB-07 | [LLM Wiki 的资料编译、查询与维护](https://blog.lacknb.cn/articles/2026/11/23/1795409726827.html)（已发布，[源稿](articles/08-rag-and-wiki/kb-07-wiki-maintenance.md)） | 把原件编成带出处的 Wiki 页，撤回或更新原件后，核对旧数字、重复页和失效链接如何被处理。 |

### 常用检索与问答基准

- [BEIR](https://github.com/beir-cellar/beir) 提供多种信息检索数据集。给定查询、文档和相关性标注，可以用 **Recall@k** 看前 k 条覆盖多少相关文档，用 **nDCG@k** 看相关文档的排序位置。
- [HotpotQA](https://hotpotqa.github.io/) 提供多跳问答及支持事实标注，可分别按官方脚本报告答案和支持事实的 EM / F1。运行时记录所选任务形态和证据范围，避免混比固定候选与开放检索结果。
- 自建 RAG 还要检查答案是否有证据支撑、引用是否指向正确位置，以及资料无法回答时是否正确拒答。复用同一批题，通过人工提供参考证据来定位差异。

RAG 评测先用 KB-12 标注参考证据，再用 KB-08 分析召回、重排和答案。复用 LE-08 的评分器和 LE-09 的成对对照。Ragas 的上下文召回与忠实度可作为自动指标，并用人工抽样核对评分。 [Ragas 上下文召回](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) · [Ragas 忠实度](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)

经典 RAG 论文提供原理起点，系列案例采用现代常见的检索工程链路。 [RAG 原论文](https://arxiv.org/abs/2005.11401) · [Karpathy 的 LLM Wiki 原文](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) · [Azure AI Search 文档级权限示例](https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview) · [BIRD Text-to-SQL 基准](https://arxiv.org/abs/2305.03111) · [Docling](https://github.com/docling-project/docling)

## 9. LLM 与 Agent 安全护栏

先画数据流和信任边界，明确模型、应用、工具代理以及隔离环境各自负责的控制。用合成数据与受控工具验证允许、拒绝、审计与恢复。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| SEC-01 | [OWASP 是什么，Top 10 怎样用](https://blog.lacknb.cn/articles/2026/11/23/1795411161291.html)（已发布，[源稿](articles/09-security-guardrails/sec-01-owasp-risk-matrix.md)） | 画一张应用资产与信任边界图，将每项风险连到负责的组件、控制和可查验的证据。 |
| SEC-02 | [OWASP Top 10 for LLM Applications](https://blog.lacknb.cn/articles/2026/11/23/1795411163039.html)（已发布，[源稿](articles/09-security-guardrails/sec-02-llm-top-ten.md)） | 将 LLM 风险映射到退款助手的输入、检索和输出，对选出的控制写出合成验收样本及预期结果。 |
| SEC-03 | [OWASP Top 10 for Agentic Applications](https://blog.lacknb.cn/articles/2026/11/23/1795411164811.html)（已发布，[源稿](articles/09-security-guardrails/sec-03-agentic-top-ten.md)） | 沿 Agent 的工具、身份、记忆和委派链路检查风险，将同一故障对应到合适的控制负责层。 |
| SEC-04 | [LLM 应用的输入、输出和数据护栏](https://blog.lacknb.cn/articles/2026/11/23/1795411166416.html)（已发布，[源稿](articles/09-security-guardrails/sec-04-input-output-rails.md)） | 用合成的合法和异常报告检查误拦、漏判和失败处理，确保输出结构和引用范围能分别验收。 |
| SEC-05 | [Agent 工具调用、审批与运行时护栏](https://blog.lacknb.cn/articles/2026/11/23/1795411168053.html)（已发布，[源稿](articles/09-security-guardrails/sec-05-tool-approval.md)） | 替换报告版本、租户或批准内容后重新检查执行结果，确认正确批准前没有写入，重投后只有一条。 |
| SEC-06 | [第三方模型、MCP Server、Skill 与 A2A 的信任链](https://blog.lacknb.cn/articles/2026/11/23/1795411169711.html)（已发布，[源稿](articles/09-security-guardrails/sec-06-third-party-trust.md)） | 用一组第三方组件登记信息验证版本变更、委派收窄和撤权分别怎样影响执行决定。 |

护栏原型可读 **NeMo Guardrails**，用一个内容规则和一个工具审批流程对照应用层、工具代理和 Sandbox 的职责；涉及用户授权的决策由可信执行边界再次检查。 [OWASP Foundation](https://owasp.org/about/) · [LLM Applications Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) · [Agentic Applications Top 10 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) · [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) · [MCP 授权规范](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)

## 10. Sandbox 分类、源码与实测

从两个维度分类。**部署位置**可分本机、远程自建和托管云。**隔离机制**可分语言运行时、宿主进程策略、共享内核容器、用户态内核以及 VM/MicroVM。浏览器、代码解释器和完整工作区属于使用场景。同一个产品在不同部署配置下可以对应不同隔离机制。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| SBX-01 | [沙箱分类与威胁建模](https://blog.lacknb.cn/articles/2026/11/23/1795416672508.html)（已发布，[源稿](articles/10-sandbox-runtime/sbx-01-taxonomy-threat-model.md)） | 为退款任务列出文件、网络、进程与清理需求，选一种隔离方式，并把边界转成行为验收。 |
| SBX-02 | [本地沙箱如何限制 Coding Agent](https://blog.lacknb.cn/articles/2026/11/23/1795416674280.html)（已发布，[源稿](articles/10-sandbox-runtime/sbx-02-local-isolation.md)） | 在受控夹具中对照普通进程、宿主策略和容器，检查工作区、子进程、出站和清理能否按预期受限。 |
| SBX-03 | [远程沙箱的生命周期与执行接口](https://blog.lacknb.cn/articles/2026/11/23/1795416676067.html)（已发布，[源稿](articles/10-sandbox-runtime/sbx-03-remote-lifecycle.md)） | 跟一条任务走完创建、就绪、执行、取产物和销毁，并检查销毁后访问与重复销毁的结果。 |
| SBX-04 | [E2B、CubeSandbox、OpenSandbox 对比与源码路径](https://blog.lacknb.cn/articles/2026/11/23/1795416677850.html)（已发布，[源稿](articles/10-sandbox-runtime/sbx-04-projects-source-tour.md)） | 沿三套沙箱实现寻找创建返回、就绪确认和产物拉取，标出 SDK、控制面和执行面各自的证据。 |
| SBX-05 | [沙箱选型、启动基准与事件响应](https://blog.lacknb.cn/articles/2026/11/23/1795416679623.html)（已发布，[源稿](articles/10-sandbox-runtime/sbx-05-benchmark-response.md)） | 从样本复算首输出与删除确认的延迟，比较复用的收益，再核对旧工作区或任务状态是否被清掉。 |

建议实验顺序为本地最小策略 → OpenSandbox 本地 Docker 生命周期 → E2B 托管 API 形态 → CubeSandbox 在具备 KVM 条件的主机上读源码并验证。可用 **gVisor / Firecracker** 辅助理解隔离层，用 **Kata Containers** 补充容器接口与 VM 的结合。各项目的许可证与托管条款、后端和平台要求按选定版本确认；仓库中的性能数字先看测试口径。 [E2B 文档](https://docs.e2b.dev/) · [E2B 源码](https://github.com/e2b-dev/E2B) · [OpenSandbox](https://github.com/opensandbox-group/OpenSandbox) · [CubeSandbox 架构](https://github.com/TencentCloud/CubeSandbox/blob/master/docs/architecture/overview.md) · [gVisor](https://gvisor.dev/docs/architecture_guide/security/) · [Firecracker](https://github.com/firecracker-microvm/firecracker) · [Kata Containers](https://github.com/kata-containers/kata-containers)

## 11. Agent 自进化与递归自我改进（RSI）

先分清本轮纠错、跨会话经验沉淀、框架配置迭代与模型权重更新。报告 Agent 在任务中重试属于执行反馈。把反复验证的处理方法保存成 Skill，可以改变未来行为。修改提示或路由策略属于应用配置变更，GRPO 则属于 F-11 的模型后训练。**递归自我改进（Recursive Self-Improvement, RSI）**关注改进如何反馈到下一轮，例如新模型参与生成或筛选下一轮训练样本，或新版 Agent 参与发现下一个改进点。实验应明确每一轮改变的是权重、训练流程，还是应用版本，并验证改进是否能延续到新任务。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| EVO-01 | [Agent 自进化有哪些模式](https://blog.lacknb.cn/articles/2026/11/23/1795417257580.html)（已发布，[源稿](articles/11-agent-evolution/evo-01-agent-improvement-modes.md)） | 用同一次用户纠正区分当场修复、长期记忆、流程知识和路由更新，再检查换会话是否仍有效。 |
| EVO-02 | [从任务反馈到可审计的经验发布](https://blog.lacknb.cn/articles/2026/11/23/1795417259258.html)（已发布，[源稿](articles/11-agent-evolution/evo-02-auditable-experience.md)） | 为一条带来源的经验设计发布门槛，分别核对留出收益、适用范围、权限和可回退版本。 |
| EVO-03 | [模型如何参与训练下一版模型](https://blog.lacknb.cn/articles/2026/11/23/1795417260943.html)（已发布，[源稿](articles/11-agent-evolution/evo-03-model-feedback-training.md)） | 拆出 STaR 和自奖励中的样本生成、验证和更新步骤，用独立留出题识别循环是否真的带来收益。延伸到 TRN-08、TRN-12、LE-03。 |
| EVO-04 | [Google DeepMind 实例，SIMA 2 与 AlphaEvolve](https://blog.lacknb.cn/articles/2026/11/23/1795417262667.html)（已发布，[源稿](articles/11-agent-evolution/evo-04-sima-alphaevolve.md)） | 对照 SIMA 2 和 AlphaEvolve，说清哪个对象被更新、评分信号从哪里来，再据此设计自己的小型候选筛选实验。 |
| EVO-05 | [给 Agent 或普通程序加一个反馈改进 Agent](https://blog.lacknb.cn/articles/2026/11/23/1795417264333.html)（已发布，[源稿](articles/11-agent-evolution/evo-05-feedback-improvement-loop.md)） | 把运行失败做成脱敏回放，按质量、权限和资源门槛评审候选，再查灰度回滚与在途任务状态。 |

实战以「前次报告把空值当成零，用户纠正了口径」为例，分别回放旧版和新版，区分当次数据规则与可跨任务复用的流程。候选经验只能调整已授权范围内的任务步骤，权限仍由可信运行时检查。 [Hermes Skills 与经验沉淀](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) · [Hermes Memory 与写入审批](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)

### 一条可以验证的应用反馈闭环

以 CSV 报告为例，应用通过常规埋点采集版本、任务结果、错误类型、工具耗时、Token 与成本，观察 Agent 按 Trace ID 聚合同类失败。发现「缺失值被算成零」后，从脱敏轨迹提炼可复现用例，通过确定性的统计结果确认问题，再提交一个修改统计 Skill 或配置的候选版本。对同一基线复测失败样本和独立留出集，检查正确率、P95、单次成功成本及权限回归。达标后交给发布流程，下一轮继续观察同类问题的发生率。普通程序也可以用同一套机制提出校验规则、重试策略或测试用例的修改。评测和遥测可复用 AE-02 与 OPS-06。 [OpenTelemetry GenAI 观测示例](https://opentelemetry.io/blog/2026/genai-observability/)

Google DeepMind 的公开资料分别记录了 **SIMA 2 利用自生成经验训练后续版本**，以及 **AlphaEvolve 改进了 Gemini 训练使用的计算内核**。这两个案例支持讨论训练反馈与训练基础设施优化。Gemini 持续发布新型号的事实，无法单独说明其全流程采用了自主递归训练和发布。 [SIMA 2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/) · [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) · [Gemini 模型卡](https://deepmind.google/models/model-cards/) · [STaR](https://arxiv.org/abs/2203.14465) · [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)

## 12. OpenClaw 与 Hermes Agent 源码案例

两篇都固定源码 tag，从一条本地测试会话入手，记录入口、路由、模型调用、工具策略、会话持久化和结果返回。再挑同一处用户纠正，观察两个实现分别写入什么、在哪一轮加载以及如何撤销。

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| CASE-01 | [OpenClaw 设计思想与运行链路](https://blog.lacknb.cn/articles/2026/11/23/1795418062770.html)（已发布，[源稿](articles/12-framework-cases/case-01-openclaw-runtime.md)） | 沿选定的 OpenClaw 版本跟踪一条退款消息，区分会话路由、工作区资料装载和真实工具结果。 |
| CASE-02 | [Hermes Agent 的设计思想与自改进链路](https://blog.lacknb.cn/articles/2026/11/23/1795418064577.html)（已发布，[源稿](articles/12-framework-cases/case-02-hermes-runtime-memory.md)） | 在 Hermes 工作区跟一条经验从候选到批准、加载和撤回，再对照下一会话的输出。 |
| CASE-03 | [同一个任务对照 OpenClaw 与 Hermes Agent](https://blog.lacknb.cn/articles/2026/11/23/1795418066265.html)（已发布，[源稿](articles/12-framework-cases/case-03-openclaw-hermes-comparison.md)） | 用同一份 CSV 对照 OpenClaw 和 Hermes，区分同会话历史记住纠正与跨会话资料已发布。 |

阅读入口：[OpenClaw Gateway 架构](https://docs.openclaw.ai/concepts/architecture) · [OpenClaw Agent Loop](https://docs.openclaw.ai/concepts/agent-loop) · [Hermes Agent 源码](https://github.com/NousResearch/hermes-agent) · [Hermes Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)

## 13. Vibe Coding 与工程化交付

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| DEV-01 | [Vibe Coding 的工作方式与原型实验](https://blog.lacknb.cn/articles/2026/11/23/1795418815589.html)（已发布，[源稿](articles/13-vibe-coding-delivery/dev-01-vibe-coding-prototype.md)） | 用两轮原型演示退款报告，拿正常、全缺失和归档状态三个任务记录改动效果。 |
| DEV-02 | [从 Vibe Coding 原型走向 Agentic Engineering](https://blog.lacknb.cn/articles/2026/11/23/1795418817314.html)（已发布，[源稿](articles/13-vibe-coding-delivery/dev-02-agentic-engineering.md)） | 把原型拆成可测试规则与页面，验证计算、输入变化、取消与审批，保留能复跑的交付证据。 |
| DEV-03 | [Computer Use 与浏览器 Agent 的工程实践](https://blog.lacknb.cn/articles/2026/11/23/1795418819068.html)（已发布，[源稿](articles/13-vibe-coding-delivery/dev-03-computer-use.md)） | 对照截图坐标和可访问语义定位，在布局变化后验证报告结果，并在归档前交还确认。 |

用长任务 Harness 的增量开发、产物接力和评估反馈作为工程化对照。 [Anthropic 长任务 Harness](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents) · [生成与评估循环的工程案例](https://www.anthropic.com/research/harness-design-long-running-apps) · [Anthropic Computer Use 文档](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)

## 14. 生产化专题

| 编号 | 文章 | 读完可以做什么 |
| --- | --- | --- |
| OPS-01 | [LLM Gateway 的限流、路由、缓存与降级](https://blog.lacknb.cn/articles/2026/11/23/1795419423601.html)（已发布，[源稿](articles/14-production/ops-01-gateway.md)） | 按请求数、Token、并发与预算设计限额，用流式首段前后的故障判断能否切换供应商。 |
| OPS-02 | [Agent 安全控制如何落到生产环境](https://blog.lacknb.cn/articles/2026/11/23/1795419425419.html)（已发布，[源稿](articles/14-production/ops-02-agent-security.md)） | 在预发布分别变更归档身份、审批对象和目标，观察代理是否记录了对应拒绝理由。 |
| OPS-03 | [Sandbox 冷启动和资源回收](https://blog.lacknb.cn/articles/2026/11/23/1795419427113.html)（已发布，[源稿](articles/14-production/ops-03-sandbox-lifecycle.md)） | 从一次运行分出排队、镜像、运行时和首命令耗时，按相同口径比较冷启动、预热和回收。 |
| OPS-04 | [Agent 服务拆分与可靠迁移](https://blog.lacknb.cn/articles/2026/11/23/1795419428781.html)（已发布，[源稿](articles/14-production/ops-04-reliable-migration.md)） | 沿任务、发件箱、队列、收据四张状态表重放事务回滚和重复交付，解释为何只保留一张收据。 |
| OPS-05 | [端到端可观测与持续发布](https://blog.lacknb.cn/articles/2026/11/23/1795419430432.html)（已发布，[源稿](articles/14-production/ops-05-progressive-delivery.md)） | 从一个故障样本写出离线回归与灰度门槛，用正确率、P95 和重复写入决定回切，并检查在途任务。 |
| OPS-07 | [Prompt、工具契约和在途任务的版本兼容](https://blog.lacknb.cn/articles/2026/11/23/1795419432161.html)（已发布，[源稿](articles/14-production/ops-07-runtime-versioning.md)） | 拿一条旧任务和一份旧版工具回包核对恢复，确认金额的元分换算正确、未知版本被挂起、重试不重复写入。 |
| OPS-06 | [用 OpenTelemetry 定位一条 Agent 慢请求](https://blog.lacknb.cn/articles/2026/11/23/1795419433902.html)（已发布，[源稿](articles/14-production/ops-06-opentelemetry-trace.md)） | 用 Spring Boot、Collector 和 Jaeger 核对四层 Span，区分业务错数、消息断链和导出失败。 |

生产链路观测参考：[OpenTelemetry 的 GenAI Trace 示例](https://opentelemetry.io/blog/2026/genai-observability/)

## 退款报告排障路线

从「退款报告等得久，还把 2 笔写成了 3 笔」的工单开始。路线里的追踪和评测命令可在仓库根目录跑通，用的是合成数据和模拟等待。

| 顺序 | 阅读入口 | 读完可以做什么 |
| --- | --- | --- |
| 1 | [OpenTelemetry 入门](articles/14-production/opentelemetry-tracing-primer.md) | 对齐入队端与消费端的 Trace ID 和父 Span ID，读出队列等待的位置。 |
| 2 | [OPS-06 实战排障与回归](articles/14-production/ops-06-opentelemetry-trace.md#实战排障与回归) | 用 Java 重放慢请求与两种错数，找到统计值第一次偏离的位置。 |
| 3 | [AE-07 失败归因](articles/07-agent-eval/ae-07-failure-attribution.md) | 用固定正确回包做一次对照，核对首个失败阶段、报告文件和归档记录。 |
| 4 | [AE-08 离线回归与灰度](articles/07-agent-eval/ae-08-regression-rollout.md) · [OPS-05 发布门槛](articles/14-production/ops-05-progressive-delivery.md) | 把故障样本纳入回归，分别核对报告正确性、尾延迟和重复收据。 |

如果耗时集中在等工具结果，可回到 [AGT-09 工具并发](articles/06-agent-runtime/agt-09-tool-scheduling.md) 检查调度依赖。如果耗时集中在沙箱租约，可用 [OPS-03 沙箱生命周期](articles/14-production/ops-03-sandbox-lifecycle.md) 的分段计时继续定位。

想接入自己的 Java 服务，可以从 [Spring Boot 接入实战](articles/14-production/ops-06-opentelemetry-trace.md#spring-boot-接入实战) 走一遍应用、Collector 和 Jaeger，再用两次故障复跑区分上下文断开与遥测导出失败。

## 首轮写作主线

专题仍按各阶段的先修顺序展开。下面用 CSV 报告做五次交付，每次只引入当前需要的技术：

1. **一次模型请求。** 打印模板和 Token IDs，理解 Prefill 与 Decode。用 Chat Completions 播放一次真实 SSE，再固定样本做 Prompt 对照。
2. **一个推理服务。** 让 Qwen/vLLM 接入同一请求，记录引擎与浏览器的首 Token 或首字指标。
3. **一个能执行工具的 Agent。** 用单 Agent 调用只读统计工具，组装上下文，把结果和产物引用写入状态。用 LE-07 建单轮评分基线，再用 AE-06 复用样本验证报告产物和归档终态。
4. **一个会恢复、可验证的任务。** 加入归档前确认、断点恢复、RAG 证据和授权测试。回归至少覆盖断流、重复执行与越权拒绝。
5. **一个可追踪的交付。** 通过隔离环境执行，连通模型、检索和工具 Trace，报告任务完成率、成本和 P95。按需要选做多 Agent、A2A、RSI 应用反馈闭环（EVO-05）或实时语音分支。

如果先写训练与部署，可从 **TRN-02 的微型预训练**起步，再完成 **TRN-04～06 和 TRN-09** 的 Qwen 微调与导出，最后执行 **SERV-01、SERV-07、SERV-12、SERV-08～09、SERV-11** 的服务、优化及选型实验。训练数据仍沿用 CSV 报告案例，服务端用同一组留出样本核对质量。DPO、PPO/GRPO 和多 GPU 部署可各自扩展为后续实验。

## 专题文章的固定结构

1. **场景与先修**。明确解决的问题、前置篇章和一个具体输入。
2. **可观察协议**。展示实际消息、Token、事件或状态，给出输出和边界。
3. **机制与源码**。按调用顺序跟到关键模块。涉及源码时记录版本、入口和行为。
4. **最小实验与验收**。提供可复现的请求或样本，验证成功路径、一个边界条件和必要指标。
5. **关联阅读**。链接上游与下一篇，以及可对照的官方文档、论文或对应题库题号。

编号用于写作索引，已有编号保留，新增编号按先修依赖插入相应位置。学习及写作顺序以章节与表格的实际行序为准。现有[题库源稿](agent-sandbox-interview-bank.md)按面试分类保留题号，[Agent 工程笔记](agent-engineering-notes.md)用来快速查阅已整理的机制。
