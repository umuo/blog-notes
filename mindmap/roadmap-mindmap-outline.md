# LLM 与 Agent 知识脑图（139/139 已发布）

> 请求主线：文本 → Token → 推理 → HTTP/SSE → Agent 工具循环 → RAG 检索 → 安全隔离 → 生产化

## 一、大模型本体与服务（63/63）

### 1. LLM 基础与推理（15/15）
- 先修：Python、矩阵和概率直觉
- 产物：解释一条提示从 Token 到首个输出
- **F-01** [Tokenizer 与 Chat Template](https://blog.lacknb.cn/articles/2026/11/19/1795068393122.html)
  - 对同一句话打印普通分词、应用聊天模板、加生成起始标记这三份 Token ID，找到序列第一次不同的位置。
- **F-02** [Transformer 如何从 Token 算出 logits](https://blog.lacknb.cn/articles/2026/11/20/1795135053358.html)
  - 画出一个 Token 经过 Embedding、注意力和输出层的张量形状，再只替换末尾 Token，验证因果遮罩是否保护前面的位置。
- **F-15** [Encoder-only、Decoder-only 与 Encoder-Decoder](https://blog.lacknb.cn/articles/2026/11/20/1795135557100.html)
  - 对同一句话画出三种 Transformer 的可见关系，再给检索、重排与生成分别选择可验收的输出。延伸到 KB-02、KB-04、KB-05。
- **F-03** [MHA、MQA、GQA 与 RoPE](https://blog.lacknb.cn/articles/2026/11/20/1795154536776.html)
  - 给定 Q 头、KV 头和每头维数，画出三种共享关系并计算 KV 缓存大小，解释同一组的 Q 头为何仍有不同结果。
- **F-13** [MoE 如何选择专家与分配计算](https://blog.lacknb.cn/articles/2026/11/20/1795161347233.html)
  - 从 Router 分数选 Top-k 专家并算出合并结果，再逐专家统计分派次数。专家并行接 SERV-10。
- **F-04** [自回归生成与采样](https://blog.lacknb.cn/articles/2026/11/20/1795161419910.html)
  - 从一组 logits 手算温度和截断后的候选概率，并用同一个小词表复现 EOS、停止词和输出上限。
- **F-05** [Prefill 与 Decode](https://blog.lacknb.cn/articles/2026/11/20/1795161425780.html)
  - 给定 21 个输入 Token，标出首个输出何时被选中，以及缓存长度何时从 21 增为 22。
- **F-06** [KV Cache 如何加速 Decode](https://blog.lacknb.cn/articles/2026/11/20/1795161430009.html)
  - 按层和 KV 头数计算增量缓存字节数，并判断改动一个输入 Token 后哪些前缀位置还能复用。
- **F-14** [FlashAttention 如何减少注意力的显存读写](https://blog.lacknb.cn/articles/2026/11/20/1795161433683.html)
  - 当新块提高最大分数时手算旧分母的重缩放，和整表 Softmax 比对；区分这个分块和 SERV-04 的 KV 缓存块。
- **F-07** [TTFT、TPOT、ITL、吞吐与端到端延迟](https://blog.lacknb.cn/articles/2026/11/20/1795161437100.html)
  - 在同一条请求上标出引擎首 Token、服务首事件、浏览器首字三个时间点，判断慢发生在生成、传输还是展示。
- **F-08** [模型精度与量化](https://blog.lacknb.cn/articles/2026/11/20/1795161440718.html)
  - 把一组权重压成 4 位编码并算出误差，分别估算权重和 KV 缓存占用；用同一题检查质量影响。
- **F-09** [多模态请求如何进入模型](https://blog.lacknb.cn/articles/2026/11/20/1795161444330.html)
  - 从图片网格数出视觉占位，比较纯文本和图文的入模长度，并检查两张图片的占位是否按顺序对应。
- **F-10** [模型从训练到可用的过程](https://blog.lacknb.cn/articles/2026/11/20/1795161450059.html)
  - 把同一退款任务改写为预训练原文、监督示范、偏好数据或强化学习奖励，标出损失使用哪段数据。
- **F-11** [GRPO 如何用于推理模型后训练](https://blog.lacknb.cn/articles/2026/11/20/1795161454784.html)
  - 用四份答案的奖励手算组内相对优势，解释全对或全错的组为何没有组内排序信号。可接着看 LE-02。
- **F-12** [推理时计算与 Reasoning 模型](https://blog.lacknb.cn/articles/2026/11/20/1795161460545.html)
  - 比较两档推理预算，把正确率、计费 Token、延迟和未完成请求放进同一张表。

### 2. LLM API 与流式协议（10/10）
- 先修：F-01～F-04、HTTP/JSON
- 产物：用原始 HTTP 跑通文本、SSE 和工具调用
- **API-01** [从原始 HTTP 请求认识 LLM API](https://blog.lacknb.cn/articles/2026/11/21/1795268752670.html)
  - 组装一次原始 HTTP 请求，从响应读出答案、完成状态、用量；按 401、429 和输出不完整选择检查项。
- **API-02** [OpenAI Chat Completions 的消息协议](https://blog.lacknb.cn/articles/2026/11/21/1795268752027.html)
  - 手写两轮 Chat Completions 的 messages，删掉第一轮的一条规则重放，找出追问丢了哪段上下文。
- **API-10** [Prompt Engineering 的可复现对照实验](https://blog.lacknb.cn/articles/2026/11/21/1795268751247.html)
  - 用相同样本比较三版提示，分别记录格式、数值和输入 Token，把一条反例交给 LE-04 的发布回归。
- **API-03** [从 Chat Completions、Assistants 到 Responses](https://blog.lacknb.cn/articles/2026/11/21/1795268750432.html)
  - 在三种接口的请求与回包里找到会话载体和工具调用标识，迁移一次工具回填，再用缺 ID 的样本检查。
- **API-04** [Function Calling 与 Structured Outputs](https://blog.lacknb.cn/articles/2026/11/21/1795268749640.html)
  - 完成一次工具定义、参数校验与结果回填，再让报告漏字段或数值错误，确认结构化输出与业务断言分别拦截。
- **API-09** [约束解码如何生成符合 Schema 的结构化输出](https://blog.lacknb.cn/articles/2026/11/21/1795268748887.html)
  - 用一个小 Schema 手算每步允许的 Token，再让格式正确但数字错误的报告通过结构检查并被业务测试拒绝。
- **API-05** [SSE 和流式工具调用](https://blog.lacknb.cn/articles/2026/11/21/1795268748102.html)
  - 把同一份流按字节切碎后重组，按调用 ID 还原工具参数，区分收到完整调用和断流时的半成品。
- **API-06** [Anthropic Messages API 的内容块与事件](https://blog.lacknb.cn/articles/2026/11/21/1795268747439.html)
  - 从一段 Anthropic 流按块索引还原工具输入，再用 tool_use_id 关联工具结果，并与 OpenAI 报文对照。
- **API-07** [模型协议适配与能力矩阵](https://blog.lacknb.cn/articles/2026/11/21/1795268746690.html)
  - 用同一份请求测试适配器的能力声明、错误与用量映射，对缺少结构化输出能力的组合给出可见处理。
- **API-08** [实时语音 Agent 的双向流](https://blog.lacknb.cn/articles/2026/11/21/1795268619097.html)
  - 用已生成、已接收和已播放的时间画出一次语音打断，验证旧音频及时停止及迟到数据不会再播。可接着看 F-09、API-05。

### 3. 评测入门与 LLM 效果评测（10/10）
- 先修：读懂模型请求和输出
- 产物：手写评分器，做可复现的模型或 Prompt 对照
- **LE-07** [第一篇，什么是评测，如何跑通第一条用例](https://blog.lacknb.cn/articles/2026/11/21/1795270270931.html)
  - 给三行退款表写一个能区分 JSON 语法错误和业务算术错误的评分器，保存每条失败理由。
- **LE-01** [从业务目标到数据集、基线和评分](https://blog.lacknb.cn/articles/2026/11/21/1795270270258.html)
  - 建一份带数据版本和分层标签的黄金集，识别训练与留出间的同源样本，并按失败类别汇总。
- **LE-08** [如何设计可测试的评分器与评分准则](https://blog.lacknb.cn/articles/2026/11/21/1795270269520.html)
  - 用已知对错的样本独立测试 Schema、公差和业务规则，再为开放解释写出人工判分标准。
- **LE-10** [常用 LLM 评测数据集测什么，应该怎么选](https://blog.lacknb.cn/articles/2026/11/21/1795270268728.html)
  - 为一个任务选择合适的公开基准类型，并说明应核对选项、最终数值、可运行测试还是其他交付证据。
- **LE-02** [公开基准实战，怎样复现并比较模型结果](https://blog.lacknb.cn/articles/2026/11/21/1795270267947.html)
  - 在固定基准子集上重放同一份生成，比较两套答案抽取口径，再计算单次通过率和 pass@k。
- **LE-03** [LLM-as-a-Judge 与人工校准](https://blog.lacknb.cn/articles/2026/11/21/1795270267279.html)
  - 给一对答案先做人工标注，再互换位置检查模型裁判，把原始位置和还原后的答案 ID 分开记。
- **LE-09** [怎样判断两版效果差异是否可信](https://blog.lacknb.cn/articles/2026/11/21/1795270266572.html)
  - 按同一道题计算候选与基线的差值，再整题取样比较区间，解释为何同题的多次运行不能拆成独立样本。
- **LE-04** [把效果评测接入 Prompt 和模型回归](https://blog.lacknb.cn/articles/2026/11/21/1795270265847.html)
  - 把数据版本、正确率、成本、延迟及失败反例写入发布检查，分别跑出放行和阻止发布的结果。
- **LE-06** [长上下文的有效性如何评测](https://blog.lacknb.cn/articles/2026/11/21/1795270265166.html)
  - 把同一证据放进长上下文的开头、中间与末尾，分开统计引用与算术正确性、首字时间和用量。结合 KB-05 对照 RAG。可接着看 AGT-04。
- **LE-05** [模型改了，线上效果真的变好了吗](https://blog.lacknb.cn/articles/2026/11/21/1795270264488.html)
  - 设计只读影子实验和用户级灰度，先核对随机分流，再比较成功率、接管率和回滚条件。

### 4. 模型预训练与后训练（12/12）
- 先修：Tokenizer、因果语言模型、训练损失与 LE 的评测口径
- 产物：跑通微型预训练和一次 Qwen 微调，产出可复现的模型版本
- **TRN-01** [预训练语料与 Token 数据管线](https://blog.lacknb.cn/articles/2026/11/21/1795272443691.html)
  - 对一组语料做文档级切分、去重、分词和打包，检查近似文档未跨集合，并解释一批移位标签。
- **TRN-02** [从随机权重预训练一个微型语言模型](https://blog.lacknb.cn/articles/2026/11/22/1795342908073.html)
  - 从一批 Token 跟到下一词损失和参数更新，再核对中途恢复是否与连续训练产生相同结果。
- **TRN-10** [分布式训练的显存账本，FSDP 与 ZeRO](https://blog.lacknb.cn/articles/2026/11/22/1795342910568.html)
  - 画出参数、梯度和优化器状态在不同分片阶段的归属，核对双进程恢复与连续训练的下一步结果。
- **TRN-03** [选继续预训练、SFT 还是偏好优化](https://blog.lacknb.cn/articles/2026/11/22/1795327133650.html)
  - 为同一退款任务分别准备领域原文、示范回答和偏好对照，并根据留出失败样本选择训练目标。
- **TRN-04** [SFT 怎么把对话变成监督信号](https://blog.lacknb.cn/articles/2026/11/22/1795353665754.html)
  - 给两轮对话标出进入损失的助手 Token，验证截断和填充前后的标签，再查独立生成能否通过业务条件。
- **TRN-05** [LoRA 与参数高效微调](https://blog.lacknb.cn/articles/2026/11/23/1795398317459.html)
  - 用目标层和秩算出 LoRA 的可训练参数量，再用同一输入确认独立加载、关闭与合并后的行为。
- **TRN-06** [QLoRA、训练量化和部署量化](https://blog.lacknb.cn/articles/2026/11/23/1795398359078.html)
  - 拆开低位权重、运算精度和可训练 Adapter 三笔账，比较显存与同题输出，再区分 F-08 的推理量化。
- **TRN-07** [偏好数据、奖励模型与 DPO](https://blog.lacknb.cn/articles/2026/11/23/1795398360808.html)
  - 构造同题优劣答案，手算一条 DPO 偏好得分，并分别检查候选排序与自由生成的通过率。
- **TRN-08** [LLM 强化学习，从 PPO 到 GRPO](https://blog.lacknb.cn/articles/2026/11/23/1795398362308.html)
  - 用同一组可验证奖励比较 PPO 与 GRPO 的优势，再用业务断言验证高奖励轨迹是否真的答对。
- **TRN-11** [知识蒸馏与学生模型评测](https://blog.lacknb.cn/articles/2026/11/23/1795398363907.html)
  - 给同一个学生对照硬标签与教师软分布，算出损失需要哪些输入，再检查分布差距变小时业务结果有无改善。
- **TRN-12** [合成训练数据与质量控制](https://blog.lacknb.cn/articles/2026/11/23/1795398365508.html)
  - 为退款任务补几条合成边界样本，重算标签、语义去重，再在隔离的人工留出集验证收益。
- **TRN-09** [用 Unsloth 跑通一次微调、评测和导出](https://blog.lacknb.cn/articles/2026/11/23/1795398367155.html)
  - 按固定模板和基座比较 LoRA 或 QLoRA 的导出产物，让同一份留出集分别跑依赖 Adapter 和合并后的配置。

### 5. 推理引擎与模型部署，vLLM 和 SGLang（16/16）
- 先修：Prefill、KV Cache、Chat Completions 与 SSE
- 产物：用同一份请求和记录对照 vLLM 与 SGLang 的服务行为
- **SERV-01** [用 Qwen 0.5B 搭一个极简 vLLM Server](https://blog.lacknb.cn/articles/2026/11/23/1795400440818.html)
  - 拿同一个探针检查模型列表、整包响应和流式结束，分别记录引擎启动进展和业务 JSON 是否合格。
- **SERV-02** [一次请求从浏览器到引擎再回浏览器](https://blog.lacknb.cn/articles/2026/11/23/1795400442549.html)
  - 按请求标识对齐服务和浏览器的时间戳，定位首字晚出现发生在调度、代理缓冲还是客户端渲染。
- **SERV-03** [读 vLLM V1 源码，从路由跟到模型前向](https://blog.lacknb.cn/articles/2026/11/23/1795400444175.html)
  - 沿固定版本的 HTTP 路由、模板处理、调度和 Worker 画出一次进程间交接，并标注哪些环节有运行证据。
- **SERV-04** [PagedAttention、Continuous Batching 和 Prefix Cache](https://blog.lacknb.cn/articles/2026/11/23/1795400445885.html)
  - 手画两次请求的 KV 块表，预测改动前缀后哪些块要重填，再看并发下缓存命中与排队是否同时变化。
- **SERV-05** [端到端流式调试和取消](https://blog.lacknb.cn/articles/2026/11/23/1795400447564.html)
  - 用一次客户端取消核对代理是否断开上游、引擎是否释放请求，按时间点分辨真正流式和缓冲后的伪流式。
- **SERV-06** [Chunked Prefill 与 Speculative Decoding](https://blog.lacknb.cn/articles/2026/11/23/1795400449358.html)
  - 分别画出分块预填充与投机解码的执行步骤，算草稿接受率，并设计能测量并发收益的单变量对照。
- **SERV-07** [SGLang 源码与 vLLM 对照](https://blog.lacknb.cn/articles/2026/11/23/1795400451069.html)
  - 沿固定版 SGLang 的 HTTP 入口跟到 Scheduler，用同前缀和改前缀请求检查复用，再与 SERV-03、SERV-04 对照。
- **SERV-12** [模型注册、产物血缘与版本发布](https://blog.lacknb.cn/articles/2026/11/23/1795400452835.html)
  - 用一次历史请求的发布号查回基座、分词器和适配器清单，再用别名切换核对升级与回滚的目标。
- **SERV-08** [从训练产物部署成单机推理服务](https://blog.lacknb.cn/articles/2026/11/23/1795400454476.html)
  - 写一份含模型、模板和 Adapter 的部署清单，用切换前、切换后和回滚后三个请求核对发布号与连接排空。
- **SERV-13** [本地推理与模型格式，llama.cpp 和 MLX-LM](https://blog.lacknb.cn/articles/2026/11/23/1795400456136.html)
  - 用同一基座对照 GGUF 与 MLX，检查输入 Token ID、加载内存、首字时间和第一处输出差异。
- **SERV-09** [部署优化，从显存预算到延迟与吞吐](https://blog.lacknb.cn/articles/2026/11/23/1795400457884.html)
  - 先手算显存，再用包含业务正确性与延迟目标的达标吞吐比较两组负载或配置。
- **SERV-14** [GPU 性能剖析，从请求时间线到内核](https://blog.lacknb.cn/articles/2026/11/23/1795400459614.html)
  - 从一条 CPU、拷贝和 CUDA 内核时间线算出内核区间并集，判断耗时集中在计算还是等待。
- **SERV-10** [多 GPU 与分离式推理部署](https://blog.lacknb.cn/articles/2026/11/23/1795400461248.html)
  - 为给定拓扑比较数据、张量、流水线和专家并行，再按传输量和带宽算 Prefill/Decode 分离的最低传输耗时。
- **SERV-11** [两种引擎的生产基准与选型报告](https://blog.lacknb.cn/articles/2026/11/23/1795400462988.html)
  - 为引擎对比冻结权重、输入和采样，先测协议与取消，再用可比较样本报告延迟、吞吐和每次成功成本。
- **SERV-15** [推理服务容量规划与弹性伸缩](https://blog.lacknb.cn/articles/2026/11/23/1795400464747.html)
  - 根据延迟目标和流量估算容量，用突发时间线说明何时触发扩容、新副本何时就绪、积压何时下降。
- **SERV-16** [离线批量推理与异步作业](https://blog.lacknb.cn/articles/2026/11/23/1795400466528.html)
  - 按批次与条目 ID 归并乱序结果，从失败点重跑单个条目并确认无重复写入；对照 SERV-04 的引擎批处理。


## 二、Agent 核心与知识库（47/47）

### 6. Agent、Context Engineering、Loop Engineering 与 Harness（25/25）
- 先修：文本和工具协议、上下文预算
- 产物：写一个可终止、可恢复的 Agent
- **AGT-01** [从工具调用到 Agent Loop](https://blog.lacknb.cn/articles/2026/11/23/1795402332959.html)
  - 把一个工具调用跑过动作校验、结果回填和终止条件，验证用户取消或预算耗尽后循环能停。
- **AGT-17** [Context Engineering 总览](https://blog.lacknb.cn/articles/2026/11/23/1795402335217.html)
  - 画出模型输入的数据流，保存阶段快照，比较上下文压缩前后是否保留了任务事实和关键限制。
- **AGT-02** [上下文组装与 Prompt 层级](https://blog.lacknb.cn/articles/2026/11/23/1795402336863.html)
  - 打印一轮包含规则、历史和工具观察的真实请求，按调用 ID 对照结果，并标出哪些文本由外部提供。
- **AGT-03** [超长工具输出怎么处理](https://blog.lacknb.cn/articles/2026/11/23/1795402338418.html)
  - 从大日志只返回概要和证据引用，再按权限和原文位置读回位于中间的错误。
- **AGT-04** [上下文裁剪](https://blog.lacknb.cn/articles/2026/11/23/1795402340040.html)
  - 给出窗口、输出预留和工具项开销，手算可用预算，并裁剪历史而保留关键约束和调用配对。
- **AGT-05** [上下文压缩](https://blog.lacknb.cn/articles/2026/11/23/1795402341623.html)
  - 保存含来源、决定和待办的摘要，再注入摘要与持久状态冲突，验证恢复不会重复写入。
- **AGT-06** [短期记忆与 Checkpoint](https://blog.lacknb.cn/articles/2026/11/23/1795402343288.html)
  - 区分历史、模型输入与持久状态，模拟中断后恢复到同一任务进度，并拒绝旧快照提交。
- **AGT-07** [长期记忆的写入、检索与治理](https://blog.lacknb.cn/articles/2026/11/23/1795402344957.html)
  - 为跨会话偏好记录租户、来源和版本，验证过期、删除或跨租户的索引命中无法进入答复。
- **AGT-08** [工具超时、取消、重试与幂等](https://blog.lacknb.cn/articles/2026/11/23/1795402346585.html)
  - 给参数错误、超时和丢回执分别选处理方式，用同一动作 ID 查证已提交的写入。
- **AGT-09** [工具并发与依赖调度](https://blog.lacknb.cn/articles/2026/11/23/1795402348200.html)
  - 画并行和串行两种依赖图，按调用 ID 归并乱序回包，并复跑字典失败时统计节点是否会启动。
- **AGT-10** [工具太多时如何做工具治理](https://blog.lacknb.cn/articles/2026/11/23/1795402349728.html)
  - 对一组工具请求分别测正确工具进候选的比例和最终选中率，判断错误在搜索还是选择。
- **AGT-11** [MCP、Tools 和 Skills 的职责](https://blog.lacknb.cn/articles/2026/11/23/1795402351437.html)
  - 画一轮退款工具调用，标出 Tool、MCP 和 Skill 分别在哪层生效，以及哪一层决定是否授权。
- **AGT-19** [MCP Server 原理与协议生命周期](https://blog.lacknb.cn/articles/2026/11/23/1795402353046.html)
  - 读一次 MCP 初始化到调用的报文，区分 Host、Client、Server，并核对对应协议版本和传输。
- **AGT-20** [从零实现一个最小 MCP Server](https://blog.lacknb.cn/articles/2026/11/23/1795402354859.html)
  - 通过客户端调用只读 CSV MCP 工具，验证非法列名、超时和结构化结果的三个行为。
- **AGT-21** [Agent Skills 的设计、加载与测试](https://blog.lacknb.cn/articles/2026/11/23/1795402356468.html)
  - 将退款规则写成 Skill，检查宿主能否发现，并为应触发和不应触发的请求各写一条验收。
- **AGT-18** [Loop Engineering 与反馈闭环](https://blog.lacknb.cn/articles/2026/11/23/1795402358094.html)
  - 让同一任务经历错误草稿、修复和再次验收，再验证连续无进展或用完预算时会退出。
- **AGT-12** [Agent Harness 源码阅读](https://blog.lacknb.cn/articles/2026/11/23/1795402359765.html)
  - 从固定版本的 Agent 源码追一条消息，标出会话、工具、事件和权限各在哪个边界交接。
- **AGT-13** [Workflow、Agent 与混合编排](https://blog.lacknb.cn/articles/2026/11/23/1795402361827.html)
  - 把退款处理拆成确定性计算、模型决策和人工审批，比较顺序、路由与并行的适用阶段。
- **AGT-14** [多 Agent 协作的几种模式](https://blog.lacknb.cn/articles/2026/11/23/1795402363518.html)
  - 给生成者和评审者划分上下文和写权限，拿两份互相冲突的统计结果设计合并规则。
- **AGT-22** [A2A 协议与跨 Agent 委派](https://blog.lacknb.cn/articles/2026/11/23/1795402365108.html)
  - 从 Agent Card 找到服务能力，发出一条 A2A 委派，并用任务和产物 ID 明确责任归属。
- **AGT-15** [Agent 的长任务与人机协作](https://blog.lacknb.cn/articles/2026/11/23/1795402366758.html)
  - 在等待审批时关闭运行时，重开后恢复到正确任务状态，拒绝旧审批并确保只写一份回执。
- **AGT-24** [持久化工作流的事件重放与状态恢复](https://blog.lacknb.cn/articles/2026/11/23/1795402368648.html)
  - 重放一次 Temporal 工作流中断，数出哪些 Activity 复用历史、哪些实际重试、收据最终有几条。
- **AGT-23** [A2A 实战，两个 Agent 如何交接长任务](https://blog.lacknb.cn/articles/2026/11/23/1795402370736.html)
  - 跟踪只读校验的同步与流式状态，断线后凭任务 ID 查回结果，并验证取消和失败状态。
- **AGT-16** [Agent 运行轨迹与调试](https://blog.lacknb.cn/articles/2026/11/23/1795402373955.html)
  - 用任务、Span、调用和动作标识串起脱敏轨迹，说明哪一步失败以及该查的下一条记录。
- **AGT-25** [Agent 前端事件协议与生成式 UI](https://blog.lacknb.cn/articles/2026/11/23/1795402375556.html)
  - 用 API-05 的流式知识驱动报告卡片，断线重连后恢复状态，并检查确认绑定的是当前报告版本。

### 7. Agent 效果评测（9/9）
- 先修：Agent Loop、工具执行、LE
- 产物：从验证一次任务终态，进阶到故障归因和发布回归
- **AE-06** [第一篇，如何验证 Agent 真的完成了任务](https://blog.lacknb.cn/articles/2026/11/23/1795403176399.html)
  - 沿 LE-07 的退款样本读回报告文件和归档表，独立核对两条、均值 5、批准前零条和批准后唯一写入。
- **AE-01** [设计 Agent 任务集与环境夹具](https://blog.lacknb.cn/articles/2026/11/23/1795403178111.html)
  - 写一条含初始状态、允许工具和预期产物的用例，每次重置环境，分别用固定回包和真实工具执行。
- **AE-02** [结果评测、轨迹评测和随机性](https://blog.lacknb.cn/articles/2026/11/23/1795403179638.html)
  - 同看两段执行轨迹，区分真正的终态失败、冗余工具调用和越权，并给出独立的指标。
- **AE-03** [使用开源框架评测自己的 Agent](https://blog.lacknb.cn/articles/2026/11/23/1795403181369.html)
  - 用 Inspect AI 组织退款样本、Agent 和评分器，验证只说「已归档」的回复会被实际证据拦住。
- **AE-07** [Agent 失败如何归因，模型、工具还是 Harness](https://blog.lacknb.cn/articles/2026/11/23/1795403183016.html)
  - 对两个都写成 3 条的报告逐段核对，用一次只替换一层的回放定位 `tool_output` 或 `artifact`。
- **AE-04** [用开源 Benchmark 对照 Agent 能力](https://blog.lacknb.cn/articles/2026/11/23/1795403184588.html)
  - 选一个适合自己目标的 Agent 基准，说明它验收的是工具参数、业务终态、仓库补丁还是可用成品。
- **AE-09** [FrontierHarness 评测全解：固定模型横评 Agent Harness、Prompt Cache 与单次通过成本](https://blog.lacknb.cn/articles/2026/11/28/1795827869958.html)
  - 固定同一底座模型与隔离沙箱，横评不同 Agent Harness 的通过率、Prompt Cache 命中率与单次通过成本；复现模型–Harness 交互反转、高缓存却更贵的陷阱及反作弊审计。
- **AE-08** [从离线 Agent 回归到线上灰度](https://blog.lacknb.cn/articles/2026/11/23/1795403186137.html)
  - 为新能力、旧行为和权限各设一道回归门槛，再用灰度数据判断保持、放量或回退。
- **AE-05** [进阶实验，Agent 的多轮强化学习](https://blog.lacknb.cn/articles/2026/11/23/1795403187704.html)
  - 用同一多轮任务比较终态奖励和步骤奖励，找出有奖励却没完成的轨迹，并用未见任务复测。延伸到 TRN-08、EVO-01。可接着看 AGT-01。

### 8. RAG、Embedding、Rerank 与 LLM Wiki（13/13）
- 先修：上下文组装、LE 与 AE 的基本口径
- 产物：对同一批资料交付可溯源问答与知识页
- **KB-11** [复杂文档解析，OCR、版面、表格与引用](https://blog.lacknb.cn/articles/2026/11/23/1795409706230.html)
  - 从一份跨页扫描表还原阅读顺序、表头和单元格，沿页码验证答复引用的原始证据。可接着看 KB-05。
- **KB-01** [文档解析、切块与索引](https://blog.lacknb.cn/articles/2026/11/23/1795409707913.html)
  - 为退款条款做两种切块，检验单位和规则号是否保留在同一块；更新文档后核对引用版本。
- **KB-02** [Embedding 与向量相似度](https://blog.lacknb.cn/articles/2026/11/23/1795409709612.html)
  - 手算两条向量的余弦相似度，再用同义词、字段名和规则编号检查召回。
- **KB-03** [BM25、向量检索与混合召回](https://blog.lacknb.cn/articles/2026/11/23/1795409711351.html)
  - 对同一查询生成词法和向量排名，手算融合顺序，再检查范围外的文档是否在召回前已被筛掉。
- **KB-04** [Rerank 怎样改善证据排序](https://blog.lacknb.cn/articles/2026/11/23/1795409713044.html)
  - 对一组候选重排，用正确证据有没有进入候选，判断应该改初召回还是重排分数。
- **KB-05** [从候选证据到可引用的 RAG 回答](https://blog.lacknb.cn/articles/2026/11/23/1795409714758.html)
  - 把检索结果组装成可点验的证据包，验证每条引用支持原句，遇到材料不足或冲突能显式反馈。
- **KB-12** [RAG 评测入门，先标问题和依据](https://blog.lacknb.cn/articles/2026/11/23/1795409716558.html)
  - 标注单段、跨段和无依据三类题，手算召回从 1/2 到 2/2，判断失败来自材料不足还是回答阶段。
- **KB-08** [RAG 的端到端质量评测](https://blog.lacknb.cn/articles/2026/11/23/1795409718320.html)
  - 对同一份 30 天政策分别注入错误数字和错误引用，从答案一致、忠实度和引用三项判出问题所在。
- **KB-13** [用 BEIR 和 HotpotQA 练习检索与证据问答评测](https://blog.lacknb.cn/articles/2026/11/23/1795409720070.html)
  - 分别对 BEIR 检索和 HotpotQA 答案与支持句算分，再为自己的退款问答保留同样的证据指标。
- **KB-09** [多租户 RAG 的索引权限与删除传播](https://blog.lacknb.cn/articles/2026/11/23/1795409721751.html)
  - 用旧引用测试撤权时的缓存和索引，记录拒绝何时生效，以及重新授权后能打开哪个版本。
- **KB-10** [结构化数据 Agent，Text-to-SQL 与指标口径](https://blog.lacknb.cn/articles/2026/11/23/1795409723408.html)
  - 手算退款统计后用 SQLite 对照去重、空值和时间区间的 SQL，再独立检查只读权限与预算。
- **KB-06** [Agentic RAG 与多跳检索](https://blog.lacknb.cn/articles/2026/11/23/1795409725088.html)
  - 根据第一跳返回的部门名写第二次查询，并从调用次数检查证据齐全、重复搜索和预算耗尽三种停止方式。
- **KB-07** [LLM Wiki 的资料编译、查询与维护](https://blog.lacknb.cn/articles/2026/11/23/1795409726827.html)
  - 把原件编成带出处的 Wiki 页，撤回或更新原件后，核对旧数字、重复页和失效链接如何被处理。


## 三、安全、隔离与自进化（16/16）

### 9. LLM 与 Agent 安全护栏（6/6）
- 先修：模型、工具和检索的信任边界
- 产物：用 OWASP 清单做威胁建模和控制映射
- **SEC-01** [OWASP 是什么，Top 10 怎样用](https://blog.lacknb.cn/articles/2026/11/23/1795411161291.html)
  - 画一张应用资产与信任边界图，将每项风险连到负责的组件、控制和可查验的证据。
- **SEC-02** [OWASP Top 10 for LLM Applications](https://blog.lacknb.cn/articles/2026/11/23/1795411163039.html)
  - 将 LLM 风险映射到退款助手的输入、检索和输出，对选出的控制写出合成验收样本及预期结果。
- **SEC-03** [OWASP Top 10 for Agentic Applications](https://blog.lacknb.cn/articles/2026/11/23/1795411164811.html)
  - 沿 Agent 的工具、身份、记忆和委派链路检查风险，将同一故障对应到合适的控制负责层。
- **SEC-04** [LLM 应用的输入、输出和数据护栏](https://blog.lacknb.cn/articles/2026/11/23/1795411166416.html)
  - 用合成的合法和异常报告检查误拦、漏判和失败处理，确保输出结构和引用范围能分别验收。
- **SEC-05** [Agent 工具调用、审批与运行时护栏](https://blog.lacknb.cn/articles/2026/11/23/1795411168053.html)
  - 替换报告版本、租户或批准内容后重新检查执行结果，确认正确批准前没有写入，重投后只有一条。
- **SEC-06** [第三方模型、MCP Server、Skill 与 A2A 的信任链](https://blog.lacknb.cn/articles/2026/11/23/1795411169711.html)
  - 用一组第三方组件登记信息验证版本变更、委派收窄和撤权分别怎样影响执行决定。

### 10. Sandbox 分类、源码与实测（5/5）
- 先修：SEC、进程与容器基础
- 产物：按相同任务与威胁模型对照执行环境
- **SBX-01** [沙箱分类与威胁建模](https://blog.lacknb.cn/articles/2026/11/23/1795416672508.html)
  - 为退款任务列出文件、网络、进程与清理需求，选一种隔离方式，并把边界转成行为验收。
- **SBX-02** [本地沙箱如何限制 Coding Agent](https://blog.lacknb.cn/articles/2026/11/23/1795416674280.html)
  - 在受控夹具中对照普通进程、宿主策略和容器，检查工作区、子进程、出站和清理能否按预期受限。
- **SBX-03** [远程沙箱的生命周期与执行接口](https://blog.lacknb.cn/articles/2026/11/23/1795416676067.html)
  - 跟一条任务走完创建、就绪、执行、取产物和销毁，并检查销毁后访问与重复销毁的结果。
- **SBX-04** [E2B、CubeSandbox、OpenSandbox 对比与源码路径](https://blog.lacknb.cn/articles/2026/11/23/1795416677850.html)
  - 沿三套沙箱实现寻找创建返回、就绪确认和产物拉取，标出 SDK、控制面和执行面各自的证据。
- **SBX-05** [沙箱选型、启动基准与事件响应](https://blog.lacknb.cn/articles/2026/11/23/1795416679623.html)
  - 从样本复算首输出与删除确认的延迟，比较复用的收益，再核对旧工作区或任务状态是否被清掉。

### 11. Agent 自进化与递归自我改进（RSI）（5/5）
- 先修：记忆与 Skills、离线评测、基础遥测与发布护栏
- 产物：用运行反馈验证并发布可回滚的改进
- **EVO-01** [Agent 自进化有哪些模式](https://blog.lacknb.cn/articles/2026/11/23/1795417257580.html)
  - 用同一次用户纠正区分当场修复、长期记忆、流程知识和路由更新，再检查换会话是否仍有效。
- **EVO-02** [从任务反馈到可审计的经验发布](https://blog.lacknb.cn/articles/2026/11/23/1795417259258.html)
  - 为一条带来源的经验设计发布门槛，分别核对留出收益、适用范围、权限和可回退版本。
- **EVO-03** [模型如何参与训练下一版模型](https://blog.lacknb.cn/articles/2026/11/23/1795417260943.html)
  - 拆出 STaR 和自奖励中的样本生成、验证和更新步骤，用独立留出题识别循环是否真的带来收益。延伸到 TRN-08、TRN-12、LE-03。
- **EVO-04** [Google DeepMind 实例，SIMA 2 与 AlphaEvolve](https://blog.lacknb.cn/articles/2026/11/23/1795417262667.html)
  - 对照 SIMA 2 和 AlphaEvolve，说清哪个对象被更新、评分信号从哪里来，再据此设计自己的小型候选筛选实验。
- **EVO-05** [给 Agent 或普通程序加一个反馈改进 Agent](https://blog.lacknb.cn/articles/2026/11/23/1795417264333.html)
  - 把运行失败做成脱敏回放，按质量、权限和资源门槛评审候选，再查灰度回滚与在途任务状态。


## 四、开源案例与生产落地（13/13）

### 12. OpenClaw 与 Hermes Agent 源码案例（3/3）
- 先修：Harness、会话、Skills、隔离与自进化
- 产物：对照同一任务在两个框架里的运行与经验沉淀
- **CASE-01** [OpenClaw 设计思想与运行链路](https://blog.lacknb.cn/articles/2026/11/23/1795418062770.html)
  - 沿选定的 OpenClaw 版本跟踪一条退款消息，区分会话路由、工作区资料装载和真实工具结果。
- **CASE-02** [Hermes Agent 的设计思想与自改进链路](https://blog.lacknb.cn/articles/2026/11/23/1795418064577.html)
  - 在 Hermes 工作区跟一条经验从候选到批准、加载和撤回，再对照下一会话的输出。
- **CASE-03** [同一个任务对照 OpenClaw 与 Hermes Agent](https://blog.lacknb.cn/articles/2026/11/23/1795418066265.html)
  - 用同一份 CSV 对照 OpenClaw 和 Hermes，区分同会话历史记住纠正与跨会话资料已发布。

### 13. Vibe Coding 与工程化交付（3/3）
- 先修：Agent Harness、评测、护栏与 Sandbox
- 产物：对比原型与可维护版本的验收证据
- **DEV-01** [Vibe Coding 的工作方式与原型实验](https://blog.lacknb.cn/articles/2026/11/23/1795418815589.html)
  - 用两轮原型演示退款报告，拿正常、全缺失和归档状态三个任务记录改动效果。
- **DEV-02** [从 Vibe Coding 原型走向 Agentic Engineering](https://blog.lacknb.cn/articles/2026/11/23/1795418817314.html)
  - 把原型拆成可测试规则与页面，验证计算、输入变化、取消与审批，保留能复跑的交付证据。
- **DEV-03** [Computer Use 与浏览器 Agent 的工程实践](https://blog.lacknb.cn/articles/2026/11/23/1795418819068.html)
  - 对照截图坐标和可访问语义定位，在布局变化后验证报告结果，并在归档前交还确认。

### 14. 生产化专题（7/7）
- 先修：服务、评测、隔离与任务执行链路
- 产物：用运行指标说明上线、限流、隔离和拆分决策
- **OPS-01** [LLM Gateway 的限流、路由、缓存与降级](https://blog.lacknb.cn/articles/2026/11/23/1795419423601.html)
  - 按请求数、Token、并发与预算设计限额，用流式首段前后的故障判断能否切换供应商。
- **OPS-02** [Agent 安全控制如何落到生产环境](https://blog.lacknb.cn/articles/2026/11/23/1795419425419.html)
  - 在预发布分别变更归档身份、审批对象和目标，观察代理是否记录了对应拒绝理由。
- **OPS-03** [Sandbox 冷启动和资源回收](https://blog.lacknb.cn/articles/2026/11/23/1795419427113.html)
  - 从一次运行分出排队、镜像、运行时和首命令耗时，按相同口径比较冷启动、预热和回收。
- **OPS-04** [Agent 服务拆分与可靠迁移](https://blog.lacknb.cn/articles/2026/11/23/1795419428781.html)
  - 沿任务、发件箱、队列、收据四张状态表重放事务回滚和重复交付，解释为何只保留一张收据。
- **OPS-05** [端到端可观测与持续发布](https://blog.lacknb.cn/articles/2026/11/23/1795419430432.html)
  - 从一个故障样本写出离线回归与灰度门槛，用正确率、P95 和重复写入决定回切，并检查在途任务。
- **OPS-07** [Prompt、工具契约和在途任务的版本兼容](https://blog.lacknb.cn/articles/2026/11/23/1795419432161.html)
  - 拿一条旧任务和一份旧版工具回包核对恢复，确认金额的元分换算正确、未知版本被挂起、重试不重复写入。
- **OPS-06** [用 OpenTelemetry 定位一条 Agent 慢请求](https://blog.lacknb.cn/articles/2026/11/23/1795419433902.html)
  - 用 Spring Boot、Collector 和 Jaeger 核对四层 Span，区分业务错数、消息断链和导出失败。


## 横向入口：题库 · 速查 · 路线图（3 份总览）

- **BANK** [Agent 与 Sandbox 面试题库：运行时、模型网关与系统设计](https://blog.lacknb.cn/articles/2026/11/19/1795049523245.html)
  - 覆盖 AG / SB / LLM / ARCH / MVP / CTX 六大面试专题与追问参考答案，用 CSV 上传→隔离分析→归档场景贯穿。
- **NOTES** Agent 工程笔记（本地：`agent-engineering-notes.md`）
  - 把多智能体编排、Agent 记忆、上下文工程、本地执行安全、模型推理与缓存浓缩成 5 层速查。
- **MAP** [LLM 与 Agent 学习路线图：从模型原理到工程落地](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
  - 串联 14 阶段 139 篇专题的置顶总览文章，含常用模型数据集与 Agent 公开基准速查表。
