---
title: "生产化 OPS-06｜用 OpenTelemetry 找到一条 Agent 慢请求到底慢在哪"
description: "把一条退款报告请求的总耗时画成可手算的时间线，依次定位排队、检索、模型、工具和沙箱，区分首字与整段耗时。按固定的 OpenTelemetry 语义约定标记模型和 Token 用量，说明跨队列关联、采样和脱敏的落地方法。"
---

![文章封面](assets/ops-06-opentelemetry-trace-cover.jpg)

> **生成工具**：Codex\
> **生成模型**：gpt-6.2-sol

## 先用大白话理解

用户点了“生成报告”，页面等了 1,850 毫秒。只看一个总时间，好比你知道快递晚了，却不知道是在仓库排队、在路上，还是卡在门卫登记。OpenTelemetry（简称 OTel）是一套生成和传递软件运行观测数据的工具与规范，可以给每段工作记开始、结束和上下游关系。

样例是同一份 CSV（逗号分隔值文件），退款是 8.00 元、2.00 元和缺失值，统计产物必须给出 2 条有效记录、均值 5.00 元。用户随后单独确认归档。这里测到报告可见为止，把人的确认时间留在另一段交互。下面每个时刻都是虚拟毫秒，从同一个请求进入的时刻计起。

## 给每段工作一个可关联的编号

Trace（一次调用链的追踪）用 Trace ID 关联一组工作。Span（工作片段）是某一步具体的计时记录，有自己的 Span ID，可引用父 Span ID。请求进来建根 Span。向队列写消息时，在消息元数据注入当前追踪上下文，工作进程从消息恢复。W3C Trace Context 是万维网联盟定义的跨进程追踪上下文格式，常见字段 `traceparent` 携带追踪 ID、父片段 ID 和追踪标记。

同步一对一调用通常能自然形成父子。队列重投或一批消息共同触发一个批处理时，消费端可使用 Span Link（关联到其他片段）保留多个来源，任务号继续用作业务对账。传播上下文解决相关性，授权仍用真实服务身份。进程内耗时用单调时钟记录，跨机器对照校准时间，也可用发送和接收的边界拆出一段更容易核实的等待。 [OpenTelemetry 上下文传递](https://opentelemetry.io/docs/concepts/context-propagation/) · [Trace 定义](https://opentelemetry.io/docs/concepts/signals/traces/)

## 读一遍时间线

| 片段 | 从何时到何时 | 耗时 | 上级 |
| --- | --- | ---: | --- |
| 请求 | 0～1850 | 1,850 毫秒 | 根 |
| 排队 | 15～715 | 700 毫秒 | 请求 |
| 检索 | 715～785 | 70 毫秒 | 请求 |
| 模型 | 785～1185 | 400 毫秒 | 请求 |
| 工具 | 1185～1765 | 580 毫秒 | 请求 |
| 沙箱租约 | 1195～1645 | 450 毫秒 | 工具 |
| 沙箱命令 | 1645～1745 | 100 毫秒 | 工具 |
| 发布报告产物 | 1765～1850 | 85 毫秒 | 请求 |

接入处理的 15 毫秒加上串行主路径的 `700 + 70 + 400 + 580 + 85`，得到 1,850 毫秒。沙箱两段已经包含在工具中，再加会重复算。两条并行路径要找挡住用户交付的那条，并观察重叠区间。此例先查 700 毫秒排队，再钻到工具内部看 450 毫秒租约。命令只占 100 毫秒，先优化统计脚本能节省的时间有限。

模型从 785 开始，模拟在 925 给出第一段用户可见文本。从模型 Span 开始计首段是 140 毫秒，从请求进入计则是 925 毫秒。两者都可以叫首段延迟，仪表盘必须标注计时起点。`Token` 是模型的文本计量单位，流式网络分块与 Token 边界可以不同，工具参数或空事件也可能先到。此例明确以第一段可见文本作为终点。

## 选一版字段，先记少量证据

本例固定 OpenTelemetry Semantic Conventions（语义约定）v1.39.0 来阅读字段。这一版本的生成式 AI 字段仍处于开发状态，升级埋点时要一起核对仪表盘。示例模型响应报告输入 620、输出 80 Token，分别放到 `gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`。保留 `gen_ai.request.model` 和调用类型 `gen_ai.operation.name`，输出失败时附短错误类型，供应商未返回用量就留空并记明来源。步骤顺序可以用应用自己的序号，如 `app.agent.step`。定义过的字段和应用自定义字段沿不同命名空间记录。统计工具和沙箱的自定义字段由应用统一约定。 [v1.39.0 的 GenAI Span 约定](https://github.com/open-telemetry/semantic-conventions/blob/v1.39.0/docs/gen-ai/gen-ai-spans.md)

正常 span 只留部署版、耗时、结果类别、输入长度或数量，原始提示词、工具参数、报告与凭据默认不采集。排查时优先用合成样本重放。确需记录内容则限制环境和可见范围、做脱敏并规定保留时间。统计指标持续收全量口径，详细 Trace 可以采样。头部采样在开始时决定保存哪些链路，尾部采样等到看到错误和耗时后再选，后者更容易保留慢例与错误，也需要聚拢同一 Trace 的片段、容量和等待窗口。 [OpenTelemetry 采样](https://opentelemetry.io/docs/concepts/sampling/)

## 动手做一个 Java 项目

先跑一个包含真实 OpenTelemetry SDK（软件开发工具包）的最小 Maven 项目。主线程把一份退款 CSV 放进队列，工作线程取出，依次走检索、模拟模型调用、统计工具和产物步骤。金额仍是 8.00 元、2.00 元和 `NA`，最后应输出两条有效记录、均值 500 分。排队和外部服务的延迟用 `Thread.sleep` 注入，统计实际用 `BigDecimal` 计算。项目不需要启动模型、消息中间件或观测后端。 [Java 演练项目](experiments/production-refund/java-otel/README.md)

```bash
mvn -q -f experiments/production-refund/java-otel/pom.xml compile exec:java
```

### 第一步，让 Span 能被看到

在 Maven 中把 `opentelemetry-sdk` 和 `opentelemetry-sdk-testing` 固定到同一版本 `1.68.0`。这与上文固定的语义约定 v1.39.0 是两条独立的版本线。前者是 Java 库，后者决定观测字段如何命名。示例的 `Tracer` 负责创建 Span，`SimpleSpanProcessor` 在 Span 结束时移交给导出器，`InMemorySpanExporter` 把结果暂存在进程内，主程序读出并打印。 [完整 Maven 配置](experiments/production-refund/java-otel/pom.xml) · [OpenTelemetry Java SDK](https://opentelemetry.io/docs/languages/java/sdk/)

```java
InMemorySpanExporter exporter = InMemorySpanExporter.create();
SdkTracerProvider provider = SdkTracerProvider.builder()
        .addSpanProcessor(SimpleSpanProcessor.create(exporter))
        .build();
try (OpenTelemetrySdk sdk = OpenTelemetrySdk.builder()
        .setTracerProvider(provider)
        .setPropagators(ContextPropagators.create(W3CTraceContextPropagator.getInstance()))
        .build()) {
    Tracer tracer = sdk.getTracer("refund-demo");
    TextMapPropagator propagator = sdk.getPropagators().getTextMapPropagator();
    // 把 tracer 和 propagator 交给接入端和工作线程。
}
```

### 第二步，把追踪编号写进队列消息

`startSpan()` 创建一个工作片段，`makeCurrent()` 把它放进当前线程的上下文。入队时必须把**生产者 Span** 设为当前，再调用 `inject`，否则发出去的可能是更外层请求的编号。SDK 把当前 Trace ID 和 Span ID 编码进 `traceparent`。队列只需保存普通字符串。消费者在另一条线程调用 `extract`，读出远端父上下文。

下面省去调度和错误处理，演示接入的两端。完整项目在统一的 `inSpan` 方法里负责恢复 Scope 和结束 Span。 [完整 Java 源码](experiments/production-refund/java-otel/src/main/java/example/refund/RefundTraceDemo.java)

```java
// 接入线程，当前上下文里已有 refund.request。
Span producer = tracer.spanBuilder("refund.enqueue")
        .setSpanKind(SpanKind.PRODUCER)
        .startSpan();
Map<String, String> headers = new HashMap<>();
try (Scope scope = producer.makeCurrent()) {
    propagator.inject(Context.current(), headers, Map::put);
    queue.add(new Job(Map.copyOf(headers), System.nanoTime(), csv));
} finally {
    producer.end();
}

// 工作线程，从消息恢复父上下文。
Context parent = propagator.extract(Context.root(), job.headers(), HEADER_GETTER);
long waitMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - job.enqueuedNanos());
Span consumer = tracer.spanBuilder("refund.process")
        .setSpanKind(SpanKind.CONSUMER)
        .setParent(parent)
        .setAttribute("app.queue.wait_ms", waitMs)
        .startSpan();
try (Scope scope = consumer.makeCurrent()) {
    // 在这个 Scope 中创建检索、模型、统计工具和产物的子 Span。
} finally {
    consumer.end();
}
```

`Scope.close()` 恢复当前线程原来的上下文。线程池会复用线程，遗漏这一步可能把下一个任务挂到上一个任务下面。`span.end()` 记下结束时刻，少了它，导出器可能一直收不到这个 Span。完整示例捕获异常后设为 `ERROR`，记录异常类型并继续抛出，避免把含用户输入的异常详情送到观测系统。

`app.queue.wait_ms` 从任务放入队列前算到消费者收到任务，包含本例注入的约 700 毫秒。消费者 Span 从取到任务才开始，不能直接拿它的持续时间当排队时间。这里两条线程共享同一个 JVM，能比较 `System.nanoTime()`。换成跨机器的 Kafka 或 RabbitMQ 后，两台机器的 `nanoTime()` 没有共同原点。用 broker（消息代理）的入队时间、消费端接收时间和同步时钟做估算，按平台定义区分首次入队、重新投递与每次处理等待。一个消息对应一个消费者时可按示例选父子关系。批量处理或追踪已自带上下文的场景，要按消息规范考虑 Span Link。 [消息 Span 约定](https://opentelemetry.io/docs/specs/semconv/messaging/messaging-spans/)

### 第三步，先用父子关系找链路，再按耗时查瓶颈

一次运行得到 `report rows=2 mean_cents=500`。下面把控制台输出整理成表，下表中的 ID 只展示前 8 位，自动化校验用完整 ID。本次所有行的 Trace ID 都以 `13b5d0ae` 开头，序号和毫秒数每次运行会略有变化。

| Span | Span ID 前 8 位 | Parent Span ID 前 8 位 | 耗时（毫秒） |
| --- | --- | --- | ---: |
| `refund.request` | `3b267c50` | `00000000` | 1832 |
| `refund.enqueue` | `0b2974ce` | `3b267c50` | 1 |
| `refund.process` | `ba7d4954` | `0b2974ce` | 1122 |
| `refund.retrieval` | `dcc01907` | `ba7d4954` | 71 |
| `chat demo-model` | `448ed01b` | `ba7d4954` | 404 |
| `refund.stats` | `e4267c28` | `ba7d4954` | 559 |
| `sandbox.lease` | `8b6405fb` | `e4267c28` | 450 |
| `sandbox.command` | `7f6e3a69` | `e4267c28` | 108 |
| `refund.artifact` | `bdf43e90` | `ba7d4954` | 86 |

另外，消费 Span 打印了 `queue_wait_ms=704`，模型 Span 打印了 `input_tokens=620 output_tokens=80` 和首段事件 `offset_ms=142`。Token 数是模拟供应商回包的值。

读输出时，先核对 `refund.process` 的 `trace` 和 `refund.enqueue` 相同，再核对其 `parent` 等于入队 Span 的 `span`。诊断时读 `queue_wait_ms`，它表示开始工作前已等了多久。然后打开 `refund.stats`，对照其子 Span `sandbox.lease` 和 `sandbox.command`。前者约 450 毫秒，后者约 100 毫秒。这个样本首先该查队列为什么等了 700 毫秒，其次查租约分配。模型 Span 的首段事件偏移约 140 毫秒，整个模型 Span 仍约 400 毫秒。这两种耗时都来自真实采集的 Span，所测等待由程序注入。

再故意制造两种故障，观察链路会怎样变化。

```bash
mvn -q -f experiments/production-refund/java-otel/pom.xml compile exec:java -Dexec.args=--drop-traceparent
mvn -q -f experiments/production-refund/java-otel/pom.xml compile exec:java -Dexec.args=--fail-tool
```

- 丢掉 `traceparent` 后，统计仍返回 500 分，但 `refund.process` 的 Trace ID 改变，父 Span ID 变为全零。线上遇到两段 Trace，沿消息发送和接收的边界核对是否保留了传播字段。
- 工具失败时，`sandbox.command`、`refund.stats`、`refund.process` 和演练的请求 Span 都能看到 `ERROR`，并且没有 `refund.artifact`。这代表本次请求确实失败。真实异步接口如果早已返回 `202 Accepted`，入口 HTTP Span 可以是成功，最终失败要到消费 Span 与任务记录里找。

### 第四步，搬进已有的 Spring Boot 项目

已有 Spring Boot 应用可以让 OpenTelemetry Java Agent 承担框架埋点和 SDK 初始化。应用依赖 `io.opentelemetry:opentelemetry-api`，在业务类用 `GlobalOpenTelemetry.getOrNoop().getTracer("refund-service")` 取 Tracer，只为沙箱租约、报表产物等应用步骤增加手工 Span。若 Agent 已经为使用的消息客户端传播上下文，先观察真实入队和消费 Span，再决定是否补自己的传播逻辑。进程内示例的 SDK 装配只用于本地演练。 [Java Agent 手工扩展](https://opentelemetry.io/docs/zero-code/java/agent/api/)

假设团队已有接收 OTLP 的 Collector。OTLP 是 OpenTelemetry Protocol，也就是应用把观测数据发给后端的协议。Collector 负责接收、处理并转发。对于配置了 HTTP 接收端口 4318 的 Collector，可用下面的环境变量启动应用。示例地址和 JAR 路径需要替换成部署环境的实际值。

```bash
export OTEL_SERVICE_NAME=refund-service
export OTEL_TRACES_EXPORTER=otlp
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otel.example.com:4318
java -javaagent:/opt/otel/opentelemetry-javaagent.jar -jar refund-service.jar
```

这里配置的是 OTLP 通用端点，HTTP Trace 请求会走 `/v1/traces`。如果改用 `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`，填写包含该路径的完整地址。启动后可在预发布提交一条合成退款请求，按 `service.name` 找入口，再顺着 Trace 打开模型和工具。入口只有 202 时，另外核对任务创建到报告可用的指标，因为短 HTTP Span 并不覆盖后台等待。上述 Collector 配置给出接入方式，本地演练验证的是 Java SDK 生成和关联 Span 的路径。 [SDK 配置与 OTLP 端点](https://opentelemetry.io/docs/languages/java/configuration/)


## 从一条慢请求回到生产排障

[Python 时间线演练](experiments/production-refund/README.md)保留了开头可手算的固定样本，可以对照 Java 版实际采集的 Span。遇到同类告警，先在同版本、同流量档看排队的分布和积压，再核对沙箱池命中率。如果有空闲执行位但任务仍等 700 毫秒，检查调度与租约分配；如果执行位全占满，结合工作池供给和请求速率看容量。修复后按 [OPS-05 的灰度口径](ops-05-progressive-delivery.md)复测正确率与端到端延迟，并保留两个版本对应的样本。 [生产化专题目录](llm-agent-learning-roadmap.md)
