# Spring Boot 退款追踪演练

向接口发一个 POST 请求，服务把三行合成金额交给工作线程。业务结果是有效 2 笔、均值 500 分。OpenTelemetry Java Agent 自动记录 HTTP 入口，应用手动补入队、消费和统计 Span。Collector 收集后再发给 Jaeger，Jaeger 用内存保存并支持查询及看瀑布图。

```text
POST 127.0.0.1:18087/refunds/report
    └─ Java Agent 记录 HTTP Span，应用记录 refund.enqueue / refund.process / refund.stats
        └─ OTLP/HTTP → Collector 127.0.0.1:14318/v1/traces
            ├─ debug → Collector 标准输出
            └─ OTLP/HTTP → Jaeger 127.0.0.1:14319/v1/traces
                └─ Jaeger UI 和查询 API → 127.0.0.1:17686
```

## 准备

演练在 macOS arm64、JDK 21 上验证。项目面向 Java 17。使用 Maven 3.6.3 或更新版本、Spring Boot 3.5.11、OpenTelemetry Java Agent 2.23.0、Collector 0.143.0、Jaeger 2.23.0。前三个进程均只监听本机地址。样例无需数据库、消息中间件或模型服务。

把对应架构的 Collector 和 Jaeger 官方发行包解压到临时目录，把 `opentelemetry-javaagent-2.23.0.jar` 放在同一目录。下载包可以对照各自发行页的校验和。以下命令都从仓库根目录运行。 [Java Agent 发行版](https://github.com/open-telemetry/opentelemetry-java-instrumentation/releases/tag/v2.23.0) · [Collector 发行版](https://github.com/open-telemetry/opentelemetry-collector-releases/releases/tag/v0.143.0) · [Jaeger 发行版](https://github.com/jaegertracing/jaeger/releases/tag/v2.23.0)

```bash
mvn -q -f experiments/production-refund/spring-otel/pom.xml package
```

在两个终端分别启动 Jaeger 和 Collector，将二进制路径替换为解压后的实际路径。Collector 配置中启用了 `debug` 导出，便于与后端查询结果对照。

```bash
/path/to/jaeger --config experiments/production-refund/spring-otel/jaeger.yaml
```

```bash
/path/to/otelcol --config experiments/production-refund/spring-otel/collector.yaml
```

本例固定的 Collector 0.143.0 用 `otlphttp/jaeger` 标识转发器。若参考其他版本的示例替换了配置，可以先用当前二进制的 `validate --config` 检查组件名称和配置。

在第三个终端启动应用。通用 OTLP 端点配置为基地址，Java Agent 会为 Trace 请求构建 `/v1/traces` 路径。批量导出的间隔在这里设为 200 毫秒，方便过一会儿查到结果。

```bash
OTEL_SERVICE_NAME=refund-spring-demo \
OTEL_PROPAGATORS=tracecontext \
OTEL_TRACES_EXPORTER=otlp \
OTEL_TRACES_SAMPLER=always_on \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:14318 \
OTEL_METRICS_EXPORTER=none \
OTEL_LOGS_EXPORTER=none \
OTEL_BSP_SCHEDULE_DELAY=200 \
java -javaagent:/path/to/opentelemetry-javaagent-2.23.0.jar \
    -jar experiments/production-refund/spring-otel/target/spring-otel-refund-1.0.0.jar \
    --server.address=127.0.0.1 --server.port=18087
```

实验结束后在各终端按 Ctrl+C 停止进程。Jaeger 使用内存，停掉后本次追踪记录随之清空。

## 请求与核对

```bash
curl -fsS -X POST http://127.0.0.1:18087/refunds/report
curl -fsS -X POST 'http://127.0.0.1:18087/refunds/report?dropTraceparent=true'
```

两个请求都应返回 `rows=2`、`meanCents=500`。正常时，返回的 `traceId` 等于 `workerTraceId`。去掉消息头后，两者不同。等一小会儿，在 Jaeger UI 打开 `http://127.0.0.1:17686/trace/TRACE_ID`，将 `TRACE_ID` 替换成响应中的实际值。也可直接查询同一个 ID。

```bash
trace_id=REPLACE_WITH_TRACE_ID
curl -fsS "http://127.0.0.1:17686/api/v3/traces/$trace_id"
```

实测正常请求可查询到 4 个 Span，父子顺序是 `POST /refunds/report` → `refund.enqueue` → `refund.process` → `refund.stats`。断链后每个 ID 可查到 2 个 Span，消费端的父 ID 为空。 [正常链路的 Jaeger 瀑布图](../../../assets/spring-otel-jaeger-waterfall.png)

测试导出故障时，保持 Collector 和 Jaeger 运行，重启同一个应用。给上面的启动命令增加 `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:14318/not-a-traces-endpoint`。这个针对 Trace 的端点会覆盖通用端点，并使用配置的完整路径。重新发一次不带 `dropTraceparent` 的 POST 请求，并用它返回的新 ID 查询。请求仍返回正确报告，应用导出器日志会报告 HTTP 404，Jaeger 查不到这次新的 Trace ID。恢复原启动命令后再发一个新请求，正常的查询路径应恢复。

代码从 [RefundController.java](src/main/java/example/refund/RefundController.java) 的 `createReport` 开始读。HTTP 请求线程注入 `traceparent`，消费端从消息头恢复父上下文，`refund.stats` 记录实际统计。相关背景见 [OpenTelemetry 入门](../../../articles/14-production/opentelemetry-tracing-primer.md) · [OPS-06 的 Spring Boot 接入实战](../../../articles/14-production/ops-06-opentelemetry-trace.md#spring-boot-接入实战)。
