# Java OpenTelemetry 退款任务示例

用一个本地队列和独立的工作线程串起入队、消费、模型、统计工具、沙箱与产物。真实运行 OpenTelemetry Java SDK，Span 用内存导出器收集并打印。CSV 是固定数据，等待由程序注入，运行不需要消息中间件、Collector 或模型服务。

## 运行

在仓库根目录执行，要求 JDK 17 或更新版本及 Maven 3.6.3 或更新版本。首次构建需要下载 Maven 依赖。

```bash
mvn -q -f experiments/production-refund/java-otel/pom.xml compile exec:java
mvn -q -f experiments/production-refund/java-otel/pom.xml compile exec:java -Dexec.args=--drop-traceparent
mvn -q -f experiments/production-refund/java-otel/pom.xml compile exec:java -Dexec.args=--fail-tool
```

- 默认运行得到 `report rows=2 mean_cents=500`。`refund.enqueue` 和 `refund.process` 共享 Trace ID。前者的 Span ID 等于后者的 Parent Span ID。排队等待约 700 毫秒，不在消费 Span 自身的持续时间里。
- `--timestamps` 额外打印 Span 和事件的 Unix 纪元纳秒时间戳。可用它们还原[入门文章的瀑布图](../../../opentelemetry-tracing-primer.md)。
- `--drop-traceparent` 模拟队列没把追踪字段交给消费者。报告照常生成，`refund.process` 另起一条 Trace，父 Span ID 为全零。
- `--fail-tool` 模拟统计工具出错。命令、工具、消费和演示请求的 Span 标为 `ERROR`，没有生成产物，也不会输出报告。这个演练已处理故障，因此 Java 进程可以正常退出。
- ID 会随运行改变，耗时会因本机调度浮动。内存导出器适合演练，生产接入方式和 OTLP 配置见 [OPS-06 源稿](../../../ops-06-opentelemetry-trace.md)。

主要入口在 [RefundTraceDemo.java](src/main/java/example/refund/RefundTraceDemo.java)。依次读 `main` 的 SDK 装配、`run` 的入队、`consume` 的上下文恢复，以及 `inSpan` 如何在成功或失败时结束 Span。
