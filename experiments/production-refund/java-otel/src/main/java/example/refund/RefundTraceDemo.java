package example.refund;

import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.context.propagation.TextMapGetter;
import io.opentelemetry.context.propagation.TextMapPropagator;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * 为一个退款任务生成真实的 OpenTelemetry Span，使用进程内队列和模拟耗时。
 */
public final class RefundTraceDemo {
    private static final String CSV = "refund_amount\n8.00\n2.00\nNA\n";
    private static final AttributeKey<Long> WAIT_MS = AttributeKey.longKey("app.queue.wait_ms");
    private static final AttributeKey<Long> INPUT_TOKENS = AttributeKey.longKey("gen_ai.usage.input_tokens");
    private static final AttributeKey<Long> OUTPUT_TOKENS = AttributeKey.longKey("gen_ai.usage.output_tokens");
    private static final AttributeKey<String> ERROR_TYPE = AttributeKey.stringKey("error.type");
    private static final TextMapGetter<Map<String, String>> HEADER_GETTER = new TextMapGetter<>() {
        @Override
        public Iterable<String> keys(Map<String, String> headers) {
            return headers.keySet();
        }

        @Override
        public String get(Map<String, String> headers, String key) {
            return headers.get(key);
        }
    };

    private RefundTraceDemo() {
    }

    public static void main(String[] args) throws Exception {
        boolean dropTrace = Arrays.asList(args).contains("--drop-traceparent");
        boolean showTimestamps = Arrays.asList(args).contains("--timestamps");
        Fault fault = faultFrom(args);
        InMemorySpanExporter exporter = InMemorySpanExporter.create();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .setResource(Resource.getDefault().merge(Resource.create(
                        Attributes.of(AttributeKey.stringKey("service.name"), "refund-demo"))))
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        try (OpenTelemetrySdk sdk = OpenTelemetrySdk.builder()
                .setTracerProvider(provider)
                .setPropagators(ContextPropagators.create(W3CTraceContextPropagator.getInstance()))
                .build()) {
            Tracer tracer = sdk.getTracer("refund-demo");
            TextMapPropagator propagator = sdk.getPropagators().getTextMapPropagator();
            try {
                Report report = run(tracer, propagator, dropTrace, fault);
                System.out.printf("report rows=%d mean_cents=%d%n", report.rows(), report.meanCents());
            } catch (ExecutionException error) {
                if (fault != Fault.TOOL_ERROR || !(error.getCause() instanceof IllegalStateException)) {
                    throw error;
                }
                System.out.println("report unavailable because the tool failed");
            }
            exporter.getFinishedSpanItems().stream()
                    .sorted(Comparator.comparingLong(SpanData::getStartEpochNanos))
                    .forEach(span -> printSpan(span, showTimestamps));
        }
    }

    private static Fault faultFrom(String[] args) {
        Fault selected = Fault.NONE;
        for (Fault fault : Fault.values()) {
            if (fault != Fault.NONE && Arrays.asList(args).contains(fault.option)) {
                if (selected != Fault.NONE) {
                    throw new IllegalArgumentException("choose only one tool or artifact failure mode");
                }
                selected = fault;
            }
        }
        return selected;
    }

    private static Report run(Tracer tracer, TextMapPropagator propagator, boolean dropTrace, Fault fault)
            throws Exception {
        BlockingQueue<Job> queue = new ArrayBlockingQueue<>(1);
        ExecutorService worker = Executors.newSingleThreadExecutor();
        try {
            return inSpan(tracer.spanBuilder("refund.request").setNoParent().startSpan(), request -> {
                inSpan(tracer.spanBuilder("refund.enqueue").setSpanKind(SpanKind.PRODUCER).startSpan(),
                        producer -> {
                            Map<String, String> headers = new HashMap<>();
                            propagator.inject(Context.current(), headers, Map::put);
                            System.out.println("producer traceparent=" + headers.get("traceparent"));
                            if (dropTrace) {
                                headers.remove("traceparent");
                            }
                            queue.add(new Job(Map.copyOf(headers), System.nanoTime(), CSV));
                            return null;
                        });
                // 故意让任务等待，使排队耗时能出现在实际采集结果中。
                Thread.sleep(700);
                return worker.submit(() -> consume(tracer, propagator, queue.take(), fault)).get();
            });
        } finally {
            worker.shutdownNow();
        }
    }

    private static Report consume(Tracer tracer, TextMapPropagator propagator, Job job, Fault fault)
            throws Exception {
        long waitMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - job.enqueuedNanos());
        Context parent = propagator.extract(Context.root(), job.headers(), HEADER_GETTER);
        Span processing = tracer.spanBuilder("refund.process")
                .setSpanKind(SpanKind.CONSUMER)
                .setParent(parent)
                .setAttribute(WAIT_MS, waitMs)
                .startSpan();
        return inSpan(processing, span -> {
            delay(tracer, "refund.retrieval", 70);
            inSpan(tracer.spanBuilder("chat demo-model").setSpanKind(SpanKind.CLIENT).startSpan(), model -> {
                model.setAttribute("gen_ai.operation.name", "chat");
                model.setAttribute("gen_ai.provider.name", "example");
                model.setAttribute("gen_ai.request.model", "demo-model");
                Thread.sleep(140);
                model.addEvent("app.model.first_visible_delta");
                Thread.sleep(260);
                model.setAttribute(INPUT_TOKENS, 620L);
                model.setAttribute(OUTPUT_TOKENS, 80L);
                return null;
            });
            Report report = inSpan(tracer.spanBuilder("refund.stats").startSpan(), tool -> {
                delay(tracer, "sandbox.lease", 450);
                return inSpan(tracer.spanBuilder("sandbox.command").startSpan(), command -> {
                    Thread.sleep(100);
                    if (fault == Fault.TOOL_ERROR) {
                        throw new IllegalStateException("simulated statistics failure");
                    }
                    return summarize(job.csv(), fault == Fault.TOOL_VALUE);
                });
            });
            if (fault == Fault.TOOL_VALUE || fault == Fault.ARTIFACT_VALUE) {
                // 只打印合成样本的阶段结果，不把业务金额写入 Span。
                System.out.printf("tool rows=%d mean_cents=%d%n", report.rows(), report.meanCents());
            }
            return inSpan(tracer.spanBuilder("refund.artifact").startSpan(), artifact -> {
                Thread.sleep(85);
                if (fault == Fault.ARTIFACT_VALUE) {
                    return new Report(report.rows() + 1, report.meanCents());
                }
                return report;
            });
        });
    }

    private static void delay(Tracer tracer, String name, long millis) throws Exception {
        inSpan(tracer.spanBuilder(name).startSpan(), span -> {
            Thread.sleep(millis);
            return null;
        });
    }

    private static Report summarize(String csv, boolean wrongToolValue) {
        var amounts = csv.lines().skip(1)
                .filter(value -> wrongToolValue || !value.equals("NA"))
                .map(value -> value.equals("NA") ? BigDecimal.ZERO : new BigDecimal(value))
                .toList();
        BigDecimal total = amounts.stream().reduce(BigDecimal.ZERO, BigDecimal::add);
        int cents = total.movePointRight(2)
                .divide(BigDecimal.valueOf(amounts.size()), 0, RoundingMode.HALF_UP)
                .intValueExact();
        return new Report(amounts.size(), cents);
    }

    private static <T> T inSpan(Span span, SpanWork<T> work) throws Exception {
        try (Scope scope = span.makeCurrent()) {
            return work.run(span);
        } catch (Exception error) {
            span.setStatus(StatusCode.ERROR);
            span.setAttribute(ERROR_TYPE, error.getClass().getName());
            throw error;
        } finally {
            span.end();
        }
    }

    private static void printSpan(SpanData span, boolean showTimestamps) {
        long durationMs = TimeUnit.NANOSECONDS.toMillis(span.getEndEpochNanos() - span.getStartEpochNanos());
        System.out.printf("%s trace=%s span=%s parent=%s duration_ms=%d status=%s%n",
                span.getName(), span.getTraceId(), span.getSpanId(), span.getParentSpanId(),
                durationMs, span.getStatus().getStatusCode());
        if (showTimestamps) {
            System.out.printf("  start_unix_ns=%d end_unix_ns=%d%n",
                    span.getStartEpochNanos(), span.getEndEpochNanos());
        }
        Long waitMs = span.getAttributes().get(WAIT_MS);
        if (waitMs != null) {
            System.out.println("  queue_wait_ms=" + waitMs);
        }
        Long inputTokens = span.getAttributes().get(INPUT_TOKENS);
        if (inputTokens != null) {
            System.out.printf("  input_tokens=%d output_tokens=%d%n",
                    inputTokens, span.getAttributes().get(OUTPUT_TOKENS));
        }
        String errorType = span.getAttributes().get(ERROR_TYPE);
        if (errorType != null) {
            System.out.println("  error_type=" + errorType);
        }
        span.getEvents().forEach(event -> {
            long offsetMs = TimeUnit.NANOSECONDS.toMillis(event.getEpochNanos() - span.getStartEpochNanos());
            System.out.printf("  event=%s offset_ms=%d%n", event.getName(), offsetMs);
            if (showTimestamps) {
                System.out.println("  event_unix_ns=" + event.getEpochNanos());
            }
        });
    }

    /**
     * 每次运行最多注入一种工具或产物故障，方便比较第一个异常阶段。
     */
    private enum Fault {
        NONE(null),
        TOOL_ERROR("--fail-tool"),
        TOOL_VALUE("--wrong-tool-value"),
        ARTIFACT_VALUE("--wrong-artifact-value");

        private final String option;

        Fault(String option) {
            this.option = option;
        }
    }

    /**
     * 模拟队列消息，只在消息元数据中保存可传播的追踪字段。
     */
    private record Job(Map<String, String> headers, long enqueuedNanos, String csv) {
    }

    /**
     * 某个阶段交出的聚合结果，平均金额以整数分表示。
     */
    private record Report(int rows, int meanCents) {
    }

    /**
     * 包装可抛出异常的操作，确保片段总会结束。
     */
    @FunctionalInterface
    private interface SpanWork<T> {
        T run(Span span) throws Exception;
    }
}
