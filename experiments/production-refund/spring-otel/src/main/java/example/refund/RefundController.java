package example.refund;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.context.propagation.TextMapGetter;
import io.opentelemetry.context.propagation.TextMapPropagator;
import jakarta.annotation.PreDestroy;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

/**
 * 观察退款任务跨工作线程的上下文传递。接口同步等待进程内工作队列中的统计结果。
 */
@RestController
class RefundController {
    private static final String CSV = "refund_amount\n8.00\n2.00\nNA\n";
    private static final TextMapGetter<Map<String, String>> HEADER_GETTER = new HeaderGetter();

    private final Tracer tracer = GlobalOpenTelemetry.get().getTracer("refund-spring-demo");
    private final TextMapPropagator propagator = GlobalOpenTelemetry.get().getPropagators().getTextMapPropagator();
    private final ThreadPoolExecutor worker = new ThreadPoolExecutor(
            1, 1, 0, TimeUnit.SECONDS, new ArrayBlockingQueue<>(8), new ThreadPoolExecutor.AbortPolicy());

    @PostMapping("/refunds/report")
    Report createReport(@RequestParam(name = "dropTraceparent", defaultValue = "false") boolean dropTraceparent)
            throws ExecutionException {
        String traceId = Span.current().getSpanContext().getTraceId();
        Map<String, String> headers = new HashMap<>();
        Span producer = tracer.spanBuilder("refund.enqueue").setSpanKind(SpanKind.PRODUCER).startSpan();
        Future<Report> result;
        long submittedAt = System.nanoTime();
        try (Scope scope = producer.makeCurrent()) {
            propagator.inject(Context.current(), headers, Map::put);
            if (dropTraceparent) {
                headers.remove("traceparent");
            }
            Map<String, String> messageHeaders = Map.copyOf(headers);
            result = worker.submit(() -> consume(messageHeaders, submittedAt, traceId));
        } catch (RejectedExecutionException failure) {
            producer.setStatus(StatusCode.ERROR);
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "refund queue is full", failure);
        } finally {
            producer.end();
        }
        try {
            return result.get(4, TimeUnit.SECONDS);
        } catch (TimeoutException failure) {
            result.cancel(true);
            throw new ResponseStatusException(HttpStatus.GATEWAY_TIMEOUT, "refund worker timed out", failure);
        } catch (InterruptedException failure) {
            result.cancel(true);
            Thread.currentThread().interrupt();
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "request interrupted", failure);
        }
    }

    private Report consume(Map<String, String> headers, long submittedAt, String traceId) {
        // 从消息重建父上下文。即使线程池自动传播了上下文，也以消息携带的标识为准。
        Context parent = propagator.extract(Context.root(), headers, HEADER_GETTER);
        long queueWaitMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - submittedAt);
        Span consumer = tracer.spanBuilder("refund.process")
                .setParent(parent)
                .setSpanKind(SpanKind.CONSUMER)
                .setAttribute("app.queue.wait_ms", queueWaitMs)
                .startSpan();
        try (Scope scope = consumer.makeCurrent()) {
            Span tool = tracer.spanBuilder("refund.stats").startSpan();
            try {
                var amounts = CSV.lines().skip(1).filter(line -> !line.equals("NA")).map(BigDecimal::new).toList();
                BigDecimal total = amounts.stream().reduce(BigDecimal.ZERO, BigDecimal::add);
                long meanCents = total.movePointRight(2)
                        .divide(BigDecimal.valueOf(amounts.size()), 0, RoundingMode.HALF_UP)
                        .longValueExact();
                return new Report(amounts.size(), meanCents, traceId, consumer.getSpanContext().getTraceId());
            } catch (RuntimeException failure) {
                tool.setStatus(StatusCode.ERROR);
                throw failure;
            } finally {
                tool.end();
            }
        } catch (RuntimeException failure) {
            consumer.setStatus(StatusCode.ERROR);
            throw failure;
        } finally {
            consumer.end();
        }
    }

    @PreDestroy
    void stopWorker() {
        worker.shutdownNow();
    }

    /**
     * 业务结果和演练用的两个追踪 ID。生产接口通常只返回受控的请求标识和业务字段。
     */
    record Report(int rows, long meanCents, String traceId, String workerTraceId) {
    }

    /**
     * 允许 W3C 传播器从模拟队列消息的头字段读取上下文。
     */
    private static final class HeaderGetter implements TextMapGetter<Map<String, String>> {
        @Override
        public Iterable<String> keys(Map<String, String> headers) {
            return headers.keySet();
        }

        @Override
        public String get(Map<String, String> headers, String key) {
            return headers.get(key);
        }
    }
}
