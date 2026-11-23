"""Measure a Chat Completions endpoint from one monotonic client clock."""

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path


SAMPLES = {
    "mixed": ("4\nNA\n6", {"count": 2, "mean": 5}),
    "zero": ("NA\n0", {"count": 1, "mean": 0}),
    "empty": ("NA\nNA", {"count": 0, "mean": None}),
    "decimal": ("2\n5\nNA", {"count": 2, "mean": 3.5}),
}


def messages(sample, prefix=""):
    rows, _ = SAMPLES[sample]
    return [
        {"role": "system", "content": (
            "Calculate refund statistics. Ignore NA and keep zero. Return a bare JSON object "
            'with keys count and mean. Set mean to null when count is zero. '
            'Example rows: 7, NA, 3. Example response: {"count":2,"mean":5}. ' + prefix
        )},
        {"role": "user", "content": f"refund\n{rows}"},
    ]


def passed(text, sample):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        result = json.loads(text, object_pairs_hook=unique)
    except ValueError:
        return False
    return (
        isinstance(result, dict) and type(result.get("count")) is int
        and (result.get("mean") is None or type(result["mean"]) in (int, float))
        and result == SAMPLES[sample][1]
    )


def request(base, path, body=None, request_id=None):
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-Id"] = request_id
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(base.rstrip("/") + path, data, headers), timeout=60)


def payload(model, sample, stream, prefix="", max_tokens=48):
    result = dict(model=model, messages=messages(sample, prefix), temperature=0,
                  max_tokens=max_tokens, stream=stream)
    if stream:
        result["stream_options"] = {"include_usage": True}
    return result


def metrics(base):
    with request(base, "/metrics") as response:
        lines = response.read().decode().splitlines()
    keys = ("prefix_cache_queries", "prefix_cache_hits", "num_requests_running",
            "num_requests_waiting", "request_queue_time_seconds",
            "request_success", "kv_cache_usage", "spec_decode_num", "cached_tokens_total",
            "cache_hit_rate", "num_running_reqs", "num_queue_reqs", "num_aborted_requests_total")
    return {line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
            for line in lines if line and not line.startswith("#") and any(key in line for key in keys)
            and not line.split("{", 1)[0].endswith("bucket")}


def one(base, model, sample, stream=True, prefix="", request_id="serv-demo",
        cancel=False, max_tokens=48, on_first_text=None):
    chunks = []
    timestamps = []
    events = []
    t0 = time.perf_counter()
    clock = lambda: round((time.perf_counter() - t0) * 1000, 3)
    params = payload(model, sample, stream, prefix, max_tokens)
    if cancel:
        params["messages"].append({"role": "user", "content": "Print numbers 1 to 500, one per line."})
    with request(base, "/v1/chat/completions", params, request_id) as response:
        headers_ms = clock()
        returned_id = response.headers.get("X-Request-Id")
        if not stream:
            raw = response.read()
            data = json.loads(raw)
            output = data["choices"][0]["message"]["content"] or ""
            return dict(request_id=request_id, response_header_id=returned_id, id=data["id"],
                        headers_ms=headers_ms, elapsed_ms=clock(), usage=data.get("usage"),
                        text=output, passed=passed(output, sample), finish=data["choices"][0]["finish_reason"])

        first_byte_ms = None
        first_event_ms = None
        done = False
        usage = None
        finish = None
        upstream_id = None
        event_lines = []
        while True:
            line = response.readline()
            if not line:
                break
            if first_byte_ms is None:
                first_byte_ms = clock()
            if line.startswith(b"data:"):
                event_lines.append(line[5:].lstrip(b" ").rstrip(b"\r\n"))
                continue
            if line.strip() or not event_lines:
                continue
            data = b"\n".join(event_lines)
            event_lines.clear()
            if first_event_ms is None:
                first_event_ms = clock()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            upstream_id = event.get("id", upstream_id)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if choices:
                choice = choices[0]
                text = choice.get("delta", {}).get("content")
                finish = choice.get("finish_reason") or finish
                if text:
                    chunks.append(text)
                    timestamps.append(clock())
                    if len(chunks) == 1 and on_first_text is not None:
                        on_first_text()
            if len(events) < 5:
                events.append(event)
            if cancel and chunks:
                break
    return dict(
        request_id=request_id, response_header_id=returned_id, id=upstream_id, headers_ms=headers_ms,
        first_line_ms=first_byte_ms, first_event_ms=first_event_ms,
        ttft_ms=timestamps[0] if timestamps else None,
        elapsed_ms=clock(), visible_event_intervals_ms=[
            round(b - a, 3) for a, b in zip(timestamps, timestamps[1:])
        ],
        usage=usage, finish=finish, done=done, text="".join(chunks),
        passed=passed("".join(chunks), sample), first_events=events,
    )


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def bench(base, model, concurrency, repetitions, name, same_prefix=True):
    samples = list(SAMPLES) * repetitions
    prefix = "Only judge valid numeric data in the report. " * 24
    start = time.perf_counter()

    def worker(item):
        i, sample = item
        context = prefix if same_prefix else f"Different request {i:04d}. " + prefix
        return one(base, model, sample, prefix=context, request_id=f"serv-{name}-{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(worker, enumerate(samples)))
    duration = time.perf_counter() - start
    ttfts = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    latencies = [r["elapsed_ms"] for r in rows if r["done"]]
    intervals = [x for r in rows for x in r["visible_event_intervals_ms"]]
    completion_tokens = sum((r["usage"] or {}).get("completion_tokens", 0) for r in rows)
    succeeded = sum(r["done"] and r["passed"] for r in rows)
    return dict(
        concurrency=concurrency, same_prefix=same_prefix, requests=len(rows), completed=len(latencies),
        correct=succeeded, duration_s=duration, p50_ttft_ms=percentile(ttfts, 0.5),
        p95_ttft_ms=percentile(ttfts, 0.95), p95_e2e_ms=percentile(latencies, 0.95),
        median_visible_event_gap_ms=statistics.median(intervals) if intervals else None,
        completion_tokens=completion_tokens, tokens_per_second=completion_tokens / duration,
        correctness_goodput=succeeded / duration,
        slo_goodput=sum(r["done"] and r["passed"] and r["elapsed_ms"] <= 1500 for r in rows) / duration,
        rows=[{k: v for k, v in r.items() if k not in ("first_events", "visible_event_intervals_ms")} for r in rows],
    )


def optional_metrics(server):
    try:
        return metrics(server)
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:18080")
    parser.add_argument("--model", help="可显式选服务端别名，例如 MLX-LM 的 default_model")
    parser.add_argument("--client-server", help="可选 HTTP 代理地址")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", default=2, type=int)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    with request(args.server, "/v1/models") as response:
        model = args.model or json.load(response)["data"][0]["id"]
    server = args.client_server or args.server
    result = dict(model=model, nonstream=one(server, model, "mixed", stream=False),
                  stream=one(server, model, "mixed"))
    if not args.quick:
        result["before"] = optional_metrics(args.server)
        result["cache_unique"] = bench(server, model, 1, args.repetitions, "unique", False)
        result["after_unique"] = optional_metrics(args.server)
        result["cache_shared"] = bench(server, model, 1, args.repetitions, "shared")
        result["after_shared"] = optional_metrics(args.server)
        result["concurrent"] = bench(server, model, 4, args.repetitions, "concurrent")
        result["after_concurrent"] = optional_metrics(args.server)
        result["cancel"] = one(server, model, "mixed", cancel=True, max_tokens=256, request_id="serv-abort")
        time.sleep(0.25)
        result["after_cancel"] = optional_metrics(args.server)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
