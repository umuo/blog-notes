"""Replay a long prefill beside a running decode and capture speculation counters."""

import argparse
import concurrent.futures
import json
import statistics
import threading
from pathlib import Path

from probe import one, optional_metrics


def mixed_pair(url, model, index):
    ready = threading.Event()
    request_id = f"serv-mix-{index}"
    # Send the long input after seeing the first visible content of the short request.
    def decode_first():
        return one(url, model, "mixed", prefix="Explain the result briefly after giving the answer. ",
                   request_id=request_id + "-decode", max_tokens=72, on_first_text=ready.set)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        early = pool.submit(decode_first)
        if not ready.wait(timeout=15):
            raise RuntimeError("短请求没有输出到达")
        prompt = f"Run {index}. " + "Check only valid entries in the table. " * 80
        long = pool.submit(one, url, model, "decimal", True, prompt, request_id + "-prefill")
        short, prompt = early.result(), long.result()
    return dict(
        decode_e2e_ms=short["elapsed_ms"],
        decode_longest_event_gap_ms=max(short["visible_event_intervals_ms"], default=0),
        prefill_ttft_ms=prompt["ttft_ms"],
        prompt_tokens=(prompt["usage"] or {}).get("prompt_tokens"),
        done=short["done"] and prompt["done"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="refund")
    parser.add_argument("--mode", required=True, choices=("mixed", "speculative"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    before = optional_metrics(args.server)
    if args.mode == "mixed":
        rows = [mixed_pair(args.server, args.model, index) for index in range(4)]
        result = dict(rows=rows,
                      median_long_ttft_ms=statistics.median(row["prefill_ttft_ms"] for row in rows),
                      median_short_worst_gap_ms=statistics.median(row["decode_longest_event_gap_ms"] for row in rows))
    else:
        prefix = ("A B C D E F G H. " * 10) + "Repeat the eight letters exactly once after answering. "
        rows = [one(args.server, args.model, sample, prefix=prefix, request_id=f"spec-{sample}")
                for sample in ("mixed", "zero", "empty", "decimal")]
        result = dict(rows=rows)
    after = optional_metrics(args.server)
    result["spec_counters"] = {
        key: after[key] - before.get(key, 0) for key in after if "spec_decode_num" in key
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
