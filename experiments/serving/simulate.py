"""Runnable teaching models. Outputs are simulated, not hardware benchmarks."""

import argparse
import hashlib
import json
import math
from collections import deque
from pathlib import Path


def paged_example():
    block_size = 4
    rows = (
        ("A", (10, 11, 12, 13, 14, 15, 16, 17, 50, 51)),
        ("B", (10, 11, 12, 13, 14, 15, 16, 17, 80, 81)),
        ("C", (8, 11, 12, 13, 14, 15, 16, 17, 80, 81)),
    )
    cache = {}
    slots = 0
    result = []
    for request, sequence in rows:
        parent = b""
        table = []
        matched = 0
        for offset in range(0, len(sequence), block_size):
            tokens = sequence[offset:offset + block_size]
            key = hashlib.sha256(parent + bytes(tokens)).digest()
            parent = key
            if key not in cache:
                slots += 1
                if len(tokens) == block_size:
                    cache[key] = slots
                table.append(slots)
            else:
                matched += len(tokens)
                table.append(cache[key])
        result.append(dict(request=request, block_table=table, cached_tokens=matched))
    return result


def prefill_and_speculate():
    # Each tick has 8 token slots. Keep one for an existing decode request.
    remaining = 18
    chunks = []
    while remaining:
        size = min(7, remaining)
        chunks.append(size)
        remaining -= size
    proposals = [[1, 2, 9], [4, 5, 6]]
    target_checks = [[1, 2, 3, 4], [4, 5, 6, 7]]
    consumed = []
    accepted_total = 0
    for draft, target in zip(proposals, target_checks):
        accepted = 0
        for proposed, verified in zip(draft, target):
            if proposed != verified:
                break
            accepted += 1
        accepted_total += accepted
        consumed += target[:accepted + 1]
    return dict(prefill_chunks=chunks, decode_slots_per_tick=1, accepted=accepted_total,
                drafted=sum(map(len, proposals)), output=consumed)


def memory_and_profile():
    weight_bytes = 494032768 * 2
    kv_bytes_per_token = 2 * 24 * 2 * 64 * 2
    occupied = math.ceil(3000 / 16) * 16
    events = [(0, 8, "cpu"), (8, 10, "copy"), (11, 15, "kernel"),
              (14, 18, "kernel"), (18, 21, "cpu")]
    # Merge overlapping intervals before measuring active wall-clock coverage.
    kernels = sorted((start, end) for start, end, name in events if name == "kernel")
    start, end = kernels[0]
    active = 0
    for left, right in kernels[1:]:
        if left <= end:
            end = max(end, right)
        else:
            active += end - start
            start, end = left, right
    active += end - start
    payload = 2048 * kv_bytes_per_token
    return dict(weight_bytes=weight_bytes, kv_bytes_per_token=kv_bytes_per_token,
                occupied_slots=occupied, kv_bytes_four_requests=4 * occupied * kv_bytes_per_token,
                events_ms=events, gpu_active_wall_ms=active, gpu_sum_kernel_ms=8,
                transfer_2048_tokens_bytes=payload, estimated_wire_ms=payload * 1000 / 2.5e9)


def capacity(strategy):
    # Tick = 1 second. Twelve seconds of 3 req/s, then a 6-second burst at 11 req/s.
    arrivals = [3] * 12 + [11] * 6 + [3] * 27
    pending = deque()
    ready = 1
    starting_at = []
    last_change = -100
    observations = []
    latencies = []
    replica_seconds = 0
    for tick, incoming in enumerate(arrivals):
        pending.extend([tick] * incoming)
        ready += starting_at.count(tick)
        starting_at = [when for when in starting_at if when != tick]
        completed = min(len(pending), ready * 5)
        replica_seconds += ready + len(starting_at)
        for _ in range(completed):
            latencies.append(tick - pending.popleft() + 1)
        observations.append(dict(tick=tick, arrivals=incoming, ready=ready, queued=len(pending)))
        if strategy == "queue" and len(pending) >= 5 and ready + len(starting_at) < 3:
            starting_at.append(tick + 4)
            last_change = tick
        if strategy == "queue" and not pending and ready > 1 and tick - last_change >= 6:
            ready -= 1
            last_change = tick
    latencies.sort()
    return dict(requests=sum(arrivals), completed=len(latencies), max_queue=max(x["queued"] for x in observations),
                p95_seconds=latencies[math.ceil(len(latencies) * 0.95) - 1],
                within_3_seconds=sum(x <= 3 for x in latencies), replicas_seconds=replica_seconds,
                timeline=observations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = dict(paged=paged_example(), decoding=prefill_and_speculate(), memory=memory_and_profile(),
                  capacity={mode: capacity(mode) for mode in ("fixed", "queue")})
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
