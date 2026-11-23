"""Deterministic refund-report production drills using only Python's standard library."""

import csv
import hashlib
import io
import json
import sqlite3
from dataclasses import dataclass, replace
from decimal import Decimal
from math import ceil

CSV = "refund_amount\n8.00\n2.00\nNA\n"


def report():
    rows = csv.DictReader(io.StringIO(CSV))
    values = [Decimal(row["refund_amount"]) for row in rows if row["refund_amount"] != "NA"]
    return {"rows": len(values), "mean_cents": int(sum(values) * 100 / len(values))}


class Gateway:
    def __init__(self):
        self.requests_left = 2
        self.tokens_left = 2200
        self.budget_left = 9
        self.inflight = 0
        self.cache = {}

    def call(self, tenant, scenario, payload_hash="csv-a", limit=1100):
        if self.requests_left == 0:
            return ["429:rpm"]
        self.requests_left -= 1
        key = (tenant, payload_hash, "report-v2", "route-v1")
        if scenario == "ok" and key in self.cache:
            return ["cache_hit"]
        if self.tokens_left < limit:
            return ["429:tpm"]
        if self.inflight >= 1:
            return ["429:concurrency"]
        if self.budget_left < 3:
            return ["budget_rejected"]

        attempts = []
        self.inflight += 1
        try:
            self.tokens_left -= limit
            self.budget_left -= 3
            attempts.append(f"A:{scenario}")
            if scenario in {"429", "timeout"}:
                if self.tokens_left < limit or self.budget_left < 3:
                    return attempts + ["fallback_budget_exhausted"]
                self.tokens_left -= limit
                self.budget_left -= 3
                attempts.append("B:ok")
            elif scenario == "partial":
                return attempts + ["stream_error:incomplete"]
            if scenario == "ok":
                self.cache[key] = report()
            return attempts + ["done"]
        finally:
            self.inflight -= 1


def gateway():
    cache = Gateway()
    limits = Gateway()
    limits.inflight = 1
    cap = Gateway()
    cap.budget_left = 2
    fallback_cap = Gateway()
    fallback_cap.budget_left = 5
    cached_calls = [cache.call("alpha", "ok"), cache.call("alpha", "ok")]
    third_call = cache.call("alpha", "ok")
    cache.requests_left = 2
    routes = {
        "upstream_429": Gateway().call("alpha", "429"),
        "upstream_timeout": Gateway().call("alpha", "timeout"),
        "partial_stream": Gateway().call("alpha", "partial"),
        "tenant_cache": cached_calls,
        "cross_tenant": cache.call("beta", "ok"),
        "request_limit": third_call,
        "token_limit": Gateway().call("alpha", "ok", limit=2201),
        "concurrency_limit": limits.call("alpha", "ok"),
        "budget_limit": cap.call("alpha", "ok"),
        "fallback_limit": fallback_cap.call("alpha", "429"),
    }
    expected = {
        "upstream_429": ["A:429", "B:ok", "done"],
        "upstream_timeout": ["A:timeout", "B:ok", "done"],
        "partial_stream": ["A:partial", "stream_error:incomplete"],
        "tenant_cache": [["A:ok", "done"], ["cache_hit"]],
        "cross_tenant": ["A:ok", "done"],
        "token_limit": ["429:tpm"],
        "concurrency_limit": ["429:concurrency"],
        "budget_limit": ["budget_rejected"],
        "fallback_limit": ["A:429", "fallback_budget_exhausted"],
        "request_limit": ["429:rpm"],
    }
    assert routes == expected
    return routes


@dataclass(frozen=True)
class ArchiveRequest:
    actor: str = "approver-a"
    tenant: str = "alpha"
    run: str = "r-17"
    digest: str = "sha256:fixture-a"
    target: str = "archive-a"
    policy: str = "p-7"
    expires_at: int = 50


def security():
    approved = ArchiveRequest()
    expected_target = {"alpha": "archive-a"}
    active_policy = "p-7"

    def decide(environment, request, now, approvals, policy_available=True):
        if not policy_available:
            return "policy_unavailable"
        if request.tenant not in expected_target or request.target != expected_target[request.tenant]:
            return "target_denied"
        if environment != "staging" or request.actor != "approver-a":
            return "identity_denied"
        if request.policy != active_policy or request.expires_at <= now or request not in approvals:
            return "approval_invalid"
        return "allow"

    cases = {
        "staging": decide("staging", approved, 40, {approved}),
        "production_identity": decide("prod", approved, 40, {approved}),
        "archive_destination": decide("staging", replace(approved, target="outside"), 40, {approved}),
        "changed_report": decide("staging", replace(approved, digest="sha256:fixture-b"), 40, {approved}),
        "expired": decide("staging", approved, 50, {approved}),
        "policy_failure": decide("staging", approved, 40, {approved}, policy_available=False),
    }
    assert list(cases.values()) == [
        "allow", "identity_denied", "target_denied", "approval_invalid", "approval_invalid",
        "policy_unavailable",
    ]
    return cases


def p95_nearest_rank(durations):
    return sorted(durations)[ceil(len(durations) * 0.95) - 1]


def sandbox():
    # Durations in virtual milliseconds. Each row is (queue, image, runtime, first_command).
    loads = [(0, 0), (0, 5), (5, 0), (10, 0), (15, 5)] * 4
    paths = {
        "cold": [(queue, 180 + jitter, 60, 20) for queue, jitter in loads],
        "cached": [(queue, 25 + jitter, 60, 20) for queue, jitter in loads],
        "warm": [(queue, 0, 10 + jitter, 20) for queue, jitter in loads],
        "clean_snapshot": [(queue, 0, 30 + jitter, 20) for queue, jitter in loads],
    }
    p95 = {path: p95_nearest_rank([sum(sample) for sample in samples])
           for path, samples in paths.items()}

    sessions = {"s-old": {"lease_ttl_ms": 80, "owner": "r-17", "generation": 1}}
    cleanup = []

    def reclaim(at):
        session = sessions.get("s-old")
        if session and at >= session["lease_ttl_ms"]:
            cleanup.extend(["revoke", "stop", "discard_overlay"])
            del sessions["s-old"]
            cleanup.append("released")

    reclaim(79)
    assert "s-old" in sessions
    reclaim(80)
    reclaim(81)
    assert not sessions and cleanup == ["revoke", "stop", "discard_overlay", "released"]
    assert p95 == {"cold": 280, "cached": 125, "warm": 50, "clean_snapshot": 70}
    return {"p95_virtual_ms": p95, "samples_per_path": len(loads), "reclaim": cleanup}


def migration():
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY, payload TEXT);
        CREATE TABLE outbox (id TEXT PRIMARY KEY, delivered INTEGER DEFAULT 0);
        CREATE TABLE receipts (id TEXT PRIMARY KEY, result TEXT);
    """)
    try:
        with db:
            db.execute("INSERT INTO jobs VALUES ('r-failed', 'csv')")
            raise RuntimeError("simulated crash before outbox insert")
    except RuntimeError:
        pass
    assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    with db:
        db.execute("INSERT INTO jobs VALUES ('r-17', 'csv')")
        db.execute("INSERT INTO outbox (id) VALUES ('r-17')")
    queue = []
    for _ in range(2):
        queue.extend(row[0] for row in db.execute("SELECT id FROM outbox WHERE delivered = 0"))
    with db:
        db.execute("UPDATE outbox SET delivered = 1 WHERE id = 'r-17'")
    for job_id in queue:
        with db:
            db.execute("INSERT INTO receipts VALUES (?, ?) ON CONFLICT (id) DO NOTHING",
                       (job_id, json.dumps(report(), sort_keys=True)))
    count = db.execute("SELECT count(*) FROM receipts").fetchone()[0]
    assert queue == ["r-17", "r-17"] and count == 1
    db.close()
    return {"rolled_back_jobs": 0, "deliveries": len(queue), "receipts": count}


def rollout():
    batch = [{"report": report(), "receipt": 1, "e2e_ms": 900} for _ in range(20)]
    batch[2] = {**batch[2], "receipt": 2}
    batch[3] = {**batch[3], "e2e_ms": 2800}
    batch[4] = {**batch[4], "e2e_ms": 2900}
    complete = sum(row["report"] == report() and row["receipt"] == 1 for row in batch)
    p95 = p95_nearest_rank([row["e2e_ms"] for row in batch])
    gate = complete == len(batch) and p95 <= 2000
    assert (complete, p95, gate) == (19, 2800, False)
    return {"correct_outcomes": complete, "batch_size": len(batch), "p95_ms": p95,
            "canary_weight_before": 5, "canary_weight_after": 5 if gate else 0}


def versions():
    recorded = {"schema": 1, "amount": "5.00", "currency": "CNY"}
    old_input = {"schema": 1, "field": "refund_amount"}
    current_input = {"schema": 2, "column": {"name": "refund_amount", "unit": "CNY_yuan"}}

    def normalize_input(tool_input):
        if tool_input == old_input:
            return {"name": tool_input["field"], "unit": "CNY_yuan"}
        if tool_input == current_input:
            return tool_input["column"]
        raise ValueError("unsupported tool input")

    assert normalize_input(old_input) == normalize_input(current_input)

    def decode(output):
        if output["schema"] == 1:
            return {"mean_cents": int(Decimal(output["amount"]) * 100), "currency": output["currency"]}
        if output["schema"] == 2:
            return {"mean_cents": output["mean_cents"], "currency": output["currency"]}
        raise ValueError("unknown schema")

    current = {"schema": 2, "mean_cents": 500, "currency": "CNY"}
    historical = decode(recorded)
    assert historical == decode(current)
    try:
        decode({"schema": 3})
    except ValueError as exc:
        unknown = str(exc)
    else:
        raise AssertionError("expected an unknown schema error")

    archive = {}
    for _ in range(2):
        key = ("alpha", "r-17", "archive")
        archive.setdefault(key, historical)
    assert len(archive) == 1
    return {"normalized_column": normalize_input(old_input),
            "historical": historical, "current": decode(current),
            "unknown": unknown, "archive_effects": len(archive)}


def tracing():
    trace_id = hashlib.sha256(b"refund:r-17").hexdigest()[:32]
    spans = [
        ["request", None, 0, 1850],
        ["queue", "request", 15, 715],
        ["retrieval", "request", 715, 785],
        ["model", "request", 785, 1185],
        ["tool", "request", 1185, 1765],
        ["sandbox.lease", "tool", 1195, 1645],
        ["sandbox.command", "tool", 1645, 1745],
        ["artifact", "request", 1765, 1850],
    ]
    steps = {name: last - first for name, _, first, last in spans}
    assert sum(steps[name] for name in ("queue", "retrieval", "model", "tool", "artifact")) + 15 == 1850
    assert steps["queue"] == 700 and steps["sandbox.lease"] == 450
    return {"trace_id": trace_id, "spans": spans, "duration_ms": steps, "first_visible_delta_ms": 925,
            "input_tokens": 620, "output_tokens": 80}


def main():
    observed = {
        "report": report(),
        "OPS-01": gateway(),
        "OPS-02": security(),
        "OPS-03": sandbox(),
        "OPS-04": migration(),
        "OPS-05": rollout(),
        "OPS-07": versions(),
        "OPS-06": tracing(),
    }
    assert observed["report"] == {"rows": 2, "mean_cents": 500}
    print(json.dumps(observed, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
