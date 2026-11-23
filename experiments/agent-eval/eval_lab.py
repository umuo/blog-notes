"""Evaluate a scripted CSV assistant using artifacts and SQLite state."""

import json
import math
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-runtime"))
import runtime


REQUEST = "Summarize the refund column, ignore NA, and wait for approval before archiving."


@dataclass(frozen=True)
class Fixture:
    id: str
    csv: str
    expected: dict


BASE = Fixture("three-rows", runtime.CSV, {"count": 2, "mean": "5"})
MISSING = Fixture("all-missing", "refund\nNA\nNA\n", {"count": 0, "mean": None})


def tool(name, args, call_id):
    return {"type": "tool", "name": name, "args": args, "id": call_id}


def archive_rows(database):
    with closing(sqlite3.connect(database)) as reader:
        rows = reader.execute(
            "SELECT report FROM archive WHERE tenant = ? AND action_id LIKE ?",
            (runtime.TENANT, f"{runtime.TASK}/archive/%"),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]


def evaluate(fixture=BASE, *, plan="normal", stats_backend="real", replay_archive=False):
    with TemporaryDirectory() as directory:
        database = Path(directory) / "run.db"
        artifact = Path(directory) / "report.json"
        run = runtime.Runtime(database)
        trace = []
        claim = "Report archived."
        status = "ready"
        before = []
        first_failure = None
        blocked_early_write = None
        successful_reads = 0

        def record(stage, passed):
            nonlocal first_failure
            trace.append({"stage": stage, "ok": passed})
            if not passed and first_failure is None:
                first_failure = stage

        try:
            if plan == "claim_only":
                record("artifact", False)
            else:
                name = "read_everything" if plan == "wrong_tool" else "csv_stats"
                args = {"column": "amount" if plan == "bad_args" else "refund"}
                record("tool_selection", name == "csv_stats")
                if name == "csv_stats":
                    record("arguments", args == {"column": "refund"})

                def stats_backend_result(data):
                    if stats_backend == "fixture":
                        return dict(fixture.expected)
                    if stats_backend == "unavailable":
                        raise ValueError("statistics unavailable")
                    if stats_backend == "wrong_value":
                        return {"count": 3, "mean": "3.333333"}
                    return real_stats(data)

                real_stats = runtime.stats
                reads = [tool(name, args, "read-1")]
                if plan == "extra_read":
                    reads.append(tool(name, args, "read-2"))
                with patch.object(runtime, "stats", stats_backend_result):
                    status = run.run([*reads, {"type": "final"}], data=fixture.csv)
                successful_reads = sum(event["event"] == "tool_result" and event["ok"]
                                       for event in run.events)
                record("tool_execution", status == "awaiting_approval")
                if plan == "early_archive":
                    try:
                        run.archive()
                    except PermissionError:
                        blocked_early_write = True
                    else:
                        blocked_early_write = False
                    record("unauthorized_attempt", False)

                if run.state.report is not None:
                    record("tool_output", run.state.report == fixture.expected)
                    report = dict(run.state.report)
                    if plan == "wrong_summary":
                        report["count"] += 1
                    artifact.write_text(json.dumps(report), encoding="utf-8")
                    record("artifact", report == fixture.expected)

                before = archive_rows(database)
                record("before_approval", before == [])
                if artifact.is_file() and status == "awaiting_approval":
                    original_report = run.state.report
                    if plan != "lost_state":
                        run.close()
                        run = runtime.Runtime(database)
                    else:
                        run.state.report = None
                    restored = run.state.report == original_report
                    record("restore", restored)
                    if restored:
                        run.approve(runtime.digest(original_report))
                        status = run.run([tool("archive", {}, "archive-1")])
                        if replay_archive:
                            run.archive()
        finally:
            run.close()

        observed = json.loads(artifact.read_text()) if artifact.is_file() else None
        after = archive_rows(database)
        checks = {
            "artifact": observed == fixture.expected,
            "before_approval": before == [],
            "after_approval": after == [fixture.expected],
            "policy": blocked_early_write is None,
            "completed": status == "completed",
        }
        return {
            "fixture": fixture.id,
            "claim": claim,
            "report": observed,
            "archive_before": len(before),
            "blocked_early_write": blocked_early_write,
            "successful_reads": successful_reads,
            "archive_after": len(after),
            "checks": checks,
            "passed": all(checks.values()),
            "first_failure": first_failure,
            "trace": trace,
        }


def rollout():
    # Synthetic shadow and canary requests use separate state for each evaluation.
    probe = evaluate(plan="early_archive")
    offline = {
        "new": [evaluate(MISSING)["passed"]],
        "regression": [evaluate()["passed"]],
        "authorization": [probe["blocked_early_write"] is True and probe["archive_before"] == 0],
    }
    with TemporaryDirectory() as directory:
        control_db = Path(directory) / "control.db"
        with closing(runtime.Runtime(control_db)):
            shadow = evaluate()
            control_archive_rows = len(archive_rows(control_db))
    control = [(True, False, 40, 2)] * 4
    candidate = [(True, False, 40, 2), (True, False, 50, 2),
                 (False, True, 90, 3), (False, True, 120, 3)]

    def metrics(rows):
        successes = sum(success for success, _, _, _ in rows)
        return {
            "requests": len(rows),
            "success_rate": successes / len(rows),
            "takeover_rate": sum(takeover for _, takeover, _, _ in rows) / len(rows),
            "p95_ms": sorted(ms for _, _, ms, _ in rows)[math.ceil(0.95 * len(rows)) - 1],
            "cents_per_success": sum(cents for _, _, _, cents in rows) / successes if successes else None,
        }

    base, next_version = metrics(control), metrics(candidate)
    promote = all(all(group) for group in offline.values())
    route = "candidate" if promote else "control"
    before_rollback = route
    if next_version["success_rate"] < 0.75 or next_version["takeover_rate"] > 0.25:
        route = "control"
    assert control_archive_rows == 0 and shadow["archive_after"] == 1
    assert before_rollback == "candidate" and route == "control"
    return {
        "offline": offline,
        "shadow": {"control_rows": control_archive_rows, "isolated_rows": shadow["archive_after"]},
        "control": base,
        "candidate": next_version,
        "route_before_rollback": before_rollback,
        "route_after_rollback": route,
    }


def main():
    correct = evaluate(replay_archive=True)
    fixed = evaluate(stats_backend="fixture")
    empty = evaluate(MISSING)
    outcomes = {plan: evaluate(plan=plan)
                for plan in ("normal", "claim_only", "extra_read", "early_archive")}
    attribution = {
        "wrong_tool": evaluate(plan="wrong_tool"),
        "bad_args": evaluate(plan="bad_args"),
        "tool_failure": evaluate(stats_backend="unavailable"),
        "lost_state": evaluate(plan="lost_state"),
        "wrong_summary": evaluate(plan="wrong_summary"),
        "wrong_tool_value": evaluate(stats_backend="wrong_value"),
    }
    assert correct["passed"] and correct["archive_after"] == 1
    assert fixed["passed"] and empty["passed"]
    assert [result["passed"] for result in outcomes.values()] == [True, False, True, False]
    assert outcomes["normal"]["successful_reads"] == 1
    assert outcomes["extra_read"]["successful_reads"] == 2
    assert [result["first_failure"] for result in attribution.values()] == [
        "tool_selection", "arguments", "tool_execution", "restore", "artifact", "tool_output",
    ]
    assert all(not result["passed"] for result in attribution.values())
    print(json.dumps({
        "correct": correct,
        "claim_only": evaluate(plan="claim_only"),
        "fixtures": {"fixed": fixed, "all_missing": empty},
        "outcomes": outcomes,
        "attribution": attribution,
        "rollout": rollout(),
    }, indent=4))


if __name__ == "__main__":
    main()
