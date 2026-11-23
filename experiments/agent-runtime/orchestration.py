"""Deterministic teaching cases for feedback loops, scheduling and tool selection."""

import argparse
import asyncio
import json
from pathlib import Path

from runtime import CSV, stats


def review(drafts, maximum=4):
    used = 0
    notices = []
    seen = set()
    expected = stats(CSV)
    for draft in drafts:
        if used >= maximum:
            return {"status": "budget_exhausted", "iterations": used, "issues": notices}
        used += 1
        signature = (draft.get("count"), draft.get("mean"))
        if draft == expected:
            return {"status": "accepted", "iterations": used, "issues": notices}
        if signature in seen:
            return {"status": "needs_human", "iterations": used, "issues": notices}
        seen.add(signature)
        notices.append([field for field in expected if draft.get(field) != expected[field]])
    return {"status": "needs_human", "iterations": used, "issues": notices}


async def scheduling():
    history = []
    started = asyncio.Event()
    slow_gate = asyncio.Event()
    slots = asyncio.Semaphore(2)

    async def read(name):
        async with slots:
            history.append("start:" + name)
            try:
                if name == "statistics":
                    started.set()
                    await slow_gate.wait()
                    return stats(CSV)
                await started.wait()
                return {"missing": "NA"}
            finally:
                history.append("finish:" + name)

    first = asyncio.create_task(read("statistics"))
    second = asyncio.create_task(read("dictionary"))
    dictionary = await second
    slow_gate.set()
    report = await first
    history.append("assemble")
    assert history == ["start:statistics", "start:dictionary", "finish:dictionary",
                       "finish:statistics", "assemble"]

    cancelled = asyncio.Event()

    async def slow_probe():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    try:
        await asyncio.wait_for(slow_probe(), timeout=0.01)
    except TimeoutError:
        assert cancelled.is_set()

    async def failed():
        raise ValueError("bad column")

    partial = await asyncio.gather(read("dictionary"), failed(), return_exceptions=True)
    assert isinstance(partial[1], ValueError)
    return dict(events=history[:5], merged={**report, **dictionary}, partial=["ok", type(partial[1]).__name__],
                timed_out_task_cancelled=cancelled.is_set())


def recall():
    # Per-tenant grants determine visibility before the lexical scorer runs.
    catalog = {
        "csv_stats": {"csv", "refund", "statistics", "count"},
        "csv_preview": {"csv", "refund", "rows", "preview"},
        "log_search": {"log", "find", "error"},
        "archive_report": {"archive", "report", "refund"},
        "other_tenant_stats": {"csv", "refund", "statistics", "count"},
    }
    authorized = {"csv_stats", "csv_preview", "log_search", "archive_report"}
    fixtures = [
        ({"csv", "refund", "count"}, "csv_stats", "csv_stats"),
        ({"csv", "refund", "preview"}, "csv_preview", "csv_stats"),
        ({"log", "error"}, "log_search", "log_search"),
        ({"save", "refund", "document"}, "archive_report", "archive_report"),
    ]
    rows = []
    for terms, expected, chooser_fixture in fixtures:
        ranked = sorted(authorized, key=lambda name: (-len(terms & catalog[name]), name))
        shortlist = ranked[:2]
        picked = chooser_fixture if chooser_fixture in shortlist else shortlist[0]
        rows.append(dict(expected=expected, candidates=shortlist, chosen=picked))
    assert sum(row["expected"] in row["candidates"] for row in rows) == 4
    assert sum(row["expected"] == row["chosen"] for row in rows) == 3
    return dict(rows=rows, recall_at_2=4, selections_correct=3, tests=4)


def branching():
    report = stats(CSV)
    candidates = [("stats-worker", report), ("review-worker", {"count": 3, "mean": "5"})]
    proposals = {name: value for name, value in candidates}
    conflicts = [field for field in report if len({value[field] for value in proposals.values()}) > 1]
    assert conflicts == ["count"]
    return dict(route="refund_statistics", worker_proposals=proposals, conflicting_fields=conflicts,
                merged=report, status="awaiting_approval",
                archive_permission={"stats-worker": False, "review-worker": False, "owner_after_approval": True})


async def main(out):
    corrected = review([{"count": 3, "mean": "5"}, stats(CSV)])
    stuck = review([{"count": 3, "mean": "5"}] * 3)
    exhausted = review([{"count": 3, "mean": "5"}, {"count": 2, "mean": "4"}, stats(CSV)], maximum=2)
    assert (corrected["status"], stuck["status"], exhausted["status"]) == (
        "accepted", "needs_human", "budget_exhausted"
    )
    result = dict(feedback={"corrected": corrected, "stuck": stuck, "exhausted": exhausted},
                  scheduling=await scheduling(), governance=recall(), collaboration=branching())
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.out))
