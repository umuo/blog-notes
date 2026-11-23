"""Temporal worker restart and history replay with offline activities."""

import argparse
import asyncio
import json
import sqlite3
import tempfile
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

from temporalio import activity, workflow
from temporalio.client import WorkflowHistory
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker


@workflow.defn
class RefundWorkflow:
    def __init__(self):
        self.approved = False
        self.phase = "started"

    @workflow.signal
    def approve(self):
        self.approved = True

    @workflow.query
    def status(self):
        return self.phase

    @workflow.run
    async def run(self, action_id: str) -> dict:
        report = await workflow.execute_activity("read_fixture", start_to_close_timeout=timedelta(seconds=5),
                                                 result_type=dict)
        self.phase = "awaiting_approval"
        await workflow.wait_condition(lambda: self.approved)
        self.phase = "archiving"
        receipt = await workflow.execute_activity(
            "archive_once", action_id,
            start_to_close_timeout=timedelta(seconds=5), result_type=str,
            retry_policy=RetryPolicy(initial_interval=timedelta(milliseconds=100), maximum_attempts=2),
        )
        self.phase = "complete"
        return {"report": report, "receipt": receipt}


@workflow.defn(name="RefundWorkflow")
class DivergentWorkflow:
    @workflow.run
    async def run(self, action_id: str) -> dict:
        await workflow.sleep(15)
        return {"receipt": action_id}


async def get_ready(handle):
    async with asyncio.timeout(15):
        while await handle.query(RefundWorkflow.status) != "awaiting_approval":
            await asyncio.sleep(0.025)


async def record():
    # The SDK may obtain its test-server binary on the first run. Workflow code
    # and activities operate on fixed input without using any model.
    with tempfile.TemporaryDirectory() as scratch:
        receipt_db = Path(scratch) / "receipts.sqlite"
        with sqlite3.connect(receipt_db) as db:
            db.execute("CREATE TABLE receipts (id TEXT PRIMARY KEY)")
        executions = {"read": 0, "archive": 0}

        @activity.defn
        async def read_fixture() -> dict:
            executions["read"] += 1
            return {"count": 2, "mean": "5"}

        @activity.defn
        async def archive_once(action_id: str) -> str:
            executions["archive"] += 1
            with sqlite3.connect(receipt_db) as db:
                db.execute("INSERT OR IGNORE INTO receipts VALUES (?)", (action_id,))
            if activity.info().attempt == 1:
                raise ApplicationError("simulated lost acknowledgment")
            return action_id

        async with await WorkflowEnvironment.start_time_skipping() as env:
            options = dict(task_queue="refund-test", workflows=[RefundWorkflow],
                           activities=[read_fixture, archive_once], max_cached_workflows=0)
            async with Worker(env.client, **options):
                handle = await env.client.start_workflow(
                    RefundWorkflow.run, "refund-01/archive", id="refund-01", task_queue="refund-test",
                )
                await get_ready(handle)
            # The first worker is closed here, then the approval is persisted.
            await handle.signal(RefundWorkflow.approve)
            async with Worker(env.client, **options):
                with env.auto_time_skipping_disabled():
                    try:
                        result = await asyncio.wait_for(handle.result(), timeout=30)
                    except TimeoutError:
                        events = await handle.fetch_history()
                        raise RuntimeError(
                            f"restart incomplete: {executions}; events {[event.event_type for event in events.events]}"
                        ) from None
            events = await handle.fetch_history()
        with sqlite3.connect(receipt_db) as db:
            writes = db.execute("SELECT count(*) FROM receipts").fetchone()[0]
        return events, result, executions, writes


async def replay(history):
    await Replayer(workflows=[RefundWorkflow]).replay_workflow(history)
    try:
        await Replayer(workflows=[DivergentWorkflow]).replay_workflow(history)
    except Exception as error:
        assert "nondetermin" in type(error).__name__.lower(), type(error).__name__
        return type(error).__name__
    raise AssertionError("unexpected replay success")


async def main(args):
    if args.record:
        history, result, executed, writes = await record()
        assert executed == {"read": 1, "archive": 2} and writes == 1
        assert result == {"report": {"count": 2, "mean": "5"}, "receipt": "refund-01/archive"}
        args.history.write_text(history.to_json() + "\n")
        evidence = dict(result=result, activity_attempts=executed, archive_rows=writes)
    else:
        history = WorkflowHistory.from_json("refund-01", args.history.read_text())
        evidence = {}
    divergent = await replay(history)
    args.out.write_text(json.dumps(dict(sdk=version("temporalio"), history_events=len(history.events),
                                        resumed=True, unchanged_replay="passed",
                                        changed_replay=divergent, **evidence),
                                   indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    asyncio.run(asyncio.wait_for(main(parser.parse_args()), timeout=150))
