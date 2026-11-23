"""Run a durable teaching batch against the same endpoint as the online probe."""

import argparse
import concurrent.futures
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from probe import SAMPLES, one, request


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:18080")
    parser.add_argument("--model")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.db.exists():
        parser.error("请选择未使用的数据库路径")
    with request(args.server, "/v1/models") as response:
        model = args.model or json.load(response)["data"][0]["id"]

    job_id = "refund-batch-01"
    fixtures = list(SAMPLES)
    online_start = time.perf_counter()
    online = [one(args.server, model, sample, stream=False, request_id=f"online-{sample}")
              for sample in fixtures]
    online_ms = (time.perf_counter() - online_start) * 1000
    with sqlite3.connect(args.db) as db:
        db.execute("""CREATE TABLE items (
            job TEXT, item TEXT, sample TEXT, status TEXT, attempts INTEGER, result TEXT,
            PRIMARY KEY(job, item)
        )""")
        accepted = time.perf_counter()
        for sample in fixtures:
            signature = hashlib.sha256(f"{job_id}:{sample}".encode()).hexdigest()[:16]
            db.execute("INSERT INTO items VALUES (?, ?, ?, 'pending', 0, NULL)", (job_id, signature, sample))
        db.commit()
        submit_ms = (time.perf_counter() - accepted) * 1000
        completion_order = []
        dedup_rows = None
        failed_first = []

        def execute(item, inject):
            item_id, sample = item
            if inject and sample == "empty":
                raise RuntimeError("teaching fixture: retryable failure before inference")
            if inject and sample == "mixed":
                time.sleep(0.4)
            return one(args.server, model, sample, stream=False,
                       request_id=f"{job_id}-{item_id}")

        for attempt in (1, 2):
            pending = db.execute("SELECT item, sample FROM items WHERE job=? AND status!='done'",
                                 (job_id,)).fetchall()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(execute, row, attempt == 1): row for row in pending}
                for future in concurrent.futures.as_completed(futures):
                    item, sample = futures[future]
                    try:
                        output = future.result()
                    except (OSError, RuntimeError) as error:
                        output = {"error": str(error)}
                        db.execute("UPDATE items SET attempts=attempts+1, status='failed', result=? WHERE job=? AND item=?",
                                   (json.dumps(output), job_id, item))
                        failed_first.append(sample)
                    else:
                        statement = """UPDATE items SET status='done', attempts=attempts+1, result=?
                                       WHERE job=? AND item=? AND status!='done'"""
                        params = (json.dumps(output, ensure_ascii=False), job_id, item)
                        db.execute(statement, params)
                        if dedup_rows is None:
                            dedup_rows = db.execute(statement, params).rowcount
                        completion_order.append(sample)
                    db.commit()
        elapsed_ms = (time.perf_counter() - accepted) * 1000
        rows = db.execute("SELECT item, sample, status, attempts, result FROM items WHERE job=?",
                          (job_id,)).fetchall()
    assert all(row[2] == "done" for row in rows) and dedup_rows == 0
    result = dict(
        job_id=job_id, model=model, input_order=fixtures, completion_order=completion_order,
        first_failures=failed_first, replayed_write_rows=dedup_rows, submit_ms=submit_ms,
        batch_elapsed_ms=elapsed_ms, online_elapsed_ms=online_ms,
        online_passed=sum(row["passed"] for row in online),
        items=[dict(item_id=item, sample=sample, status=status, attempts=attempts, result=json.loads(data))
               for item, sample, status, attempts, data in rows],
    )
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run()
