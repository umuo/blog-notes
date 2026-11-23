"""Executable CSV-agent runtime examples with a scripted model and offline tools."""

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path


CSV = "refund\n4\nNA\n6\n"
TASK = "refund-01"
TENANT = "team-a"


def stats(data):
    reader = csv.DictReader(io.StringIO(data))
    if reader.fieldnames != ["refund"]:
        raise ValueError("expected refund column")
    total = Decimal(0)
    count = 0
    for row in reader:
        if None in row or row["refund"] is None:
            raise ValueError("malformed row")
        value = row["refund"].strip()
        if value == "NA":
            continue
        number = Decimal(value)
        if not number.is_finite():
            raise ValueError("amount must be finite")
        total += number
        count += 1
    return {"count": count, "mean": str(total / count) if count else None}


def digest(value):
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass
class State:
    step: int = 0
    report: dict | None = None
    confirmed: str | None = None
    status: str = "ready"
    seen: tuple = ()


class Runtime:
    def __init__(self, filename, *, task=TASK, tenant=TENANT):
        self.db = sqlite3.connect(filename)
        self.task = task
        self.tenant = tenant
        self.key = f"{tenant}/{task}"
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints(
                key TEXT PRIMARY KEY, version INTEGER NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS archive(
                tenant TEXT NOT NULL, action_id TEXT NOT NULL, report_hash TEXT NOT NULL,
                report TEXT NOT NULL, PRIMARY KEY(tenant, action_id));
        """)
        row = self.db.execute("SELECT version, body FROM checkpoints WHERE key=?", (self.key,)).fetchone()
        self.version, self.state = (0, State()) if row is None else (row[0], State(**json.loads(row[1])))
        self.events = []

    def event(self, kind, **fields):
        number = len(self.events) + 1
        self.events.append({"run_id": self.task, "seq": number, "event": kind, **fields})

    def checkpoint(self):
        expected = self.version
        store = json.dumps(asdict(self.state), ensure_ascii=False)
        with self.db:
            if expected == 0:
                write = self.db.execute("INSERT OR IGNORE INTO checkpoints VALUES (?,1,?)", (self.key, store))
            else:
                write = self.db.execute(
                    "UPDATE checkpoints SET version=version+1,body=? WHERE key=? AND version=?",
                    (store, self.key, expected),
                )
            if write.rowcount != 1:
                raise RuntimeError("stale checkpoint")
        self.version += 1
        self.event("checkpoint", version=self.version, status=self.state.status)

    def close(self):
        self.db.close()

    def approve(self, artifact_hash):
        if self.state.status != "awaiting_approval" or digest(self.state.report) != artifact_hash:
            raise ValueError("approval is for a different report")
        self.state.confirmed = artifact_hash
        self.state.status = "approved"
        self.checkpoint()

    def archive(self, *, lose_receipt=False):
        current = digest(self.state.report)
        if self.state.confirmed != current or self.state.report is None:
            raise PermissionError("archive requires approval for this report")
        action_id = f"{self.task}/archive/{current}"
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO archive VALUES (?,?,?,?)",
                (self.tenant, action_id, current, json.dumps(self.state.report)),
            )
        if lose_receipt:
            raise TimeoutError("committed but acknowledgment lost")
        return action_id

    def recover_archive(self):
        current = digest(self.state.report)
        if self.state.confirmed != current or self.state.report is None:
            raise PermissionError("cannot reconcile unapproved report")
        key = f"{self.task}/archive/{current}"
        found = self.db.execute(
            "SELECT report_hash FROM archive WHERE tenant=? AND action_id=?", (self.tenant, key)
        ).fetchone()
        if found != (current,):
            return False
        self.state.status = "completed"
        self.checkpoint()
        self.event("reconciled", action_id=key)
        return True

    def run(self, actions, *, limit=6, budget=6, cancelled=False, data=CSV):
        if self.state.status in {"completed", "cancelled", "failed", "limit", "budget"}:
            return self.state.status
        observations = []
        scripted = None if callable(actions) else iter(actions)
        while True:
            if cancelled:
                self.state.status = "cancelled"
                break
            if self.state.step >= limit:
                self.state.status = "limit"
                break
            if budget <= 0:
                self.state.status = "budget"
                break
            action = actions(tuple(observations)) if scripted is None else next(scripted, None)
            if action is None:
                break
            budget -= 1
            self.state.step += 1
            kind = action["type"]
            span = f"step-{self.state.step}"
            self.event("decision", span_id=span, decision_kind=kind)
            if kind == "clarify":
                self.state.status = "needs_input"
                break
            if kind == "final":
                self.state.status = "awaiting_approval" if self.state.report is not None else "failed"
                break
            call_id = action.get("id")
            if kind != "tool" or not isinstance(call_id, str) or call_id in self.state.seen:
                self.event("validation_failed", error="invalid or duplicate call id", call_id=call_id, span_id=span)
                self.state.status = "failed"
                break
            self.state.seen = (*self.state.seen, call_id)
            try:
                if action.get("name") == "csv_stats" and action.get("args") == {"column": "refund"}:
                    self.state.report = stats(data)
                    result = {"report": self.state.report}
                elif action.get("name") == "archive" and action.get("args") == {}:
                    result = {"action_id": self.archive()}
                    self.state.status = "completed"
                else:
                    raise ValueError("unknown tool or invalid arguments")
            except (ArithmeticError, ValueError, PermissionError) as error:
                self.event("tool_result", call_id=call_id, ok=False, error=type(error).__name__, span_id=span)
                self.state.status = "failed"
                break
            observations.append({"tool_call_id": call_id, "result": result})
            self.event("tool_result", call_id=call_id, ok=True, result=result, span_id=span)
            self.checkpoint()
            if self.state.status == "completed":
                return self.state.status
        self.checkpoint()
        return self.state.status


def scenarios(folder):
    file = folder / "state.sqlite"
    read = {"type": "tool", "name": "csv_stats", "id": "c-1", "args": {"column": "refund"}}
    done = {"type": "final"}
    run = Runtime(file)
    model_inputs = []

    def scripted_model(observations):
        model_inputs.append(observations)
        if not observations:
            return read
        assert observations[-1] == {"tool_call_id": "c-1", "result": {"report": {"count": 2, "mean": "5"}}}
        return done

    assert run.run(scripted_model) == "awaiting_approval"
    first_events, state = run.events, asdict(run.state)
    snapshot = digest(run.state.report)
    run.close()

    resumed = Runtime(file)
    assert resumed.state.report == {"count": 2, "mean": "5"}
    try:
        resumed.approve("stale-report")
    except ValueError:
        stale_rejected = True
    else:
        raise AssertionError("stale approval accepted")
    resumed.approve(snapshot)
    try:
        resumed.archive(lose_receipt=True)
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected acknowledgment loss")
    resumed.close()
    receipt_lost_state = Runtime(file)
    assert receipt_lost_state.state.status == "approved"
    assert receipt_lost_state.recover_archive()
    rows = receipt_lost_state.db.execute("SELECT count(*) FROM archive").fetchone()[0]
    assert rows == 1
    recovered = receipt_lost_state.events
    receipt_lost_state.close()
    other = Runtime(file, tenant="team-b")
    assert other.state.report is None
    other.close()
    checks = {}
    case_events = {}
    for key, scripted, kwargs in [
        ("clarify", [{"type": "clarify"}], {}),
        ("early_final", [done], {}),
        ("invalid_args", [{**read, "args": {"column": "other"}}], {}),
        ("write_without_approval", [{"type": "tool", "name": "archive", "id": "w-1", "args": {}}], {}),
        ("duplicate_call", [read, read], {}),
        ("cancel", [read], {"cancelled": True}),
        ("limit", [read, done], {"limit": 1}),
        ("budget", [read, done], {"budget": 1}),
    ]:
        sample = Runtime(file, task=key)
        checks[key] = sample.run(scripted, **kwargs)
        if key in {"invalid_args", "duplicate_call", "write_without_approval"}:
            case_events[key] = sample.events
        sample.close()
    assert checks == dict(clarify="needs_input", early_final="failed", invalid_args="failed",
                          write_without_approval="failed", duplicate_call="failed", cancel="cancelled",
                          limit="limit", budget="budget")

    writer, stale = Runtime(file, task="concurrent"), Runtime(file, task="concurrent")
    writer.run([read])
    try:
        stale.checkpoint()
    except RuntimeError:
        stale_checkpoint_rejected = True
    else:
        raise AssertionError("lost update")
    writer.close()
    stale.close()
    return dict(state_at_approval=state, stale_approval_rejected=stale_rejected,
                model_observations=model_inputs, archive_rows=rows, recovery=recovered,
                cases=checks, trace=first_events, failure_traces=case_events,
                stale_checkpoint_rejected=stale_checkpoint_rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        result = scenarios(Path(directory))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
