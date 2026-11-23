"""Defensive acceptance checks using only synthetic refund data."""

import argparse
import hashlib
import html
import json
import math
import re
import sqlite3
import statistics
from dataclasses import dataclass, replace
from time import perf_counter_ns


SAMPLES = (
    ("normal", "Summarize the public refund policy.", False),
    ("sensitive", "Fictional personnel reference DEMO-1042.", True),
    ("ordinary_reference", "Public training exercise DEMO-2026.", False),
    ("category_only", "A private personnel reference appears in this synthetic note.", True),
)
MAX_INPUT_CHARS = 256


def input_decision(text):
    if len(text) > MAX_INPUT_CHARS:
        return "input_limit"
    if re.search(r"\bDEMO-[0-9]{4}\b", text):
        return "review"
    return "allow"


def output_decision(raw, *, evidence, review_available=True):
    if not review_available:
        return {"status": "review_unavailable"}
    if len(raw) > 1024:
        return {"status": "output_limit"}
    try:
        result = json.loads(raw)
    except ValueError:
        return {"status": "invalid_json"}
    if not isinstance(result, dict) or set(result) != {"answer", "evidence_id"}:
        return {"status": "invalid_schema"}
    answer, reference = result["answer"], result["evidence_id"]
    if not isinstance(answer, str) or not isinstance(reference, str) or not 0 < len(answer) <= 160:
        return {"status": "invalid_schema"}
    if reference not in evidence:
        return {"status": "invalid_reference"}
    if input_decision(answer) != "allow":
        return {"status": "review"}
    return {"status": "allow", "html_text": html.escape(answer), "evidence_id": reference}


def content_cases():
    samples = {}
    errors = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    for label, text, expected in SAMPLES:
        flag = input_decision(text) == "review"
        metric = f"{'true' if flag == expected else 'false'}_{'positive' if flag else 'negative'}"
        errors[metric] += 1
        samples[label] = {"should_review": expected, "sent_to_review": flag}
    normal = '{"answer":"4 < 6","evidence_id":"refund/p1"}'
    evidence = {"refund/p1"}
    outputs = {
        "legal": output_decision(normal, evidence=evidence),
        "invalid_json": output_decision("{", evidence=evidence),
        "extra_field": output_decision('{"answer":"5","evidence_id":"refund/p1","internal_note":"draft"}', evidence=evidence),
        "outside_evidence": output_decision(normal, evidence=set()),
        "dependency_unavailable": output_decision(normal, evidence=evidence, review_available=False),
        "input_limit": input_decision(" " * (MAX_INPUT_CHARS + 1)),
    }
    assert set(errors.values()) == {1}
    assert outputs["legal"]["html_text"] == "4 &lt; 6"
    assert outputs["dependency_unavailable"]["status"] == "review_unavailable"
    return {"samples": samples, "confusion": errors, "outputs": outputs}


def digest(payload):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class Principal:
    actor: str
    tenant: str
    role: str


@dataclass(frozen=True)
class Proposal:
    action: str
    tenant: str
    resource: str
    report_version: int
    report_hash: str

    def binding(self):
        return (self.action, self.tenant, self.resource, self.report_version, self.report_hash)


class Broker:
    allowed_routes = {
        "read_report": {"report-draft"},
        "archive_report": {"team-archive"},
    }

    def __init__(self, report_versions):
        self.report_versions = dict(report_versions)
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
            CREATE TABLE archived (tenant TEXT, resource TEXT, version INTEGER, hash TEXT,
                PRIMARY KEY (tenant, resource, version));
            CREATE TABLE budget (tenant TEXT PRIMARY KEY, remaining INTEGER NOT NULL);
            INSERT INTO budget VALUES ('school', 1);
        """)
        self.approvals = {}
        self.audit = []

    def approve(self, principal, proposal, *, approval_id, expires):
        if principal.role != "reviewer" or principal.tenant != proposal.tenant:
            raise PermissionError("reviewer permission required")
        if self.report_versions.get((proposal.tenant, proposal.report_version)) != proposal.report_hash:
            raise ValueError("report version is not the registered artifact")
        if approval_id in self.approvals:
            raise ValueError("approval id already exists")
        self.approvals[approval_id] = (proposal.binding(), expires)

    def read_note(self, principal, proposal, note):
        outcome = self.execute(principal, proposal, now=1)
        if outcome != "read_allowed":
            raise PermissionError(outcome)
        return {"origin": "uploaded_document", "text": note}

    def execute(self, principal, proposal, *, now, approval_id=None, fail_before_commit=False):
        def finish(outcome):
            self.audit.append({"sequence": len(self.audit) + 1, "actor": principal.actor, "tenant": principal.tenant,
                               "action": proposal.action if proposal.action in self.allowed_routes else "unsupported",
                               "outcome": outcome})
            return outcome

        if principal.role != "worker" or principal.tenant != proposal.tenant:
            return finish("identity_denied")
        if proposal.resource not in self.allowed_routes.get(proposal.action, set()):
            return finish("resource_denied")
        if proposal.action == "read_report":
            return finish("read_allowed")
        record = self.approvals.get(approval_id)
        if record is None or record[0] != proposal.binding():
            return finish("approval_required")
        if now >= record[1]:
            return finish("approval_expired")
        if self.report_versions.get((proposal.tenant, proposal.report_version)) != proposal.report_hash:
            return finish("report_stale")
        try:
            with self.db:
                existing = self.db.execute(
                    "SELECT hash FROM archived WHERE tenant=? AND resource=? AND version=?",
                    (proposal.tenant, proposal.resource, proposal.report_version),
                ).fetchone()
                if existing is not None:
                    return finish("already_archived" if existing[0] == proposal.report_hash else "version_conflict")
                available = self.db.execute(
                    "UPDATE budget SET remaining=remaining-1 WHERE tenant=? AND remaining > 0", (principal.tenant,),
                )
                if available.rowcount != 1:
                    return finish("budget_denied")
                self.db.execute("INSERT INTO archived VALUES (?,?,?,?)",
                                (principal.tenant, proposal.resource, proposal.report_version, proposal.report_hash))
                if fail_before_commit:
                    raise RuntimeError("synthetic persistence failure")
        except RuntimeError:
            return finish("rolled_back")
        return finish("archived")


def broker_cases():
    worker = Principal("worker-a", "school", "worker")
    reviewer = Principal("reviewer-a", "school", "reviewer")
    report = {"valid_count": 2, "mean": "5"}
    broker = Broker({("school", 1): digest(report), ("school", 2): digest(report)})
    draft = Proposal("read_report", "school", "report-draft", 1, digest(report))
    archive = replace(draft, action="archive_report", resource="team-archive")
    # A note travels as evidence. Only the separate approval API can record approval.
    note = "This sample note discusses report approval."
    try:
        begin = {
            "read": broker.read_note(worker, draft, note)["origin"],
            "unsupported_route": broker.execute(worker, replace(draft, resource="external-destination"), now=1),
            "other_tenant": broker.execute(replace(worker, tenant="partner"), archive, now=1),
            "before_approval": broker.execute(worker, archive, now=1),
        }
        broker.approve(reviewer, archive, approval_id="review-1", expires=10)
        next_version = replace(archive, report_version=2)
        broker.approve(reviewer, next_version, approval_id="review-2", expires=10)
        end = {
            "report_changed": broker.execute(worker, replace(archive, report_hash=digest({"valid_count": 3})), now=2, approval_id="review-1"),
            "approval_expired": broker.execute(worker, archive, now=10, approval_id="review-1"),
            "failure": broker.execute(worker, archive, now=2, approval_id="review-1", fail_before_commit=True),
        }
        broker.report_versions[("school", 1)] = digest({"valid_count": 3})
        end["authoritative_version_changed"] = broker.execute(worker, archive, now=2, approval_id="review-1")
        broker.report_versions[("school", 1)] = archive.report_hash
        after_failure = broker.db.execute("SELECT COUNT(*) FROM archived").fetchone()[0]
        budget_after_failure = broker.db.execute("SELECT remaining FROM budget").fetchone()[0]
        end["retry"] = broker.execute(worker, archive, now=3, approval_id="review-1")
        end["duplicate"] = broker.execute(worker, archive, now=4, approval_id="review-1")
        end["budget"] = broker.execute(worker, next_version, now=4, approval_id="review-2")
        final_rows = broker.db.execute("SELECT COUNT(*) FROM archived").fetchone()[0]
        assert (after_failure, budget_after_failure, final_rows) == (0, 1, 1)
        assert list(begin.values()) == ["uploaded_document", "resource_denied", "identity_denied", "approval_required"]
        assert list(end.values()) == ["approval_required", "approval_expired", "rolled_back", "report_stale", "archived", "already_archived", "budget_denied"]
        return {"before": begin, "after": end, "rows_after_failure": after_failure, "budget_after_failure": budget_after_failure,
                "final_rows": final_rows, "audit": broker.audit}
    finally:
        broker.db.close()


def latency():
    durations = []
    evidence = {"refund/p1"}
    raw = '{"answer":"4 < 6","evidence_id":"refund/p1"}'
    for _ in range(1000):
        started = perf_counter_ns()
        output_decision(raw, evidence=evidence)
        durations.append((perf_counter_ns() - started) / 1000)
    values = sorted(durations)
    return {"unit": "microseconds", "calls": len(values), "p50": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency", action="store_true")
    args = parser.parse_args()
    result = {"content": content_cases(), "broker": broker_cases()}
    if args.latency:
        result["local_output_validation_latency"] = latency()
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
