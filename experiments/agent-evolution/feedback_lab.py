"""Replay synthetic report corrections and a strictly reviewed release."""

import json
import math
from collections import Counter
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class Case:
    id: str
    org: str
    schema: str
    values: tuple
    expected: tuple


@dataclass(frozen=True)
class Candidate:
    version: str
    org: str
    schema: str
    rule: str
    source: str
    requested_capabilities: tuple = ("read_fixture", "write_draft")


DISCOVERY = Case("correction-1", "finance", "refund-v2", (4, "NA", 6), (2, 5))
HOLDOUT = (
    Case("unseen-missing", "finance", "refund-v2", (2, "NA", 8, 8), (3, 6)),
    Case("unseen-empty", "finance", "refund-v2", ("NA", "NA"), (0, None)),
    Case("unseen-zero", "finance", "refund-v2", (0, 8), (2, 4)),
    Case("legacy", "finance", "legacy-zero", ("NA", 6), (2, 3)),
    Case("other-tenant", "support", "refund-v2", ("NA", 6), (2, 3)),
)
BASE = Candidate("v1", "finance", "refund-v2", "zero_fill", "approved-baseline")
BETTER = Candidate("v2", "finance", "refund-v2", "skip_missing", DISCOVERY.id)
OVERBROAD = Candidate("global", "*", "*", "skip_missing", DISCOVERY.id)
UNAUTHORIZED = Candidate("bad-policy", "finance", "refund-v2", "skip_missing", DISCOVERY.id,
                         ("read_fixture", "write_draft", "archive_report"))


def report(case, candidate, *, override=None):
    rule = "zero_fill"
    if candidate.org in {case.org, "*"} and candidate.schema in {case.schema, "*"}:
        rule = candidate.rule
    if override is not None:
        rule = override
    keep = [0 if item == "NA" else item for item in case.values if rule != "skip_missing" or item != "NA"]
    return len(keep), mean(keep) if keep else None


def replay(candidate):
    checks = {case.id: report(case, candidate) == case.expected for case in HOLDOUT}
    authority = candidate.requested_capabilities == BASE.requested_capabilities
    source = candidate.source == DISCOVERY.id
    fixes_example = report(DISCOVERY, candidate) == DISCOVERY.expected
    return {"checks": checks, "passed": sum(checks.values()), "total": len(checks),
            "known_correction_passed": fixes_example, "same_capabilities": authority,
            "verified_source": source, "eligible": authority and source and fixes_example and all(checks.values())}


class ReleaseBook:
    def __init__(self):
        self.active = BASE
        self.candidates = {}
        self.audit = []

    def submit(self, actor, candidate):
        if actor != "observer" or candidate.version in self.candidates:
            raise ValueError("candidate submit denied")
        record = replay(candidate)
        self.candidates[candidate.version] = (candidate, record)
        self.audit.append(["submitted", candidate.version, "observer"])
        return record

    def activate(self, reviewer, version):
        if reviewer != "reviewer" or not self.candidates[version][1]["eligible"]:
            self.audit.append(["denied", version, reviewer])
            return False
        self.active = self.candidates[version][0]
        self.audit.append(["activated", version, reviewer])
        return True

    def rollback(self, reviewer):
        if reviewer != "reviewer":
            raise PermissionError("review required")
        self.audit.append(["rollback", self.active.version, reviewer])
        self.active = BASE


def observe(events):
    groups = Counter()
    seen = set()
    for entry in events:
        assert set(entry) == {"trace", "version", "error"}
        if entry["trace"] not in seen:
            groups[(entry["version"], entry["error"])] += 1
            seen.add(entry["trace"])
    return {f"{version}/{error}": count for (version, error), count in groups.items()}


def release_metrics(synthetic_requests):
    success = sum(row[0] for row in synthetic_requests)
    durations = sorted(row[1] for row in synthetic_requests)
    return {"requests": len(durations), "success_rate": success / len(durations),
            "p95_ms": durations[math.ceil(0.95 * len(durations)) - 1],
            "cents_per_success": sum(row[2] for row in synthetic_requests) / success if success else None}


def main():
    observations = [
        {"trace": "synthetic-01", "version": "v1", "error": "wrong_missing_rule"},
        {"trace": "synthetic-01", "version": "v1", "error": "wrong_missing_rule"},
        {"trace": "synthetic-02", "version": "v1", "error": "wrong_missing_rule"},
        {"trace": "synthetic-03", "version": "v1", "error": "format_error"},
    ]
    clusters = observe(observations)
    assert clusters["v1/wrong_missing_rule"] == 2
    book = ReleaseBook()
    fixed = book.submit("observer", BETTER)
    broad = book.submit("observer", OVERBROAD)
    wrong_scope = book.submit("observer", UNAUTHORIZED)
    unverified = book.submit("observer", Candidate("unverified", BETTER.org, BETTER.schema, BETTER.rule, "unverified-note"))
    assert unverified["passed"] == 5 and not unverified["verified_source"] and not unverified["eligible"]
    assert fixed["eligible"] and not broad["eligible"] and not wrong_scope["eligible"]
    assert not book.activate("observer", BETTER.version)
    assert not book.activate("reviewer", OVERBROAD.version)
    assert book.activate("reviewer", BETTER.version)
    active_before_rollback = book.active.version
    control = release_metrics([(True, 70, 2), (True, 80, 2), (True, 85, 2), (True, 95, 2)])
    canary = release_metrics([(True, 80, 2), (True, 90, 2), (True, 110, 2), (False, 190, 3)])
    if canary["success_rate"] < 0.9 or canary["p95_ms"] > 150 or canary["cents_per_success"] > 3.0:
        book.rollback("reviewer")
    assert book.active == BASE
    modes = {
        "old_report": report(DISCOVERY, BASE),
        "same_session_retry": report(DISCOVERY, BASE, override="skip_missing"),
        "next_session_without_release": report(DISCOVERY, BASE),
        "next_session_with_release": report(DISCOVERY, BETTER),
        "legacy_on_new_release": report(HOLDOUT[3], BETTER),
        "other_org_on_new_release": report(HOLDOUT[4], BETTER),
    }
    print(json.dumps({
        "modes": modes, "clusters": clusters, "baseline": replay(BASE), "scoped": fixed,
        "overbroad": broad, "capability_expansion": wrong_scope, "unknown_provenance": unverified,
        "canary_simulation": {"control": control, "candidate": canary, "active_before_rollback": active_before_rollback,
                              "active_after_rollback": book.active.version}, "audit": book.audit,
    }, indent=4))


if __name__ == "__main__":
    main()
