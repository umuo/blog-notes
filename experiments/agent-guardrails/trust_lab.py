"""Check version enrollment and delegation against fake providers."""

import json
from dataclasses import dataclass, replace

from guardrails_lab import digest


@dataclass(frozen=True)
class Registration:
    provider: str
    service: str
    tenant: str
    release: str
    artifact: str
    manifest: str


@dataclass(frozen=True)
class Grant:
    id: str
    issuer: str
    tenant: str
    subject: str
    audience: str
    task: str
    actions: frozenset[str]
    expires: int
    generation: int


class Registry:
    def __init__(self, registration):
        self.registration = registration
        self.policy_generation = 1
        self.live_grants = {}

    def mint(self, grant):
        if (grant.issuer != "workspace-policy" or grant.generation != self.policy_generation
                or grant.tenant != self.registration.tenant or grant.audience != self.registration.service):
            raise ValueError("grant must come from the active workspace policy")
        if grant.id in self.live_grants:
            raise ValueError("grant ID must be unique")
        self.live_grants[grant.id] = grant

    def revoke(self, grant_id):
        self.live_grants.pop(grant_id)

    def authorize(self, observed, *, grant_id, actor, tenant, task, action, now):
        pinned = self.registration
        grant = self.live_grants.get(grant_id)
        if (observed.provider, observed.service, observed.tenant) != (pinned.provider, pinned.service, pinned.tenant):
            return "service_identity_denied"
        if (observed.release, observed.artifact, observed.manifest) != (pinned.release, pinned.artifact, pinned.manifest):
            return "version_review_required"
        if grant is None or grant.generation != self.policy_generation or now >= grant.expires:
            return "grant_inactive"
        if (grant.subject, grant.tenant, grant.audience, grant.task) != (actor, tenant, pinned.service, task):
            return "scope_denied"
        if action not in grant.actions or action not in {"read_report", "archive_report"}:
            return "scope_denied"
        if action == "archive_report":
            return "send_to_approval_broker"
        return "allow_read"


def run():
    baseline = {"name": "refund-report-tool", "description": "Reads an approved report summary."}
    registration = Registration("catalog-a", "records-service", "school", "v1",
                                digest({"build": "fixture-1"}), digest(baseline))
    registry = Registry(registration)
    grant = Grant("delegation-1", "workspace-policy", "school", "agent-worker", "records-service",
                  "refund-task", frozenset({"read_report", "archive_report"}), 10, 1)
    registry.mint(grant)
    audit = []

    def check(name, observed=registration, *, action="read_report", now=1, task="refund-task", grant_id=grant.id):
        decision = registry.authorize(observed, grant_id=grant_id, actor="agent-worker", tenant="school",
                                      task=task, action=action, now=now)
        audit.append({"case": name, "provider": observed.provider, "release": observed.release,
                      "grant_id": grant_id, "decision": decision})
        return decision

    cases = {
        "registered": check("registered"),
        "description_changed": check("description_changed", replace(
            registration, manifest=digest({**baseline, "description": "Reads a quarterly summary."}))),
        "artifact_changed": check("artifact_changed", replace(registration, artifact=digest({"build": "fixture-2"}))),
        "unregistered_identity": check("unregistered_identity", replace(registration, provider="catalog-b")),
        "different_task": check("different_task", task="other-task"),
        "write_routed_to_broker": check("write_routed_to_broker", action="archive_report"),
    }
    registry.policy_generation += 1
    cases["old_generation"] = check("old_generation")
    narrowed = replace(grant, id="delegation-2", actions=frozenset({"read_report"}),
                       generation=registry.policy_generation)
    registry.mint(narrowed)
    cases["narrow_read"] = check("narrow_read", grant_id=narrowed.id)
    cases["narrow_write"] = check("narrow_write", action="archive_report", grant_id=narrowed.id)
    cases["expired"] = check("expired", now=10, grant_id=narrowed.id)
    registry.revoke(narrowed.id)
    cases["revoked"] = check("revoked", grant_id=narrowed.id)
    assert list(cases.values()) == [
        "allow_read", "version_review_required", "version_review_required", "service_identity_denied",
        "scope_denied", "send_to_approval_broker", "grant_inactive", "allow_read", "scope_denied",
        "grant_inactive", "grant_inactive",
    ]
    return {"cases": cases, "audit": audit}


if __name__ == "__main__":
    print(json.dumps(run(), indent=4))
