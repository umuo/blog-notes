"""Deterministic lifecycle/tenant fixture, using a fixed job rather than arbitrary commands."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


@dataclass
class Session:
    id: str
    tenant: str
    workspace: Path
    expires_at: int


class ControlPlane:
    def __init__(self, root):
        self.root = root
        self.sessions = {}
        self.requests = {}
        self.audit = []

    def create(self, tenant, request_id, now):
        key = (tenant, request_id)
        if key in self.requests:
            known = self.requests[key]
            return known if known in self.sessions else "request_already_closed"
        if any(row.tenant == tenant for row in self.sessions.values()):
            return "quota_exceeded"
        identifier = f"session-{len(self.requests) + 1}"
        folder = self.root / identifier
        folder.mkdir()
        self.sessions[identifier] = Session(identifier, tenant, folder, now + 10)
        self.requests[key] = identifier
        self.audit.append([identifier, "created"])
        return identifier

    def owner(self, tenant, identifier, now):
        session = self.sessions.get(identifier)
        if session is None or session.tenant != tenant:
            return None
        if now >= session.expires_at:
            self.destroy(tenant, identifier)
            return None
        return session

    def execute(self, tenant, identifier, now):
        session = self.owner(tenant, identifier, now)
        if session is None:
            return "unavailable"
        result = subprocess.run(
            [sys.executable, str(HERE / "refund_job.py"), str(HERE / "refunds.csv"),
             str(session.workspace / "report.json")],
            check=True, capture_output=True, timeout=5,
        )
        assert result.stdout.splitlines()[0] == b"READY"
        self.audit.append([identifier, "job_completed"])
        return "completed"

    def artifact(self, tenant, identifier, now):
        session = self.owner(tenant, identifier, now)
        file = session.workspace / "report.json" if session is not None else None
        if file is None or not file.is_file():
            return "unavailable"
        return json.loads(file.read_text())

    def destroy(self, tenant, identifier):
        session = self.sessions.get(identifier)
        if session is None:
            return "already_absent"
        if session.tenant != tenant:
            return "unavailable"
        rmtree(session.workspace)
        del self.sessions[identifier]
        self.audit.append([identifier, "destroyed"])
        return "destroyed"


def main():
    with TemporaryDirectory(prefix="lifecycle-lab-") as workspace:
        owner = "school"
        control = ControlPlane(Path(workspace))
        identifier = control.create(owner, "refund-1", 0)
        result = {
            "id": identifier,
            "create_retry": control.create(owner, "refund-1", 0),
            "quota": control.create(owner, "refund-2", 0),
            "before_run": control.artifact(owner, identifier, 1),
            "completed": control.execute(owner, identifier, 1),
            "other_tenant": control.artifact("partner", identifier, 2),
            "artifact": control.artifact(owner, identifier, 2),
            "destroy": control.destroy(owner, identifier),
            "destroy_retry": control.destroy(owner, identifier),
            "after_destroy": control.artifact(owner, identifier, 3),
        }
        expired = control.create(owner, "refund-2", 3)
        result["after_expiry"] = control.execute(owner, expired, 13)
        result["remaining"] = len(control.sessions)
        assert not list(Path(workspace).iterdir())
        assert result == {
            "id": "session-1", "create_retry": "session-1", "quota": "quota_exceeded",
            "before_run": "unavailable", "completed": "completed", "other_tenant": "unavailable",
            "artifact": {"valid_count": 2, "mean": 5.0}, "destroy": "destroyed",
            "destroy_retry": "already_absent", "after_destroy": "unavailable",
            "after_expiry": "unavailable", "remaining": 0,
        }
        result["audit"] = control.audit
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
