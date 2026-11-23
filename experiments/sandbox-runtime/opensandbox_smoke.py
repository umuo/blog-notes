"""Exercise an authorized loopback OpenSandbox API with a fixed synthetic job."""

import asyncio
import json
import os
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from opensandbox import Sandbox, SandboxManager
from opensandbox.config import ConnectionConfig
from opensandbox.models.execd import RunCommandOpts
from opensandbox.models.filesystem import WriteEntry
from opensandbox.models.sandboxes import NetworkPolicy, SandboxFilter


HERE = Path(__file__).resolve().parent
IMAGE = "python:3.12.12-slim"


def status_after_delete(domain, api_key, sandbox_id):
    request = Request(f"http://{domain}/v1/sandboxes/{sandbox_id}",
                      headers={"OPEN-SANDBOX-API-KEY": api_key})
    try:
        with urlopen(request, timeout=5) as response:
            return response.status
    except HTTPError as error:
        return error.code


async def run():
    domain = os.environ.get("OPEN_SANDBOX_LAB_DOMAIN", "127.0.0.1:18090")
    if urlsplit(f"http://{domain}").hostname != "127.0.0.1":
        raise ValueError("use an authorized loopback lab endpoint")
    key = os.environ["OPEN_SANDBOX_API_KEY"]
    config = ConnectionConfig(domain=domain, api_key=key, use_server_proxy=True, disable_metrics=True,
                              request_timeout=timedelta(seconds=90))
    tags = {"purpose": "synthetic-refund-blog-fixture", "experiment": uuid4().hex}
    try:
        sandbox = await Sandbox.create(
            IMAGE, entrypoint=["sleep", "3600"], timeout=timedelta(minutes=3),
            ready_timeout=timedelta(seconds=45), metadata=tags,
            resource={"cpu": "500m", "memory": "256Mi"},
            network_policy=NetworkPolicy(defaultAction="deny"), connection_config=config,
        )
    except Exception:
        manager = await SandboxManager.create(config)
        try:
            matches = await manager.list_sandbox_infos(SandboxFilter(metadata=tags))
            for candidate in matches.sandbox_infos:
                if all((candidate.metadata or {}).get(k) == v for k, v in tags.items()):
                    await manager.kill_sandbox(candidate.id)
        finally:
            await manager.close()
        raise
    try:
        initial = await sandbox.get_info()
        await sandbox.files.write_files([
            WriteEntry(path="/tmp/refund_job.py", data=(HERE / "refund_job.py").read_bytes(), mode=0o644),
            WriteEntry(path="/tmp/refunds.csv", data=(HERE / "refunds.csv").read_bytes(), mode=0o644),
        ])
        event = await sandbox.commands.run(
            "python /tmp/refund_job.py /tmp/refunds.csv /tmp/report.json",
            opts=RunCommandOpts(timeout=timedelta(seconds=5)),
        )
        assert event.error is None
        report = json.loads(await sandbox.files.read_file("/tmp/report.json"))
        assert report == {"valid_count": 2, "mean": 5.0}
        await sandbox.renew(timedelta(minutes=4))
        renewed = await sandbox.get_info()
        assert renewed.expires_at > initial.expires_at
    finally:
        try:
            await sandbox.kill()
        finally:
            await sandbox.close()
    last = await asyncio.to_thread(status_after_delete, domain, key, sandbox.id)
    assert last == 404, last
    return {"sdk": version("opensandbox"), "workload_image": IMAGE, "ready_state": initial.status.state,
            "requested_egress": "deny", "report": report, "expiry_extended": True, "status_after_delete": last}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), indent=4))
