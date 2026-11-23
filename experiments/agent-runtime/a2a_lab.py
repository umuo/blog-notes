"""A2A 0.3 JSON-RPC and SSE exercised over in-process ASGI, without a listener."""

import argparse
import asyncio
import json
from importlib.metadata import version
from pathlib import Path

import httpx
from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TextPart
from a2a.utils import new_task

from runtime import CSV, digest, stats


class Validator(AgentExecutor):
    def __init__(self):
        self.pending = {}
        self.calls = 0

    async def execute(self, context, queue):
        self.calls += 1
        task = context.current_task or new_task(context.message)
        updater = TaskUpdater(queue, task.id, task.context_id)
        await queue.enqueue_event(task)
        clue = context.get_user_input()
        if clue == "hold":
            gate = asyncio.Event()
            self.pending[task.id] = (gate, updater)
            try:
                await updater.start_work()
                await gate.wait()
            finally:
                self.pending.pop(task.id, None)
            return
        await updater.start_work()
        if clue == "bad-fixture":
            await updater.failed()
            return
        report = stats(CSV)
        await updater.add_artifact(
            [Part(root=TextPart(text=json.dumps(report)))], artifact_id="validation-01",
            name="refund-validation", metadata={"report_hash": digest(report), "read_only": True},
        )
        await updater.complete()

    async def cancel(self, context, queue):
        gate, updater = self.pending[context.task_id]
        await TaskUpdater(queue, context.task_id, context.context_id).cancel()
        gate.set()


def app(executor):
    card = AgentCard(
        name="Refund validator", description="Read-only offline report checks",
        url="https://validator.example.test/", version="1.0", protocol_version="0.3.0",
        capabilities=AgentCapabilities(streaming=True), default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="refund-check", name="Check refund report", description="Recompute count and mean",
                           tags=["refund", "csv"])],
    )
    return A2AStarletteApplication(card, DefaultRequestHandler(executor, InMemoryTaskStore())).build()


def payload(method, params):
    return dict(jsonrpc="2.0", id="rpc-" + method.replace("/", "-"), method=method, params=params)


def message(text, id_, blocking=True):
    return dict(
        message=dict(kind="message", role="user", messageId=id_,
                     parts=[dict(kind="text", text=text)]),
        configuration=dict(blocking=blocking),
    )


async def rpc(client, method, params):
    response = await client.post("/", json=payload(method, params))
    response.raise_for_status()
    return response.json()


async def run(out):
    executor = Validator()
    endpoint = app(executor)
    other = app(Validator())
    base = "https://validator.example.test"
    async with (httpx.AsyncClient(transport=httpx.ASGITransport(app=endpoint), base_url=base) as client,
                httpx.AsyncClient(transport=httpx.ASGITransport(app=other), base_url=base) as isolated):
        card = (await client.get("/.well-known/agent-card.json")).json()
        assert card["protocolVersion"] == "0.3.0"
        ledger = {}

        async def submit(correlation):
            if correlation not in ledger:
                response = await rpc(client, "message/send", message("check", "message-" + correlation))
                ledger[correlation] = response["result"]["id"]
            return ledger[correlation]

        task_id = await submit("application-001")
        assert await submit("application-001") == task_id and executor.calls == 1
        result = (await rpc(client, "tasks/get", {"id": task_id}))["result"]
        assert result["status"]["state"] == "completed"
        artifact, = result["artifacts"]
        assert json.loads(artifact["parts"][0]["text"]) == stats(CSV)
        cross_scope = await rpc(isolated, "tasks/get", {"id": task_id})
        assert "error" in cross_scope
        response = await client.post("/", json=payload("message/stream", message("check", "stream-01")))
        assert "text/event-stream" in response.headers["content-type"]
        updates = [json.loads(line.removeprefix("data:").strip())["result"]
                   for line in response.text.splitlines() if line.startswith("data:")]
        assert updates[0]["kind"] == "task" and updates[-1]["status"]["state"] == "completed"
        assert any(item["kind"] == "artifact-update" for item in updates)
        cached_task = updates[0]["id"]
        recovered = await rpc(client, "tasks/get", {"id": cached_task})
        assert recovered["result"]["status"]["state"] == "completed"
        ongoing = (await rpc(client, "message/send", message("hold", "cancel-01", False)))["result"]
        ongoing_id = ongoing["id"]
        cancelled = (await rpc(client, "tasks/cancel", {"id": ongoing_id}))["result"]
        final = (await rpc(client, "tasks/get", {"id": ongoing_id}))["result"]
        assert cancelled["status"]["state"] == final["status"]["state"] == "canceled"
        bad = (await rpc(client, "message/send", message("bad-fixture", "fail-01")))["result"]
        assert bad["status"]["state"] == "failed"
        result = dict(
            sdk=version("a2a-sdk"), protocol=card["protocolVersion"], advertised=card["skills"][0]["id"],
            application_id="application-001", registry_task_reused=True, artifact=artifact,
            stream_kinds=[item["kind"] for item in updates], status_after_new_request=recovered["result"]["status"]["state"],
            cancel_ack=cancelled["status"]["state"], failure_status=bad["status"]["state"],
            cross_scope_error=cross_scope["error"]["code"],
        )
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(run(args.out), timeout=20))
