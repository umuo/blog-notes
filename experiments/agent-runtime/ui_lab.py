"""AG-UI SSE encoding, A2UI schema validation and application-level recovery."""

import argparse
import json
import tempfile
from pathlib import Path
from urllib.parse import urljoin

from ag_ui.core.events import (
    CustomEvent, RunFinishedEvent, RunStartedEvent, StateSnapshotEvent,
    ToolCallArgsEvent, ToolCallEndEvent, ToolCallResultEvent, ToolCallStartEvent,
)
from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from runtime import Runtime, digest


THREAD = "refund-thread"
MODEL_RUN = "report-01"
TOOL = {"type": "tool", "name": "csv_stats", "id": "ui-csv-1", "args": {"column": "refund"}}
SCHEMA_ROOT = "https://a2ui.org/specification/v1_0/"


def validate_surface(path, messages):
    names = ("agent_to_renderer.json", "common_types.json", "catalog.json")
    entries = {name: json.loads((path / name).read_text()) for name in names}
    registry = Registry()
    for name, resource in entries.items():
        resource = Resource.from_contents(resource)
        registry = registry.with_resource(urljoin(SCHEMA_ROOT, name), resource)
        if resource.id() is not None:
            registry = registry.with_resource(resource.id(), resource)
    catalog = entries["catalog.json"]["$id"]
    registry = registry.with_resource(urljoin(catalog, "common_types.json"),
                                      Resource.from_contents(entries["common_types.json"]))
    registry = registry.with_resource(urljoin(SCHEMA_ROOT, "catalog.json"),
                                      Resource.from_contents(entries["catalog.json"]))
    validator = Draft202012Validator(entries["agent_to_renderer.json"], registry=registry)
    for message in messages:
        validator.validate(message)
    assert not validator.is_valid({"version": "1.0", "updateComponents": {
        "surfaceId": "refund-card", "components": [{"id": "root", "component": "UnknownWidget"}],
    }})


def surface(report):
    return [
        {"version": "1.0", "createSurface": {
            "surfaceId": "refund-card",
            "catalogId": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
        }},
        {"version": "1.0", "updateComponents": {"surfaceId": "refund-card", "components": [
            {"id": "root", "component": "Card", "child": "summary"},
            {"id": "summary", "component": "Text", "text": {"@path": "/summary"}},
        ]}},
        {"version": "1.0", "updateDataModel": {
            "surfaceId": "refund-card", "path": "/summary",
            "value": f"有效 {report['count']} 条，均值 {report['mean']}",
        }},
    ]


def run_frames(report, surface_messages):
    hash_ = digest(report)
    return [
        RunStartedEvent(thread_id=THREAD, run_id=MODEL_RUN, protocol_version="1.0"),
        ToolCallStartEvent(tool_call_id="ui-csv-1", tool_call_name="csv_stats"),
        ToolCallArgsEvent(tool_call_id="ui-csv-1", delta='{"column":"refund"}'),
        ToolCallEndEvent(tool_call_id="ui-csv-1"),
        ToolCallResultEvent(message_id="report-observation", tool_call_id="ui-csv-1",
                            content=json.dumps(report), role="tool"),
        CustomEvent(name="refund/a2ui", value=surface_messages),
        StateSnapshotEvent(snapshot={"status": "awaiting_approval", "report": report, "report_hash": hash_}),
        RunFinishedEvent(thread_id=THREAD, run_id=MODEL_RUN, outcome={
            "type": "interrupt", "interrupts": [{"id": "confirm-01", "reason": "report_approval",
                                                "metadata": {"reportHash": hash_}}],
        }),
    ]


class Projection:
    def __init__(self):
        self.cursor = 0
        self.state = {}

    def consume(self, seq, frame):
        if seq <= self.cursor:
            return "duplicate"
        if seq != self.cursor + 1 and frame["type"] != "STATE_SNAPSHOT":
            return "gap"
        if frame["type"] == "STATE_SNAPSHOT":
            self.state = frame["snapshot"]
        self.cursor = seq
        return "applied"


def main(args):
    encoder = EventEncoder()
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "ui.sqlite"
        runtime = Runtime(db)
        assert runtime.run([TOOL, {"type": "final"}]) == "awaiting_approval"
        report = runtime.state.report
        interface = surface(report)
        validate_surface(args.schemas, interface)
        frames = run_frames(report, interface)
        sse = [encoder.encode(frame) for frame in frames]
        events = [json.loads(frame.removeprefix("data: ")) for frame in sse]

        view = Projection()
        for index, event in enumerate(events[:3], start=1):
            assert view.consume(index, event) == "applied"
        gap = view.consume(6, events[5])
        resume = view.consume(7, events[6])
        duplicate = view.consume(7, events[6])
        assert (gap, resume, duplicate) == ("gap", "applied", "duplicate")
        assert view.state["report"] == report
        assert surface(view.state["report"]) == interface
        reply = RunAgentInput(thread_id=THREAD, run_id="report-02", protocol_version="1.0",
                              messages=[], resume=[{
                                  "interruptId": "confirm-01", "status": "resolved",
                                  "payload": {"reportHash": view.state["report_hash"]},
                              }])

        try:
            runtime.approve("old-report")
        except ValueError:
            stale_rejected = True
        else:
            raise AssertionError("stale approval accepted")
        answer, = reply.resume
        assert answer.interrupt_id == "confirm-01"
        runtime.approve(answer.payload["reportHash"])
        assert runtime.run([{"type": "tool", "name": "archive", "id": "approved-write", "args": {}}]) == "completed"
        writes = runtime.db.execute("SELECT count(*) FROM archive").fetchone()[0]
        runtime.close()

        cancelling = Runtime(db, task="cancelled-run")
        assert cancelling.run([TOOL], cancelled=True) == "cancelled"
        cancelled = RunFinishedEvent(thread_id=THREAD, run_id="report-cancel", outcome={"type": "cancelled"})
        cancelling.close()

    assert writes == 1
    evidence = dict(ag_ui_version="1.0", a2ui_version="1.0",
                    events=events, sample_sse=sse[1], a2ui=interface,
                    resume_request=reply.model_dump(mode="json", by_alias=True),
                    after_gap=gap, after_snapshot=resume, after_replayed_event=duplicate,
                    old_approval_rejected=stale_rejected, current_approval_archive_rows=writes,
                    cancel=json.loads(encoder.encode(cancelled).removeprefix("data: ")))
    args.out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemas", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
