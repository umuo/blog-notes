"""Compare pinned OpenClaw and Hermes turns through the same fake provider."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from fixture_api import RULE, fixture


HERE = Path(__file__).resolve().parent
QUESTION = "Read the synthetic refunds.csv and report the count and mean."
CORRECTION = f"{RULE}: In this refund fixture, exclude NA and keep true zeroes. {QUESTION}"
NOTE = f"{RULE}: Exclude NA when summarizing this synthetic refund CSV. Keep true zeros."


def feed(workspace):
    workspace.mkdir()
    shutil.copyfile(HERE / "refunds.csv", workspace / "refunds.csv")


def observations(events, begin, outcome):
    batch = [event for event in events[begin:] if event["event"] != "missing_read_tool"]
    assert [event["event"] for event in batch] == ["read_requested", "reply"], batch
    assert batch[1]["csv_in_tool_result"] and batch[1]["provider_model"] == "refund-fixture"
    return {"reply": outcome, "tool": batch[0]["tool"],
            "memory_in_system": batch[1]["memory_in_system"],
            "correction_in_history": batch[1]["correction_in_history"]}


def openclaw(root, executable):
    workspace = root / "workspace"
    feed(workspace)
    home = root / "home"
    home.mkdir()
    memory = workspace / "MEMORY.md"
    config = root / "fixture.json"
    with fixture(workspace) as (url, events):
        config.write_text(json.dumps({
            "agents": {"defaults": {"workspace": str(workspace), "model": {"primary": "fixture/refund-fixture"}}},
            "models": {"providers": {"fixture": {"baseUrl": url, "api": "openai-completions",
                                               "apiKey": "fixture-only", "models": [
                                                   {"id": "refund-fixture", "name": "Refund Fixture",
                                                    "contextWindow": 65536, "maxTokens": 256}]}}},
            "tools": {"allow": ["read"], "fs": {"workspaceOnly": True}},
        }))
        env = {"PATH": os.environ["PATH"], "LANG": "en_US.UTF-8", "NO_PROXY": "127.0.0.1",
               "OPENCLAW_HOME": str(home), "OPENCLAW_STATE_DIR": str(root / "state"),
               "OPENCLAW_CONFIG_PATH": str(config), "OPENCLAW_DISABLE_TELEMETRY": "1"}

        def turn(number, prompt):
            start = len(events)
            args = ["node", str(executable), "agent", "--local", "--json", "--timeout", "22",
                    "--session-id", str(UUID(int=number)), "--message", prompt]
            proc = subprocess.run(args, env=env, cwd=workspace, capture_output=True, text=True, timeout=35)
            if proc.returncode:
                raise RuntimeError(proc.stderr[-1600:] + proc.stdout[-500:])
            data = json.loads(proc.stdout)["meta"]
            assert data["toolSummary"] == {"calls": 1, "tools": ["read"], "failures": 0}, data.get("toolSummary")
            return observations(events, start, json.loads(data["finalAssistantVisibleText"]))

        turns = {"baseline": turn(1, QUESTION), "corrected": turn(1, CORRECTION),
                 "same_session_followup": turn(1, QUESTION), "fresh_before_publish": turn(2, QUESTION)}
        memory.write_text(f"# Approved fixture note\n\n{NOTE}\n")
        turns["fresh_after_publish"] = turn(3, QUESTION)
        memory.unlink()
        turns["fresh_after_remove"] = turn(4, QUESTION)
        return {"tag": "v2026.11.7", "entry": "agent --local", "application_fixture_wrote_memory": True, "turns": turns}


def hermes(root):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from persistence.conversation.messages import ConversationRequest
    from persistence_assembly import open_persistence, profile_scope
    from run_agent import AIAgent
    from tools import write_approval as approval
    from tools.memory_tool import load_on_disk_store, memory_tool
    from tools.skill_manager_tool import skill_manage
    from tools.skills_tool import skill_view

    workspace = root / "workspace"
    feed(workspace)
    home = Path(os.environ["HERMES_HOME"])
    memory_file = home / "memories" / "MEMORY.md"
    with fixture(workspace) as (url, events), open_persistence(profile_scope(), eager=True) as db:
        def turn(number, prompt):
            start = len(events)
            session_id = f"refund-fixture-{number}"
            history = [message.as_openai_message()
                       for message in db.messages.read_conversation(session_id, ConversationRequest())]
            agent = AIAgent(model="refund-fixture", base_url=url, api_key="fixture-only", provider="custom",
                            api_mode="chat_completions", enabled_toolsets=["file"], quiet_mode=True,
                            skip_context_files=True, skip_background_review=True, max_iterations=3, max_tokens=256,
                            cwd=str(workspace), session_id=session_id, persistence=db)
            # Keep the model-visible file tool to the single read operation for this fixture.
            agent.tools = [tool for tool in agent.tools if tool.get("function", {}).get("name") == "read_file"]
            try:
                result = agent.run_conversation(prompt, conversation_history=history or None)
                assert result["completed"] and result["api_calls"] == 2
                saved = db.messages.read_conversation(session_id, ConversationRequest())
                assert any(row.as_openai_message().get("role") == "tool" for row in saved)
                return observations(events, start, json.loads(result["final_response"]))
            finally:
                agent.close()

        turns = {"baseline": turn(1, QUESTION), "corrected": turn(1, CORRECTION), "same_session_followup": turn(1, QUESTION)}
        store = load_on_disk_store()
        pending = json.loads(memory_tool(action="add", target="memory", content=NOTE, store=store))
        assert pending["staged"] and not memory_file.exists()
        turns["fresh_before_publish"] = turn(2, QUESTION)
        handle_pending_subcommand(approval.MEMORY, ["approve", pending["pending_id"]], memory_store=store)
        assert NOTE in memory_file.read_text() and not approval.list_pending(approval.MEMORY)
        turns["fresh_after_publish"] = turn(3, QUESTION)
        remove = json.loads(memory_tool(action="remove", target="memory", old_text=NOTE, store=store))
        assert remove["staged"] and NOTE in memory_file.read_text()
        handle_pending_subcommand(approval.MEMORY, ["approve", remove["pending_id"]], memory_store=store)
        assert not memory_file.is_file() or NOTE not in memory_file.read_text()
        turns["fresh_after_remove"] = turn(4, QUESTION)

        slug = "refund-csv-fixture"
        body = ("---\nname: refund-csv-fixture\ndescription: Read the synthetic refund CSV safely.\n---\n\n"
                "# Refund CSV\n\nRead data, then verify report count and mean.\n")
        draft = json.loads(skill_manage(action="create", name=slug, content=body))
        assert draft["staged"] and not list((home / "skills").rglob("SKILL.md"))
        handle_pending_subcommand(approval.SKILLS, ["approve", draft["pending_id"]])
        assert "verify report count and mean" in skill_view(slug, preprocess=False)
        edit = json.loads(skill_manage(action="patch", name=slug, old_string="verify report count and mean",
                                       new_string="verify count, mean, and true zeroes"))
        assert edit["staged"] and "verify report count and mean" in skill_view(slug, preprocess=False)
        handle_pending_subcommand(approval.SKILLS, ["reject", edit["pending_id"]])
        assert not approval.list_pending(approval.SKILLS) and "verify report count and mean" in skill_view(slug, preprocess=False)
        assert (home / "state.db").is_file()
        return {"tag": "v0.23.3", "entry": "AIAgent.run_conversation", "profile_state_db": True, "native_transcript_contains_tool_result": True,
                "memory_staged_approved_removed": True, "skill_staged_approved_edit_rejected": True, "turns": turns}


def validate(result):
    expected_counts = (3, 2, 2, 3, 2, 3)
    turns = result["turns"]
    assert tuple(turn["reply"]["count"] for turn in turns.values()) == expected_counts, turns
    for turn, count in zip(turns.values(), expected_counts):
        assert turn["reply"]["mean"] == (5 if count == 2 else 10 / 3)
    assert turns["same_session_followup"]["correction_in_history"]
    assert turns["fresh_after_publish"]["memory_in_system"]
    assert not turns["fresh_before_publish"]["memory_in_system"] and not turns["fresh_after_remove"]["memory_in_system"]


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--openclaw", type=Path)
    choice.add_argument("--hermes", action="store_true")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="refund-framework-") as folder:
        root = Path(folder).resolve()
        if args.openclaw:
            result = openclaw(root, args.openclaw.resolve())
        else:
            home = root / "profile"
            home.mkdir()
            os.environ["HERMES_HOME"] = str(home)
            (home / "config.yaml").write_text("memory:\n  write_approval: true\nskills:\n  write_approval: true\n")
            result = hermes(root)
        validate(result)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
