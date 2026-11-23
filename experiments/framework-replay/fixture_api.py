"""A loopback-only Chat Completions fixture for the synthetic refund exercise."""

import json
import re
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread


RULE = "REFUND_SKIP_NA"


def contains_fixture(value):
    if isinstance(value, list):
        return any(contains_fixture(part) for part in value)
    if isinstance(value, dict):
        return any(contains_fixture(value.get(key)) for key in ("content", "text"))
    if not isinstance(value, str):
        return False
    if value.lstrip().startswith(("{", "[")):
        try:
            return contains_fixture(json.loads(value))
        except ValueError:
            pass
    lines = [re.sub(r"^\s*\d+\|", "", line).strip() for line in value.splitlines()]
    return any(lines[index:index + 4] == ["refund", "4", "NA", "6"] for index in range(len(lines) - 3))


@contextmanager
def fixture(workspace):
    observations = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, value, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.reply(404, b"{}", "application/json")
                return
            size = int(self.headers.get("Content-Length", 0))
            if size > 3_000_000:
                self.reply(400, b"{}", "application/json")
                return
            request = json.loads(self.rfile.read(size))
            messages = request["messages"]
            user_index = max(i for i, msg in enumerate(messages) if msg["role"] == "user")
            system_has_rule = any(RULE in json.dumps(msg.get("content")) for msg in messages if msg["role"] in {"system", "developer"})
            history_has_rule = any(RULE in json.dumps(msg.get("content")) for msg in messages if msg["role"] == "user")
            tool_results = [msg for msg in messages[user_index + 1:] if msg["role"] == "tool"]
            names = [tool["function"]["name"] for tool in request.get("tools", []) if tool.get("type") == "function"]
            if tool_results:
                saw_rows = contains_fixture(tool_results[-1].get("content"))
                observations.append({"event": "reply", "provider_model": request["model"],
                                     "memory_in_system": system_has_rule, "correction_in_history": history_has_rule,
                                     "csv_in_tool_result": saw_rows})
                if not saw_rows:
                    answer = {"error": "fixture did not see the synthetic CSV"}
                elif system_has_rule or history_has_rule:
                    answer = {"count": 2, "mean": 5}
                else:
                    answer = {"count": 3, "mean": 10 / 3}
                message = {"role": "assistant", "content": json.dumps(answer)}
                reason = "stop"
            elif any(name in names for name in ("read", "read_file")):
                tool = "read" if "read" in names else "read_file"
                observations.append({"event": "read_requested", "tool": tool})
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"index": 0, "id": "call_refunds", "type": "function",
                     "function": {"name": tool, "arguments": json.dumps({"path": str(workspace / "refunds.csv")})}}
                ]}
                reason = "tool_calls"
            else:
                observations.append({"event": "missing_read_tool", "tools": names})
                message = {"role": "assistant", "content": '{"error":"read tool unavailable"}'}
                reason = "stop"
            base = {"id": "chatcmpl-fixture", "created": 1, "model": request["model"]}
            usage = {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
            if request.get("stream"):
                delta = {key: value for key, value in message.items() if value is not None}
                parts = [
                    {**base, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {**base, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
                    {**base, "object": "chat.completion.chunk", "choices": [], "usage": usage},
                ]
                wire = "".join(f"data: {json.dumps(part)}\n\n" for part in parts) + "data: [DONE]\n\n"
                self.reply(200, wire.encode(), "text/event-stream")
            else:
                message.get("tool_calls", [{}])[0].pop("index", None)
                result = {**base, "object": "chat.completion",
                          "choices": [{"index": 0, "message": message, "finish_reason": reason}], "usage": usage}
                self.reply(200, json.dumps(result).encode(), "application/json")

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", observations
        finally:
            server.shutdown()
            thread.join()
