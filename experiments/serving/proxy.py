"""A loopback-only demo proxy with header, flush, limit and release probes."""

import argparse
import http.client
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


HTML = Path(__file__).with_name("trace.html")


def make_handler(upstream, mode, release, limit):
    origin = urlsplit(upstream)
    slots = threading.BoundedSemaphore(limit)

    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path == "/":
                self.respond(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif self.path in ("/health", "/v1/models"):
                self.relay(None)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            try:
                size = int(self.headers["Content-Length"])
                if not 0 < size <= 1_048_576:
                    raise ValueError()
            except (ValueError, TypeError):
                self.send_error(400)
                return
            body = self.rfile.read(size)
            if not slots.acquire(blocking=False):
                self.respond(429, b'{"error":"concurrency limit"}')
                return
            try:
                self.relay(body)
            finally:
                slots.release()

        def respond(self, status, data, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Release-Id", release)
            self.end_headers()
            self.wfile.write(data)

        def relay(self, body):
            raw_id = self.headers.get("X-Request-Id", "")
            request_id = raw_id if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", raw_id) else uuid.uuid4().hex
            began = time.perf_counter()
            times = {}

            def record(name):
                times[name] = round((time.perf_counter() - began) * 1000, 3)

            upstream = http.client.HTTPConnection(origin.hostname, origin.port, timeout=60)
            sent_headers = False
            outcome = "complete"
            try:
                upstream.request(self.command, self.path, body=body, headers={
                    "Content-Type": "application/json", "X-Request-Id": request_id,
                })
                response = upstream.getresponse()
                record("upstream_headers_ms")
                content_type = response.getheader("Content-Type", "application/octet-stream")
                streaming = content_type.startswith("text/event-stream")
                data = None if streaming and mode == "flush" else response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", content_type)
                self.send_header("X-Request-Id", request_id)
                self.send_header("X-Release-Id", release)
                if data is not None:
                    self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.flush()
                sent_headers = True
                record("client_headers_ms")
                if data is not None:
                    self.wfile.write(data)
                    record("first_write_ms")
                else:
                    for line in iter(response.readline, b""):
                        self.wfile.write(line)
                        self.wfile.flush()
                        if "first_write_ms" not in times:
                            record("first_write_ms")
            except (BrokenPipeError, ConnectionResetError):
                outcome = "connection_closed"
                if not sent_headers:
                    self.respond(502, b'{"error":"connection closed"}')
            except OSError:
                outcome = "upstream_unavailable"
                if not sent_headers:
                    self.respond(502, b'{"error":"upstream unavailable"}')
            finally:
                upstream.close()
                record("elapsed_ms")
                print(json.dumps(dict(id=request_id, outcome=outcome, mode=mode, **times)), flush=True)

        def log_message(self, format, *args):
            return

    return Proxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="http://127.0.0.1:18080")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--mode", choices=("flush", "buffer"), default="flush")
    parser.add_argument("--release", default="refund-01")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port),
                            make_handler(args.upstream, args.mode, args.release, args.limit)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
