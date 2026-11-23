"""Exercise local routing labels, health and concurrent request rejection."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError

from probe import payload, request


def send(server, model, stream=False):
    body = payload(model, "mixed", stream, max_tokens=256 if stream else 48)
    if stream:
        body["messages"] = [{"role": "user", "content": "Print numbers 1 to 500, one per line."}]
    return request(server, "/v1/chat/completions", body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="http://127.0.0.1:18139")
    parser.add_argument("--new", default="http://127.0.0.1:18140")
    parser.add_argument("--model", default="default_model")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    health = {}
    for name, server in (("old", args.old), ("new", args.new)):
        with request(server, "/health") as response:
            health[name] = response.status
    changes = []
    for step, server in (("old", args.old), ("promote", args.new), ("rollback", args.old)):
        with send(server, args.model) as response:
            content = json.load(response)["choices"][0]["message"]["content"]
            changes.append(dict(step=step, release=response.headers["X-Release-Id"], text=content))
    limited = None
    with send(args.old, args.model, stream=True) as held:
        first = held.readline()
        if not first:
            raise RuntimeError("长请求未开始输出")
        try:
            with send(args.old, args.model):
                raise RuntimeError("并发闸门未拒绝新的请求")
        except HTTPError as error:
            limited = error.code
            if limited != 429:
                raise
    start = time.perf_counter()
    while True:
        try:
            with send(args.old, args.model) as response:
                recovered_status = response.status
                break
        except HTTPError as error:
            if error.code != 429 or time.perf_counter() - start > 10:
                raise
            time.sleep(0.05)
    assert changes[0]["release"] == changes[2]["release"] != changes[1]["release"]
    args.out.write_text(json.dumps(dict(health=health, route=changes, concurrent_status=limited,
                                        recovery_status=recovered_status),
                                   indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
