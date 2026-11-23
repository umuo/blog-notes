"""Measure a fresh local server, then terminate only the process this script started."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from probe import request, one


def rss_bytes(pid):
    process = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    value = process.stdout.strip()
    return int(value) * 1024 if value else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("在 -- 之后给出本地引擎启动命令")
    try:
        with request(args.server, "/health"):
            pass
    except OSError:
        pass
    else:
        parser.error("已有服务占用目标地址，请为实验选择独立端口")
    log = args.out.with_suffix(".log")
    with log.open("w") as console:
        began = time.perf_counter()
        server = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT)
        peak_loading = 0
        try:
            while True:
                peak_loading = max(peak_loading, rss_bytes(server.pid))
                if server.poll() is not None:
                    raise RuntimeError(f"服务器启动退出，请查看 {log}")
                if time.perf_counter() - began > 120:
                    raise TimeoutError(f"服务器未在 120 秒内就绪，请查看 {log}")
                try:
                    with request(args.server, "/health"):
                        break
                except OSError:
                    time.sleep(0.1)
            ready = time.perf_counter() - began
            cold = one(args.server, args.model, "mixed", request_id="cold")
            warm = one(args.server, args.model, "mixed", request_id="warm")
            rss_after = rss_bytes(server.pid)
            output = dict(
                ready_seconds=ready, peak_sampled_loading_rss_bytes=peak_loading,
                rss_bytes_after_warm=rss_after, cold=cold, warm=warm,
            )
            args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        finally:
            server.terminate()
            try:
                server.wait(timeout=7)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
