"""Measure a Darwin process policy and a local Docker VM with a fixed CSV job."""

import argparse
import json
import math
import os
import platform
import selectors
import shutil
import socket
import statistics
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns


HERE = Path(__file__).resolve().parent
JOB = HERE / "refund_job.py"
INPUT = HERE / "refunds.csv"
DEFAULT_IMAGE = "python:3.12.12-slim"
REPORT = {"valid_count": 2, "mean": 5.0}


def shell(*args):
    return subprocess.check_output(args, cwd=HERE, text=True, timeout=24).strip()


def measure(job, started):
    with subprocess.Popen(job, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=20):
                process.kill()
                raise TimeoutError("no worker output before timeout")
            marker = process.stdout.readline()
            first = (perf_counter_ns() - started) / 1_000_000
        try:
            body, error = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        if marker != b"READY\n" or process.returncode:
            raise RuntimeError(f"unexpected worker output: {marker!r} {error!r}")
    result = json.loads(body)
    assert result["report"] == REPORT
    return first, result


def stats(values):
    ordered = sorted(values)
    return {"n": len(values), "p50_ms": round(statistics.median(values), 2),
            "p95_ms": round(ordered[math.ceil(0.95 * len(values)) - 1], 2),
            "samples_ms": [round(value, 2) for value in values]}


def clean_container(identifier):
    started = perf_counter_ns()
    shell("docker", "rm", "-f", identifier)
    elapsed = (perf_counter_ns() - started) / 1_000_000
    visible = shell("docker", "ps", "-aq", "--filter", f"id={identifier}")
    assert not visible
    return elapsed


def mount(source, target, readonly=False):
    mode = ",readonly" if readonly else ""
    return f"type=bind,source={source},target={target}{mode}"


def run(image, samples):
    with TemporaryDirectory(prefix="refund-sandbox-") as temp, socket.socket() as listener:
        root = Path(temp).resolve()
        workspace = root / "workspace"
        workspace.mkdir(mode=0o777)
        workspace.chmod(0o777)
        hidden = root / "team-b-marker.txt"
        hidden.write_text("synthetic evidence belonging to another workspace\n")
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = str(listener.getsockname()[1])

        profile = root / "fixture.sb"
        quoted = json.dumps(str(hidden))
        profile.write_text(f"(version 1)\n(allow default)\n(deny file-read* (literal {quoted}))\n(deny network*)\n")
        plain = [sys.executable, str(JOB), str(INPUT), str(workspace / "report.json"),
                 "--other", str(hidden), "--port", port]
        mac = ["/usr/bin/sandbox-exec", "-f", str(profile)]
        _, baseline = measure(plain, perf_counter_ns())
        assert baseline["checks"]["fixture_listener"] == "allowed"
        connection, _ = listener.accept()
        connection.close()
        _, scoped = measure([*mac, *plain], perf_counter_ns())
        assert scoped["checks"] == {"other_file": "denied", "child_other_file": "denied", "fixture_listener": "denied"}

        common = ["docker", "create", "--pull", "never", "--network", "none", "--read-only",
                  "--user", "65534:65534", "--pids-limit", "32", "--memory", "128m",
                  "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                  "--mount", mount(JOB, "/refund_job.py", True),
                  "--mount", mount(INPUT, "/refunds.csv", True),
                  "--mount", mount(workspace, "/workspace"), image]
        running_job = ["python", "/refund_job.py", "/refunds.csv", "/workspace/report.json"]
        identifier = shell(*common, *running_job, "--other", str(hidden), "--port", port, "--readonly", "/refunds.csv")
        try:
            _, container = measure(["docker", "start", "-a", identifier], perf_counter_ns())
            checks = container["checks"]
            assert {key: checks[key] for key in ("other_file", "child_other_file", "fixture_listener", "readonly_mount")} == {
                "other_file": "not_visible", "child_other_file": "not_visible",
                "fixture_listener": "not_reachable", "readonly_mount": "denied",
            }, checks
            assert "lo" in checks["interfaces"] and "eth0" not in checks["interfaces"] and not checks["ipv4_default_route"], checks
        finally:
            clean_container(identifier)
        assert json.loads((workspace / "report.json").read_text()) == REPORT

        cold, clean, host, warm = [], [], [], []
        for _ in range(samples):
            start = perf_counter_ns()
            identifier = shell(*common, *running_job)
            try:
                duration, _ = measure(["docker", "start", "-a", identifier], start)
                cold.append(duration)
            finally:
                clean.append(clean_container(identifier))
            local, _ = measure([*mac, sys.executable, str(JOB), str(INPUT), str(workspace / "report.json")], perf_counter_ns())
            host.append(local)

        resident = shell(*common, "python", "-c", "from time import sleep; sleep(120)")
        try:
            shell("docker", "start", resident)
            for _ in range(samples):
                duration, _ = measure(["docker", "exec", resident, *running_job], perf_counter_ns())
                warm.append(duration)
            shell("docker", "stop", "-t", "1", resident)
        finally:
            clean_container(resident)
        bind_survived = (workspace / "report.json").is_file()
        assert bind_survived
        metadata = {
            "host": f"{platform.system()} {platform.mac_ver()[0]} {platform.machine()}",
            "docker": shell("docker", "version", "--format", "{{.Server.Version}}"),
            "docker_context": shell("docker", "context", "show"),
            "image": shell("docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image),
            "mac_profile": "fixture.sb: v1 targeted deny fixture",
        }
    return {
        "environment": metadata, "boundaries": {"baseline": baseline, "mac_policy": scoped, "docker": container},
        "timings": {"mac_process": stats(host), "docker_cached_create_to_first": stats(cold),
                    "docker_resident_exec_to_first": stats(warm), "docker_remove_ack": stats(clean)},
        "cleanup": {"bind_artifact_remains_after_container_remove": bind_survived,
                    "fixture_root_exists_after_cleanup": root.exists()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--samples", type=int, default=10, choices=range(1, 31))
    args = parser.parse_args()
    if platform.system() != "Darwin" or not shutil.which("docker"):
        parser.error("requires macOS and a running Docker engine")
    if not os.path.isfile("/usr/bin/sandbox-exec"):
        parser.error("requires the legacy macOS sandbox-exec tool")
    print(json.dumps(run(args.image, args.samples), indent=4))


if __name__ == "__main__":
    main()
