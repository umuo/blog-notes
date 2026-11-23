"""Compute a fixed synthetic refund report and optionally check declared boundaries."""

import argparse
import csv
import errno
import json
import socket
import subprocess
import sys
from pathlib import Path


def boundary_cases(other, port, readonly):
    def status(operation):
        try:
            operation()
        except PermissionError:
            return "denied"
        except FileNotFoundError:
            return "not_visible"
        except ConnectionRefusedError:
            return "not_reachable"
        except OSError as error:
            if error.errno != errno.EROFS:
                raise
            return "denied"
        return "allowed"

    child = subprocess.run(
        [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).read_bytes()", str(other)],
        capture_output=True, timeout=4,
    )
    if child.returncode == 0:
        child_state = "allowed"
    elif b"PermissionError" in child.stderr:
        child_state = "denied"
    elif b"FileNotFoundError" in child.stderr:
        child_state = "not_visible"
    else:
        raise RuntimeError(child.stderr.decode())

    def dial():
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass

    result = {"other_file": status(other.read_bytes), "child_other_file": child_state,
              "fixture_listener": status(dial)}
    if readonly is not None:
        result["readonly_mount"] = status(lambda: readonly.open("rb+").close())
    if Path("/sys/class/net").is_dir():
        result["interfaces"] = {entry.name: (entry / "operstate").read_text().strip()
                                for entry in sorted(Path("/sys/class/net").iterdir()) if (entry / "operstate").is_file()}
        result["ipv4_default_route"] = any(line.split()[1] == "00000000"
                                            for line in Path("/proc/net/route").read_text().splitlines()[1:])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--other", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--readonly", type=Path)
    args = parser.parse_args()
    print("READY", flush=True)
    with args.input.open(newline="") as source:
        rows = [int(row["refund"]) for row in csv.DictReader(source) if row["refund"] != "NA"]
    report = {"valid_count": len(rows), "mean": sum(rows) / len(rows)}
    args.output.write_text(json.dumps(report) + "\n")
    data = {"report": report, "python": sys.version.split()[0]}
    if args.other is not None:
        data["checks"] = boundary_cases(args.other, args.port, args.readonly)
    print(json.dumps(data), flush=True)


if __name__ == "__main__":
    main()
