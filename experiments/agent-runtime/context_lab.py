"""Context, provenance, log windows and scoped-memory replay without an LLM."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from runtime import CSV, digest, stats


RULE = "归档前必须确认当前报告摘要。"
USER = "统计退款 CSV 的有效数量和均值。"
REPORT = stats(CSV)


def prompt(phase):
    system = {"role": "system", "content": RULE}
    user = {"role": "user", "content": USER}
    evidence = {"role": "tool", "tool_call_id": "csv-1", "content": {
        "source": "refund.csv", "status": "ok", "report": REPORT,
        "untrusted_note": "仅供引用的 CSV 字段内容",
    }}
    call = {"role": "assistant", "tool_calls": [{"id": "csv-1", "name": "csv_stats",
                                                 "arguments": {"column": "refund"}}]}
    if phase == "prepare":
        return [system, user]
    if phase == "report":
        return [system, user, call, evidence]
    if phase == "approve":
        state = {"role": "system", "content": {"report_hash": digest(REPORT), "approval": "missing"}}
        return [system, state, user, call, evidence]
    raise ValueError("unknown phase")


def trim():
    limits = dict(window=120, output=30, margin=10)
    mandatory = 20 + 18 + 14
    room = limits["window"] - limits["output"] - limits["margin"] - mandatory
    groups = [("old_weather", 21), ("csv-1 request and observation", 18), ("small_preview", 10)]
    used = sum(cost for _, cost in groups)
    removed = []
    for name, cost in groups:
        if used <= room:
            break
        removed.append(name)
        used -= cost
    retained = [name for name, _ in groups if name not in removed]
    assert room == 28 and removed == ["old_weather"] and used == room
    return dict(**limits, mandatory=mandatory, available=room, optional_before=49, optional_after=used,
                removed=removed, retained=retained, report_before=REPORT, report_after=REPORT)


def reconcile_summary():
    report_hash = digest(REPORT)
    summary = {"sources": ["refund.csv#L2-L4", "csv-1"], "report_hash": report_hash,
               "decisions": ["ignore NA"], "pending": ["archive"]}
    # A prior view might say "pending" after the actual write was committed.
    persisted = {"archive_status": "completed", "action_id": "refund-01/archive/" + report_hash}
    resume = "skip_archive" if persisted["archive_status"] == "completed" else "inspect_receipt"
    assert resume == "skip_archive"
    return dict(summary=summary, authoritatively_persisted=persisted, resume_decision=resume)


def logs(folder):
    resource = folder / "worker.log"
    candidate_rows = []
    offsets = {}
    hasher = hashlib.sha256()
    with resource.open("wb") as file:
        for number in range(1, 50_001):
            error = number in (24_731, 34_912)
            line = f"{number:05d} {'ERROR' if error else 'INFO'} {'refund row total mismatch' if error else 'row read'}\n".encode()
            if error:
                offsets[number] = file.tell()
                candidate_rows.append({"line": number, "kind": "refund_mismatch"})
            file.write(line)
            hasher.update(line)
    with resource.open("rb") as file:
        file.seek(offsets[24_731])
        first = file.readline().decode().strip()
    summary = dict(lines=50_000, byte_count=resource.stat().st_size, error_count=len(candidate_rows),
                   resource_sha256=hasher.hexdigest(), candidates=candidate_rows)
    wire = json.dumps(summary, ensure_ascii=False).encode()
    assert first.startswith("24731 ERROR") and len(wire) < 512
    return dict(**summary, returned_summary_bytes=len(wire), read_back=first)


@dataclass
class Preference:
    tenant: str
    user: str
    key: str
    value: str
    version: int
    source: str
    deleted: bool = False


def memories():
    stored = {}
    stale_index = []

    def write(tenant, user, key, value, source, previous):
        scope = (tenant, user, key)
        prior = stored.get(scope)
        if source != "explicit_user" or (prior.version if prior else 0) != previous:
            return False
        stored[scope] = Preference(*scope, value, previous + 1, source)
        stale_index.append((scope, previous + 1))
        return True

    def find(tenant, user):
        results = []
        for scope, index_version in stale_index:
            current = stored[scope]
            if scope[:2] == (tenant, user) and current.version == index_version and not current.deleted:
                results.append(asdict(current))
        return results

    assert write("a", "u", "number_format", "two-decimals", "explicit_user", 0)
    bad = write("a", "u", "number_format", "guess", "model_inference", 1)
    assert write("a", "u", "number_format", "one-decimal", "explicit_user", 1)
    assert write("b", "u", "number_format", "percent", "explicit_user", 0)
    outdated = write("a", "u", "number_format", "stale", "explicit_user", 1)
    before = find("a", "u")
    found_other = find("b", "u")
    record = stored[("a", "u", "number_format")]
    record.deleted = True
    record.value = ""
    record.version += 1
    after = find("a", "u")
    assert len(before) == len(found_other) == 1 and before[0]["value"] == "one-decimal"
    assert not after and not outdated and not bad
    return dict(previous=before, other_scope=found_other, after_delete=after,
                inferred_written=bad, stale_write_succeeded=outdated, index_entries_remaining=len(stale_index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshots = {phase: prompt(phase) for phase in ("prepare", "report", "approve")}
    digest_values = {phase: digest(view) for phase, view in snapshots.items()}
    assert len(set(digest_values.values())) == 3 and snapshots["approve"][1]["content"]["approval"] == "missing"
    with tempfile.TemporaryDirectory() as directory:
        data = dict(snapshots=snapshots, snapshot_hashes=digest_values, trim=trim(),
                    compaction=reconcile_summary(), log=logs(Path(directory)), memory=memories())
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
