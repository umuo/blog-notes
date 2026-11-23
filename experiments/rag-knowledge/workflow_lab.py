"""Reproduce evidence gaps, multi-hop search, and Wiki maintenance."""

import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from retrieval_lab import HERE, chunk_documents, corpus, evaluate, grounded, rerank, retrieve


def multi_hop(chunks, budget):
    query = "refund archive requests"
    trace = []
    evidence = {}
    answer = None
    while len(trace) < budget and query not in {call["query"] for call in trace}:
        candidates = retrieve(query, chunks, limit=4)
        if query == "refund archive requests":
            candidates = [entry for entry in candidates if entry.heading.startswith("Support routing")]
        ranked = rerank(query, candidates)[:1]
        trace.append({"query": query, "hits": [entry.id for entry in ranked]})
        for entry in ranked:
            evidence[entry.id] = entry
            if entry.id == "alias":
                found = re.search(r"handled by the (.+?)\.", entry.text)
                if found:
                    query = f"{found[1]} opens weekdays"
            if entry.id == "window":
                found = re.search(r"\d{2}:\d{2}", entry.text)
                if found and "alias" in evidence:
                    answer = found[0]
        if answer:
            break
    return {"answer": answer, "citations": [entry.citation for entry in evidence.values()] if answer else [],
            "calls": len(trace), "trace": trace, "stop": "complete" if answer else "budget" if len(trace) == budget else "repeat"}


def source_link(entry):
    return f"raw/{entry.source}-v{entry.version}-p{entry.page}-{entry.id}.md"


def wiki_pages(directory, chunks):
    definitions = {"refund-amount": ("amount", "missing"), "archive-policy": ("approval", "retention")}
    by_id = {chunk.id: chunk for chunk in chunks}
    (directory / "raw").mkdir(exist_ok=True)
    for entry in chunks:
        raw = directory / source_link(entry)
        if not raw.exists():
            raw.write_text(f"# {entry.heading}\n\n{entry.text}\n")
    for topic, sources in definitions.items():
        page = [f"# {topic}", ""]
        for source in sources:
            entry = by_id.get(source)
            if entry:
                page.append(f"- [{entry.citation}]({source_link(entry)}) {entry.text}")
            else:
                page.append(f"- Source unavailable: {source}")
        (directory / f"{topic}.md").write_text("\n".join(page) + "\n")


def lint(directory, chunks):
    valid = {chunk.citation for chunk in chunks}
    paths = sorted(directory.glob("*.md"))
    topics = Counter(path.read_text().splitlines()[0] for path in paths)
    return {
        "duplicates": sorted(topic for topic, count in topics.items() if count > 1),
        "invalid_refs": sorted({ref for path in paths
                                for ref, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", path.read_text())
                                if ref not in valid or not (directory / link).is_file()}),
    }


def wiki_case(chunks):
    with TemporaryDirectory() as root:
        directory = Path(root)
        wiki_pages(directory, chunks)
        policy = directory / "archive-policy.md"
        before = policy.read_text()
        duplicate = directory / "archive-copy.md"
        duplicate.write_text(before)
        conflicting = [replace(chunk, version=3, text=chunk.text.replace("30 days", "45 days"))
                       if chunk.id == "retention" else chunk for chunk in chunks]
        versions = [chunk for chunk in chunks + conflicting if chunk.id == "retention"]
        conflict = [{"ref": entry.citation, "days": re.search(r"(\d+) days", entry.text)[1]} for entry in versions]
        assert len({item["days"] for item in conflict}) == 2
        lint_before = lint(directory, conflicting)
        duplicate.unlink()
        wiki_pages(directory, conflicting)
        after = policy.read_text()
        assert "45 days" in after and "30 days" not in after
        retention_link = next(link for ref, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", after)
                              if ref.endswith("#retention"))
        verified_source = (directory / retention_link).read_text().splitlines()[-1]
        assert "45 days" in verified_source
        available = [chunk for chunk in conflicting if chunk.id != "retention"]
        (directory / retention_link).unlink()
        lint_deleted = lint(directory, available)
        wiki_pages(directory, available)
        deleted = policy.read_text()
        assert "Source unavailable: retention" in deleted
        assert lint(directory, available) == {"duplicates": [], "invalid_refs": []}
        return {"before": before, "version_conflict": conflict, "lint_before": lint_before,
                "after_update": after, "verified_source": verified_source, "lint_after_source_removed": lint_deleted, "after_rebuild": deleted}


def diagnostics(chunks):
    questions = {item["id"]: item for item in json.loads((HERE / "questions.json").read_text())}
    cross = questions["cross"]
    initial = retrieve(cross["query"], chunks)
    by_id = {chunk.id: chunk for chunk in chunks}
    gold = [by_id[identifier] for identifier in cross["evidence"]]
    retention = questions["retention"]
    available = [by_id["retention"]]
    wrong_answer = [{"text": "Keep an archived refund report for 90 days after approval.", "citation": available[0].citation}]
    wrong_reference = [{"text": available[0].text, "citation": by_id["approval"].citation}]
    return {
        "retrieval_budget_1": evaluate(cross, initial[:1], grounded(cross, initial[:1])),
        "oracle_2": evaluate(cross, gold, grounded(cross, gold)),
        "wrong_claim_right_ref": evaluate(retention, available, wrong_answer),
        "correct_claim_wrong_ref": evaluate(retention, [*available, by_id["approval"]], wrong_reference),
    }


def main():
    chunks = list(chunk_documents(corpus()))
    one, two = multi_hop(chunks, 1), multi_hop(chunks, 2)
    repeated = multi_hop([chunk for chunk in chunks if chunk.id != "alias"], 3)
    assert one["answer"] is None and two["answer"] == "09:00" and repeated["stop"] == "repeat"
    report = diagnostics(chunks)
    assert report["retrieval_budget_1"]["recall"] == 0.5
    assert report["oracle_2"]["faithful"] and report["oracle_2"]["answer_equal"]
    assert not report["wrong_claim_right_ref"]["faithful"]
    print(json.dumps({"multi_hop": {"one": one, "two": two, "repeat": repeated},
                      "diagnostics": report, "wiki": wiki_case(chunks)}, indent=4))


if __name__ == "__main__":
    main()
