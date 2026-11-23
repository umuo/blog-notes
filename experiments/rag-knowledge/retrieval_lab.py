"""Small inspectable retrieval, citation, and evaluation pipeline."""

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path


HERE = Path(__file__).resolve().parent
ALIASES = {
    "reimbursement": "refund", "refunds": "refund", "blank": "missing",
    "missing": "missing", "na": "missing", "archived": "archive",
    "approved": "approval", "amounts": "amount", "retained": "keep",
}


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


@dataclass(frozen=True)
class Chunk:
    source: str
    version: int
    tenant: str
    id: str
    heading: str
    page: int
    start: int
    end: int
    text: str

    @property
    def citation(self):
        return f"{self.source}/v{self.version}/p{self.page}#{self.id}"


def corpus():
    return json.loads((HERE / "corpus.json").read_text())


def chunk_documents(documents, *, width=None):
    for document in documents:
        for section in document["sections"]:
            body = section["text"]
            boundaries = range(0, len(body), width or len(body) or 1)
            for start in boundaries:
                end = min(start + (width or len(body)), len(body))
                identifier = section["id"] if width is None else f"{section['id']}-{start}"
                yield Chunk(
                    document["id"], document["version"], document["tenant"], identifier,
                    section["heading"], section["page"], start, end, body[start:end],
                )


def bm25(query, documents):
    counts = [Counter(tokens(document)) for document in documents]
    lengths = [sum(row.values()) for row in counts]
    average = sum(lengths) / len(lengths) if lengths else 1
    k1, b = 1.2, 0.75
    scores = [0.0] * len(documents)
    for term in sorted(set(tokens(query))):
        frequency = sum(term in row for row in counts)
        inverse = math.log1p((len(documents) - frequency + 0.5) / (frequency + 0.5))
        for index, row in enumerate(counts):
            occurrences = row[term]
            if occurrences:
                length_factor = 1 - b + b * lengths[index] / average
                scores[index] += inverse * occurrences * (k1 + 1) / (occurrences + k1 * length_factor)
    return scores


def vector(text):
    """Handwritten alias map for an offline vector exercise."""
    return Counter(ALIASES.get(term, term) for term in tokens(text))


def cosine(left, right):
    dot = sum(left[key] * right[key] for key in sorted(left.keys() & right.keys()))
    norm = math.sqrt(sum(value ** 2 for value in left.values()) * sum(value ** 2 for value in right.values()))
    return dot / norm if norm else 0.0


def ranked(chunks, scores, limit):
    pairs = sorted(zip(chunks, scores), key=lambda pair: (-pair[1], pair[0].id))
    return [chunk for chunk, score in pairs if score > 0][:limit]


def retrieve(query, chunks, *, tenant="school", mode="hybrid", limit=4):
    visible = [chunk for chunk in chunks if chunk.tenant == tenant]
    texts = [chunk.text for chunk in visible]
    lexical = ranked(visible, bm25(query, texts), limit)
    semantic = ranked(visible, [cosine(vector(query), vector(text)) for text in texts], limit)
    if mode == "bm25":
        return lexical
    if mode == "vector":
        return semantic
    fused = Counter()
    for ranking in (lexical, semantic):
        for position, chunk in enumerate(ranking, 1):
            fused[chunk] += 1 / (60 + position)
    return sorted(fused, key=lambda chunk: (-fused[chunk], chunk.id))[:limit]


def pair_score(query, chunk):
    query_terms = set(vector(query))
    content = set(vector(chunk.text))
    score = len(query_terms & content)
    if {"approval", "archive"} <= query_terms and {"approval", "archive"} <= content:
        score += 3
        if "only" in content:
            score += 2
    if "days" in query_terms and "days" in content:
        score += 4
    return score


def rerank(query, candidates):
    return sorted(candidates, key=lambda chunk: -pair_score(query, chunk))


def toy_cross_attention():
    """Expose one query-to-document attention row with handwritten vectors."""
    query = (1, 0)
    documents = {"exact_action": [(1, 0), (0, 1)], "generic_topic": [(0, 1), (0, 1)]}
    output = {}
    for name, keys in documents.items():
        logits = [sum(a * b for a, b in zip(query, key)) / math.sqrt(len(query)) for key in keys]
        exponentials = [math.exp(logit) for logit in logits]
        weights = [value / sum(exponentials) for value in exponentials]
        output[name] = {"logits": logits, "weights": weights, "first_value_weight": weights[0]}
    return output


def grounded(question, evidence):
    """Deterministic extractive answerer used to exercise the interface."""
    terms = [term.lower() for term in question["terms"]]
    if not terms or not all(any(term in chunk.text.lower() for chunk in evidence) for term in terms):
        return None
    selected = [chunk for chunk in evidence if any(term in chunk.text.lower() for term in terms)]
    statements = []
    for chunk in selected:
        sentences = re.split(r"(?<=\.) ", chunk.text)
        matching = [sentence for sentence in sentences if any(term in sentence.lower() for term in terms)]
        statements.append({"text": " ".join(matching), "citation": chunk.citation})
    return statements


def evaluate(question, candidates, answer):
    gold = set(question["evidence"])
    actual = {chunk.id for chunk in candidates}
    evidence_by_ref = {chunk.citation: chunk for chunk in candidates}
    cited = [evidence_by_ref.get(statement["citation"]) for statement in answer or []]
    expected = question["answer"]
    gain = sum(1 / math.log2(rank + 1) for rank, chunk in enumerate(candidates, 1) if chunk.id in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), len(candidates)) + 1))
    return {
        "ndcg": gain / ideal if ideal else None,
        "recall": len(gold & actual) / len(gold) if gold else None,
        "mrr": next((1 / rank for rank, chunk in enumerate(candidates, 1) if chunk.id in gold), 0) if gold else None,
        "faithful": all(chunk is not None and statement["text"] in chunk.text
                        for statement, chunk in zip(answer or [], cited)) if answer else None,
        "citations": all(chunk is not None and chunk.id in gold for chunk in cited) if answer else None,
        "answer_equal": " ".join(statement["text"] for statement in answer) == expected if answer else expected is None,
        "refused": answer is None,
    }


def tenant_case(chunks):
    by_id = {chunk.id: chunk for chunk in chunks}
    identities = {"school": {"policy": 2}, "partner": {"partner": 1}}
    old_cache = {"school": by_id["retention"].citation}

    def authorize(tenant, ref):
        source, version, _ = ref.split("/", 2)
        return int(version[1:]) == identities[tenant].get(source)

    ref = old_cache["school"]
    query = "keep archived refund report days"
    initial = [chunk.id for chunk in retrieve(query, chunks, limit=2) if authorize("school", chunk.citation)]
    allowed_before = authorize("school", ref)
    other_tenant = authorize("partner", ref)
    identities["school"].pop("policy")
    denied_stale = not authorize("school", ref)  # same stale index and old cache
    during_update = [chunk.id for chunk in retrieve(query, [c for c in chunks if authorize("school", c.citation)], limit=2)]
    old_cache.pop("school")
    index = [chunk for chunk in chunks if chunk.source != "policy"]
    after_purge = all(chunk.source != "policy" for chunk in index)
    republished = replace(by_id["retention"], version=3, text="Keep an archived refund report for 45 days after approval.")
    identities["school"]["policy"] = 3
    updated = retrieve(query, [*index, republished], limit=1)
    assert not authorize("school", ref) and authorize("school", republished.citation)
    assert updated == [republished] and denied_stale and not other_tenant
    return {"initial": initial, "before": allowed_before, "other_tenant": other_tenant,
            "during_update": during_update, "stale_denied": denied_stale,
            "old_cache_present": bool(old_cache), "purged": after_purge, "new_ref": updated[0].citation}


def main():
    docs = corpus()
    chunks = list(chunk_documents(docs))
    questions = json.loads((HERE / "questions.json").read_text())
    by_id = {chunk.id: chunk for chunk in chunks}
    result = {}
    for question in questions:
        query = question["query"]
        candidates = retrieve(query, chunks)
        reordered = rerank(query, candidates)
        known = [by_id[identifier] for identifier in question["evidence"]]
        answer = grounded(question, reordered[:2])
        result[question["id"]] = {
            "initial": [chunk.id for chunk in candidates],
            "reordered": [chunk.id for chunk in reordered],
            "pair_scores": {chunk.id: pair_score(query, chunk) for chunk in candidates},
            "before_top1": evaluate(question, candidates[:1], grounded(question, candidates[:1])),
            "after_top1": evaluate(question, reordered[:1], grounded(question, reordered[:1])),
            "before": evaluate(question, candidates[:2], grounded(question, candidates[:2])),
            "after": evaluate(question, reordered[:2], answer),
            "answer": answer,
            "gold_only": evaluate(question, known, grounded(question, known)),
        }
    query = "blank reimbursement"
    vector_case = {mode: [chunk.id for chunk in retrieve(query, chunks, mode=mode, limit=3)]
                   for mode in ("bm25", "vector", "hybrid")}
    assert not vector_case["bm25"] and vector_case["vector"][0] == "missing"
    assert result["archive"]["after_top1"]["recall"] == 1 and result["archive"]["before_top1"]["recall"] == 0
    assert result["retention"]["after"]["answer_equal"] and result["unknown"]["after"]["refused"]
    fixed = list(chunk_documents(docs[:1], width=35))
    changed = [replace(chunk, version=3, text=chunk.text.replace("CNY", "USD")) for chunk in chunks if chunk.source == "dictionary"]
    rebuilt = [chunk for chunk in chunks if chunk.source != "dictionary"] + changed
    assert "USD" in next(chunk.text for chunk in rebuilt if chunk.id == "amount")
    assert all(chunk.version == 3 for chunk in rebuilt if chunk.source == "dictionary")
    print(json.dumps({"vector_case": vector_case, "fixed_amount": [asdict(c) for c in fixed if c.id.startswith("amount-")],
                      "structure_amount": asdict(by_id["amount"]), "index_changed": [c.citation for c in changed],
                      "toy_cross_attention": toy_cross_attention(), "questions": result, "tenant": tenant_case(chunks)}, indent=4))


if __name__ == "__main__":
    main()
