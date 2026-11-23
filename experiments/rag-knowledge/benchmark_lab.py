"""Evaluate a fixed public-dataset slice without model inference."""

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from retrieval_lab import bm25, tokens


def scifact(directory):
    with duckdb.connect() as db:
        corpus = db.execute("SELECT _id, title, text FROM read_parquet(?) ORDER BY _id", [str(directory / "corpus.parquet")]).fetchall()
        queries = dict(db.execute("SELECT _id, text FROM read_parquet(?)", [str(directory / "queries.parquet")]).fetchall())
    relevance = defaultdict(dict)
    with (directory / "test.tsv").open() as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if int(row["score"]) > 0:
                relevance[row["query-id"]][row["corpus-id"]] = int(row["score"])
    sample_ids = sorted(relevance, key=int)[:10]
    texts = [f"{title or ''} {body}" for _, title, body in corpus]
    terms = [set(tokens(text)) for text in texts]

    def metrics(scores, gold):
        ordered = sorted(zip(corpus, scores), key=lambda item: (-item[1], item[0][0]))[:10]
        gains = [gold.get(row[0], 0) for row, _ in ordered]
        ideal = sorted(gold.values(), reverse=True)[:10]

        def dcg(values):
            return sum(value / math.log2(rank + 1) for rank, value in enumerate(values, 1))

        return sum(value > 0 for value in gains) / len(gold), dcg(gains) / dcg(ideal)

    output = {}
    for name in ("overlap", "bm25"):
        scores = []
        for identifier in sample_ids:
            query = queries[identifier]
            term_set = set(tokens(query))
            similarity = bm25(query, texts) if name == "bm25" else [len(term_set & doc) for doc in terms]
            scores.append(metrics(similarity, relevance[identifier]))
        output[name] = {"recall_at_10": sum(row[0] for row in scores) / len(scores),
                        "ndcg_at_10": sum(row[1] for row in scores) / len(scores)}
    return {"corpus_count": len(corpus), "query_ids": sample_ids, "metrics": output}


def hotpot(directory):
    data = json.loads((directory / "hotpot-rows.json").read_text())
    examples = [row["row"] for row in data["rows"]]
    gold, answers, evidence = [], {}, {}
    for row in examples:
        identifier = row["id"]
        support = list(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
        gold.append({"_id": identifier, "answer": row["answer"], "supporting_facts": support})
        question_terms = set(tokens(row["question"]))
        candidates = [
            (title, position, len(question_terms & set(tokens(sentence))))
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
            for position, sentence in enumerate(sentences)
        ]
        ordered = sorted(candidates, key=lambda row: (-row[2], row[0], row[1]))
        evidence[identifier] = [candidate[:2] for candidate in ordered[:2]]
        answers[identifier] = "yes"

    with TemporaryDirectory() as root:
        gold_file = Path(root) / "expected.json"
        predicted_file = Path(root) / "predicted.json"
        gold_file.write_text(json.dumps(gold))
        experiments = {
            "constant_yes_lexical_support": {"answer": answers, "sp": evidence},
            "gold_answer_lexical_support": {"answer": {row["_id"]: row["answer"] for row in gold}, "sp": evidence},
            "gold_answer_and_support": {"answer": {row["_id"]: row["answer"] for row in gold},
                                        "sp": {row["_id"]: row["supporting_facts"] for row in gold}},
        }
        results = {}
        for name, predictions in experiments.items():
            predicted_file.write_text(json.dumps(predictions))
            evaluator = directory / "hotpot_evaluate_v1.py"
            output = subprocess.check_output([sys.executable, str(evaluator), str(predicted_file), str(gold_file)], text=True)
            results[name] = ast.literal_eval(output.strip().splitlines()[-1])
    assert results["gold_answer_and_support"]["joint_f1"] == 1
    return {"ids": [row["id"] for row in examples], "metrics": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    directory = args.data_dir
    files = ("corpus.parquet", "queries.parquet", "test.tsv", "hotpot-rows.json", "hotpot_evaluate_v1.py")
    print(json.dumps({"inputs_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files},
                      "scifact": scifact(directory), "hotpot_distractor": hotpot(directory)}, indent=4))


if __name__ == "__main__":
    main()
