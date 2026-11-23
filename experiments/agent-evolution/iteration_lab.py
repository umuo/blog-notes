"""Demonstrate fixed feedback and candidate selection with no model."""

import json
from collections import Counter
from fractions import Fraction


TRAIN_ROUNDS = (
    (("mixed", (4, "NA", 6), (2, Fraction(5))),),
    (("empty", ("NA", "NA"), (0, None)),),
)
HOLDOUT = (
    ("mixed", (2, "NA", 8, 8), (3, Fraction(6))),
    ("mixed", (0, "NA", 8), (2, Fraction(4))),
    ("empty", ("NA",), (0, None)),
    ("numeric", (3, 9), (2, Fraction(6))),
)
ANALOGY_HOLDOUT = (
    ((4, "NA", 6), (2, Fraction(5))),
    (("NA", "NA"), (0, None)),
    ((0, 8), (2, Fraction(4))),
    ((2, "NA", 8, 8), (3, Fraction(6))),
)


def apply(values, rule):
    keep = [0 if value == "NA" else value for value in values if rule != "skip_missing" or value != "NA"]
    return len(keep), Fraction(sum(keep), len(keep)) if keep else None


def current_rule(votes, category):
    return "skip_missing" if votes[category] else "zero_fill"


def holdout_results(votes):
    return [apply(values, current_rule(votes, category)) == expected for category, values, expected in HOLDOUT]


def iterations():
    votes = Counter()
    runs = [{"round": 0, "held_out": holdout_results(votes)}]
    for number, dataset in enumerate(TRAIN_ROUNDS, 1):
        accepted = []
        for category, values, expected in dataset:
            existing = current_rule(votes, category)
            for candidate in (existing, "skip_missing"):
                if apply(values, candidate) == expected:
                    if candidate == "skip_missing":
                        votes[category] += 1
                    accepted.append({"category": category, "rule": candidate})
                    break
        runs.append({"round": number, "accepted": accepted, "held_out": holdout_results(votes),
                     "rule_votes": dict(votes)})
    assert [sum(row["held_out"]) for row in runs] == [1, 3, 4]
    return runs


def proxy_judge():
    values = (4, "NA", 6)
    expected = 2, Fraction(5)
    choices = {"zero_fill": "A polished answer with a heading, an explanation, and a short conclusion.",
               "skip_missing": "Exclude the missing item."}
    by_style = max(choices, key=lambda key: len(choices[key].split()))
    by_oracle, = (rule for rule in choices if apply(values, rule) == expected)
    assert by_style != by_oracle
    return {"proxy_preferred": by_style, "oracle_preferred": by_oracle,
            "description_word_count": {key: len(value.split()) for key, value in choices.items()}}


def search_analogy():
    def two_pass(values):
        total = sum(value for value in values if value != "NA")
        count = sum(value != "NA" for value in values)
        return (count, Fraction(total, count) if count else None), 2 * len(values)

    def one_pass(values):
        total = count = 0
        for value in values:
            if value != "NA":
                total += value
                count += 1
        return (count, Fraction(total, count) if count else None), len(values)

    def wrong_policy(values):
        return apply(values, "zero_fill"), len(values)

    ranking = {}
    for name, candidate in (("two_pass", two_pass), ("zero_fill", wrong_policy), ("one_pass", one_pass)):
        pairs = [candidate(values) for values, _ in ANALOGY_HOLDOUT]
        ranking[name] = {"correct": sum(actual == expected for (actual, _), (_, expected) in zip(pairs, ANALOGY_HOLDOUT)),
                         "visits": sum(visits for _, visits in pairs)}
    valid = [name for name, score in ranking.items() if score["correct"] == len(ANALOGY_HOLDOUT)]
    winner = min(valid, key=lambda name: ranking[name]["visits"])
    assert ranking == {"two_pass": {"correct": 4, "visits": 22}, "zero_fill": {"correct": 1, "visits": 11},
                       "one_pass": {"correct": 4, "visits": 11}}
    assert winner == "one_pass"
    return {"candidates": ranking, "selected": winner}


if __name__ == "__main__":
    print(json.dumps({"iterations": iterations(), "judge": proxy_judge(), "search": search_analogy()}, indent=4))
