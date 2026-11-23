"""退款表上的可复现教学实验。只依赖 Python 标准库。"""

import argparse
import json
import math
import random
import statistics
from fractions import Fraction
from pathlib import Path


def answer(rows):
    valid = [value for value in rows if value is not None]
    return len(valid), Fraction(sum(valid), len(valid)) if valid else None


def group(rows):
    return tuple(sorted(value for value in rows if value is not None))


def verify(rows, candidate):
    count, mean = candidate
    return type(count) is int and count == answer(rows)[0] and mean == answer(rows)[1]


def online_demo():
    """从当前分类策略在线采样。每组仅更新一次，更新时概率比为 1。"""
    rng = random.Random(2026)
    train = [(0, 1, None), (0, None), (3, None), (3, 5), (None, None, None)]
    heldout = [(4, None, 6), (None, 0), (6, 0, None), (None, None)]

    def completion(rows, action):
        if action == 0:
            return len(rows), Fraction(sum(value or 0 for value in rows), len(rows))
        if action == 1:
            return answer([value for value in rows if value])
        return answer(rows)

    def distribution(logits):
        exp = [math.exp(score - max(logits)) for score in logits]
        return [value / sum(exp) for value in exp]

    def greedy_pass(logits):
        action = max(range(len(logits)), key=logits.__getitem__)
        return sum(verify(rows, completion(rows, action)) for rows in heldout)

    logits = [0.2, 0.1, 0.0]
    before = greedy_pass(logits)
    initial_probabilities = distribution(logits)
    mean_rewards = []
    zero_spread = 0
    for step in range(30):
        rows = train[step % len(train)]
        probabilities = distribution(logits)
        actions = rng.choices(range(len(logits)), weights=probabilities, k=4)
        rewards = [int(verify(rows, completion(rows, action))) for action in actions]
        center = statistics.mean(rewards)
        std = statistics.pstdev(rewards)
        mean_rewards.append(center)
        if std == 0:
            zero_spread += 1
            continue
        for index, probability in enumerate(probabilities):
            gradient = statistics.mean(
                (reward - center) / std * (int(action == index) - probability)
                for action, reward in zip(actions, rewards)
            )
            logits[index] += 0.4 * gradient
    return dict(initial_probabilities=initial_probabilities, final_probabilities=distribution(logits),
                before=before, after=greedy_pass(logits), heldout=len(heldout),
                steps=len(mean_rewards), zero_spread=zero_spread, mean_rewards=mean_rewards)


def reinforce():
    rows = (4, None, 6)
    candidates = [(2, 5), (3, Fraction(10, 3)), (1, 4), (2, 5)]
    rewards = [int(verify(rows, candidate)) for candidate in candidates]
    mean = statistics.mean(rewards)
    sigma = statistics.pstdev(rewards)
    grpo = [(reward - mean) / sigma if sigma else 0.0 for reward in rewards]
    ppo = [reward - 0.4 for reward in rewards]
    ratios = [1.3, 0.7, 1.1, 0.9]
    clipped = [
        min(ratio * adv, min(max(ratio, 0.8), 1.2) * adv)
        for ratio, adv in zip(ratios, grpo)
    ]
    # 训练阶段只验 JSON 对象的两个值。单独留出跨解析器的一致性检查。
    output = '{"count":3,"count":2,"mean":5}'
    permissive_pass = verify(rows, tuple(json.loads(output)[key] for key in ("count", "mean")))

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("重复键")
            result[key] = value
        return result

    try:
        json.loads(output, object_pairs_hook=unique_object)
        strict_accepts = True
    except ValueError:
        strict_accepts = False
    return dict(rewards=rewards, mean=mean, pop_std=sigma, grpo=grpo, ppo=ppo,
                ratios=ratios, clipped=clipped, duplicate_key_permissive=permissive_pass,
                duplicate_key_strict=strict_accepts, online=online_demo())


def fit(examples, soft=False):
    # 二分类候选排序器。两个权重对应「样例中金额是否非空」和「金额是否非零」。
    weights = [0.0, 0.0]
    for _ in range(100):
        gradient = [0.0, 0.0]
        for (present, nonzero), label in examples:
            features = [present, nonzero]
            teacher_prob = 0.9 if label else 0.1
            target = teacher_prob if soft else label
            prediction = 1 / (1 + math.exp(-sum(w * x for w, x in zip(weights, features))))
            for axis, feature in enumerate(features):
                gradient[axis] += (prediction - target) * feature
        weights = [w - 0.75 * grad / len(examples) for w, grad in zip(weights, gradient)]
    return weights


def features(value):
    # 用 ±1 编码二元提示特征，便于观察 0 与 NA 在特征中有什么区别。
    return (1 if value is not None else -1, 1 if value not in (0, None) else -1)


def examples(rows):
    return [(features(value), int(value is not None)) for row in rows for value in row]


def evaluate(weights, rows):
    correct = 0
    predictions = []
    for row in rows:
        kept = [
            value for value in row if sum(w * x for w, x in zip(weights, features(value))) > 0
        ]
        count = len(kept)
        mean = Fraction(sum(value or 0 for value in kept), count) if count else None
        correct += int(verify(row, (count, mean)))
        predictions.append([count, str(mean) if mean is not None else None])
    return dict(passed=correct, total=len(rows), predictions=predictions,
                weights=[round(w, 6) for w in weights])


def distill():
    train = [(0, 9, None), (None, 0, 3), (None, None, 6), (7, 1)]
    heldout = [(4, None, 6), (None, 0), (None, None), (0, 8, None)]
    hard = fit(examples(train))
    soft = fit(examples(train), soft=True)
    teacher_positive = [0.9 if y else 0.1 for _, y in examples(heldout)]

    def brier(weights):
        probs = [1 / (1 + math.exp(-sum(w * x for w, x in zip(weights, f))))
                 for f, _ in examples(heldout)]
        return statistics.mean((p - q) ** 2 for p, q in zip(probs, teacher_positive))

    return dict(hard=evaluate(hard, heldout), soft=evaluate(soft, heldout),
                hard_teacher_mse=brier(hard), soft_teacher_mse=brier(soft))


def synthetic():
    original = [(3, None), (5, None, 7), (1, 4, None), (9, None, 2)]
    heldout = [(4, None, 6), (None, 0), (None, None), (0, 8, None)]
    train_groups = {group(row) for row in original}
    heldout_groups = {group(row) for row in heldout}
    candidates = [
        ("重复", (None, 3), (1, 3)),
        ("留出近重复", (6, 4, None, None), (2, 5)),
        ("错误标签", (0, 1, None), (3, Fraction(1, 3))),
        ("零与缺失", (0, 1, None), (2, Fraction(1, 2))),
        ("零与缺失", (None, 0, 2), (2, 1)),
        ("零与缺失", (0, 3, None), (2, Fraction(3, 2))),
        ("合法普通项", (7, 2, None), (2, Fraction(9, 2))),
        ("留出近重复", (None, None, None), (0, None)),
    ]
    accepted = []
    rejected = []
    for source, row, label in candidates:
        key = group(row)
        if key in heldout_groups:
            reason = "heldout_overlap"
        elif key in train_groups:
            reason = "duplicate"
        elif not verify(row, label):
            reason = "wrong_answer"
        else:
            accepted.append(row)
            train_groups.add(key)
            continue
        rejected.append(dict(source=source, row=row, reason=reason))
    # 并列时选择后一条规则，以复现覆盖不足时选择错误规则的情形。
    rules = [lambda value: value is not None, lambda value: bool(value)]

    def select(train):
        pairs = [(value, value is not None) for row in train for value in row]
        return max(enumerate(rules), key=lambda pair: (
            sum(pair[1](value) == label for value, label in pairs), pair[0]
        ))[0]

    def score(rule):
        return sum(
            verify(row, answer([value for value in row if rule(value)])) for row in heldout
        )

    baseline = select(original)
    augmented = select(original + accepted)
    return dict(accepted=accepted, rejected=rejected, baseline_rule=baseline,
                augmented_rule=augmented, baseline_pass=score(rules[baseline]),
                augmented_pass=score(rules[augmented]), heldout=len(heldout))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = dict(reinforce=reinforce(), distill=distill(), synthetic=synthetic())
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
