---
title: "LLM 评测 LE-04｜把效果评测接入 Prompt 和模型回归"
description: "冻结退款样本版本，用相同评分器比较基线与候选，在一次回归中汇总准确性、输入 Token 和延迟，保存失败记录。通过两个可运行场景验证发布门槛会放行达标版本，并拦住总分提高但关键样本退化的版本。"
keywords: [llm, eval, regression, python]
---

![发布前冻结样本、运行回归、检查阈值并记录发布决定。](assets/le-04-regression-gates-cover.jpg)

从新版提示得到更多正确结果以后，团队准备替换退款助手。新版本虽然总分更高，却仍把其中一条带负金额和空值的表算错。假如这条结果直接进入财务报表，平均通过率的上涨也不能覆盖这次错误。

**回归评测（regression evaluation）**用已知样本持续检查行为是否退化。本篇把 [LE-01](le-01-dataset-baseline.md) 冻结的数据、[LE-08](le-08-testable-graders.md) 的结构与业务判定，以及 [LE-09](le-09-paired-comparison.md) 的成对比较放在同一次发布检查。对接现成框架时，Promptfoo 等项目可管理测试和断言。这里用标准库展示数据进出及拦截条件，便于核对发布决定。 [Promptfoo 断言文档](https://www.promptfoo.dev/docs/configuration/expected-outputs/)

## 1. 每次运行固定什么？

| 需要固定或记录 | 示例 |
| --- | --- |
| 数据版本 | `refund-cases-v1.jsonl`、文件 SHA-256、测试集的五条 ID。 |
| 评分器版本 | 格式检查与业务值检查，规则改变时给新版本。 |
| 候选与基线 | Prompt 或模型标识、解码设置、超时和重试策略。 |
| 每条结果 | 样本 ID、候选版本、原始回应、失败类型、输入 Token 和时延。 |
| 发布规则 | 最低业务通过率，关键案例必须通过，另设成本和延迟阈值。 |

**P95（第 95 百分位）延迟**是描述尾部等待的一种方式。本文用最近秩法，将每条完整请求的时延从小到大排序，取第 `ceil(0.95 × n)` 条。`n` 是请求数，`ceil` 表示向上取整。五条样本时它就是最大值，适合验证门槛程序。生产环境需要更多样本并明确是否将超时计入。两版都用相同的起止点测量，分别报告首 Token 和完成时延。

## 2. 先定义发布门槛

本文用可手算的教学阈值。候选业务通过率至少 80%，两条空值相关的关键样本必须通过。平均输入 Token 至多比基线多 25%，P95 时延至多多 100 ms。这些数值用于演算，生产门槛可按业务需求和预算设定。

另有一条从 API-10 保存的真实历史输出：在 `mixed` 上，V1 的回复带 Markdown 围栏，严格解析失败。可以单独重放它来验证评分器这一分支，保留原始响应。它与本篇冻结的留出集角色不同，不能混入本篇两版的通过率分母。 [API-10 实测与失败记录](https://blog.lacknb.cn/articles/2026/11/21/1795268751247.html)

## 3. 固定输入，实跑两次发布检查

先运行 LE-01 的脚本得到 `refund-cases-v1.jsonl`。把下列代码存为 `eval_gate.py`。其中模型回应、Token 和延迟采用教学夹具。这样能重复触发不同的发布决定。接入真实生成时，保留记录结构，把 `replay` 改成调用接口并测量真实响应。

```python
import argparse
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import mean

DATA_HASH = "e52384255123e2f1e2792b36b184d9eed317200ca7e1e773430d6de818025d84"
CRITICAL = {"test-02", "test-03"}
FIELDS = ("count", "missing", "mean")


def read_cases(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != DATA_HASH:
        raise ValueError("unexpected dataset version")
    all_rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    return [row for row in all_rows if row["split"] == "test"]


def score(raw, expected):
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        return "invalid_json"
    if (
        not isinstance(answer, dict) or set(answer) != set(FIELDS)
        or any(type(answer[k]) is not int for k in FIELDS[:2])
        or (answer["mean"] is not None and type(answer["mean"]) not in (int, float))
    ):
        return "wrong_schema"
    failed = [key for key in FIELDS if answer[key] != expected[key]]
    return "wrong_" + "_".join(failed) if failed else None


def replay(cases, version):
    for i, case in enumerate(cases):
        answer = case["expected"]
        if case["id"] == "test-02" and version in {"baseline", "regression"}:
            answer = {"count": 4, "missing": 0, "mean": -1.5}
        if case["id"] == "test-03" and version == "baseline":
            answer = {"count": 2, "missing": 0, "mean": 0}
        raw = json.dumps(answer)
        yield {
            "id": case["id"], "version": version, "raw": raw,
            "error": score(raw, case["expected"]),
            "input_tokens": {"baseline": 100, "candidate": 118, "regression": 136}[version] + i,
            "latency_ms": {"baseline": 200, "candidate": 240, "regression": 335}[version] + i * 10,
        }


def metrics(rows):
    times = sorted(row["latency_ms"] for row in rows)
    return {
        "business_pass": sum(row["error"] is None for row in rows) / len(rows),
        "format_pass": sum(row["error"] not in {"invalid_json", "wrong_schema"}
                           for row in rows) / len(rows),
        "input_mean": mean(row["input_tokens"] for row in rows),
        "p95_ms": times[ceil(0.95 * len(times)) - 1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("refund-cases-v1.jsonl"))
    parser.add_argument("--scenario", choices=["candidate", "regression"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", type=Path, help="optional API-10 prompt-runs.jsonl")
    args = parser.parse_args()
    cases = read_cases(args.cases)
    assert CRITICAL <= {case["id"] for case in cases}
    baseline = list(replay(cases, "baseline"))
    candidate = list(replay(cases, args.scenario))
    before, after = metrics(baseline), metrics(candidate)
    flags = {
        "business": after["business_pass"] >= 0.8,
        "critical": all(row["error"] is None for row in candidate if row["id"] in CRITICAL),
        "tokens": after["input_mean"] <= before["input_mean"] * 1.25,
        "latency": after["p95_ms"] <= before["p95_ms"] + 100,
    }
    args.out.mkdir(exist_ok=True)
    report = {
        "dataset_sha256": DATA_HASH, "grader": "refund-v1", "records_per_arm": len(cases),
        "baseline": before, "candidate": after, "checks": flags, "allow": all(flags.values()),
    }
    if args.history:
        past = [json.loads(line) for line in args.history.read_text(encoding="utf-8").splitlines()]
        mixed, = [row for row in past if row["version"] == "V1" and row["case"] == "mixed"]
        report["replayed_historical_failure"] = score(mixed["raw"], mixed["expected"])
    failures = [row for row in baseline + candidate if row["error"]]
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "failures.jsonl").write_text("".join(json.dumps(row) + "\n" for row in failures))
    print(json.dumps(report, indent=2))
    return 0 if report["allow"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

依次执行下面两条命令。第二次预期返回退出码 1，持续集成（CI，Continuous Integration）可以据此拦截。两个目录都保留了报告与失败记录。

```bash
python3 eval_gate.py --scenario candidate --out passing-run
python3 eval_gate.py --scenario regression --out blocked-run
```

| 场景 | 业务通过率 | 平均输入 Token | P95 延迟 | 发布决定 |
| --- | ---: | ---: | ---: | --- |
| 基线 | 3/5 | 102 | 240 ms | 参照 |
| 达标候选 | 5/5 | 120 | 280 ms | 放行 |
| 退化候选 | 4/5 | 138 | 375 ms | 拦截 |

退化候选总分高于基线，也达到 80% 的底线。但 `test-02` 失败，Token 和时延都越界。用 `--history` 指向 API-10 的 `prompt-runs.jsonl`，还会看到历史回应被识别为 `invalid_json`。这条记录仅用于确认评分器能识别历史失败。

## 4. 接到真正的 CI 时记录什么？

轻量、确定性的回归可在每次改动时运行。真实模型较慢或需要费用，可以在候选发布前单独运行一遍，保留原始生成和配置。数据文件与评分器锁定版本后，同时跑基线、候选，并保留报告作为流水线产物。把日志中涉及私人请求的部分脱敏后再归档。服务超时、外部依赖故障与模型输出错误记不同标签，便于区别重试环境和修复版本。

## 下一篇

[**LE-06《长上下文的有效性如何评测》**](le-06-long-context-eval.md) 会把评测对象从短表扩展到较长的字段字典，移动同一条证据，观察正确率、引用与首 Token 时间。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
