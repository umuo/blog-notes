---
title: "LLM 评测 LE-01｜从业务目标到数据集、基线和评分"
description: "把单条退款评测扩成可版本化的数据集，加入黄金样本、边界情形和历史失败，用固定切分与相同评分器比较两种统计口径，记录参考答案、分层和失败原因，再用 SHA-256 核对数据版本。"
keywords: [llm, eval, dataset, python]
---

![黄金样本、边界样本和失败样本进入有版本的评测数据集。](assets/le-01-dataset-baseline-cover.jpg)

[LE-07](le-07-first-eval.md) 用一条样本核对统计程序。真实用户还会上传全空金额、只含零值或带负金额的表。如果评测只有 `10、空、0`，一个写死回复「两条、均值 5」的版本也能过。现在把要求写成一组可复用的数据，明确哪些用来改提示，哪些留作最后评价。

## 1. 从业务规则推出数据分层

本次任务合同是「金额空白算缺失，数值 0 算有效。如果所有金额都缺失，均值返回 JSON `null`」。为了让边界易于核对，教学集只用有限整数，允许负数，均值不做货币舍入。正式项目要明确币种、金额精度和舍入规则，再用相应参考计算。

**黄金样本**有人工核对过的参考结果，可用于判断候选是否满足合同。**边界样本**覆盖全空、全零、小数均值等容易漏掉的条件。**历史失败样本**来自一个已观察到的问题，修复时用来防止同类错误再次出现。一个样本可以有多个属性，本文给每条记录一个主要分层以便汇总。

| 数据角色 | 样本例子 | 手算依据 |
| --- | --- | --- |
| 调试集里的黄金样本 | 8、4 | 有效 2，缺失 0，均值 6。 |
| 调试集里的历史失败 | 10、空、0 | API-10 的 V1 输出过 `mean: null`。当前合同的参考值是 5。 |
| 调试集里的边界样本 | 空、空、空 | 有效 0，缺失 3，均值为 `null`。 |
| 留出集里的独立情形 | -4、空、-2、空 | 有效 2，缺失 2，均值 -3。 |

同一原始文件的改写、增删标题和重复请求要留在同一划分。这里用 `family` 表示来源或案例族，使划分检查能发现同一族横跨调试集和测试集。API-10 已跑过的金额组合放进调试集或留在历史记录中。此篇留出集另外选了几组输入。公开教学数据便于复跑，真实发布门槛应使用有访问控制的独立测试集。 [评测样本与边缘情形](https://developers.openai.com/api/docs/guides/evaluation-best-practices)

## 2. 先冻结输入，再跑两条候选逻辑

**版本化**把一批样本的内容和标识固定下来。样本中保存 `id`、`family`、`split`、`stratum`、原始 CSV 和参考结果。脚本按固定顺序写 JSON Lines（每行一条 JSON），再打印 SHA-256 摘要。只要其中一个样本或参考答案改动，摘要就会变化。评分器也应记录版本，避免把两套判定规则的分数直接比较。

下面比较两段可观察的程序：基线把空值当 0，候选使用合同规定的空值处理。让两者都返回同样的三个字段。这样可以手算两组预期差异，也能看出哪些样本对这项改动没有区分力。模型或 Prompt 接入后，沿用同一个样本格式及评分规则。

## 3. 运行并按分层找出失败

存为 `refund_dataset.py`，在空目录运行 `python3 refund_dataset.py`。代码生成 `refund-cases-v1.jsonl`，再对留出集跑两组候选。历史失败保存在调试集。为避免泄漏，输出和预期是独立定义的。

```python
from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

# 格式：ID、集合、分层、案例族、各行金额、人工核对过的期望。
specs = [
    ("dev-01", "dev", "golden", "positive-a", [8, 4], (2, 0, 6)),
    ("dev-02", "dev", "historical", "api10-mixed", [10, None, 0], (2, 1, 5)),
    ("dev-03", "dev", "edge", "empty-a", [None, None, None], (0, 3, None)),
    ("test-01", "test", "golden", "positive-b", [6, 10], (2, 0, 8)),
    ("test-02", "test", "edge", "negative-b", [-4, None, -2, None], (2, 2, -3)),
    ("test-03", "test", "edge", "empty-b", [None, None], (0, 2, None)),
    ("test-04", "test", "edge", "zero-b", [0, 0, 0], (3, 0, 0)),
    ("test-05", "test", "golden", "fraction-b", [2, 5], (2, 0, 3.5)),
]
fields = ("count", "missing", "mean")


def dataset():
    rows = []
    for key, split, stratum, family, amounts, expected in specs:
        lines = ["id,amount"]
        lines.extend(f"{i},{a if a is not None else ''}" for i, a in enumerate(amounts, 1))
        rows.append({
            "id": key, "split": split, "stratum": stratum, "family": family,
            "csv": "\n".join(lines), "expected": dict(zip(fields, expected)),
        })
    dev = {row["family"] for row in rows if row["split"] == "dev"}
    test = {row["family"] for row in rows if row["split"] == "test"}
    assert dev.isdisjoint(test)
    assert len({row["id"] for row in rows}) == len(rows)
    return rows


def predict(raw_csv, keep_empty_as_zero):
    amounts = [row["amount"].strip() for row in csv.DictReader(StringIO(raw_csv))]
    valid = [int(value) for value in amounts if value]
    missing = len(amounts) - len(valid)
    if keep_empty_as_zero:
        valid.extend([0] * missing)
        missing = 0
    return {
        "count": len(valid), "missing": missing,
        "mean": sum(valid) / len(valid) if valid else None,
    }


rows = dataset()
text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
payload = text.encode("utf-8")
Path("refund-cases-v1.jsonl").write_bytes(payload)
print("dataset v1", sha256(payload).hexdigest())

test = [row for row in rows if row["split"] == "test"]
totals = Counter(row["stratum"] for row in test)
for label, keep_empty in [("baseline", True), ("rule", False)]:
    passes = Counter()
    for row in test:
        actual = predict(row["csv"], keep_empty)
        wrong = [key for key in fields if actual[key] != row["expected"][key]]
        if wrong:
            print(label, row["id"], "failed:", ",".join(wrong))
        else:
            passes[row["stratum"]] += 1
    print(label, f"overall={sum(passes.values())}/{len(test)}",
          " ".join(f"{k}={passes[k]}/{total}" for k, total in sorted(totals.items())))
```

预期结果中，基线在 `test-02` 和 `test-03` 上失败，总分 **3/5**。它在黄金层全通过，边界层只通过一个全零样本。候选规则总分 **5/5**。如果仅保留通用情形，就看不到空值错误。保存 SHA-256 摘要与各条原始输出后，可以复用同一数据版本，也可以用新评分器重新计算。

## 4. 把评分和数据维护放进同一工作流

判定能对上程序字段时，用结构和数值规则。对于退款报告的口头解释，人工可以按「有没有交代保留 0」「有没有说明分母来源」判分，记录准则和分档示例。每次加入真实失败，复制必要输入并脱敏，将其标记为回归样本，再将后续版本的独立测试样本单独保管。报告至少写明样本数、各层覆盖和失败类型。五条留出样本足以验收教学逻辑，规模扩大后才能描述实际流量表现。

## 下一篇

[**LE-08《如何设计可测试的评分器与评分准则》**](le-08-testable-graders.md) 会给评分器本身写测试，区分 JSON 结构、数值公差、业务规则和解释质量。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
