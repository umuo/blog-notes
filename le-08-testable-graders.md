---
title: "LLM 评测 LE-08｜如何设计可测试的评分器与评分准则"
description: "为退款统计评分器写已知正误的测试，从 JSON Schema 过渡到数值公差和业务不变量，再用解释是否交代空值、零值和分母来定义人工评分准则，并明确平局、拒答和工具故障的处理。"
keywords: [llm, eval, python]
---

![数值断言和开放式准则分别检查退款报告的结构、业务结果和解释质量。](assets/le-08-testable-graders-cover.jpg)

[LE-01](le-01-dataset-baseline.md) 有一条历史失败是「有效 2 条，缺失 1 条，均值为 null」。JSON 结构可以允许金额为空，样本里仍然有两条有效金额 10 和 0，均值应当为 5。评分器要能说清失败发生在哪一步。写完评分器，还要给它一组提前知道对错的输入，确认自身的判定稳定。

## 1. 格式、跨字段规则和参考数值分别检查

**JSON Schema** 是描述 JSON 结构与字段类型的规则。以下 Schema 要求三个字段，禁止附加字段。条数用非负整数，均值可为数值或空。它允许形如 `{"count":2,"missing":1,"mean":null}` 的对象进入下一层。

```json
{
  "type": "object",
  "properties": {
    "count": {"type": "integer", "minimum": 0},
    "missing": {"type": "integer", "minimum": 0},
    "mean": {"type": ["number", "null"]}
  },
  "required": ["count", "missing", "mean"],
  "additionalProperties": false
}
```

根据合同，这个样本还要满足两条**业务不变量**：有效数加缺失数等于总行数 3。均值只有在有效数为 0 时才为空。然后对照人工参考结果 `(2, 1, 5)`，检查两个条数及均值。对小数，本文设绝对公差 **0.01 个输入金额单位**，包含边界。若参考值是 5，预测 5.01 通过，5.011 失败。实际系统按业务决定精度，本文的公差用于演示。 [JSON Schema 数值约束](https://json-schema.org/understanding-json-schema/reference/numeric)

## 2. 用已知对错的样本测试评分器

把下面保存为 `check_grader.py`，安装 `jsonschema` 后运行。库负责通用结构校验，业务函数只关注退款合同。解析时用 `Decimal` 保留小数文本的十进制值，以便复现公差边界。重复字段、`NaN` 这类非标准数值会在进入结构校验前被拦下。

```bash
python -m pip install "jsonschema==4.25.1"
python check_grader.py
```

```python
from decimal import Decimal
import json

from jsonschema import Draft202012Validator

FIELDS = {
    "count": {"type": "integer", "minimum": 0},
    "missing": {"type": "integer", "minimum": 0},
    "mean": {"type": ["number", "null"]},
}
VALIDATOR = Draft202012Validator({
    "type": "object", "properties": FIELDS, "required": list(FIELDS),
    "additionalProperties": False,
})
EXPECTED = {"count": 2, "missing": 1, "mean": 5}
TOLERANCE = Decimal("0.01")


def no_duplicates(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate key")
    return result


def invalid_constant(text):
    raise ValueError(f"non-JSON constant: {text}")


def grade(raw, status="completed", expected=EXPECTED):
    if status in {"refusal", "tool_error", "unjudgeable"}:
        return status
    if status != "completed":
        raise ValueError(f"unknown status: {status}")
    try:
        data = json.loads(
            raw, parse_float=Decimal,
            parse_constant=invalid_constant, object_pairs_hook=no_duplicates,
        )
    except (ValueError, TypeError):
        return "invalid_json"
    counts = ("count", "missing")
    if not VALIDATOR.is_valid(data) or any(type(data[k]) is not int for k in counts):
        return "wrong_schema"
    expected_rows = sum(expected[k] for k in counts)
    if sum(data[k] for k in counts) != expected_rows or (
        (data["mean"] is None) != (data["count"] == 0)
    ):
        return "inconsistent_report"
    if any(data[k] != expected[k] for k in counts):
        return "wrong_counts"
    if data["mean"] is not None:
        error = abs(Decimal(data["mean"]) - Decimal(str(expected["mean"])))
        if error > TOLERANCE:
            return "outside_tolerance"
    return "pass"


cases = [
    ('{"count":2,"missing":1,"mean":5.01}', "pass"),
    ('{"count":2,"missing":1,"mean":5.011}', "outside_tolerance"),
    ('{"count":3,"missing":0,"mean":3.33}', "wrong_counts"),
    ('{"count":2,"missing":1,"mean":null}', "inconsistent_report"),
    ('{"count":2,"missing":0,"mean":5}', "inconsistent_report"),
    ('{"count":true,"missing":1,"mean":5}', "wrong_schema"),
    ('{"count":2,"missing":1,"mean":NaN}', "invalid_json"),
    ('{"count":2,"count":2,"missing":1,"mean":5}', "invalid_json"),
]
for raw, expected in cases:
    actual = grade(raw)
    assert actual == expected, (raw, expected, actual)
    print(actual)
for state in ("refusal", "tool_error", "unjudgeable"):
    assert grade("", state) == state
assert grade('{"count":0,"missing":3,"mean":null}',
             expected={"count": 0, "missing": 3, "mean": None}) == "pass"
print("12 个判定均符合预期")
```

计数额外要求 JSON 整数写法，所以 `2.0` 即使数学上是整数，也会被这份应用合同拒绝。评分器故意把第一处失败作为结果，便于定位错误。若产品要看同一份答案错了几个字段，可以像 LE-07 一样在结构通过后继续记录各个断言。拒答、工具失败与无法判定保留独立状态。它们是否计入端到端失败率，要看指标分母是在统计全部任务，还是仅统计成功交付的回答。

## 3. 给开放式解释准备评分准则

**评分准则（rubric）**把难以完全程序化的质量拆成具体可观察项。对「为什么均值为 5」的解释，可以请标注者逐项打分，每项 0 或 1 分：

| 条目 | 给 1 分的证据 |
| --- | --- |
| 空金额处理 | 明确说明空金额被排除。 |
| 真实零值处理 | 明确说明 0 仍计入有效记录。 |
| 分母和计算 | 说明有效金额为 10 和 0，分母为 2，算出 5。 |

「忽略空金额，0 保留，所以 (10 + 0) ÷ 2 = 5」得 3 分。「平均就是 5，因为有两行」缺少金额、空值和保留 0 的解释，三项均未达到。只要出现错误计算或与输入矛盾的结论，先记录硬断言失败，再评解释的表达质量。这样能避免一段长解释掩盖错误结论。 [评分器与人工标注](https://developers.openai.com/api/docs/guides/evaluation-best-practices)

**逐项打分**方便找原因。**成对比较**把两版解释放在同一个样本上比较，例如解释 A 完整交代分母，B 只给结论，则 A 胜。两版同样充分时选平局。判定所需证据缺失时标为无法判定。先用已知对错的案例独立验证评分器，再把它装进模型对照流程。 [LLM 评测最佳实践](https://developers.openai.com/api/docs/guides/evaluation-best-practices)

## 下一篇

[**LE-10《常用 LLM 评测数据集测什么，应该怎么选》**](le-10-public-datasets.md) 会比较四种公开数据集，并读一条样本及其具体评分方法。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
