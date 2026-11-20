---
title: "LLM 入门 F-12｜推理时计算与 Reasoning 模型怎样比较？"
description: "以四道退款汇总题为固定样本，区分逐 Token 解码、服务端推理强度和应用层多次采样。用同一套评分脚本比较正确率、Token 用量、延迟和输出预算触顶率，并提供可离线运行的样例和可选的真实接口路径。"
keywords: [llm, python]
---

![同一问题在不同推理预算下可以花费不同的时间与 Token。](assets/f-12-inference-time-reasoning-cover.jpg)

同一道退款统计题，某次回答很快，另一次等了更久，但两次呈现的文字长度几乎一样。是在重复计算，还是请求用了不同的推理预算？想比较这些设置，先要知道究竟调整了哪一层。

**推理时计算（Inference-time Compute）**是在模型权重已经确定的情况下，为一个请求额外投入的计算。F-11 的 GRPO 是训练阶段的更新方法。本文用已部署的模型来做实验，比较回答表现和资源用量。

## 1. 三层计算可以各自改变

| 层次 | 在哪里运行 | 这篇用什么来观察 |
| --- | --- | --- |
| 一般自回归解码 | 模型和推理引擎 | 根据当前前缀计算下一 Token 分数、选一个、追加，再继续。Token 是分词后的处理单位 |
| 可配置的推理强度 | 供应商或模型的推理接口 | 同一个模型支持多个强度时，用 `reasoning.effort` 选择强度，查看输出与推理用量 |
| 多次采样与验证 | 用户的应用，或一个组合系统 | 给同一题取多份候选，再用确定的规则验证，或在缺少验证器时把相同答案归组投票 |

第一层是 F-04 的基本循环。**Reasoning 模型**通常指模型在推理任务上做了专门训练或暴露了推理相关能力；产品接口可能让应用配置推理强度。给 `effort` 设置 `low` 或 `medium`，表示低档或中档，实际支持的档位与默认值由模型决定。它影响模型愿意投入的推理工作，返回的可见字数不一定同步变化。第三层的多次采样会产生多次调用，成本和延迟按实际调用合计。报表场景有计算器或业务规则，优先用可验证的结果选答案。 [OpenAI Reasoning 模型指南](https://developers.openai.com/api/docs/guides/reasoning) · [Self-Consistency 多样采样论文](https://arxiv.org/abs/2203.11171)

## 2. 固定四道题，再比较一项设置

规则统一为「空值不计入合计，缺失数只数空值，输出合计,缺失数」。代码中的 `None` 表示空值。

| 题号 | 退款列 | 标准答案 | 需要留意的边界 |
| --- | --- | --- | --- |
| A | 12，空，8 | `20,1` | 空值和普通金额混合 |
| B | 0，5，空，−2 | `3,1` | 0 是已知值，负数参与合计 |
| C | 空，空，7 | `7,2` | 多个空值 |
| D | 4，−4，0 | `0,0` | 合计为 0 与全列缺失含义不同 |

对比时固定模型版本、提示、输出格式、输出上限 `512`、串行并发以及使用的接口。每道题都跑两档，只改变 `effort`。正式对比时应扩大题集、多轮交错重复，并固定是否预热、网络条件和工具可用范围。四道题适合核对统计链路。脚本用**完全匹配**打分，必须返回完整字符串，格式与数字都通过才计一题正确。

## 3. 同时记录准确、用量和退出原因

**API（Application Programming Interface，应用程序接口）**在这里返回答案以及用量。以 OpenAI Responses API 为例，同一条记录可以取出：

| 观测项 | 字段或计算 | 读数的意义 |
| --- | --- | --- |
| 正确率 | 完整且正确的请求数 ÷ 全部请求数 | 截断的部分答案仍计入分母 |
| 输入和缓存输入 Token | `usage.input_tokens`、`input_tokens_details.cached_tokens` | 缓存输入是输入总数的子集，账单上可能按不同单价 |
| 输出和推理 Token | `usage.output_tokens`、`output_tokens_details.reasoning_tokens` | 在该接口里，推理 Token 已包含于输出总量 |
| 客户端延迟 | 发起调用至收到整个响应的秒数 | 本例用中位数，正式测量也可报告尾部 |
| 输出预算触顶率 | 返回 `incomplete` 且 `reason=max_output_tokens` 的请求数 ÷ 总请求数 | 可能只有推理用量，还没有可见答案 |

`max_output_tokens` 是给输出的 Token 上限。这里把这个状态作为触顶口径；如果上下文接近上限，服务也可能返回同一原因，因此实验刻意使用短提示。若要估计账单，按「未缓存输入、缓存输入、输出」分别乘对应模型和计费方式的单价，再按账单单位换算。推理 Tokens 已是输出的一部分，计价时用输出总量。 [Responses 推理用量与不完整响应说明](https://developers.openai.com/api/docs/guides/reasoning)

## 4. 先离线校验，再选择是否跑真实接口

保存为 `reasoning_budget.py`。直接运行 `python3 reasoning_budget.py` 会加载固定教学样例，不发请求。这四题及其标准答案是真正参与评分的输入，样例响应中的时间和 Token 用量是构造数据，只用于检查汇总。

若需要实测，并有可用的 API 账户，在本地安装 `python -m pip install "openai==2.44.0"`，用环境变量 `OPENAI_API_KEY` 配置认证，再运行 `python reasoning_budget.py --model YOUR_MODEL_ID`。这里的模型代号要替换为账户可用且支持 `low`、`medium` 的版本。该路径会发出 8 次计费用的合成报表请求，逐条打印返回的模型标识、状态、用量和延迟。设置 `max_retries=0` 关闭自动重试，让客户端异常与模型预算触顶保持独立。 [Responses API 官方文档](https://developers.openai.com/api/docs/guides/reasoning)

```python
import argparse
from statistics import median
from time import perf_counter

CASES = [
    ("A", [12, None, 8], "20,1"),
    ("B", [0, 5, None, -2], "3,1"),
    ("C", [None, None, 7], "7,2"),
    ("D", [4, -4, 0], "0,0"),
]
EFFORTS = ("low", "medium")
CAP = 512


def row(text, status, reason, inputs, cached, outputs, reasoning, seconds):
    assert 0 <= cached <= inputs and 0 <= reasoning <= outputs
    return {
        "text": text.strip(),
        "status": status,
        "cap_hit": status == "incomplete" and reason == "max_output_tokens",
        "counts": (inputs, cached, outputs, reasoning),
        "seconds": seconds,
    }


def fixture():
    # 答案、输出 Token、其中的推理 Token、秒。只用于验证统计。
    samples = {
        "low": [("20,1", 50, 40, 0.4), ("3,0", 64, 54, 0.6),
                ("7,2", 76, 66, 0.5), ("0,1", 56, 46, 0.8)],
        "medium": [("20,1", 110, 100, 0.9), ("3,1", 132, 122, 1.1),
                   ("", CAP, CAP, 2.5), ("0,0", 112, 102, 1.2)],
    }
    return {
        effort: [
            row(text, "completed" if text else "incomplete",
                None if text else "max_output_tokens", 60, 0, outputs, reason_tokens, latency)
            for text, outputs, reason_tokens, latency in samples[effort]
        ]
        for effort in EFFORTS
    }


def live(model):
    from openai import OpenAI

    client = OpenAI(max_retries=0, timeout=60)
    groups = {effort: [] for effort in EFFORTS}
    for case_id, values, _ in CASES:
        for effort in EFFORTS:
            prompt = f"退款列为 {values}，None 表示缺失。求非缺失之和与缺失数。只输出总和,缺失数。"
            start = perf_counter()
            response = client.responses.create(
                model=model, input=prompt, reasoning={"effort": effort},
                max_output_tokens=CAP, store=False,
            )
            elapsed = perf_counter() - start
            usage = response.usage
            if usage is None:
                raise RuntimeError(f"{case_id}/{effort} 未返回用量")
            record = row(
                response.output_text, response.status,
                getattr(response.incomplete_details, "reason", None),
                usage.input_tokens, usage.input_tokens_details.cached_tokens,
                usage.output_tokens, usage.output_tokens_details.reasoning_tokens, elapsed,
            )
            groups[effort].append(record)
            print(case_id, effort, response.model, record)
    return groups


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="传入支持 low、medium 的模型以发出真实请求")
options = parser.parse_args()
groups = live(options.model) if options.model else fixture()
for effort in EFFORTS:
    records = groups[effort]
    correct = sum(
        record["status"] == "completed" and record["text"] == case[2]
        for record, case in zip(records, CASES)
    )
    hits = sum(record["cap_hit"] for record in records)
    counts = [sum(column) for column in zip(*(record["counts"] for record in records))]
    latency = median(record["seconds"] for record in records)
    print(
        f"{effort}: 正确 {correct}/{len(CASES)}, 到达预算 {hits}/{len(CASES)}, "
        f"输入/缓存输入/输出/推理 Token {counts}, 延迟中位数 {latency:.2f} s"
    )
```

离线实跑结果：

```text
low: 正确 2/4, 到达预算 0/4, 输入/缓存输入/输出/推理 Token [240, 0, 246, 206], 延迟中位数 0.55 s
medium: 正确 3/4, 到达预算 1/4, 输入/缓存输入/输出/推理 Token [240, 0, 866, 836], 延迟中位数 1.15 s
```

这组教学数据里，中档正确数从 2 增到 3，但有一题用完预算。总输出 Token 从 246 增到 866，延迟中位数也增加。如果只看返回了完整答案的三道题，就会漏掉那条确实占用 Token 的失败请求。线上对照使用真实返回，可以同时比较正确率、出答率和单位正确答案的成本。本地已复核教学样例，真实请求需要用自己的授权账户执行。

接下来进入 **LLM API 与流式协议**。先修可回看 F-04 的采样参数和 F-07 的延迟观测，再用 API-01 从原始请求和响应字段追踪模型标识、用量和失败原因。 [学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)

[F-11 GRPO 如何用于推理模型后训练](f-11-grpo.md)
