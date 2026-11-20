---
title: "LLM 入门 F-04｜自回归生成与采样：下一个 Token 怎么选？"
description: "用只有四个候选的教学词表，把原始分数变成概率，再手算贪心选择、温度缩放、Top-k 和 Top-p。沿着一条退款说明逐步生成，区分结束标记、停止文本和输出预算，并用可直接运行的 Python 脚本验证每种设置。"
keywords: [llm, python]
---

![下一 Token 的生成流程。先打分，再筛选和选择，追加到序列后继续计算。](assets/f-04-autoregressive-sampling-cover.jpg)

假设报表助手准备补完一句话「退款金额为空时，记为……」。模型给出了几个候选：「缺失」「为零」「。」和一个结束标记。它会直接选最高分，还是有机会选第二名？选完「缺失」后，句号又是怎样决定的？

本文把这段话当作提示的结尾，教学词表里只放这四个候选。**Token** 是分词器把文本切出来的单位，这里约定「缺失」和「为零」各占一个 Token，真实模型的切法可能不同。第四项 **EOS（End of Sequence，序列结束标记）**也是一个候选，选中它时，这条序列可以结束。示例设置的分数只用于演算。报表实际应采用哪条规则，要由数据字典或其他证据决定。 [F-01 的分词实验](https://blog.lacknb.cn/articles/2026/11/19/1795068393122.html)

## 1. 自回归就是选一个、追加一个、再算下一个

在 F-02 的实验里，Transformer 为输入中的每个位置给出了一排 **logits（模型给候选 Token 的原始分数）**。对于这里这种从左往右生成的模型，当前输入末尾的那一排，正好对应下一 Token 的候选分数。 [F-02 的 logits 实验](https://blog.lacknb.cn/articles/2026/11/20/1795135053358.html)

模型在一轮里只决定当前要追加的一个 Token。把它接到已有输入后面，这段更长的序列就成为下一轮的**前缀**。**自回归（autoregressive）**指下一步依赖已经有的输入以及前几步选出的结果。第一步若追加「缺失」，第二步会根据「退款金额为空时，记为缺失」重新打分。若第一步选了「为零」，后续模型看到的内容也会变。

```text
已有前缀
   ↓
模型计算末尾的下一 Token 分数
   ↓
按解码规则选一个 Token，追加到前缀
   ↓
满足停止条件？  是 → 结束
                 否 → 用更新后的前缀继续
```

这里的**解码**指从模型分数决定接下来输出哪些 Token 的过程。应用最终还要用分词器把选出来的 Token 序列还原成可见文本。 [Transformers v5.13.0 生成文档](https://huggingface.co/docs/transformers/v5.13.0/en/main_classes/text_generation)

## 2. 从原始分数得到四个候选的概率

第一轮先用一组好算的分数 `[ln(6), ln(3), ln(2), ln(1)]`。这里的 `ln` 是自然对数，`exp` 是它的反运算，例如 `exp(ln(6)) = 6`。**Softmax** 先对每个候选分数取指数，再除以所有指数的总和。在本例里，这个和是 `6 + 3 + 2 + 1 = 12`。

| 候选 | 原始分数 | 取指数后 | Softmax 概率 |
| --- | --- | ---: | ---: |
| 缺失 | `ln(6)` | 6 | `6/12 = 0.5000` |
| 为零 | `ln(3)` | 3 | `3/12 = 0.2500` |
| 。 | `ln(2)` | 2 | `2/12 ≈ 0.1667` |
| EOS | `ln(1)` | 1 | `1/12 ≈ 0.0833` |

例如「缺失」拿到分母 12 中的 6 份，概率就是 0.5。概率描述的是按当前分数**抽签时的抽中机会**。真正会不会抽签，要看选用的解码策略。这个例子对候选词表做 Softmax。在注意力内部，Softmax 归一化的是可读取的位置。

## 3. Greedy、Temperature、Top-k、Top-p 分别改了什么？

### Greedy 总选当前最高分

**Greedy（贪心选择）**每次直接取当前分数最大的候选。代码里常叫 `argmax`，意思是取最大值所在的位置。上表的最高分是 `ln(6)`，所以这一轮选「缺失」。Softmax 保留分数排序，直接比 logits 就够了。

**Sampling（采样）**是根据概率抽一个候选。直接按上表抽签时，「缺失」有 50% 的抽中机会，「为零」也有 25%。接下来的三个设置都可以用来调整这张抽签表。 [Transformers 的 Greedy 与采样实现](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/generation/utils.py#L2896)

### Temperature 拉开或缩小候选之间的差距

**Temperature（温度）**用 `T` 表示。先把每个 logit 除以正数 `T`，再算 Softmax。这个参数调整的是分数差距带来的概率落差。

- `T = 1` 时，沿用原始分数。
- `T = 0.5` 时，`ln(6) ÷ 0.5 = 2 × ln(6)`，取指数就是 `6² = 36`。四项变成 `[36, 9, 4, 1]`，总和 50，「缺失」的概率升到 `36/50 = 0.72`。
- `T = 2` 时，取指数后得到 `[√6, √3, √2, 1]`，最大的「缺失」约占 0.3714，其他候选的机会也更接近。

| 温度 | 缺失 | 为零 | 。 | EOS |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.7200 | 0.1800 | 0.0800 | 0.0200 |
| 1 | 0.5000 | 0.2500 | 0.1667 | 0.0833 |
| 2 | 0.3714 | 0.2626 | 0.2144 | 0.1516 |

除以正数会保持排名。只调低温度时，第一名更常中签，其他候选仍有机会。在 Transformers 的单路径生成中，可用 `do_sample=False` 关闭采样，直接做贪心选择。单路径对应 `num_beams=1`，即每轮只维护一条生成序列。上面的温度公式要求 `T > 0`。 [Transformers 生成参数](https://huggingface.co/docs/transformers/v5.13.0/en/main_classes/text_generation) · [温度处理实现](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/generation/logits_process.py#L238)

### Top-k 固定候选数，Top-p 固定累计概率门槛

**Top-k** 中的 `k` 是保留几个候选。例如 `k = 2` 时，只保留分数最高的「缺失」和「为零」。这两项原概率是 0.50 和 0.25，合计 0.75。用各自的概率除以 0.75，重新得到合计为 1 的抽签概率 `[2/3, 1/3, 0, 0]`。

**Top-p** 中的 `p` 是累计概率门槛。这种方法也叫 **Nucleus Sampling（核采样）**。先把候选按概率从高到低排列，从第一项往后累加，保留到第一次达到门槛。以 `p = 0.8` 为例：

| 加入候选 | 单项概率 | 到这里累计 | 达到 0.8 吗？ |
| --- | ---: | ---: | --- |
| 缺失 | 0.5000 | 0.5000 | 否 |
| 为零 | 0.2500 | 0.7500 | 否 |
| 。 | 0.1667 | 0.9167 | 是，保留这一项后停止 |
| EOS | 0.0833 | — | 此轮被过滤 |

Top-p 保留三项，重新归一化后是 `[6/11, 3/11, 2/11, 0]`。阈值 0.8 表示要留下原分布至少 80% 的概率质量。下一轮如果第一项本身就占 90%，同样的 Top-p 只需留下第一项。这里的 Top-k 是对**词表候选**做筛选。到 MoE（Mixture of Experts，混合专家）里看到同一个术语时，它筛选的是要执行的专家。 [核采样原论文](https://arxiv.org/abs/1904.09751) · [Transformers 的 Top-p 和 Top-k 实现](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/generation/logits_process.py#L473)

**一起用时，在哪张表上累加？** 在 Transformers v5.13.0 的普通采样路径里，这三项按「温度 → Top-k → Top-p」处理。举例说，原分布单独用 `top_p=0.65` 会留下前两项，因为第一项是 0.50。若先用 `top_k=2`，Top-p 看到的是剩余候选归一化后的概率，第一项已有 `2/3 ≈ 0.667`，于是只留下第一项。不同接口的参数组合和默认值要以对应实现为准。 [Transformers v5.13.0 的处理顺序](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/generation/utils.py#L1261)

## 4. 走两轮生成，再看三种停止条件

现在用 Greedy 走完整个示例。下表中的四个数是每步的「取指数后的分数」。它们和第 2 节的 6、3、2、1 是同一个量，大小排序和 logits 相同。后续各轮使用新前缀所对应的分数。

| 第几轮 | 已追加到提示的内容 | 取指数后的分数，顺序是「缺失、为零、。、EOS」 | 本轮选择 |
| ---: | --- | --- | --- |
| 1 | 空 | `[6, 3, 2, 1]` | 缺失 |
| 2 | 缺失 | `[1, 1, 8, 2]` | 。 |
| 3 | 缺失。 | `[1, 1, 2, 8]` | EOS |

前两轮生成可见的「缺失。」。是否进入第三轮，由停止条件决定。

| 停止条件 | 这次何时结束 | 示例里展示的文本 |
| --- | --- | --- |
| EOS，序列结束标记 | 第三轮选中了结束标记 | 缺失。 |
| `max_new_tokens=2`，最多追加 2 个 Token | 第二轮后预算用完，没有第三轮 | 缺失。 |
| 停止文本设为「。」 | 第二轮新增文本末尾命中「。」 | 缺失，本例把匹配到的停止文本剔除 |

EOS 通过特别的 Token 编号识别。停止文本通过字符内容匹配，一段停止文本可能跨越多个 Token。停止文本是否保留在返回内容里，取决于库或服务的约定。`max_new_tokens` 限制的是在提示之后新增的 Token 数，提示本身的长度另算。因为 EOS 也要经过一次选择，若想在这张演算表里实际生成 EOS，预算至少要允许三轮。 [Transformers 生成参数](https://huggingface.co/docs/transformers/v5.13.0/en/main_classes/text_generation) · [Transformers 停止条件实现](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/generation/stopping_criteria.py)

## 5. 用 Python 跑一遍候选分布和停止原因

把下面保存为 `decode_demo.py`，运行 `python3 decode_demo.py`。只用 Python 标准库。`VOCAB` 是教学词表，沿用上面四个候选的顺序。`candidate_probs` 先调温度，再筛选并重新归一化。代码从所有分数里减去同一个最大值 `peak`，可以避免取指数时数值过大，归一化后的概率保持相同。`SCORE_MASSES` 按已经生成的 Token 保存下一轮取指数后的四项分数，模拟模型随前缀变化重新打分。

```python
from math import exp, isclose, log
from random import Random

VOCAB = ("缺失", "为零", "。", "<EOS>")
EOS = VOCAB[-1]
FIRST_LOGITS = [log(mass) for mass in (6, 3, 2, 1)]


def candidate_probs(logits, temperature=1.0, top_k=None, top_p=1.0):
    scores = [score / temperature for score in logits]
    peak = max(scores)
    masses = [exp(score - peak) for score in scores]
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    kept_by_k = ranked[:top_k]
    total = sum(masses[index] for index in kept_by_k)

    kept_by_p = []
    cumulative = 0.0
    for index in kept_by_k:
        kept_by_p.append(index)
        cumulative += masses[index] / total
        if cumulative >= top_p:
            break

    remaining_mass = sum(masses[index] for index in kept_by_p)
    return [masses[i] / remaining_mass if i in kept_by_p else 0.0 for i in range(len(scores))]


for name, options in (
    ("T=1", {}),
    ("T=0.5", {"temperature": 0.5}),
    ("T=2", {"temperature": 2.0}),
    ("Top-k=2", {"top_k": 2}),
    ("Top-p=0.8", {"top_p": 0.8}),
    ("Top-k=2, Top-p=0.65", {"top_k": 2, "top_p": 0.65}),
):
    probs = candidate_probs(FIRST_LOGITS, **options)
    assert isclose(sum(probs), 1.0)
    print(name, [round(prob, 4) for prob in probs])

weights = candidate_probs(FIRST_LOGITS, top_k=2)
picked = Random(0).choices(VOCAB, weights=weights, k=1)[0]
print("固定种子抽一次", picked)

# 用固定分数模拟模型。键是相对于提示已经追加的 Token。
SCORE_MASSES = {
    (): (6, 3, 2, 1),
    ("缺失",): (1, 1, 8, 2),
    ("缺失", "。"): (1, 1, 2, 8),
}


def greedy_generate(max_new_tokens, stop_text=None):
    generated = []
    for _ in range(max_new_tokens):
        logits = [log(mass) for mass in SCORE_MASSES[tuple(generated)]]
        index = max(range(len(logits)), key=lambda i: logits[i])
        token = VOCAB[index]
        generated.append(token)

        if token == EOS:
            return generated, "".join(generated[:-1]), "EOS"

        text = "".join(generated)
        if stop_text and text.endswith(stop_text):
            return generated, text[:-len(stop_text)], "stop_text"

    return generated, "".join(generated), "max_new_tokens"


for limit, stop in ((4, None), (2, None), (4, "。")):
    generated, text, reason = greedy_generate(limit, stop)
    print("逐步选中的 Token", generated, "可见文本", repr(text), "停止于", reason)

assert greedy_generate(4) == (["缺失", "。", EOS], "缺失。", "EOS")
assert greedy_generate(2) == (["缺失", "。"], "缺失。", "max_new_tokens")
assert greedy_generate(4, "。") == (["缺失", "。"], "缺失", "stop_text")
```

实跑输出：

```text
T=1 [0.5, 0.25, 0.1667, 0.0833]
T=0.5 [0.72, 0.18, 0.08, 0.02]
T=2 [0.3714, 0.2626, 0.2144, 0.1516]
Top-k=2 [0.6667, 0.3333, 0.0, 0.0]
Top-p=0.8 [0.5455, 0.2727, 0.1818, 0.0]
Top-k=2, Top-p=0.65 [1.0, 0.0, 0.0, 0.0]
固定种子抽一次 为零
逐步选中的 Token ['缺失', '。', '<EOS>'] 可见文本 '缺失。' 停止于 EOS
逐步选中的 Token ['缺失', '。'] 可见文本 '缺失。' 停止于 max_new_tokens
逐步选中的 Token ['缺失', '。'] 可见文本 '缺失' 停止于 stop_text
```

`Random(0)` 固定这次实验的**随机种子**，也就是生成随机数序列的起点。这次抽到了第二名「为零」，改用 `Random(1)` 再运行，就能看到它抽中第一名。停止文本的匹配示例只检查已生成文本的末尾。把展示循环里最后一组 `(4, "。")` 改成 `(4, "缺失。")`，两个 Token 合起来也会命中停止文本，展示结果变成空字符串。

文中的温度、Top-k、Top-p 和组合筛选共五组概率，已用 Transformers v5.13.0 对照验证。逐步生成用的是上表的教学分数，便于重复检查三种停止条件。

## 下一篇

**F-05《Prefill 与 Decode》**会沿着同一条请求往下看。Prefill（预填充）先处理整段提示并算出首轮分数，Decode（逐步解码）在选出 Token 后继续计算下一轮。届时会具体区分第一轮和后续轮次各自在算什么。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
