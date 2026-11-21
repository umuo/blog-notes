---
title: "LLM 评测 LE-10｜常用 LLM 评测数据集测什么，应该怎么选"
description: "从 MMLU、C-Eval、GSM8K 和 HumanEval 各读一条有来源的公开样本，比较选择题、数值答案与代码补全的判定方式，说明准确率、答案抽取、功能测试以及数据和用途限制怎样影响选型。"
keywords: [llm, eval, benchmark, dataset]
---

![MMLU、C-Eval、GSM8K 和 HumanEval 对应不同样本形态和评分方式。](assets/le-10-public-datasets-cover.jpg)

退款助手的要求是忽略空金额、保留 0，并解释分母为什么是 2。团队选模型时，经常会看到几个公开基准的分数。哪个分数能帮助做这个选择？先读一条真实题目、看它交给模型的输入，再看评分器究竟检查哪一项。

**公开基准（benchmark）**是一组公开定义的任务及评测协议。下表的样本用中文概述，原始题面、选项和答案以相应数据集为准。

## 1. MMLU 与 C-Eval，都是选择题，覆盖侧重点不同

**MMLU（Massive Multitask Language Understanding，大规模多任务语言理解）**覆盖 57 个学科或任务。以 `abstract_algebra/test` 第 0 条为例，题目问扩域 `ℚ(√2, √3, √18)` 相对有理数的次数，四个选项依次是 0、4、2、6。数据中的 `answer: 1` 用零起始索引表示第二个选项 B。把数字键误当字母或按 1 起始解码，会让同一个预测得到不同成绩。 [MMLU 原论文](https://arxiv.org/abs/2009.03300) · [MMLU 数据卡与样本](https://huggingface.co/datasets/cais/mmlu)

**C-Eval**是一套中文多层级、多学科评测，论文覆盖 52 门学科。看 `accountant/val` 里的 `id: 4`。它用四个选项询问关于战略管理哪种说法错误，答案 C 对应把战略管理视作一次性工作的选项。数据行分别存 `A` 到 `D`、`answer` 和题目。这个样本检查的是公开题库中的知识选择。仓库标注 CC BY-NC-SA 4.0，二次分发和商用场景需留意所用版本及其许可。 [C-Eval 原论文](https://arxiv.org/abs/2305.08322) · [C-Eval 样本](https://huggingface.co/datasets/ceval/ceval-exam/viewer/accountant/val) · [数据卡](https://huggingface.co/datasets/ceval/ceval-exam)

选择题的一个常见指标是**准确率**，即正确选项数除以已评分题数。获得选项有两种常见途径。一种生成文本，再按规则提取选项字母。一种计算各选项在给定题面下的条件对数概率，再选择得分最高的一项。前者会受到格式和答案抽取影响，后者要固定候选形式、Token 边界及评分规则。比较公开分数时还要对齐各科汇总口径。例如按题数汇总与先算每科再等权平均，对题数不同的科目权重不同。 [LM Evaluation Harness 任务接口](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

## 2. GSM8K，答案是最终数值

**GSM8K（Grade School Math 8K）**包含小学阶段的英文数学文字题。以 `main/train` 第 0 条为例，Natalia 四月卖出 48 件发夹，五月数量是四月的一半，问两个月共卖多少。参考结果的最后一个数是 **72**，原始参考文本把它放在 `####` 标记之后。数据还包含计算说明，评分协议可以只比较末尾数值。 [GSM8K 数据、样本和许可](https://huggingface.co/datasets/openai/gsm8k) · [OpenAI 数据集说明](https://github.com/openai/grade-school-math)

**归一化**把允许的等价写法转换为可比较的形式。例如在课后练习里可以把 `72.0` 归一成数值 72。正式基准要先看所用评分器的抽取和归一化逻辑，有些只剥去逗号、货币记号等符号，然后做字符串精确匹配。模型说「我答 70，后来改为 72」，也要靠协议确定取哪个位置。如果任务要求解释过程，最终数值评分只能说明最后的答案是否匹配。 [Harness 的 GSM8K 任务配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml)

## 3. HumanEval，答案是一段可执行的函数

HumanEval 的 `HumanEval/0` 给出 `has_close_elements` 的函数签名和说明，要求检测列表中是否存在两个不同位置的数，其差的绝对值小于阈值。数据行有 `prompt`、`test`、`entry_point` 和 `canonical_solution`。评测把候选函数与测试配合执行。即使一种实现先排序、另一种枚举所有数对，只要满足规定测试，都可以通过。候选输出的好坏靠程序行为判断。 [HumanEval 原始仓库](https://github.com/openai/human-eval) · [HumanEval 数据卡与样本](https://huggingface.co/datasets/openai/openai_humaneval)

**单元测试（unit test）**用预先写好的输入和断言检查函数。单个候选通过全部给定断言才计通过。多次生成可以进一步计算 **pass@k**，衡量 k 个候选中至少有一个通过的机会。正式执行模型生成的代码需要隔离执行、限制资源和超时。本篇只运行下面手写的函数和断言。 [HumanEval 原论文](https://arxiv.org/abs/2107.03374)

## 4. 用四条样本检查四种判定

把下面存为 `read_benchmarks.py`。代码中 MMLU 与 C-Eval 的标签及 GSM8K 的参考数值来自上述公开行，其余输入是演示数据。它演示三类评分机制，不作为四个基准的实现。

```python
from decimal import Decimal
from itertools import combinations

def choice_correct(prediction, expected):
    return prediction.strip().upper() == expected


# MMLU：模型给出 B，数据中的 1 指第二个选项。
assert choice_correct(" B", "ABCD"[1])
# C-Eval：按字母标签判定。
assert choice_correct("c", "C")
assert not choice_correct("A", "C")


def demo_gsm8k_answer(text):
    marker, sep, answer = text.rpartition("####")
    if not sep:
        raise ValueError("missing final-answer marker")
    return Decimal(answer.strip().replace(",", ""))


assert demo_gsm8k_answer("A model calculation. #### 72.0") == Decimal(72)


def has_close_elements(values, threshold):
    return any(abs(a - b) < threshold for a, b in combinations(values, 2))


assert has_close_elements([0.0, 1.0, 3.0], 1.1)
assert not has_close_elements([0.0, 1.0, 3.0], 1.0)
assert not has_close_elements([0.0], 1.0)
print("选择题键、最终数值和三条函数断言均通过")
```

如果返回模型目录比较能力，可以优先看匹配任务的这一类分数。中文知识、数学文字题、函数补全各有自己的协议。如果验收的是退款助手，还要用 [LE-01](le-01-dataset-baseline.md) 的数据检查空值、0、计数与输出字段，并验证业务样本的证据和实际运行结果。公开样本可能出现在模型训练数据里，这会影响把成绩解释为泛化能力。记录任务及数据版本、划分、提示、答案提取规则和是否测试过本地业务集。

## 下一篇

[**LE-02《公开基准实战，怎样复现并比较模型结果》**](le-02-reproduce-benchmark.md) 会用评分框架固定 GSM8K 的一小组题，实际跑出可回查的记录。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
