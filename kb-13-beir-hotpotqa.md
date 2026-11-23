---
title: "知识库工程 KB-13｜用 BEIR 和 HotpotQA 练习两种证据评测"
description: "从 BEIR SciFact 正式测试划分取十个查询，对完整语料比较词重叠与 BM25。再用 HotpotQA 五道干扰段落题和官方脚本单独算答案、支持事实及联合分，把评测口径迁回自己的退款问答。"
keywords: [llm, rag, knowledge-base]
---

![分别练习文档排名和多跳问答的证据评分。](assets/kb-13-beir-hotpotqa-cover.jpg)

> **生成工具**：Codex\
> **生成模型**：gpt-6.2-sol

你可以问「找得到正确文档吗」，也可以问「答案和证据都对吗」。前者适合用检索基准 BEIR，后者可用带支持句标注的 HotpotQA 练习。一组分只回答一组问题。

## 1. SciFact 用查询、语料与相关性标注

BEIR（异构零样本信息检索基准）收录多种检索数据集。我们取其中的 SciFact，从测试相关性标签选编号排序前 10 个查询，每个查询仍检索完整的 5,183 篇文档。qrels（查询与文档的相关性标注）给出查询 ID、文档 ID 和相关分。脚本对标题与正文分别跑共同词数排序和 BM25。平分时按文档 ID 固定顺序。 [BEIR 论文](https://arxiv.org/abs/2104.08663) · [SciFact 数据入口](https://huggingface.co/datasets/BeIR/scifact)

| 检索方法 | Recall@10 | nDCG@10 |
| --- | ---: | ---: |
| 共同词数 | 0.550 | 0.406 |
| BM25 | 0.750 | 0.646 |

Recall@10 是前 10 个命中的相关文档数除以该题所有相关文档数。nDCG@10 用 `1/log2(1+r)` 折损第 `r` 名的相关分，再除以理想排序的分数。这里对 10 个查询等权平均，结果适用于本次小切片。

## 2. HotpotQA 检查答案和支持句

HotpotQA 的 QA 是 Question Answering（问答）。本例取 Distractor（给固定候选段落并混入干扰段落）验证集开头 5 题。支持事实用 Wikipedia 条目标题与从 0 开始的句子号表示。对每道题，词面规则只按共同词数挑两句。答案先刻意统一填 `yes`，再分别注入标准答案、标准句，借同一官方评分器检查各项分数会怎样变化。 [HotpotQA 数据集](https://hotpotqa.github.io/) · [官方评分器](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)

| 输入 | 答案 EM | 支持句 F1 | 联合 F1 |
| --- | ---: | ---: | ---: |
| 固定 yes + 词面挑句 | 0.200 | 0.437 | 0.000 |
| 标准答案 + 词面挑句 | 1.000 | 0.437 | 0.437 |
| 标准答案 + 标准句 | 1.000 | 1.000 | 1.000 |

EM（Exact Match，精确匹配）在规范化后比对是否一样。支持句 F1 综合选对句子的精确率与覆盖标注句的召回率。官方联合 F1 先把答案、支持句的精确率分别相乘，召回率也相乘，再算调和平均。因此第一组虽然有一道答对，联合评分仍为零。把这个方法迁到退款题，分别记录候选是否包含原文、答复是否正确、页码能否支撑答复。保留切片 ID、输入校验值和测试模式，避免把固定段落结果与全开放检索混比。 [复跑命令](experiments/rag-knowledge/README.md) · [实际输出](experiments/rag-knowledge/benchmark-results.json)
