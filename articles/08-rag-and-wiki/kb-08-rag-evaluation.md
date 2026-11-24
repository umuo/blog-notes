---
title: "知识库工程 KB-08｜RAG 端到端质量怎样拆开检查"
description: "用退款字典的跨段题和保留期题，把证据召回、排序、答案忠实度与引用正确性分开评测。实测只召回一个段落、给齐标准证据、正确出处配错误陈述及正确陈述配错误出处四种情况。"
keywords: [llm, rag, knowledge-base]
---

![分别检查证据有没有进来、答案有没有依据、引用能不能点验。](../../assets/kb-08-rag-evaluation-cover.jpg)

> **生成工具**：Codex\
> **生成模型**：gpt-6.2-sol

退款助手说「保留 90 天」，还附上学校政策第 2 页。点过去发现写的是 30 天。这份回答的引用地址指向相关条款，但陈述仍不忠实。端到端评测要能把这两件事拆开。

## 1. 给每一站自己的问题

| 阶段 | 要看什么 | 本例 |
| --- | --- | --- |
| 文档 | 原文是否解析完整，位置和版本是否保留 | 扫描件 `NA` 行是否能从第 2 页检索 |
| 初召回 | 前 K 个候选覆盖多少相关证据 | 跨段题前 1 只有 `missing`，覆盖 1/2 |
| 重排 | 有用的证据是否更靠前 | 正式审批条款从第 4 升至第 1 |
| 答案忠实度 | 每个陈述是否由引用片段支持 | 学校政策写 30 天，答案说 90 天就失败 |
| 引用正确性 | 引用是否指向人工标注的正确条款 | 写对天数但引用审批条款仍失败 |

K 是取到的候选条数。忠实度本例用原文字符串去对照，可信，但认不出等义改写。引用检查确认地址是否属于本题可用的金标准证据。自动判分还要用一小批人工样本校准。 [Ragas 上下文召回](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) · [Ragas 忠实度](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)

## 2. 先预测四次回放会在哪一项失分

先不看结果表。想象检索器已经给出了学校政策，原文写的是保留 30 天。甲写「90 天」但引用保留期条款，乙写「30 天」却引用审批条款。请分别预测答案文本是否匹配、引用是否命中，以及所引原文能否支持答复。再看跨段题，预测只给一段和给齐两段时哪个会拒答。

| 特意制造的差异 | 召回覆盖 | 忠实 | 引用命中 |
| --- | ---: | --- | --- |
| 跨段题只给第一条 | 0.5 | 无答案 | 无答案 |
| 跨段题给齐标准材料 | 1.0 | 通过 | 通过 |
| 给正确证据，手工注入 90 天 | 1.0 | 失败 | 通过 |
| 写对 30 天，指向批准规则 | 1.0 | 失败 | 失败 |

最后一行的「忠实失败」是相对它实际引用的条款而言。证据包别处有正确原句，所以严格答案文本仍能匹配。这个差别说明为何答案、陈述和引用需要各自的分数。对无答案题，另记正确拒答和误拒答。人工抽查可以同时选自动分高却被投诉的样本和自动分低但同义改写的样本，再按首个失败阶段修复。

## 3. 运行以后核对首个异常

```bash
rag_check=$(mktemp)
python3 experiments/rag-knowledge/workflow_lab.py > "$rag_check"
python3 - "$rag_check" <<'PY'
import json
import sys
from pathlib import Path

results = json.loads(Path(sys.argv[1]).read_text())["diagnostics"]
for name, row in results.items():
    print(name, {key: row[key] for key in ("recall", "answer_equal", "faithful", "citations", "refused")})
PY
```

甲对应 `wrong_claim_right_ref`，引用命中，答案文本和忠实度失分。先查陈述是怎么生成的。乙对应 `correct_claim_wrong_ref`，答案文本匹配，引用和忠实度失分。先查出处 ID 怎么跟陈述配对。跨段题若只召回了一半，先查证据供给，再用 [KB-12 的标准证据对照](kb-12-rag-gold-evidence.md) 检查给齐材料后能否回答。 [诊断代码](../../experiments/rag-knowledge/workflow_lab.py) · [实际结果](../../experiments/rag-knowledge/workflow-results.json)
