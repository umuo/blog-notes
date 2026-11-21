---
title: "LLM 评测 LE-06｜长上下文的有效性如何评测"
description: "把退款数据字典的同一条当前证据分别放到上下文开头、中段和末尾，改变输入长度和干扰项，实测答案、引用和本地首 Token 延迟，再说明如何与检索增强生成使用同一组样本对照。"
keywords: [llm, eval, context, python]
---

![同一条退款政策证据在长上下文的开头、中段和末尾移动。](assets/le-06-long-context-eval-cover.jpg)

用户上传金额为 10、空、0 的退款表，问有效条数和均值。应用附上电子表格字段字典。当前规则编号 **R7** 规定排除空值、保留 0，正确结果是 2 条、均值 5。较长的字典里还可能夹着一条退役规则，说空金额按 0 计入。模型能容纳整份字典，也需要在这份输入里找到当前规则。

**上下文窗口（context window）**是模型一次可以处理的 Token（文本编码单元）容量。这里评测的是在不同长度和证据位置下，模型能否交出正确统计并引用当前规则。先读 [F-07 的延迟口径](https://blog.lacknb.cn/articles/2026/11/20/1795161437100.html) 和 [LE-01](le-01-dataset-baseline.md) 的样本设计。关于长上下文中证据位置的研究，可对照 Lost in the Middle 和 LongBench。 [Lost in the Middle](https://arxiv.org/abs/2307.03172) · [LongBench](https://arxiv.org/abs/2308.14508)

## 1. 三个因素各自怎么改

| 因素 | 本篇取值 | 保持固定的东西 |
| --- | --- | --- |
| 证据位置 | 文件片段的头、中、尾 | 用同一条 R7 原文，当前规则与问题总是相同。 |
| 输入长度 | 8 或 48 条填充记录 | 只多加同类填充。按实际分词器记录输入 Token。 |
| 干扰项 | 一条普通元数据或一条过期退款规则 | 始终放在最后一条填充记录里，当前规则 R7 的正文保持不变。 |

实际请求的末尾始终放 CSV 问题。**证据位置**指它在字段字典中的位置。即使放到字典末尾，后面仍有问题和消息包装。脚本记录证据首个 Token 在实际输入序列中的位置，方便检查实验变化。每种组合在 CPU 上跑一次，共 12 次。对比两个因素时，先固定第三个因素，例如在 48 条、含过期规则时只比较三个位置。

## 2. 答对、引用对、速度怎样计

- **格式成功**用正则识别三项字段，分别是 `COUNT`（条数）、`MEAN`（与输入相同的金额单位）及 `SOURCE`（证据编号）。
- **数值正确**要求 2 和 5 同时正确。**引用正确**要求来源写 R7。两项分别记录，再报告两项都正确的次数。输出编号可以被核对，不能仅据此证明模型内部用到了该段证据。
- **TTFT（Time to First Token，首 Token 延迟）**在本地从调用 `model.generate` 到首个生成 Token 交给回调。分词已完成，不包含排队、网络和界面显示。代码先热身一次。
- **成本代理**记录输入和生成的 Token 数。用 API 做同样的对照时，应另外记录服务返回的用量和计费口径。本机模型运行只记录 Token 数。

## 3. 运行并保存每次原始结果

把下列代码存为 `context_positions.py`。它沿用 API-10 和 LE-03 的 Python 3.12、PyTorch 2.11.0、Transformers 5.13.0 与同一份 Qwen 权重。首次运行需要下载公开权重。把程序放在空目录运行 `python context_positions.py`。报告把格式、数值、引用、Token 和首 Token 延迟都保留在 `context-runs.jsonl`。

```python
import json
from pathlib import Path
import re
from time import perf_counter

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EVIDENCE = "CURRENT refund amount policy R7: exclude blanks; keep numeric zero."
OLD_RULE = "RETIRED refund amount policy X2: count an empty amount as zero."
QUESTION = (
    "Apply the CURRENT refund amount policy to this CSV:\nid,amount\n1,10\n2,\n3,0\n"
    "Return only COUNT=<valid count>;MEAN=<mean>;SOURCE=<policy id>."
)
ANSWER = re.compile(
    r"COUNT\s*=\s*(\d+)\s*;\s*MEAN\s*=\s*([-+]?\d+(?:\.\d+)?)"
    r"\s*;\s*SOURCE\s*=\s*([A-Z]\d+)", re.I,
)


def make_input(length, position, distractor):
    records = [
        f"Catalog note D{i:02d}: the warehouse export stores a date, "
        "a row identifier, and a file description for reconciliation."
        for i in range(length)
    ]
    if distractor:
        records[-1] = OLD_RULE
    slot = {"head": 0, "middle": length // 2, "tail": length}[position]
    records.insert(slot, EVIDENCE)
    return "Field dictionary:\n" + "\n".join(records) + "\n\n" + QUESTION


class FirstToken:
    def __init__(self):
        self.prompt_seen = False
        self.first = None

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first is None:
            self.first = perf_counter()

    def end(self):
        pass


def check(raw):
    match = ANSWER.fullmatch(raw.strip())
    if not match:
        return {"format_ok": False, "numbers_ok": False, "citation_ok": False}
    count, value, source = match.groups()
    return {"format_ok": True, "numbers_ok": int(count) == 2 and float(value) == 5.0,
            "citation_ok": source.upper() == "R7"}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    model.eval()
    config = GenerationConfig(
        max_new_tokens=36, do_sample=False,
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )

    def run(length, position, distractor):
        prompt = make_input(length, position, distractor)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        batch = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True,
                          return_tensors="pt")
        offset = rendered.index(EVIDENCE)
        spans = batch.pop("offset_mapping")[0].tolist()
        evidence_token = next(i for i, (start, end) in enumerate(spans) if start <= offset < end)
        count = batch["input_ids"].shape[-1]
        clock = FirstToken()
        with torch.inference_mode():
            start = perf_counter()
            generated = model.generate(**batch, generation_config=config, streamer=clock)[0][count:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "length": length, "position": position, "distractor": distractor,
            "context_tokens": count, "evidence_start_token": evidence_token,
            "generated_tokens": len(generated), "ttft_ms": round((clock.first - start) * 1000, 1),
            "raw": raw, **check(raw),
        }

    run(8, "head", False)  # 热身，不计入结果。
    with Path("context-runs.jsonl").open("w", encoding="utf-8") as log:
        for distractor in (False, True):
            for length in (8, 48):
                for pos in ("head", "middle", "tail"):
                    row = run(length, pos, distractor)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    assert check("COUNT=2;MEAN=5;SOURCE=R7")["numbers_ok"]
    assert not check("COUNT=3;MEAN=3.33;SOURCE=R7")["numbers_ok"]
    assert not check("COUNT=2;MEAN=5;SOURCE=X2")["citation_ok"]
    main()
```

## 4. 先测完整输出，再用二选一探针检查证据

这次本地完整报告实测 **0/12** 通过格式，有 10 次用满 36 个生成 Token。短字典输入为 272 至 280 Token，首 Token 延迟中位数约 **390 ms**。长字典为 1,232 至 1,240 Token，中位数约 **1,690 ms**。一次「长字典、含过期规则、证据放尾部」的原始回复写出 `SOURCE=X2`，数值也错误，其余不少回复仍在解释步骤时已耗尽预算。完整报告的失败混合了证据识别、计算、格式遵从和生成预算，单凭这轮分数无法判断哪个位置更有效。

取完全相同的字段字典，再做一个**证据识别探针**。仅问「当前退款规则的编号是什么」，把 R7、X2 放在 A/B 两个位置分别测试。直接比较两个单 Token 选择在模型下一步的原始分数（logits），不要求模型先生成解释或按固定格式写完整报告。把下面代码保存为 `context_probe.py`，与上面的脚本放在同一目录，再运行 `python context_probe.py`。

```python
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from context_positions import MODEL, REVISION, QUESTION, make_input

torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
model.eval()
letters = ("A", "B")
candidates = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in letters]
assert all(len(ids) == 1 for ids in candidates)
candidate_ids = [ids[0] for ids in candidates]

records = []
for distractor in (False, True):
    for length in (8, 48):
        for position in ("head", "middle", "tail"):
            handbook = make_input(length, position, distractor).removesuffix(QUESTION)
            for order in (("R7", "X2"), ("X2", "R7")):
                prompt = handbook + (
                    "Which ID belongs to the CURRENT refund amount policy? "
                    f"A: {order[0]}. B: {order[1]}. Answer with the letter."
                )
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True,
                )
                batch = tokenizer(rendered + "Answer:", add_special_tokens=False,
                                  return_tensors="pt")
                with torch.inference_mode():
                    scores = model(**batch, logits_to_keep=1).logits[0, -1, candidate_ids]
                picked = order[int(scores.argmax())]
                records.append(dict(length=length, position=position, distractor=distractor,
                                    order=order, picked=picked, correct=picked == "R7"))

Path("probe-runs.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8",
)
for distractor in (False, True):
    for length in (8, 48):
        for position in ("head", "middle", "tail"):
            subset = [row for row in records if (row["distractor"], row["length"], row["position"])
                      == (distractor, length, position)]
            print(distractor, length, position, f"{sum(row['correct'] for row in subset)}/2")
```

每格两次分别交换 R7 和 X2 的选项位置。无过期规则时六格都是 **2/2**，合计 **12/12**。有过期规则时，结果如下：

| 含过期规则的字典 | 头部 | 中段 | 尾部 |
| --- | ---: | ---: | ---: |
| 8 条填充记录 | 2/2 | 2/2 | 1/2 |
| 48 条填充记录 | 1/2 | 2/2 | 1/2 |

同一批字典里，模型在二选一中识别出 R7 的次数远多于完整任务，它仍会受过期规则和选项位置影响。每格只有一份字典、两次选项排列，这组结果适合诊断。要估计位置影响，需要扩充独立样本并随机化规则编号和选项顺序。

## 5. 把结果接到检索对照

先按长度和干扰项分组，对照头、中、尾三个位置。数值正确且引用错误时，说明模型写出的输出缺少对当前证据的正确定位。格式失败时保留原文，以便检查是多解释了一句，还是本来就漏掉关键字段。当前规则的 `evidence_start_token` 随位置移动，输入长度应主要由填充记录数决定。单个教学案例的耗时和结果都不适合用来声称某个模型普遍有「中间遗忘」。

将来完成 KB-05 后，可以用同一题对比全量上下文与 **RAG（Retrieval-Augmented Generation，检索增强生成）**。RAG 先从文档找少量片段再交给模型。先记录 R7 是否被检索到，再固定评分器对照数值、引用、输入 Token 和端到端耗时。检索没找到 R7、找到了但排序靠后、模型看见仍答错，各自需要不同的修复。 [RAG 原论文](https://arxiv.org/abs/2005.11401)

## 下一篇

[**LE-05《模型改了，线上效果真的变好了吗》**](le-05-online-evaluation.md) 会把离线的成对样本接到线上影子评测和灰度对照，记录人工接管、延迟和回滚条件。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
