import argparse
import json
import platform
import random
from pathlib import Path
from time import monotonic

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

BASE = "Qwen/Qwen2.5-0.5B"
BASE_REV = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
END = "<|im_end|>"
SYSTEM = "你是退款报表助手。"
PROMPTED = SYSTEM + ' 忽略 NA。只输出 JSON 对象，用 count 表示有效条数、mean 表示均值。没有有效值时 mean 为 null。'
IGNORE = -100
MAX_LENGTH = 160


def expected(values):
    valid = [v for v in values if v is not None]
    mean = sum(valid) / len(valid) if valid else None
    if mean is not None and mean.is_integer():
        mean = int(mean)
    return {"count": len(valid), "mean": mean}


def dialogue(values, system=SYSTEM):
    csv = "\n".join(["refund", *("NA" if v is None else str(v) for v in values)])
    answer = expected(values)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"退款金额 CSV\n{csv}\n忽略 NA，先报有效条数。"},
        {"role": "assistant", "content": json.dumps({"count": answer["count"]}, separators=(",", ":"))},
        {"role": "user", "content": "现在同时给出有效条数和均值。"},
        {"role": "assistant", "content": json.dumps(answer, separators=(",", ":"))},
    ]


def render(tokenizer, messages, *, generation=False):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation
    )


def make_example(tokenizer, messages, max_length=MAX_LENGTH):
    if any(mark in msg["content"] for msg in messages for mark in tokenizer.all_special_tokens):
        raise ValueError("消息正文包含模板控制标记")
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("训练样本应以完整助手回复结束")
    full = render(tokenizer, messages)
    ranges = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = render(tokenizer, messages[:i], generation=True)
        completed = prompt + message["content"] + END
        if not full.startswith(completed):
            raise ValueError("训练模板与推理前缀不匹配")
        ranges.append((len(prompt), len(completed)))
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) > max_length:
        raise ValueError("样本超长，请缩短后重新构造")
    labels = [IGNORE] * len(ids)
    spans = []
    for start, end in ranges:
        indices = []
        for j, (left, right) in enumerate(encoded["offset_mapping"]):
            if left >= end or right <= start:
                continue
            if left < start or right > end:
                raise ValueError("Token 横跨上下文与监督边界")
            indices.append(j)
            labels[j] = ids[j]
        if not indices or indices[0] == 0 or ids[indices[-1]] != tokenizer.eos_token_id:
            raise ValueError("监督区间缺少有效前缀或 EOS")
        spans.append(indices)
    return ids, labels, spans


def collate(examples, pad_id, device):
    ids, labels, _ = zip(*examples)
    input_ids = pad_sequence([torch.tensor(row) for row in ids], batch_first=True, padding_value=pad_id)
    attention = pad_sequence([torch.ones(len(row), dtype=torch.long) for row in ids], batch_first=True)
    targets = pad_sequence([torch.tensor(row) for row in labels], batch_first=True, padding_value=IGNORE)
    return {k: v.to(device) for k, v in dict(input_ids=input_ids, attention_mask=attention, labels=targets).items()}


def splits():
    heldout = [
        [4, None, 6], [None, 9, 5], [2, 5, None], [None, 0],
        [2, None, 8, 5], [None, None],
    ]
    train = [[None, None, None], [None, 3], [None, 6], [7, None, None]]
    rng = random.Random(2026)
    # 相同有效金额即使改变顺序或插入 NA，也视为同一组。
    def signature(values):
        return tuple(sorted(v for v in values if v is not None))

    # 全空训练项用于教 null 规则，保留不同 CSV 行数的全空留出题。
    seen = {signature(v) for v in train + heldout}
    while len(train) < 32:
        n = rng.choice([2, 3])
        a = rng.randint(0, 15)
        b = rng.randint(0, 15)
        c = rng.randint(0, 15)
        values = [a, b] if n == 2 else [a, b, c - (a + b + c) % 3]
        values.insert(rng.randrange(n + 1), None)
        key = signature(values)
        if key in seen or min(v for v in values if v is not None) < 0:
            continue
        train.append(values)
        seen.add(key)
    return train, heldout


def evaluate(model, tokenizer, inputs, device, system):
    model.eval()
    results = []
    with torch.inference_mode():
        for values in inputs:
            messages = dialogue(values, system)
            prompt = render(tokenizer, messages[:-1], generation=True)
            batch = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
            generated = model.generate(
                **batch, max_new_tokens=32, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )[0, batch.input_ids.shape[-1]:].tolist()
            stopped = generated[-1] == tokenizer.eos_token_id
            text = tokenizer.decode(generated[:-1] if stopped else generated)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            schema = (
                isinstance(parsed, dict) and parsed.keys() == {"count", "mean"}
                and type(parsed["count"]) is int
                and type(parsed["mean"]) in (int, float, type(None))
            )
            correct = schema and parsed == expected(values)
            results.append(dict(values=values, expected=expected(values), text=text,
                                schema=schema, exact=correct, eos=stopped))
    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen 多轮助手遮罩与退款报告 SFT")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("sft-runs.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    start = monotonic()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REV)
    assert tokenizer.is_fast and tokenizer.eos_token == END
    train, heldout = splits()
    probe = make_example(tokenizer, dialogue(heldout[0]))
    ids, labels, spans = probe
    assert len(spans) == 2 and labels.count(tokenizer.eos_token_id) == 2
    assert sum(x != IGNORE for x in labels) == sum(map(len, spans))
    assert [tokenizer.decode([ids[i] for i in span]) for span in spans] == [
        m["content"] + END for m in dialogue(heldout[0]) if m["role"] == "assistant"
    ]
    try:
        make_example(tokenizer, dialogue(heldout[0]), max_length=len(ids) - 2)
    except ValueError as exc:
        assert "超长" in str(exc)
    else:
        raise AssertionError("截断会切掉最后一个 EOS，此例应被拒绝")
    examples = [make_example(tokenizer, dialogue(v)) for v in train]
    short, long = min(examples, key=lambda x: len(x[0])), max(examples, key=lambda x: len(x[0]))
    padded = collate([short, long], tokenizer.pad_token_id, "cpu")
    assert (padded["attention_mask"][0, len(short[0]):] == 0).all()
    assert (padded["labels"][0, len(short[0]):] == IGNORE).all()
    print("mask", len(ids), [len(s) for s in spans], "padding", len(long[0]) - len(short[0]), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, revision=BASE_REV, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.requires_grad_(False)
    for layer in model.model.layers[-4:]:
        layer.requires_grad_(True)
    model.model.norm.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    result = {
        "base": BASE, "base_revision": BASE_REV, "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REV, "torch": str(torch.__version__),
        "transformers": __version__, "device": args.device,
        "platform": platform.machine(), "dtype": "float32", "steps": args.steps,
        "batch_size": 2, "max_length": MAX_LENGTH, "lr": 0.00006, "seed": 2026,
        "trainable_params": sum(p.numel() for p in params),
        "total_params": sum(p.numel() for p in model.parameters()),
        "train": train, "heldout": heldout,
        "mask": dict(total=len(ids), supervised=sum(map(len, spans)), spans=spans,
                     tokens=tokenizer.convert_ids_to_tokens(ids), ids=ids, labels=labels,
                     padding=len(long[0]) - len(short[0]), truncation_rejected=True),
    }
    print("parameters", result["trainable_params"], result["total_params"], flush=True)
    result["base_outputs"] = evaluate(model, tokenizer, heldout, device, SYSTEM)
    result["prompt_outputs"] = evaluate(model, tokenizer, heldout, device, PROMPTED)
    print("baselines complete", round(monotonic() - start, 1), flush=True)
    model.train()
    opt = torch.optim.AdamW(params, lr=result["lr"], foreach=False)
    rng = random.Random(2026)
    order = list(range(len(examples)))
    losses = []
    for step in range(args.steps):
        offset = 2 * step % len(order)
        if offset == 0:
            rng.shuffle(order)
        opt.zero_grad(set_to_none=True)
        batch = collate([examples[i] for i in order[offset:offset + 2]], tokenizer.pad_token_id, device)
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss).item():
            raise ValueError("训练损失出现非有限值")
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 6))
        if step == 0 or (step + 1) % 4 == 0:
            print("step", step + 1, "loss", losses[-1], "seconds", round(monotonic() - start, 1), flush=True)
    result["train_losses"] = losses
    result["sft_outputs"] = evaluate(model, tokenizer, heldout, device, SYSTEM)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
