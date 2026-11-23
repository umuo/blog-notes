"""在固定 Qwen Instruct 基座上验证 LoRA、合并和离线 DPO。"""

import argparse
import importlib.util
import json
from pathlib import Path

import peft
import torch
import torch.nn.functional as F
import transformers

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SYSTEM = "忽略 NA，只返回 JSON，包含有效条数 count 和均值 mean。无有效值时 mean 为 null。"
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sft_example", HERE.parent / "trn-04/run_sft.py")
SFT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SFT)


def messages(values, answer=None):
    rows = "\n".join("NA" if value is None else str(value) for value in values)
    result = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"退款金额 CSV\nrefund\n{rows}\n返回统计。"},
    ]
    if answer is not None:
        result.append({"role": "assistant", "content": json.dumps(answer, separators=(",", ":"))})
    return result


def evaluate(model, tokenizer, rows, device):
    model.eval()
    results = []
    for values in rows:
        prompt = SFT.render(tokenizer, messages(values), generation=True)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = model.generate(
                **inputs, do_sample=False, max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )[0, inputs.input_ids.shape[-1]:].tolist()
        stopped = ids[-1] == tokenizer.eos_token_id
        output = tokenizer.decode(ids[:-1] if stopped else ids).strip()
        try:
            parsed = json.loads(output)
            correct = isinstance(parsed, dict) and parsed == SFT.expected(values)
        except ValueError:
            correct = False
        results.append(dict(values=values, output=output, passed=stopped and correct))
    return results


def pair_scores(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
    labels = batch["labels"][:, 1:]
    valid = labels.ne(-100)
    scores = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = labels.masked_fill(~valid, 0)
    sequence = (scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * valid).sum(1)
    return sequence[0] - sequence[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.set_float32_matmul_precision("highest")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32
    ).to(args.device)
    train, heldout = SFT.splits()
    # 这个实验把全空语义组完整留给评测。
    train = [values for values in train if any(value is not None for value in values)][:12]
    result = dict(
        model=MODEL, revision=REVISION, torch=torch.__version__, transformers=transformers.__version__,
        peft=peft.__version__, device=args.device, train_rows=train,
    )
    result["base"] = evaluate(base, tokenizer, heldout, args.device)
    config = peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(base, config)
    result["trainable_total"] = model.get_nb_trainable_parameters()
    examples = [SFT.make_example(tokenizer, messages(values, SFT.expected(values))) for values in train]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    result["sft_losses"] = []
    model.train()
    for example in examples:
        optimizer.zero_grad(set_to_none=True)
        loss = model(**SFT.collate([example], tokenizer.pad_token_id, args.device), use_cache=False).loss
        loss.backward()
        optimizer.step()
        result["sft_losses"].append(loss.item())
    adapter_dir = args.out.parent / "peft-adapter"
    model.save_pretrained(adapter_dir, selected_adapters=["default"])
    result["sft"] = evaluate(model, tokenizer, heldout, args.device)
    model.load_adapter(adapter_dir, adapter_name="reloaded")
    probe = SFT.collate([examples[0]], tokenizer.pad_token_id, args.device)["input_ids"]
    with torch.no_grad():
        model.set_adapter("default")
        separate = model(probe).logits[:, -1]
        with model.disable_adapter():
            frozen = model(probe).logits[:, -1]
        model.set_adapter("reloaded")
        reloaded = model(probe).logits[:, -1]
    result["reload_max_abs"] = (separate - reloaded).abs().max().item()
    result["base_max_abs"] = (separate - frozen).abs().max().item()
    model.set_adapter("default")

    def pair(values):
        good = SFT.expected(values)
        bad = {**good, "count": len(values)}
        if bad == good:
            bad["count"] += 1
        return SFT.collate(
            [SFT.make_example(tokenizer, messages(values, candidate)) for candidate in (good, bad)],
            tokenizer.pad_token_id, args.device,
        )

    pairs = [pair(values) for values in train]
    heldout_pairs = [pair(values) for values in heldout]
    with torch.no_grad():
        reference = [pair_scores(model, batch).detach() for batch in pairs]
        result["sft_heldout_margins"] = [pair_scores(model, batch).item() for batch in heldout_pairs]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0005)
    result["dpo_losses"] = []
    model.train()
    for batch, ref in zip(pairs, reference):
        optimizer.zero_grad(set_to_none=True)
        loss = F.softplus(-0.1 * (pair_scores(model, batch) - ref))
        loss.backward()
        optimizer.step()
        result["dpo_losses"].append(loss.item())
    result["dpo"] = evaluate(model, tokenizer, heldout, args.device)
    with torch.no_grad():
        result["dpo_heldout_margins"] = [pair_scores(model, batch).item() for batch in heldout_pairs]
        unmerged = model(probe).logits[:, -1]
        merged = model.merge_and_unload(safe_merge=True).eval()
        result["merge_max_abs"] = (merged(probe).logits[:, -1] - unmerged).abs().max().item()
    result["merged"] = evaluate(merged, tokenizer, heldout, args.device)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
