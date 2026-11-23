"""用 Unsloth Zoo 的 Apple Silicon 后端验证 LoRA 和 4-bit 训练。"""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from unsloth_zoo.mlx.loader import FastMLXModel
from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SYSTEM = "忽略 NA，只返回 JSON，包含有效条数 count 和均值 mean。无有效值时 mean 为 null。"
SEED = 2026
HERE = Path(__file__).resolve().parent


def messages(values, expected=False):
    csv = "\n".join("NA" if value is None else str(value) for value in values)
    dialogue = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"退款金额 CSV\nrefund\n{csv}\n返回统计。"},
    ]
    if expected:
        label = json.dumps(answer(values), separators=(",", ":"))
        dialogue.append({"role": "assistant", "content": label})
    return dialogue


def answer(values):
    valid = [value for value in values if value is not None]
    mean = sum(valid) / len(valid) if valid else None
    if mean is not None and mean.is_integer():
        mean = int(mean)
    return {"count": len(valid), "mean": mean}


def score(model, tokenizer, rows):
    model.eval()
    result = []
    for values in rows:
        prompt = tokenizer.apply_chat_template(messages(values), tokenize=False, add_generation_prompt=True)
        output = generate(model, tokenizer, prompt=prompt, max_tokens=32, verbose=False).strip()
        try:
            parsed = json.loads(output)
            passed = isinstance(parsed, dict) and parsed == answer(values)
        except ValueError:
            passed = False
        result.append(dict(values=values, output=output, passed=passed))
    return result


def training_row(tokenizer, values):
    prompt = tokenizer.apply_chat_template(messages(values), tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages(values, expected=True), tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("训练文本与推理前缀不匹配")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) > 128:
        raise ValueError("训练回复跨越分词边界或超过长度限制")
    completion = full[len(prompt):].removesuffix("\n")
    if not completion.endswith("<|im_end|>"):
        raise ValueError("训练回复缺少模板结束标记")
    return {"prompt": prompt, "completion": completion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=[4, 16], required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((HERE / "peft-results.json").read_text())
    train = dataset["train_rows"]
    heldout = [item["values"] for item in dataset["base"]]

    model, tokenizer = FastMLXModel.from_pretrained(
        MODEL, revision=REVISION, load_in_4bit=args.bits == 4, dtype="float16",
        text_only=True, max_seq_length=128, random_state=SEED, trust_remote_code=False,
    )
    packed = getattr(model, "_unsloth_quantization_config", None)
    if args.bits == 4 and (not packed or packed.get("bits") != 4):
        raise RuntimeError(f"没有实际加载 4-bit 权重。实际配置为 {packed!r}")
    if args.bits == 16 and packed:
        raise RuntimeError(f"16-bit 对照组加载了量化权重。实际配置为 {packed!r}")
    weight_bytes = sum(array.nbytes for _, array in tree_flatten(model.parameters()))
    base = score(model, tokenizer, heldout)
    mx.random.seed(SEED)
    model = FastMLXModel.get_peft_model(
        model, r=4, lora_alpha=8, lora_dropout=0.0, random_state=SEED,
        target_modules=["q_proj", "v_proj"], use_gradient_checkpointing=False,
    )
    trainable = sum(array.size for _, array in tree_flatten(model.trainable_parameters()))
    data = [training_row(tokenizer, values) for values in train]
    config = MLXTrainingConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=12,
        learning_rate=0.001, warmup_steps=0, lr_scheduler_type="constant",
        weight_decay=0.0, logging_steps=1, max_seq_length=128, seed=SEED,
        compile=False, use_cce=False, gradient_checkpointing=False, save_steps=0,
        eval_steps=0, append_eos=False, completion_only_loss=True,
        output_dir=str(args.work_dir / "trainer"),
    )
    trainer = MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=data, args=config)
    losses = []
    memory = []

    def on_step(step, total, loss, lr, tok_s, peak_gb, elapsed, num_tokens, grad_norm=None):
        losses.append(float(loss))
        memory.append(int(mx.get_peak_memory()))

    trainer.add_step_callback(on_step)
    mx.reset_peak_memory()
    trainer.train()
    if len(losses) != config.max_steps:
        raise RuntimeError(f"只读到 {len(losses)} 条损失")
    trained = score(model, tokenizer, heldout)
    lora = args.work_dir / "lora"
    model.save_pretrained_merged(str(lora), tokenizer=tokenizer, save_method="lora")
    lora_bytes = sum(path.stat().st_size for path in lora.glob("*.safetensors"))
    reloaded, reloaded_tokenizer = FastMLXModel.from_pretrained(
        str(lora), load_in_4bit=args.bits == 4, dtype="float16", text_only=True, max_seq_length=128,
    )
    reloaded_outputs = score(reloaded, reloaded_tokenizer, heldout)
    del reloaded
    result = dict(
        platform=platform.machine(), python=platform.python_version(), zoo=version("unsloth-zoo"),
        mlx=version("mlx"), mlx_lm=version("mlx-lm"), transformers=version("transformers"),
        base_model=MODEL, revision=REVISION, bits=args.bits, quantization=packed,
        weight_bytes=weight_bytes, trainable=trainable, seed=SEED, lora_bytes=lora_bytes,
        peak_training_mlx_bytes=max(memory), losses=losses,
        base=base, trained=trained, reloaded=reloaded_outputs,
    )
    if args.bits == 16:
        merged = args.work_dir / "merged"
        model.save_pretrained_merged(str(merged), tokenizer=tokenizer, save_method="merged_16bit")
        merged_model, merged_tokenizer = load(str(merged))
        result["merged"] = score(merged_model, merged_tokenizer, heldout)
        result["merged_bytes"] = sum(path.stat().st_size for path in merged.glob("*.safetensors"))
    (args.work_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
