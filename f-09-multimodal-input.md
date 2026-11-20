---
title: "LLM 入门 F-09｜多模态请求如何进入模型？"
description: "用一张带蓝色单元格的报表截图，追踪图文消息、像素小块、视觉向量和语言模型的输入位置。实测同一问题在纯文本和两种图像尺寸下占用多少位置，再用小模型验证图文输入如何汇到同一个文本输出头。"
keywords: [llm, python]
---

![表格图片和文本走各自处理路径，在语言模型的输入序列中会合。](assets/f-09-multimodal-input-cover.jpg)

用户发了一张报表截图，问「截图里是否有蓝色高亮的单元格？」。如果只发送文字，模型看到的是问题；把图片也放入同一条消息后，系统还要把像素变成模型能用的数字。**多模态**指请求中包含文本、图像等不同类型的信息。这里专看图像和文本如何会合。

F-01 里，**Tokenizer（分词器）**把词片段变成 Token ID，也就是词表编号。图像路径另有一个处理器和视觉网络。本文用 **Qwen2.5-VL-3B-Instruct** 做例子。VL 指 Vision-Language（视觉语言），3B 是约三十亿级的模型规模名称。不同模型的图像切块与计费规则各有差异。 [Qwen2.5-VL 技术报告](https://arxiv.org/abs/2502.13923) · [F-01 的分词实验](https://blog.lacknb.cn/articles/2026/11/19/1795068393122.html)

## 1. 一条消息为什么用多个内容块？

在所用的 Transformers 接口里，一条用户消息可包含一个图像块和一个文字块。块的 `type` 标出内容类型，处理器知道该走哪种读取方式。以下是结构示意，实际实验传入合成图片对象。

```json
{
    "role": "user",
    "content": [
        {"type": "image", "image": "report.png"},
        {"type": "text", "text": "截图里是否有蓝色高亮的单元格？"}
    ]
}
```

同样的结构在不同 **API（Application Programming Interface，应用程序接口）**中可能换字段名，例如由服务接收图片地址、文件或编码内容。本文 JSON 对应的是本地处理器的组织方式。**Chat Template（对话模板）**把角色、图像段边界和问题拼进统一的序列格式。图像段有专用的占位 Token，后面会换成视觉向量，文本段照常查嵌入表。 [Qwen2.5-VL 模型卡的消息示例](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## 2. 224×224 像素怎样得到 64 个视觉位置？

1. **预处理像素。** 将图像转成红绿蓝三通道，按模型要求缩放和归一化。这里的两张合成图已经在支持的网格上。
2. **切成小块。** Qwen2.5-VL 的空间切块边长为 14 像素。224 像素一边切成 16 份，网格为 `16 × 16 = 256` 个空间块。处理器的 `image_grid_thw` 三轴是时间、高、宽，本例是 `[1, 16, 16]`。原始像素张量每行含 1176 个数，对应 `2 × 3 × 14 × 14`；这个实现的时间块宽为 2，单图预处理会填充到所需时间块。
3. **编码并合并。** 视觉网络把小块变成学习到的向量，再把每 `2 × 2` 个相邻块合到一个语言模型输入位置。256 个空间块合成 64 个视觉位置。
4. **放入文本序列。** 模型用视觉向量替换这些占位符对应的嵌入。语言模型随后对同一条图文序列算下一文本 Token 的原始分数，也就是 **logits**。 [视觉模型与合并源码](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)

| 同一个问题 | 总输入位置 | 视觉占位数 | 处理器网格 |
| --- | ---: | ---: | --- |
| 纯文本 | 29 | 0 | 无 |
| 加 224×224 的图 | 95 | 64 | `[1, 16, 16]` |
| 加 448×448 的图 | 287 | 256 | `[1, 32, 32]` |

小图多出的 `95−29=66` 个位置里，64 个是视觉位置，另有图像段起止标记。图像边长翻倍后，视觉位置约变为四倍。实际尺寸可能受缩放上下限影响，输入位置也不能直接当作所有服务的账单。更高分辨率有机会保留细节，同时会增加输入预算。

## 3. 用两个尺寸验算处理器，再观察输出形状

使用 Python 3.11 或 3.12，在已有环境安装 `python -m pip install "torch==2.11.0" "torchvision==0.26.0" "transformers[chat-template]==5.13.0" "pillow==12.3.0"`。保存为 `multimodal_shapes.py` 后运行 `python multimodal_shapes.py`。首次取的是处理器和词表。已有同一版本的处理器文件时，也可以用环境变量 `VL_PROCESSOR_DIR` 指向其本地目录。后半段把纯文本与 448×448 图文输入送进一层随机小模型，查看同一个输出头的形状，无需下载完整权重。配置里的 `mrope_section` 为多模态旋转位置编码分配时间、高度和宽度的分量，目的是保留图像块在网格里的位置关系。

```python
import os

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
processor = AutoProcessor.from_pretrained(
    os.environ.get("VL_PROCESSOR_DIR", MODEL), revision=REVISION,
)
question = "截图里是否有蓝色高亮的单元格？"
text = [{"type": "text", "text": question}]


def make_image(size):
    picture = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(picture)
    step = size // 4
    draw.rectangle((step, step, 2 * step, 2 * step), fill="royalblue")
    for index in range(1, 4):
        position = index * step
        draw.line((0, position, size, position), fill="gray", width=2)
        draw.line((position, 0, position, size), fill="gray", width=2)
    return picture


def prepare(content):
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )


plain = prepare(text)
print("纯文本输入位置", plain["input_ids"].shape[-1])
assert "pixel_values" not in plain
for size in (224, 448):
    result = prepare([{"type": "image", "image": make_image(size)}, *text])
    grid = result["image_grid_thw"][0].tolist()
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    placeholders = (result["input_ids"] == image_id).sum().item()
    print(
        f"{size}×{size}: 入模位置 {result['input_ids'].shape[-1]}, "
        f"视觉占位 {placeholders}, 网格 {grid}, "
        f"像素张量 {tuple(result['pixel_values'].shape)}"
    )
    assert placeholders == (size // 28) ** 2

torch.manual_seed(7)
config = Qwen2_5_VLConfig(
    text_config={
        "vocab_size": 152064, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
        "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]},
    },
    vision_config={
        "depth": 1, "hidden_size": 32, "out_hidden_size": 32,
        "num_heads": 4, "intermediate_size": 64, "fullatt_block_indexes": [0],
    },
)
model = Qwen2_5_VLForConditionalGeneration(config).eval()
with torch.inference_mode():
    plain_logits = model(**plain, logits_to_keep=1).logits
    image_logits = model(**result, logits_to_keep=1).logits
print("随机小模型首步分数", tuple(plain_logits.shape), tuple(image_logits.shape))
assert plain_logits.shape == image_logits.shape == (1, 1, 152064)
```

实跑结果：

```text
纯文本输入位置 29
224×224: 入模位置 95, 视觉占位 64, 网格 [1, 16, 16], 像素张量 (256, 1176)
448×448: 入模位置 287, 视觉占位 256, 网格 [1, 32, 32], 像素张量 (1024, 1176)
随机小模型首步分数 (1, 1, 152064) (1, 1, 152064)
```

## 4. 从输入形状追到模型输出

纯文本的输入没有 `pixel_values`，因此无图时不会送图像张量。两条首步输出的三个轴分别为批次、保留的输出位置数和词表大小。随机权重用于检查路径和形状。换成训练好的模型时，可将这两个输入分别交给 `generate()`，只解码新输出。验收时比较回答是否识别到蓝格，以及无图片时能否说明缺少证据，避免把碰巧猜对当作图像识别。 [官方生成调用示例](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

源码可以从 `Qwen2_5_VLProcessor` 的图像占位符展开入手，再看模型内的 `get_image_features()` 和 `masked_scatter()`，前者产生视觉表示，后者把它们放进语言模型输入。音频还有采样和时间轴，实时语音还有传输与打断，路线图在 API-08 继续展开。 [固定版处理器](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py) · [固定版模型](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)

下一篇 **F-10《模型从训练到可用的过程》**会继续问，能读到这些输入的模型，怎样学会按要求输出报表答案。 [学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)

[F-08 模型精度与量化](f-08-model-precision-and-quantization.md) · [F-10 模型从训练到可用的过程](f-10-model-training-stages.md)
