---
title: "LLM 入门 F-08｜模型精度与量化：权重怎样缩到 4 位？"
description: "从报表模型的一次点积出发，手算比例因子、整数编码、恢复误差和低比特打包。比较 FP16、BF16、INT8、INT4，区分权重量化与键值缓存量化，再看 GPTQ、AWQ 分别如何处理模型质量。"
keywords: [llm, python]
---

![浮点权重用比例因子映射到离散的低比特编码。](assets/f-08-model-precision-and-quantization-cover.jpg)

「请汇总 CSV。」的模型能够正常跑起来，但部署设备上留给权重的内存有限。CSV 是逗号分隔值文件。**量化（Quantization）**把原本用较多位存的数映射到较少的表示等级。它常用于压缩模型权重，也可用于压缩推理时不断增长的缓存。先在一条只有四个权重的计算里观察，压缩以后模型究竟读到了什么。

## 1. 位宽、数值范围和精度

| 名称 | 意思 | 怎样理解 |
| --- | --- | --- |
| **FP16**（16 位浮点数） | 常见 IEEE 半精度，1 位符号、5 位指数、10 位小数 | 指数决定能覆盖的大小，小数位决定同一区间里能分多细 |
| **BF16**（BFloat16，脑浮点 16 位） | 1 位符号、8 位指数、7 位小数 | 和常规 32 位浮点数的指数位数相同，在同一区间的离散间距比 FP16 粗 |
| **INT8**（8 位整数） | 8 位编码最多有 256 种状态 | 要配合比例等信息才能解释原始小数单位 |
| **INT4**（4 位整数） | 4 位最多有 16 种状态 | 两个码可以放进一个字节，实际精度取决于分组、范围与比例 |

FP16 最大有限正值为 65504。像 100000 这样的数超出这个范围，但 BF16 可以表示其附近的值。两种都是 2 字节一个数，动态范围与分辨率不同。量化权重的存储位宽、激活位宽和乘加内部的累积位宽可以各不相同。**激活**就是样本通过网络时产生的中间数字。 [NVIDIA 浮点格式说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)

## 2. 用一个比例把四个权重放到整数格

设某次线性计算的权重是 `[-1.4, -0.73, 0.12, 1.0]`，输入向量是 `[1, 2, 3, -1]`。用**对称量化**，让 0 对应 0，把 INT4 的整数范围选为 `[-7, 7]`，正负两边一样宽。这里留出最负的编码，不同格式可以做别的选择。

取这一组中最大的绝对值 1.4，用它除以最大码 7，得到**比例因子** `s = 0.2`。编码时用原值除以 `s` 再四舍五入到整数；恢复近似值时用整数乘 `s`。逐项算如下：

| 原值 | 除以比例后约为 | 整数码 | 恢复近似值 |
| ---: | ---: | ---: | ---: |
| -1.40 | -7.00 | -7 | -1.40 |
| -0.73 | -3.65 | -4 | -0.80 |
| 0.12 | 0.60 | 1 | 0.20 |
| 1.00 | 5.00 | 5 | 1.00 |

用原权重做点积是 `−3.500`，恢复后做同样的点积是 `−3.400`。本例没有超出整数范围，单个权重的最大舍入误差不超过半格，即 `s/2`。模型通常会按组为权重存比例，减少一个大权重让整组的格子都变粗的情况。若使用**非对称量化**，往往还要保存零点，也就是实数 0 对应哪个整数码。

## 3. 权重预算和缓存预算分开算

十亿个权重按 FP16、INT8、INT4 存放，裸数据分别约为 2 GB、1 GB 和 0.5 GB。这里 GB 按十亿字节计。若 INT4 每 128 个权重额外保存一个 FP16 比例，仅这些比例就再占约 15.6 MB。实际模型还可能有未量化的层、对齐填充和格式元数据。

**权重量化**压缩的是模型参数，跟着模型加载。**KV Cache 量化**压缩的是注意力的 K（Key，匹配用的键）和 V（Value，读取用的值），跟着请求、批次和上下文长度变化。F-06 的 48 MiB 是固定条件下的 16 位缓存内容。如果只把每个缓存数改为 8 位，裸缓存内容约减半，仍需计入比例等额外数据。只发布 INT4 权重文件时，运行时的缓存仍可能按 BF16 保存。 [KIVI 键值缓存量化论文](https://arxiv.org/abs/2402.02750)

## 4. GPTQ 与 AWQ 各在保护什么？

两者都是 **PTQ（Post-Training Quantization，训练后量化）**的权重量化方法。校准是用一小批代表性输入观察网络数值，以帮助决定怎样压缩。

- **GPTQ** 是针对预训练生成式 Transformer 的权重量化方法。它用校准输入近似估计权重误差对层输出的影响，逐步量化并补偿其他权重。 [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- **AWQ（Activation-aware Weight Quantization，激活感知权重量化）**从校准激活判断哪些输入通道敏感，通过等价缩放保护这些通道，再做低位量化。 [AWQ 论文](https://arxiv.org/abs/2306.00978)

部署前记录原模型和量化版本，核对目标引擎能否读对应的打包格式，以及硬件是否有适合该格式的算子。再用同一组报表样本对比字段口径、数值正确性、加载空间与运行延迟。纯 Python 点积展示的是误差如何进入计算，实际模型质量需要在留出任务上单独验证。

## 5. 用 Python 复核

保存为 `quantize_demo.py` 并运行 `python3 quantize_demo.py`，只需标准库。代码用二进制补码放入每个 4 位位置，`& 15` 取低 4 位，`<< 4` 把一半移到字节的高位。打包结果只包含权重码，比例另外存。全零输入采用比例 1 避免除以 0。

```python
from math import isclose

weights = [-1.4, -0.73, 0.12, 1.0]
inputs = [1.0, 2.0, 3.0, -1.0]


def quantize(values, bits):
    limit = 2 ** (bits - 1) - 1
    peak = max(abs(value) for value in values)
    scale = peak / limit if peak else 1.0
    codes = [max(-limit, min(limit, round(value / scale))) for value in values]
    restored = [code * scale for code in codes]
    return scale, codes, restored


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


reference = dot(weights, inputs)
print(f"浮点点积 {reference:.3f}")
for bits in (4, 8):
    scale, codes, restored = quantize(weights, bits)
    max_error = max(abs(a - b) for a, b in zip(weights, restored))
    assert max_error <= scale / 2 + 1e-12
    print(
        f"INT{bits}: 比例 {scale:.6f}, 编码 {codes}, "
        f"最大误差 {max_error:.6f}, 恢复后点积 {dot(restored, inputs):.3f}"
    )
_, codes, restored = quantize(weights, 4)
packed = bytes(((codes[i] & 15) << 4) | (codes[i + 1] & 15) for i in (0, 2))
print("四个 4 位整数打包", packed.hex(), f"共 {len(packed)} 字节")
assert isclose(dot(restored, inputs), -3.4)
assert quantize([0.0, 0.0], 4)[2] == [0.0, 0.0]
```

实跑结果：

```text
浮点点积 -3.500
INT4: 比例 0.200000, 编码 [-7, -4, 1, 5], 最大误差 0.080000, 恢复后点积 -3.400
INT8: 比例 0.011024, 编码 [-127, -66, 11, 91], 最大误差 0.003150, 恢复后点积 -3.494
四个 4 位整数打包 9c15 共 2 字节
```

下一篇 **F-09《多模态请求如何进入模型》**会把报表截图送进视觉编码器，看看图像又是怎样占用输入位置的。 [学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)

[F-07 TTFT、TPOT、ITL、吞吐与端到端延迟](f-07-inference-latency-metrics.md) · [F-09 多模态请求如何进入模型](f-09-multimodal-input.md)
