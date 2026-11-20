---
title: "LLM 入门 F-14｜FlashAttention 如何减少注意力的显存读写？"
description: "从一行注意力分数手算整行与分块 Softmax，理解 FlashAttention 为什么能保留数学结果，又少搬运完整的分数矩阵。文章给出 CPU 数值对照及可选的 GPU 测量脚本，区分临时计算和跨步骤保存的键值缓存。"
keywords: [llm, python]
---

![FlashAttention 将一次注意力计算分成较小的块，减少在大显存和片上存储之间搬运中间结果。](assets/f-14-flashattention-cover.jpg)

还是那条「退款金额缺失」的报告任务。假如助手同时收到很长的字段字典，一次注意力要为许多输入位置互相打分。问题不只有「算多少次」，还有「这些分数写到哪里、下一步又从哪读」。FlashAttention 优化的正是这个计算过程。先修见 F-02 的注意力实验和 F-06 的键值缓存。 [F-02 手算示例](https://blog.lacknb.cn/articles/2026/11/20/1795135053358.html)

## 1. 看懂要搬的那张数字表

在单头注意力中，**Q（Query，查询）**和 **K（Key，键）**点积产生分数，再用 **Softmax** 把一行分数变成总和为 1 的权重，最后按权重混合 **V（Value，值）**。注意力的一个「头」表示一组独立的查询和读值计算。假设有 `N` 个查询位置、`N` 个键位置，分数表就是 `N × N`。例如 `N = 4096`，单头一张采用 16 位浮点数的完整表就占 `4096² × 2 = 32 MiB`。MiB 表示 `1024²` 字节。批次和注意力头数还会放大这个数字。

在图形处理器 **GPU（Graphics Processing Unit）** 上，**HBM（High Bandwidth Memory，高带宽显存）**容量大，**SRAM（Static Random-Access Memory，静态随机存取存储器）**等片上存储离计算单元近，但空间更小。朴素实现可能先把整张分数表写到显存，再读回来做 Softmax，接着写权重，后面又把权重读回来乘 V。这里的 **I/O（Input/Output）**指不同层次存储之间的数据搬运。 [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

FlashAttention 把 Q、K、V 按块送到计算单元附近，尽量在块内完成打分与加权，只保留计算下一块所需的状态与最终输出。全量注意力的互相打分在数学上仍然成立，常规稠密场景的算术量仍随序列长度平方增长。改变的是中间数据落地与搬运方式。

## 2. Softmax 怎么边看边算？

用「金额」这个查询位置举例。为方便观察，设它能读取四条已出现的证据，缩放并应用可见规则后的分数为 `[0, 1, 2, 3]`，每条的 V 只取一维，分别为 `[2, 4, 6, 8]`。整行计算时，可以先从每项分数减去最大值 3，取指数后求加权平均。这种平移避免指数过大，最后的归一化会约掉共同倍率。

分块时，先看前两项，后两项还没读到。为每个查询记三个数：

| 符号 | 当前含义 | 前两项处理完 |
| --- | --- | ---: |
| `m` | 已见分数的最大值 | 1 |
| `l` | 以当前 `m` 为基准的指数和，也就是分母 | 1.367879 |
| `a` | 以当前 `m` 为基准的加权 V 之和，也就是分子 | 4.735759 |

第二块 `[2, 3]` 把最大值提高到 3。旧结果以 1 为基准，新结果以 3 为基准，直接相加会混用尺度。先把旧的 `l`、`a` 同时乘 `exp(1−3)`，再加上第二块按新基准算出的两部分。更新结束后 `m = 3`、`l ≈ 1.553002`、`a ≈ 10.848192`，输出 `a / l ≈ 6.985305`。真实注意力里 `a` 是向量，里面每个分量都能用同一个倍率更新。 [FlashAttention 论文 Algorithm 1](https://arxiv.org/abs/2205.14135)

这叫**在线 Softmax**。只要状态按同一基准更新，它和整行 Softmax 在实数数学上等价。有限精度下，运算顺序和舍入会造成小差异，验收时用误差容限。因果注意力里，整块不可见可以直接跳过，部分不可见的位置按遮罩排除。教学输入的四个位置全部可见。

## 3. 先在 CPU 上验算同一行

CPU 是中央处理器。下面只用 Python 标准库，保存为 `online_softmax.py` 后运行 `python3 online_softmax.py`。`exp` 表示取指数，`isclose` 按指定容差检查两条路径足够接近。

```python
from math import exp, isclose

scores = [0.0, 1.0, 2.0, 3.0]
values = [2.0, 4.0, 6.0, 8.0]
peak = max(scores)
weights = [exp(score - peak) for score in scores]
reference = sum(weight * value for weight, value in zip(weights, values)) / sum(weights)

peak = float("-inf")
mass = weighted_sum = 0.0
for start in range(0, len(scores), 2):
    block = scores[start:start + 2]
    new_peak = max(peak, max(block))
    old_scale = exp(peak - new_peak)
    weights = [exp(score - new_peak) for score in block]
    mass = mass * old_scale + sum(weights)
    weighted_sum = weighted_sum * old_scale + sum(
        w * v for w, v in zip(weights, values[start:start + 2])
    )
    peak = new_peak
    print(f"m={peak:.0f}, l={mass:.6f}, a={weighted_sum:.6f}")

result = weighted_sum / mass
assert isclose(result, reference, abs_tol=1e-12)
print(f"整行={reference:.6f}, 分块={result:.6f}, 误差={abs(result-reference):.3g}")
```

实跑结果：

```text
m=1, l=1.367879, a=4.735759
m=3, l=1.553002, a=10.848192
整行=6.985305, 分块=6.985305, 误差=1.78e-15
```

## 4. 在支持 CUDA 的设备上量显存与耗时

**CUDA** 是 NVIDIA 的 GPU 计算平台。下面是可在支持 Flash 后端的 CUDA 环境中运行的对照脚本，使用 PyTorch 的 **SDPA（Scaled Dot Product Attention，缩放点积注意力）**接口，显式指定 Flash 后端。按官方安装说明配置 CUDA 版 PyTorch，把脚本存为 `compare_flash.py`，运行 `python compare_flash.py`。两条路径固定同一组张量与因果规则，四个轴依次是批次、头数、位置数和头宽。记录最大绝对误差、预热后每轮平均毫秒数、已分配显存相对输入常驻状态的额外峰值。因果遮罩也预先分配，计时发生在 GPU 上。 [PyTorch SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用，跳过 FlashAttention 显存和耗时对照。")

torch.manual_seed(7)
shape = (1, 4, 2048, 64)
q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))
causal = torch.ones((2048, 2048), device="cuda", dtype=torch.bool).tril()


def naive():
    scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    scores.masked_fill_(~causal, float("-inf"))
    return scores.softmax(dim=-1) @ v


def flash():
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def measure(operation):
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        output = operation()
        del output
    end.record()
    end.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    return start.elapsed_time(end) / 20, peak / 1024**2


with torch.inference_mode():
    reference = naive().cpu().float()
    actual = flash().cpu().float()
    torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.02)
    print(torch.__version__, torch.cuda.get_device_name())
    print("最大绝对误差", (actual - reference).abs().max().item())
    for name, operation in (("naive", naive), ("flash", flash)):
        milliseconds, peak_mib = measure(operation)
        print(f"{name}: {milliseconds:.3f} ms, 运行额外峰值 {peak_mib:.1f} MiB")
```

本文实跑了 CPU 数值对照。本机无 CUDA，GPU 的耗时和显存需要在支持的设备上测得。若后端被拒绝，先记录 GPU 型号、PyTorch 版本、数据类型和头宽。强制 Flash 后端可以让不支持的组合直接暴露，避免把回退结果误读为 FlashAttention。

## 5. 跟 KV Cache 和 PagedAttention 对照

F-06 的 **KV Cache（键值缓存）**保留历史 K、V，生命周期跨多个解码步。**PagedAttention（分页注意力）**将这些缓存位置组织成块，减少分配和共享的浪费。FlashAttention 主要优化一次注意力计算内部的分数与权重搬运。两处都提到「块」，一个块面向临时计算，另一个面向缓存管理。后续 SERV-04 会从引擎的分块和调度继续讲。 [PagedAttention 论文](https://arxiv.org/abs/2309.06180)

下一篇 **F-07《TTFT、TPOT、ITL、吞吐与端到端延迟》**将用同一份固定负载观察引擎和界面两边的时间。 [学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)

[F-06 KV Cache 如何加速 Decode](f-06-kv-cache.md) · [F-07 TTFT、TPOT、ITL、吞吐与端到端延迟](f-07-inference-latency-metrics.md)
