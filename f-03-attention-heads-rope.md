---
title: "LLM 入门 F-03｜MHA、MQA、GQA 与 RoPE：注意力头怎样共享 K/V？"
description: "从「退款金额缺失」出发，逐步手算 Q、K、V 怎样从同一份隐藏向量投影出来，又怎样用打分和加权汇总更新「金额」的位置。再对照 MHA、GQA、MQA 如何共享 K/V，用 Qwen2.5 的配置算缓存，并用二维旋转理解 RoPE 如何把相对位置带进注意力。"
keywords: [llm, python]
---

![MHA 的 K/V 各自独立，GQA 按组共享，MQA 全部共享。RoPE 用向量旋转给打分带入相对位置。](assets/f-03-attention-heads-rope-cover.jpg)

还是上一篇的 CSV 备注「退款金额缺失」。当助手生成说明时，有的注意力头可能更关注「退款金额」对应哪个字段，有的头可能更多参考「缺失」对应什么处理规则。这只是说明多个头可以形成不同的读取权重，实际模型里的头未必有这么固定的分工。

如果有四个 Query 头，得为它们各存一套 K 和 V 吗？可以各存一套，也可以两两共用，甚至四个共用一套。这就是 **MHA、GQA、MQA** 的区别。同样是给注意力打分，**RoPE** 关心的是另一件事：两个 Token 的位置相隔多远，怎样反映在 Q 和 K 的点积里。

本篇沿用 [F-02 的 Q/K/V 与点积例子](https://blog.lacknb.cn/articles/2026/11/20/1795135053358.html)。把「退款」「金额」「缺失」「。」当作四个教学 Token，真实分词仍以目标模型的 Tokenizer 为准。

## 1. Q、K、V 是什么，从哪里来？

先只看一个头。对「退款」「金额」「缺失」这三个位置，自注意力的输入是一张隐藏向量表 `X`，每行对应一个位置。在第一层，这些表示来自 Token 嵌入及该层的输入处理。到后面的层，输入已经包含前面各层算出的上下文信息。自注意力里的 Q、K、V 都从这同一份 `X` 出发，各用一套学到的权重做投影。 [Transformer 原论文，第 3.2.2～3.2.3 节](https://arxiv.org/html/1706.03762#S3.SS2)

| 投影结果 | 可以怎么理解 | 在计算里怎么用 |
| --- | --- | --- |
| Q（Query，查询） | 我在当前位置想找什么 | 取当前查询位置的 Q，和各个可见位置的 K 做点积 |
| K（Key，键） | 我这个位置有哪些可用于匹配的特征 | 对每个查询给出一个匹配分数 |
| V（Value，值） | 我这个位置实际能提供什么信息 | 用匹配权重乘各个位置的 V，再把结果相加 |

「想找什么」只是帮助理解的说法。Q、K、V 都是一串数字，权重矩阵在训练中学得。下面令每行隐藏向量只有两个数，使用三张手写的 `2 × 2` 权重矩阵。忽略偏置和位置旋转，就能用纸笔复算各步。

```text
X 的三行分别对应「退款」「金额」「缺失」
X   = [[1, 0], [0, 1], [1, 1]]
W_Q = [[0, 1], [1, 0]]  # 交换两个分量
W_K = [[1, 0], [0, 1]]  # 保留两个分量
W_V = [[2, 0], [0, 4]]  # 两个分量分别放大 2 倍和 4 倍

Q = X × W_Q    K = X × W_K    V = X × W_V
```

矩阵乘法在这里就是让 `X` 的每一行去乘同一张权重矩阵。例如「金额」这一行是 `[0, 1]`，乘 `W_Q` 的两列分别得到 `0×0 + 1×1 = 1` 和 `0×1 + 1×0 = 0`，所以它的 Q 是 `[1, 0]`。用同一行乘 `W_K` 得到 `[0, 1]`，乘 `W_V` 得到 `[0, 4]`。三个位置都照此计算，就得到三张形状为 `[3, 2]` 的表。

| 位置 | 输入向量 | 这个位置发问用的 Q | 供别人匹配的 K | 供别人读取的 V |
| --- | --- | --- | --- | --- |
| 1「退款」 | `[1, 0]` | `[0, 1]` | `[1, 0]` | `[2, 0]` |
| 2「金额」 | `[0, 1]` | `[1, 0]` | `[0, 1]` | `[0, 4]` |
| 3「缺失」 | `[1, 1]` | `[1, 1]` | `[1, 1]` | `[2, 4]` |

### 只算「金额」这一个查询位置

1. **用 Q 和 K 打分。** 取「金额」的 Q `[1, 0]`，依次和三行 K 做点积，得到 `[1, 0, 1]`。Q 和 K 要同宽才能逐分量相乘再相加。按头宽 `d_k = 2` 缩放后，分数约为 `[0.707, 0, 0.707]`。所有位置的 Q 都算一遍会得到一张 `3 × 3` 的打分表，这里只取它的第 2 行。
2. **用遮罩限制可见位置，再归一化。** 「缺失」在第 2 个位置之后，先把它对应的分数置为负无穷。Softmax 对其余分数取指数后归一化，三个位置的权重约为 `[0.670, 0.330, 0]`。
3. **用权重混合 V。** 计算 `0.670 × [2, 0] + 0.330 × [0, 4] + 0 × [2, 4]`。用完整精度算，输出约为 `[1.340, 1.321]`。这一行就是这个头为「金额」算出的注意力结果，包含了可见位置贡献的信息。

把全部查询一起写，公式就是 `Softmax(Q × Kᵀ / √d_k + mask) × V`。`Kᵀ` 是把 K 的行列互换，乘出来的打分表每行对应一个查询，每列对应一个被查位置。`mask` 给可见位置加 0，给不可见位置加负无穷。Softmax 沿每一行归一化。V 必须和 K 按位置一一对应，它的向量宽度决定了这个头的输出宽度。

可以用一个小改动区分 **K 负责匹配**和 **V 负责提供内容**。保持 Q 和 K 不变，把「退款」的 V 从 `[2, 0]` 改成 `[4, 0]`。权重仍约为 `[0.670, 0.330, 0]`，但输出的第一项从 `1.340` 变成约 `2.679`。因此，权重和参与加权的内容是两个可以分开检查的量。 [Transformer 原论文，第 3.2.1 节](https://arxiv.org/html/1706.03762#S3.SS2.SSS1)

## 2. MHA、GQA、MQA 怎样分配 K/V？

多个头并行做上一节的计算。每个 Query 头有自己的 Q 投影，各自给可见位置打分并算出一行结果。模型把各个头的结果拼起来，再做一次输出投影，供后续层使用。 [Transformer 原论文，第 3.2.2 节](https://arxiv.org/html/1706.03762#S3.SS2.SSS2)

接下来把 Query 头固定为四个，记作 Q₁～Q₄。「KV₁」指第 1 套 Key 和 Value，包含两个张量。它们为序列里的每个 Token 保有一行向量。下图的箭头表示「这个 Query 头读取哪一套 K/V」，每行都描述同一个注意力层。

```text
MHA · 四套 K/V
Q1 → KV1    Q2 → KV2    Q3 → KV3    Q4 → KV4

GQA · 两套 K/V
Q1 → KV1    Q2 → KV1    Q3 → KV2    Q4 → KV2

MQA · 一套 K/V
Q1 → KV1    Q2 → KV1    Q3 → KV1    Q4 → KV1
```

| 名称 | Query 头数 | K/V 头数 | 分配方式 |
| --- | ---: | ---: | --- |
| MHA（Multi-Head Attention） | 4 | 4 | 每个 Query 头配一套自己的 K/V |
| GQA（Grouped-Query Attention） | 4 | 2 | 两个 Query 头一组，每组配一套 K/V |
| MQA（Multi-Query Attention） | 4 | 1 | 所有 Query 头共用一套 K/V |

在常见的等分实现里，Query 头数要能被 K/V 头数整除。比如四个 Query 头分到两组，每组两个。分组是模型结构里的固定关系，每次推理会用同样的分组。 [MQA 原论文](https://arxiv.org/abs/1911.02150) · [GQA 原论文，图 2](https://arxiv.org/html/2305.13245#S2.SS2)

**共用 K/V 后，两个 Query 头会得到同一个结果吗？** 沿用上一节的 K 和 V，仍然只算「金额」这个位置。Q₁ 取上一节的 `[1, 0]`，让第二个头用自己的投影得到 Q₂ `[0, 1]`。这时两个头读的是同一套 K/V，点积和结果却可以不同：

| Query 头 | 和「退款」「金额」的点积（缩放前） | Softmax 权重（含遮罩） | 加权后的 V |
| --- | --- | --- | --- |
| Q₁ `[1, 0]` | `[1, 0]` | `[0.670, 0.330, 0]` | `[1.340, 1.321]` |
| Q₂ `[0, 1]` | `[0, 1]` | `[0.330, 0.670, 0]` | `[0.660, 2.679]` |

共享的是**每个位置的 K 和 V 向量**。各头仍会用自己的 Q 算一排权重，分别混合相同的 V。下面的缓存公式数的是需要保存多少套 K/V，因此用 K/V 头数。

例如在第 6 节的小模型里，输入有 4 个位置，隐藏宽度 32，每头宽度 8。使用 GQA 时，每个位置从同一行隐藏向量投影出 `4 × 8 = 32` 个 Q 分量，和 `2 × 8 = 16` 个 K 分量以及 16 个 V 分量。拆成头以后，Q 是 `[1, 4, 4, 8]`，K 和 V 各是 `[1, 2, 4, 8]`。这里的四个轴依次是批次、头数、序列位置、头宽。

## 3. 算一次 K/V 缓存能省多少

助手逐个生成 Token 时，可以保留先前位置已经算好的 K 和 V，供下一步直接读取。这些留下来的张量叫 **KV Cache**。每个新位置用自己的 Q 来发问，读取历史位置的 K 和 V，所以常规 KV Cache 保存的是各层的 K 和 V。 [Transformers 缓存说明](https://huggingface.co/docs/transformers/v5.13.0/en/cache_explanation)

以一条序列为例，假设各层配置相同、K/V 使用相同的数值类型、完整缓存所有位置，张量本身的容量这样算：

```text
K/V 总字节数
= 2 × 层数 × 序列长度 × K/V 头数 × 每头宽度 × 每个数的字节数

2 表示 K 和 V 两份张量
```

拿 **Qwen2.5-0.5B-Instruct** 来代入。公开配置里有 24 层、14 个 Query 头、2 个 K/V 头，隐藏宽度 896，每头宽度是 `896 ÷ 14 = 64`。每个 K/V 头服务 7 个 Query 头。假设缓存用 BF16，每个数占 2 字节。如果单条序列累计保留 4096 个位置，GQA 的容量就是 `2 × 24 × 4096 × 2 × 64 × 2 = 50,331,648` 字节，也就是 **48 MiB**。 [固定版本的 Qwen 模型配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json)

把 Query 头数、宽度和其余条件都固定，只在纸上切换 K/V 头数，可以得到这三种结果：

| 方案 | Query 头数 | K/V 头数 | 每缓存 1 个 Token，计入全部层 | 缓存 4096 个 Token |
| --- | ---: | ---: | ---: | ---: |
| MHA 对照配置 | 14 | 14 | 84 KiB | 336 MiB |
| Qwen 实际头数配置（GQA） | 14 | 2 | 12 KiB | 48 MiB |
| MQA 对照配置 | 14 | 1 | 6 KiB | 24 MiB |

`1 KiB = 1024 B`，`1 MiB = 1024 KiB`。这里的 GQA 缓存是 MHA 对照的 **1/7**。这张表只计算 K/V 张量内容，在实际服务里还要另计模型权重、临时张量、缓存块与运行时的开销。缓存更小可以减少解码时的 K/V 读取量，端到端延迟还取决于注意力实现、硬件和请求如何批处理。MHA 与 MQA 两行是用来理解比例的对照，动手实验会创建三组独立的教学模型。 [MQA 论文的解码带宽讨论](https://arxiv.org/abs/1911.02150) · [GQA 论文的实测设置](https://arxiv.org/abs/2305.13245)

## 4. RoPE 怎样让打分带上相对位置

用「退款金额缺失」来想。如果两个 Token 的初始表示一样，模型得有办法在打分时区分它们一个在前、一个在后。**RoPE（Rotary Position Embedding，旋转位置编码）**先把 Q 和 K 的分量成对看成许多二维向量。每一对分量对应自己的旋转频率，再按 Token 位置决定旋转角度。旋转发生在算 Q·K 之前。在 Qwen2 的这里，V 仍使用原投影。 [RoPE 原论文](https://arxiv.org/abs/2104.09864) · [Qwen2 旋转与缓存调用](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2/modeling_qwen2.py#L217)

先只拿一对分量。假设原始 Q 和 K 都是向右的单位向量 `[1, 0]`，每增加一个位置就转 30°。两个单位向量的点积，等于它们夹角的余弦。

| Q 的位置 | K 的位置 | 两者转过的角度 | 夹角 | 旋转后点积 |
| ---: | ---: | --- | ---: | ---: |
| 3 | 1 | 90°、30° | 60° | 0.500 |
| 4 | 2 | 120°、60° | 60° | 0.500 |
| 3 | 2 | 90°、60° | 30° | 0.866 |

第一行整体右移一个位置，两个箭头都多转 30°，相对夹角没变，点积仍然是 0.500。只把 K 从位置 1 移到位置 2，夹角缩小，点积也就变了。

这个性质在一般情形同样成立。把位置 `p` 对应的旋转记为 `R(p)`，旋转后的点积 `(R(p)Q) · (R(s)K)` 等于 `Q · R(s-p)K`。位置对点积的影响通过相对位移 `s-p` 进入。真实模型的分量对采用不同频率，Q、K 也包含各自的内容，因此点积变化还受向量内容影响。上面 0.500 到 0.866 是这对教学向量的结果。 [RoPE 原论文，第 3.2 节](https://arxiv.org/abs/2104.09864)

在 Qwen2.5 里，这几件事有明确的先后顺序：

```text
隐藏状态 → Q/K/V 投影（头数由 GQA 配置决定）
         → 按各自位置旋转 Q、K（RoPE）
         → 缓存旋转后的 K 和原投影的 V
         → 每个 Q 头读取同组的 K/V，打分并处理因果遮罩
         → 对分数做 Softmax，按得到的权重混合 V
```

到了下一个生成步，历史 K 已经在正确的位置上转好了，可以直接从缓存读出来。新的 Q 和 K 要按新 Token 的位置旋转，再进入同一条计算链路。 [Qwen2 前向计算](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2/modeling_qwen2.py#L217)

## 5. 用 Python 验证相对位移和缓存容量

只需 Python 标准库。把下面的代码保存为 `rope_and_cache.py`，运行 `python3 rope_and_cache.py`。它先验证同时平移 Q、K 不改变点积，再复算 4096 个位置的容量表。

```python
from math import cos, isclose, radians, sin


def rotated_dot(query_position, key_position):
    # 教学情形：原始 Q 和 K 都是 (1, 0)，每个位置转 30°。
    query_angle = radians(30 * query_position)
    key_angle = radians(30 * key_position)
    return cos(query_angle) * cos(key_angle) + sin(query_angle) * sin(key_angle)


original = rotated_dot(3, 1)
shifted = rotated_dot(4, 2)
closer = rotated_dot(3, 2)
assert isclose(original, shifted)
assert not isclose(original, closer)
print(f"RoPE 同时右移：{original:.3f} → {shifted:.3f}")
print(f"RoPE 缩短间隔：{original:.3f} → {closer:.3f}")

layers = 24
sequence_length = 4096
head_dim = 64
bytes_per_value = 2
for name, kv_heads in (("MHA", 14), ("GQA", 2), ("MQA", 1)):
    per_token = 2 * layers * kv_heads * head_dim * bytes_per_value
    total = sequence_length * per_token
    print(f"{name}: {per_token / 1024:.0f} KiB/Token, {total / 1024**2:.0f} MiB")
```

实测输出：

```text
RoPE 同时右移：0.500 → 0.500
RoPE 缩短间隔：0.500 → 0.866
MHA: 84 KiB/Token, 336 MiB
GQA: 12 KiB/Token, 48 MiB
MQA: 6 KiB/Token, 24 MiB
```

可以把 `sequence_length` 改成 `8192` 再运行。三种方案的每 Token 容量不变，总容量翻倍。

## 6. 在真实 Qwen2 模块里看缓存形状

这一节可选。用三个随机初始化的单层模型，对照同一份四 Token 输入。沿用前几篇的 Python 3.11 或 3.12 虚拟环境即可。环境里还没有依赖时，运行下面的安装命令。脚本在 CPU 上直接创建模型，运行时无需下载权重。

```bash
python -m pip install "torch==2.11.0" "transformers==5.13.0" "tokenizers==0.22.2"
```

把代码存成 `inspect_attention_cache.py`，再运行 `python inspect_attention_cache.py`。本次用 FP32，每个数占 4 字节。缓存形状的四个轴依次是「批次数、K/V 头数、已缓存位置数、每头宽度」。

```python
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

# 四个教学 Token：退款、金额、缺失、句号。
input_ids = torch.tensor([[2, 3, 4, 5]])
for name, kv_heads in (("MHA", 4), ("GQA", 2), ("MQA", 1)):
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(config).float().eval()

    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, output_attentions=True)
        cache = output.past_key_values.layers[0]

    attention = model.model.layers[0].self_attn
    assert cache.keys.shape == cache.values.shape == (1, kv_heads, 4, 8)
    assert output.attentions[0].shape == (1, 4, 4, 4)
    assert output.logits.shape == (1, 4, 16)
    total_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in (cache.keys, cache.values)
    )
    print(
        f"{name}: Q/K 投影宽度 "
        f"{attention.q_proj.out_features}/{attention.k_proj.out_features}, "
        f"K/V 各 {tuple(cache.keys.shape)}, 合计 {total_bytes} B"
    )
```

实测输出：

```text
MHA: Q/K 投影宽度 32/32, K/V 各 (1, 4, 4, 8), 合计 1024 B
GQA: Q/K 投影宽度 32/16, K/V 各 (1, 2, 4, 8), 合计 512 B
MQA: Q/K 投影宽度 32/8, K/V 各 (1, 1, 4, 8), 合计 256 B
```

观察缓存的第二维。K/V 头数从 4 变 2 再变 1，缓存容量跟着按比例减少。Q 的投影宽度始终是 32，因为四个 Query 头各占 8 个分量。注意力权重的形状始终是 `[1, 4, 4, 4]`，四个轴分别是批次、Query 头、查询位置和被查询位置。每个 Query 头仍然要做一份自己的打分。

想跟源码时，打开固定版本的 `Qwen2Attention.forward()`，沿三处往下看：

1. 找 `q_proj`、`k_proj`、`v_proj`。Q 投影宽度来自 Query 头数，K 和 V 投影宽度来自 K/V 头数。 [Qwen2 投影层](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2/modeling_qwen2.py#L200)
2. 找 `apply_rotary_pos_emb()` 和 `past_key_values.update()`。先旋转 Q、K，再把旋转后的 K 和原投影的 V 加到缓存。 [Qwen2 前向计算](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2/modeling_qwen2.py#L217)
3. 进入本实验选用的 `eager_attention_forward()`。它用 `repeat_kv()` 把 K/V 在计算时按组对齐到 Query 头，缓存里仍是扩展前的头数。 [Qwen2 的 eager 注意力实现](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen2/modeling_qwen2.py#L161)

## 下一篇

路线图接下来是 **F-13《MoE 如何选择专家与分配计算》**。我们会从注意力回到每一层的 FFN，看一个 Token 怎样被送到部分专家，再把各专家的结果合起来。想继续看缓存怎样随逐步生成而增长，可以到后面的 **F-06「KV Cache 如何加速 Decode」**。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
