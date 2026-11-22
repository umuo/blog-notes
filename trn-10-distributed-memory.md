---
title: "LLM 训练 TRN-10｜分布式训练的显存账本，FSDP 与 ZeRO"
description: "沿用 TRN-02 的 14,496 参数模型，拆开参数、梯度、AdamW 状态、激活和临时缓冲的内存开销。对照 FSDP 与 ZeRO 的通信路径，并用两个 CPU 进程核对实际分片、同批更新和检查点恢复。"
keywords: [llm, training, pytorch, deepspeed]
---

![分布式进程分别持有训练状态的分片，通过集合通信取回层权重，并保存分片检查点。](assets/trn-10-distributed-memory-cover.jpg)

> **生成工具**：Codex\
> **生成模型**：gpt-6.2-sol

[TRN-02](trn-02-tiny-language-model.md) 用 14,496 个参数训练退款语料。一次前向处理两条样本，每条六个 Token。把这个训练放到两块设备上，如果每块各算一行，为什么每块仍可能保留完整的权重？如果把权重各放一半，轮到某一层时又怎样拿齐计算所需的参数？

**FSDP（Fully Sharded Data Parallel，全分片数据并行）**和 **ZeRO（Zero Redundancy Optimizer，零冗余优化器）**都通过分片减少训练状态副本。本篇先把一个优化器步拆成可以数清的字节，再实际跑两个 CPU 进程，验证 FSDP 怎样持有分片以及怎样恢复。GPU（Graphics Processing Unit，图形处理器）的峰值显存和吞吐需要在目标硬件上另测。 [PyTorch FSDP2](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html) · [ZeRO 原论文](https://arxiv.org/abs/1910.02054)

## 1. 单设备里占内存的是什么？

| 项目 | 跨训练步骤怎样使用 | 本例的计算 |
| --- | --- | --- |
| 参数 | 前向和反向都要读取，优化器更新它。 | 14,496 个 FP32 元素，每项 4 字节，共 57,984 字节。 |
| 梯度 | 反向产生，供优化器更新。后续可以清空。 | 这里保留一次更新后的梯度，仍是 57,984 字节。 |
| 优化器状态 | AdamW 为每个参数保留梯度的一阶和二阶滑动统计。 | 两个 FP32 张量，共 115,968 字节。每个参数张量还有一个小的步数标量。 |
| 激活 | 前向产生的中间量，反向可能需要重用。 | 例如 `[B,T,H]=[2,6,32]` 的一个 FP32 表示要 1,536 字节。仅此一个张量，整层还会产生更多中间量。 |
| 临时缓冲 | 算子工作区、通信缓冲、临时聚合权重和分配器缓存。 | 大小由内核、分片单元、并发通信和运行时决定。 |

`B` 是当前进程的样本数，`T` 是每条序列的 Token 数，`H` 是隐藏向量宽度。**FP32** 是每个元素占 32 位的浮点格式。只算参数、梯度和两个等长的 AdamW 状态，本例是 **231,936 字节，即 226.5 KiB**，其中 1 KiB 等于 1,024 字节。AdamW 的矩状态通常在第一次更新时才建立，所以只观察初始化后的内存会低估训练用量。

常见混合精度账本也可能凑成每参数 16 字节，例如 2 字节计算权重、2 字节梯度、4 字节 FP32 主权重和 8 字节 Adam 的两份矩状态。这和本例的 16 字节恰巧相同，组成不同。FSDP2 可让优化使用原精度的权重分片，并按混合精度策略构造临时计算权重。要以实际训练配置和张量类型为准。激活重计算（activation checkpointing）通过反向时重新执行部分前向，减少保存的激活，这与把完整训练状态写到磁盘的检查点各有用途。 [FSDP2 混合精度配置](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html)

## 2. 哪些东西分片，哪些通信会发生？

**Rank** 是分布式进程编号。以下设四个 rank，`N=4`，且各分片一样大。**DDP（Distributed Data Parallel，分布式数据并行）**让各 rank 处理不同样本，通常各自保存完整训练状态，再同步梯度。**分片（sharding）**让不同 rank 持有训练状态的不同部分。

| 方式 | 参数在各 rank 的常驻方式 | 梯度 | 优化器矩状态 | 按本例估算每 rank |
| --- | --- | --- | --- | ---: |
| 单设备或 DDP | 完整 | 完整 | 完整 | 231,936 字节 |
| ZeRO-1 | 完整 | 完整 | 切成 N 份 | 144,960 字节 |
| ZeRO-2 | 完整 | 切成 N 份 | 切成 N 份 | 101,472 字节 |
| ZeRO-3 或全分片 FSDP | 切成 N 份 | 切成 N 份 | 切成 N 份 | 57,984 字节 |

表里的估算只计训练状态的理想等分，先省略步数标量、填充、激活与临时缓冲。例如 ZeRO-2 的计算是 `57,984 + (57,984 + 115,968) / 4`。框架不同，即使分片范围相同，分组、预取、同步时点、偏移和配置项也可能不同。ZeRO-1 和 ZeRO-2 各 rank 更新自己负责的参数，再把更新后的权重同步到需要完整副本的 rank。 [DeepSpeed ZeRO 阶段定义与配置](https://deepspeed.readthedocs.io/en/latest/zero3.html)

用两个 rank 做一层的手算。rank 0 暂存权重 `[w0,w1]`，rank 1 暂存 `[w2,w3]`。

1. **All-gather（全收集）**把两份拼齐，各 rank 得到这一组完整权重来算自己的样本。以 FSDP2 的 `reshard_after_forward=True` 为例，前向后释放聚合权重，反向计算前再收齐。
2. 两个 rank 对同一组参数分别产生梯度 `[2,4,6,8]` 和 `[4,6,8,10]`。**Reduce-scatter（规约散发）**先合并对应位置，再各发一段。若先求和再除以 2，rank 0 得 `[3,5]`，rank 1 得 `[7,9]`，各自只保留负责的部分。
3. 每个 rank 用本地梯度分片和本地优化器状态更新对应参数，下一轮需要读取时再聚合。

对照 DDP 的 **All-reduce（全规约）**，上例得到的四项平均梯度会发回每个 rank。全分片的峰值仍受每次收齐多少权重、通信是否预取、梯度累积，以及激活何时释放影响。把整个模型作为一次聚合单元更好理解，训练大模型通常按 Transformer 块组织分片单元。 [PyTorch FSDP2 的分片与通信](https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html)

## 3. 用两个 CPU 进程复核字节数和续训

在空目录先运行 TRN-01 的数据准备代码，保留 `train.bin` 和 `manifest.json`，再保存 TRN-02 的 `train_tiny.py`。复用 Python 3.12 和 PyTorch 2.11.0。此处固定两步的窗口起点，单进程每步算两行，两进程每个 rank 算一行，**全局 batch 都是两行**。训练第一步的两行各有 6 和 5 个有效标签，因此损失要除以全局 11 项。FSDP 在规约时按 rank 平均局部梯度，本例把每个局部损失总和先乘 rank 数，再除以全局有效标签数。

把下面代码保存为 `shard_demo.py`。单进程先给出基线，再用两进程对照。代码用 Gloo（PyTorch 支持的 CPU 集合通信后端），并用 **DCP（Distributed Checkpoint，PyTorch 分布式检查点）**保存和恢复分片。

```python
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from train_tiny import IGNORE, TinyLM, batch, load_shard, summed_loss


def tensor_bytes(tensor):
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.numel() * local.element_size()


def ledger(model, optimizer):
    states = optimizer.state.values()
    moments = sum(tensor_bytes(state[key]) for state in states
                  for key in ("exp_avg", "exp_avg_sq"))
    return {
        "parameters": sum(tensor_bytes(p) for p in model.parameters()),
        "gradients": sum(tensor_bytes(p.grad) for p in model.parameters()),
        "adam_moments": moments,
        "adam_steps": sum(tensor_bytes(state["step"]) for state in states),
    }


def build(vocab, mesh):
    torch.manual_seed(2026)
    model = TinyLM(vocab)
    if mesh is not None:
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
    return model, torch.optim.AdamW(model.parameters(), lr=0.003, eps=1e-5, foreach=False)


def update(model, optimizer, shard, step, rank, world):
    starts = torch.tensor([0, 6] if step == 1 else [1, 7])
    x, y, mask = batch(shard, starts[rank::world])
    valid = y.ne(IGNORE).sum()
    if world > 1:
        dist.all_reduce(valid, op=dist.ReduceOp.SUM)
    optimizer.zero_grad(set_to_none=True)
    # FSDP averages gradients over ranks. Scale the local numerator by world.
    (summed_loss(model, x, y, mask) * world / int(valid)).backward()
    optimizer.step()


def snapshot(model, optimizer, step):
    weights, state = get_state_dict(model, optimizer)
    return {"model": weights, "optimizer": state, "step": torch.tensor(step)}


def train(rank, world):
    manifest = json.loads(Path("manifest.json").read_text())
    vocab = manifest["vocab_size"]
    data = load_shard(Path("."), "train.bin", manifest["shards"]["train.bin"], vocab)
    mesh = init_device_mesh("cpu", (world,)) if world > 1 else None
    model, opt = build(vocab, mesh)

    update(model, opt, data, 1, rank, world)
    print(json.dumps({"world": world, "rank": rank, **ledger(model, opt)}), flush=True)

    if world == 1:
        update(model, opt, data, 2, rank, world)
        torch.save(model.state_dict(), "single-reference.pt")
        return

    dcp.save(snapshot(model, opt, 1), checkpoint_id="sharded-ckpt")
    resumed, resumed_opt = build(vocab, mesh)
    state = snapshot(resumed, resumed_opt, 0)
    dcp.load(state, checkpoint_id="sharded-ckpt")
    set_state_dict(resumed, resumed_opt,
                   model_state_dict=state["model"], optim_state_dict=state["optimizer"])
    assert state["step"].item() == 1

    update(model, opt, data, 2, rank, world)
    update(resumed, resumed_opt, data, 2, rank, world)
    for direct, restored in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(direct.to_local(), restored.to_local())

    single = torch.load("single-reference.pt", weights_only=True, map_location="cpu")
    max_diff = 0.0
    for name, param in model.named_parameters():
        global_param = param.full_tensor().detach()
        max_diff = max(max_diff, float((global_param - single[name]).abs().max()))
        torch.testing.assert_close(global_param, single[name], atol=2e-6, rtol=2e-5)
    if rank == 0:
        print(json.dumps({"resume_exact": True, "global_batch": 2,
                          "max_diff_from_single": max_diff}), flush=True)


def main():
    torch.set_num_threads(1)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (1, 2):
        raise ValueError("run with one process or two")
    if world == 1:
        train(0, 1)
        return
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        train(dist.get_rank(), world)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

```bash
python shard_demo.py
GLOO_SOCKET_IFNAME=lo0 torchrun --master-addr=127.0.0.1 --master-port=29649 \
  --nproc-per-node=2 shard_demo.py
```

命令在 macOS 上把 Gloo 绑定到回环接口 `lo0`。Linux 一般使用 `lo`，端口可替换为本机未占用的端口。同一目录要能让两个进程读写。示例只在本机通信。

| 实测张量载荷 | 单进程 | FSDP 两进程，每 rank |
| --- | ---: | ---: |
| 参数 | 57,984 字节 | 28,992 字节 |
| 梯度 | 57,984 字节 | 28,992 字节 |
| AdamW 两份矩 | 115,968 字节 | 57,984 字节 |
| AdamW 步数标量 | 68 字节 | 68 字节 |

前三项在每个 rank 上正好减半。步数标量仍按参数张量逐个保存。这个数字按本地张量元素数和元素大小计算，记录时点是第一次 `optimizer.step()` 之后，梯度仍在。它描述所列张量的载荷，无法替代进程总内存或 GPU 峰值。实验固定两边的 AdamW 设置，包括 `eps=1e-5`。聚合两进程的权重后，与单进程两步结果的最大绝对差为 **2.61 × 10⁻⁷**，在代码设定的公差内。两进程本身从检查点恢复后再跑一步，与未中断路径逐元素相等。

## 4. 分片检查点里需要保存什么？

上面的 DCP 实验让两个 rank 共同保存权重分片、优化器分片和已完成的步数。重建相同的分片模型后，先给 DCP 一份目标状态结构，再把读出的数据交给 `set_state_dict`。真实训练中还要保存学习率调度、随机数、数据采样或数据流位置，以及混合精度缩放状态。这个实验的输入窗口按步数固定，所以恢复到第 1 步就能直接取第 2 步窗口。DCP 支持在符合条件时加载到新的分片布局，改变 rank 数仍要重新验证训练语义和数据进度。本例在相同的两进程布局中验证恢复。 [PyTorch 2.11 Distributed Checkpoint](https://docs.pytorch.org/docs/2.11/distributed.checkpoint.html)

DeepSpeed ZeRO 的训练检查点由各进程共同写出和恢复。官方文档提醒，ZeRO-3 若在保存后立刻恢复，应重新初始化引擎，再加载检查点。导出用的合并 FP32 权重便于推理、分发或初始化另一轮训练。要续训，应保留对应的优化器、进度和其它训练状态。 [DeepSpeed 检查点与权重导出](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)

## 5. 有 GPU 时怎样比较峰值和吞吐？

以单 GPU 为基线，记录 GPU 型号、框架与通信后端版本、精度策略和模型分片单元。再跑 DDP、FSDP2 或 DeepSpeed ZeRO 的对应配置。比较时锁定同一份数据、窗口长度、优化器和每次更新的全局 batch。

1. **锁定工作量。** 每 rank 微批样本数 × rank 数 × 累积微步数等于全局样本数。例如单卡微批 2、累积 4 步，与两卡每 rank 微批 2、累积 2 步都得到 8 条。若有忽略标签，同时核对有效 Token 数及损失归一化。
2. **先热身再测峰值。** 跑过至少一次优化器更新，确保 AdamW 状态已建立。热身后在各 rank 同步设备，再调用 `torch.cuda.reset_peak_memory_stats()`。在相同的若干优化器步结束后，再同步设备并读取 `torch.cuda.max_memory_allocated()`，按 rank 报告并取最大值。另记 `max_memory_reserved()`，以区分张量分配与缓存分配器保留的内存。
3. **按完整训练步算吞吐。** 同步起止，记录这段时间所有 rank 处理的有效 Token。用全局有效 Token 总数除以最慢 rank 的耗时，得到 Token/s。把写检查点的耗时另列，除非指标明确包含保存。
4. **对照恢复。** 在等量步数打断一次，用同一套检查点工具恢复，再检查已完成步数、接下来的样本、损失与权重。若对照跨布局恢复，预先约定数值公差和数据采样规则。

本地实测覆盖 CPU 分片、更新和恢复。CUDA 峰值、GPU 吞吐和 DeepSpeed 运行结果由对应环境的实测补充。 [PyTorch CUDA 内存说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-memory-management)

## 下一篇

按路线图行序接下来是 [**TRN-03《选继续预训练、SFT 还是偏好优化》**](trn-03-training-objectives.md)。继续以退款报告为例，比较原始文档、示范答复和偏好对各自提供了什么训练信号。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
