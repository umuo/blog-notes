---
title: "LLM 入门 API-06｜Anthropic Messages API 的内容块与事件"
description: "从 Anthropic 的原始 HTTP 请求出发，观察顶层系统指令、普通文字块、工具提议和工具结果。沿用退款统计函数重建流式参数并对照 OpenAI 的消息形状，区分组装时的块索引与下次请求使用的工具 ID。"
keywords: [llm, api, anthropic, sse]
---

![Anthropic Messages 用内容块表达普通文字、工具提议和工具结果。](assets/api-06-anthropic-messages-cover.jpg)

同一个退款 CSV 包含 10、空、0。让 Claude 调用应用的统计函数，算出两条有效、一条缺失、均值 5。在 [API-04](api-04-function-calling-structured-outputs.md) 的 OpenAI Responses 例子里，工具提议是一项 `function_call`。Anthropic Messages 把它放在 **assistant 消息的内容块**中，应用再用另一条消息里的结果块回传。

**Content block（内容块）**是消息 `content` 数组中带类型的一项。它可以承载文字、图像、工具提议或结果。整体仍是一轮消息，但同一条消息可以有多个块。 [Claude Messages API](https://platform.claude.com/docs/en/api/messages) · [Anthropic 工具回填规则](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)

## 1. 从原始 HTTP 请求开始

下面省略了 curl 会补齐的传输头。示例使用 Anthropic 的 `2026-10-21` API 版本和官方流式文档中出现的模型 ID。调用时需配置当前项目可用的模型和密钥。

```http
POST /v1/messages HTTP/1.1
Host: api.anthropic.com
x-api-key: <ANTHROPIC_API_KEY>
anthropic-version: 2026-10-21
Content-Type: application/json

{
  "model": "claude-opus-5-5",
  "max_tokens": 256,
  "system": "生成退款报告。空金额排除，0 保留。",
  "messages": [{
    "role": "user",
    "content": [{"type":"text","text":"请调用统计工具。\nid,amount\n1,10\n2,\n3,0"}]
  }],
  "tools": [{
    "name": "summarize_refunds",
    "description": "按金额数组统计有效条数、缺失条数和均值。",
    "input_schema": {
      "type": "object",
      "properties": {"amounts": {"type":"array","items":{"type":["number","null"]}}},
      "required": ["amounts"],
      "additionalProperties": false
    }
  }]
}
```

`max_tokens` 限制生成长度。`system` 是顶层系统指令，应用不要把它照搬为 `messages[].role: "system"`。`messages` 在本篇只使用 `user` 和 `assistant` 两类角色。显式的 `text` 块让纯文本和后续工具块能用同样的方式遍历。工具参数定义在 `input_schema`，实际的块在响应中叫 `input`。不同版本头支持的选项可能不同，例如 `2026-10-21` 已从 Messages 请求移除 `temperature`、`top_p`、`top_k`。 [Claude API 版本说明](https://platform.claude.com/docs/en/api/versioning)

## 2. 从 `tool_use` 到紧邻的 `tool_result`

首次响应可能先给一句提示，再给工具提议。下面保留了与后续回填相关的字段：

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "我来统计。"},
    {
      "type": "tool_use",
      "id": "toolu_refund_1",
      "name": "summarize_refunds",
      "input": {"amounts": [10, null, 0]}
    }
  ],
  "stop_reason": "tool_use"
}
```

应用保留整条 assistant 消息，读取 `tool_use.input`，按工具名、参数和业务上下文校验。程序算出 **(10 + 0) ÷ 2 = 5** 后，新建一条 `user` 消息紧跟在这条 assistant 消息后面：

```json
{
  "role": "user",
  "content": [{
    "type": "tool_result",
    "tool_use_id": "toolu_refund_1",
    "content": "{\"count\":2,\"missing\":1,\"mean\":5}"
  }]
}
```

该消息里的工具结果块需要排在普通文本块之前。若同一条 assistant 消息提出多个应用工具调用，在接下来的 user 消息中依次提供对应结果。应用侧错误可通过结果块的 `is_error` 标记。发起第二个 `POST /v1/messages` 时回传原来的 user 消息、带工具提议的 assistant 消息，以及带结果的 user 消息，并保留本轮需要的顶层配置。服务通常用后续 assistant 的 `text` 内容和 `end_turn` 停止原因交还可见答案。 [工具结果要求](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)

| 同一语义 | Anthropic Messages | OpenAI Chat Completions |
| --- | --- | --- |
| 应用级指令 | 顶层 `system` | 由应用提供的 `developer` 消息 |
| 模型提议 | assistant 的 `tool_use` 块，`input` 是对象 | assistant 的 `tool_calls[]`，`function.arguments` 是 JSON 字符串 |
| 执行结果 | user 的 `tool_result` 块，用 `tool_use_id` 关联 | `role: "tool"` 消息，用 `tool_call_id` 关联 |
| 终止原因 | `stop_reason` 如 `tool_use`、`end_turn` | `finish_reason` 如 `tool_calls`、`stop` |

映射时应把普通文字与工具块都保留，再按目标协议编码。ID 在当前供应商的一次往返内匹配。跨供应商回放历史时，需要按目标接口校验所有消息和标识，并为调用和结果维护同一个映射。

## 3. 流式块索引和工具 ID 有不同职责

加上 `stream: true` 后，同样的响应以 SSE（服务器发送事件）编码。[API-05](api-05-sse-streaming-tools.md) 讲了怎样把字节还原成一帧数据。Anthropic 的这组帧还带有命名事件，JSON 里的 `type` 标出事件含义。一次示例按以下顺序运行：

| 事件 | 这个案例中的动作 |
| --- | --- |
| `message_start` | 服务返回消息元数据。 |
| `content_block_start`、`content_block_delta`、`content_block_stop`，`index: 0` | 更新第一块文字「我来统计。」 |
| `content_block_start`，`index: 1` | 建立工具块并保存 `id: "toolu_refund_1"`，此时 `input` 可以是空对象。 |
| 两次 `input_json_delta`，`index: 1` | 累加 `partial_json`，合并出完整参数。 |
| `content_block_stop`，`index: 1` | 当前工具块已收齐，可以尝试解析。 |
| `message_delta`、`message_stop` | 读取 `stop_reason: "tool_use"`，确认整条消息结束再交付。 |

块 `index` 负责当前流里的组装，`tool_use.id` 负责下一次 HTTP 请求里的调用关联。真实流还可能包含 `ping`、错误或新增事件类型。对未知事件留出处理空间，错误或未收到终态时结束当前处理，避免把只收到一部分的数据交给工具。 [Anthropic 流式事件说明](https://platform.claude.com/docs/en/build-with-claude/streaming)

## 4. 动手把同一次提议转换往返

把下面存为 `claude_blocks.py`，运行 `python3 claude_blocks.py`。输入是已从 SSE 解出的合成事件，程序按块索引重建工具提议，再把提议转成 Chat Completions 形状并转回，最后构造 Anthropic 的结果消息。只展示这个纯文本和单一自定义工具的用例。

```python
import json

events = [
    {"type": "content_block_start", "index": 1, "content_block": {
        "type": "tool_use", "id": "toolu_refund_1", "name": "summarize_refunds", "input": {},
    }},
    {"type": "content_block_delta", "index": 1, "delta": {
        "type": "input_json_delta", "partial_json": '{"amounts":[10,',
    }},
    {"type": "content_block_delta", "index": 1, "delta": {
        "type": "input_json_delta", "partial_json": 'null,0]}',
    }},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    {"type": "message_stop"},
]


def rebuild(events):
    pending = {}
    calls = {}
    stop_reason = None
    for event in events:
        kind = event["type"]
        index = event.get("index")
        if kind == "content_block_start" and event["content_block"]["type"] == "tool_use":
            pending[index] = []
            calls[index] = dict(event["content_block"])
        elif kind == "content_block_delta" and index in pending:
            delta = event["delta"]
            if delta["type"] == "input_json_delta":
                pending[index].append(delta["partial_json"])
        elif kind == "content_block_stop" and index in pending:
            raw = "".join(pending.pop(index))
            if raw:
                calls[index]["input"] = json.loads(raw)
        elif kind == "message_delta":
            stop_reason = event["delta"].get("stop_reason")
        elif kind == "message_stop" and stop_reason == "tool_use" and not pending:
            return [calls[index] for index in sorted(calls)]
    raise RuntimeError("tool turn incomplete")


call, = rebuild(events)
chat = {"id": call["id"], "type": "function", "function": {
    "name": call["name"], "arguments": json.dumps(call["input"], separators=(",", ":")),
}}
restored = {"id": chat["id"], "type": "tool_use", "name": chat["function"]["name"],
            "input": json.loads(chat["function"]["arguments"])}
assert restored == call
assert restored["input"] == {"amounts": [10, None, 0]}

values = restored["input"]["amounts"]
valid = [value for value in values if value is not None]
summary = {"count": len(valid), "missing": len(values) - len(valid),
           "mean": sum(valid) / len(valid) if valid else None}
reply = {"role": "user", "content": [{
    "type": "tool_result", "tool_use_id": restored["id"], "content": json.dumps(summary),
}]}
assert reply["content"][0]["tool_use_id"] == call["id"]
assert summary == {"count": 2, "missing": 1, "mean": 5}
try:
    rebuild(events[:-1])
except RuntimeError:
    print("缺少消息终态：不会交付工具调用")
else:
    raise AssertionError("truncated stream was accepted")
print(json.dumps(reply, ensure_ascii=False))
```

预期第一行报告缺少终态时不会交付工具调用，第二行的 `tool_result` 包含 `toolu_refund_1` 和均值 `5.0`。这个夹具验证应用如何重建、映射、回填。实际接入还需覆盖混合文字块、多个工具、拒答和未知块，按目标模型及接口版本验证支持范围。

## 下一篇

[**API-07《模型协议适配与能力矩阵》**](api-07-provider-adapters.md) 会把两家都能表达的消息与事件放进同一个应用接口，再保留各自的能力与原始字段。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
