---
title: "LLM 入门 API-05｜SSE 和流式工具调用"
description: "从 HTTP 响应字节开始解析 SSE，区分网络读取边界、完整事件和模型增量。对照 Chat Completions 和 Responses 的流式字段，用一字节一块的实验拼回退款工具参数，检查 UTF-8 跨块、断流、终态和取消。"
keywords: [llm, api, sse, agent]
---

![从响应字节到 SSE 帧、模型增量，再到可校验的完整工具参数。](assets/api-05-sse-streaming-tools-cover.jpg)

[API-04](api-04-function-calling-structured-outputs.md) 的退款工具一次收到完整参数 `{"amounts":[10,null,0]}`。打开流式输出后，客户端可能先看到 `{"amounts":[10,`，再看到 `null,0]}`。它们都是字符串片段。在中途拿其中任意一段做 JSON 解析，都会因为结构未闭合而失败。

这次跟着字节走。**SSE（Server-Sent Events，服务器发送事件）**是 HTTP 响应体中编码事件的一种文本格式。模型接口用这种格式逐个送出消息事件。JSON 对象嵌在事件的 `data` 字段里，工具入参又嵌在 JSON 对象的字符串字段里。三层边界都要各自处理。 [SSE 标准](https://html.spec.whatwg.org/multipage/server-sent-events.html) · [Chat Completions 流式参考](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)

## 1. 从 HTTP 响应到完整事件

在 Chat Completions 和 Responses 中，请求体里的 `stream: true` 启动流式返回。常见成功响应类型为 `text/event-stream`。客户端先判断 HTTP 状态和响应类型，再按 SSE 解析正文。

| 层 | 交付给下一层的内容 | 边界在哪？ |
| --- | --- | --- |
| TCP、HTTP 与客户端读取 | 响应体的一块字节 | 应用一次读取的大小由网络、HTTP 实现和库决定，可能切在中文字符或 JSON 字符串中间。 |
| SSE | 一条完整事件 | 空行结束一帧。多行 `data:` 用换行拼接。以冒号开头的注释可以忽略。 |
| 模型协议 | 事件中的文本增量、调用增量、终态 | 用供应商字段判断该事件的含义和关联 ID。 |
| 界面与应用 | 可见文本、完整工具参数、最终状态 | 文本可以增量展示。工具参数待全部收到并校验后才能执行。 |

**UTF-8** 是 Unicode 文本的一种编码。一个中文字符编码成的多个字节可能分到不同读取块。使用增量解码器可以保留半个字符的字节，等下一次读取补齐。SSE 还允许 `\r\n`、`\r` 和 `\n` 表示行结束，行尾字符本身也可能跨读取块。

下面是两条合成 SSE 事件的正文，省略了各包中无关的模型字段。第一条给调用 ID、名字和第一段参数，第二条只给同一调用的后续参数：

```text
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"summarize_refunds","arguments":"{\"amounts\":[10,"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"null,0]}"}}]},"finish_reason":null}]}

```

通过 `choices[].index` 和 `tool_calls[].index` 定位同一个候选、同一个工具调用。遇到首段时记录调用 ID 与工具名，后续把参数原文逐段追加。示例收到 `finish_reason: "tool_calls"`，并等到 Chat Completions 的 `data: [DONE]` 之后，才交付本轮完整调用。若启用了流式用量，终止前还可能收到 `choices: []` 的用量块。 [Chat Completions 创建接口](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)

## 2. Responses 的事件粒度

Responses 把不同用途写进事件类型，客户端从事件的 `type` 判断如何处理。SSE 层的工作保持一致，模型协议层换成如下映射：

| Chat Completions | Responses | 应用怎么用 |
| --- | --- | --- |
| `delta.content` | `response.output_text.delta` | 逐段更新显示文本。 |
| `delta.tool_calls[].function.arguments` | `response.function_call_arguments.delta` | 按调用索引或 Item 标识累积参数。 |
| 工具结束的 `finish_reason` | `response.function_call_arguments.done`、`response.output_item.done` | 读取完整参数或调用 Item，用 `call_id` 对应工具结果。 |
| `[DONE]` 作为流边界 | `response.completed` 等响应级终态 | 区分完整响应、失败、未完成或缺少终态。 |

Responses 中参数的 `done` 表示这个部分结束，应用还应读取响应级终态，处理错误、未完成和取消。后续需要发回的 `function_call_output` 使用调用 Item 的 `call_id`。对接时按所选接口处理终态，避免把一个文本分片的结束当作整个请求结束。 [Responses 流式指南](https://developers.openai.com/api/docs/guides/streaming-responses) · [Responses 流式事件参考](https://developers.openai.com/api/reference/resources/responses/streaming-events)

## 3. 用一字节一块复现

把代码保存为 `stream_tools.py`，运行 `python3 stream_tools.py`。这个最小例子消费合成 Chat Completions 流，故意让每次读取只有一个字节。用于全量接入时，还要处理供应商错误、资源上限、背压与工具回填。本例把可执行工具调用留到终态之后交付。

```python
import codecs
import json


def events(chunks):
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    line_chars = []
    data = []
    previous_cr = False
    for chunk in chunks:
        for char in decoder.decode(chunk):
            if previous_cr and char == "\n":
                previous_cr = False
                continue
            previous_cr = char == "\r"
            if char not in ("\r", "\n"):
                line_chars.append(char)
                continue
            line = "".join(line_chars)
            line_chars.clear()
            if not line:
                if data:
                    yield "\n".join(data)
                    data.clear()
                continue
            field, sep, value = line.partition(":")
            if field == "data":
                data.append(value.removeprefix(" ") if sep else "")
    decoder.decode(b"", final=True)
    # 流断开时，未由空行结束的帧不交给模型协议解析器。


def read_chat(chunks, cancelled=lambda: False):
    text = []
    calls = {}
    reasons = {}
    done = False
    for data in events(chunks):
        if cancelled():
            return "cancelled", "".join(text), []
        if data == "[DONE]":
            done = True
            break
        packet = json.loads(data)
        for choice in packet["choices"]:
            index = choice["index"]
            delta = choice["delta"]
            if index == 0 and delta.get("content"):
                text.append(delta["content"])
            for partial in delta.get("tool_calls", []):
                key = (index, partial["index"])
                call = calls.setdefault(key, {"parts": []})
                function = partial.get("function", {})
                if "id" in partial:
                    call["id"] = partial["id"]
                if "name" in function:
                    call["name"] = function["name"]
                call["parts"].append(function.get("arguments", ""))
            if choice.get("finish_reason"):
                reasons[index] = choice["finish_reason"]

    if cancelled():
        return "cancelled", "".join(text), []
    if not done or 0 not in reasons:
        return "interrupted", "".join(text), []
    tools = []
    if reasons[0] == "tool_calls":
        for (choice_index, _), call in sorted(calls.items()):
            if choice_index == 0:
                tools.append((call["id"], call["name"], json.loads("".join(call["parts"]))))
    return reasons[0], "".join(text), tools


def frame(delta, reason=None):
    body = {"choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
    return ("data: " + json.dumps(body, ensure_ascii=False) + "\r\n\r\n").encode()


payload = b"".join([
    b": heartbeat\r\n\r\n",
    frame({"content": "正在核对。"}),
    frame({"tool_calls": [{
        "index": 0, "id": "call_1", "type": "function",
        "function": {"name": "summarize_refunds", "arguments": '{"amounts":[10,'},
    }]}),
    frame({"tool_calls": [{"index": 0, "function": {"arguments": "null,0]}"}}]}),
    frame({}, "tool_calls"),
])
full = payload + b"data: [DONE]\r\n\r\n"
one_byte = (full[i:i + 1] for i in range(len(full)))
status, text, tools = read_chat(one_byte)
assert status == "tool_calls" and text == "正在核对。"
assert tools == [("call_1", "summarize_refunds", {"amounts": [10, None, 0]})]
assert read_chat([payload]) == ("interrupted", text, [])
assert read_chat([full], cancelled=lambda: True) == ("cancelled", "", [])
assert list(events([b"data: a\ndata: b\n\n"])) == ["a\nb"]
assert list(events([b"data: final\r", b"\r"])) == ["final"]
assert list(events([b"data: unfinished\n"])) == []
print(status, text, tools)
```

期望能恢复「正在核对。」和完整 `amounts`。删掉末尾 `[DONE]` 的帧，即使当前参数看起来已经完整，也得到 `interrupted`，工具调用列表为空。真实客户端取消时要关闭正在读取的响应或让 HTTP 任务中止，并在应用状态里标记取消。已经展示的增量可视为未完成文本。连接中断后是否重试取决于应用状态，尤其要查上一轮有没有已经执行过的动作。

## 下一篇

[**API-06《Anthropic Messages API 的内容块与事件》**](api-06-anthropic-messages.md) 用同一个退款工具对照块索引、参数增量、`tool_use` 与 `tool_result`。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
