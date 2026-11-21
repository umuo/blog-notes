---
title: "LLM 入门 API-01｜从原始 HTTP 请求认识 LLM API"
description: "用一份三行退款 CSV，把一次 Chat Completions 调用拆成请求行、认证头、JSON 请求体与响应。读懂模型标识、结束原因、Token 用量及常见错误，再用 curl 和本地模拟服务复现成功、认证失败和限流。"
keywords: [llm, api, http]
---

![HTTP 请求和响应示意图：客户端发出 POST /v1/chat/completions，服务端返回 200 OK 和报告。](assets/api-01-http-request-cover.jpg)

你收到了三条退款记录，想请大语言模型（Large Language Model，LLM）写一句统计结论。**空金额要排除，金额为 0 的记录要保留。** 用 CSV（Comma-Separated Values，逗号分隔值）写出来，数据是这样的：

```csv
id,amount
1,10
2,
3,0
```

手算一遍：第 2 条金额缺失，第 1 条和第 3 条参加计算，有效金额 **2** 条，缺失 **1** 条，均值 **(10 + 0) ÷ 2 = 5**。如果把空金额也算成 0，分母就成了 3。这三个数等会儿用来验收模型输出。

在 [F-01《Tokenizer 与 Chat Template》](https://blog.lacknb.cn/articles/2026/11/19/1795068393122.html) 和 [F-05《Prefill 与 Decode》](https://blog.lacknb.cn/articles/2026/11/20/1795161425780.html) 里，我们从模型内部观察输入怎样变成 Token、又怎样逐步生成输出。这次站在应用一侧，用 HTTP（Hypertext Transfer Protocol，超文本传输协议）访问 API（Application Programming Interface，应用程序编程接口）。先修只需要看过 URL 和 JSON，手边有 curl 就能跟着发送请求，有 Python 3 就能在本机复现。

## 1. 先看一条原始请求

下面是发给 OpenAI Chat Completions 接口的 HTTP/1.1 请求示意。密钥用占位符表示，保留了讲解所需的头部。实际发送时，curl 会补齐用于传输的头部，包括记录请求体字节数的 `Content-Length`。

```http
POST /v1/chat/completions HTTP/1.1
Host: api.openai.com
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "model": "gpt-6-luna",
  "messages": [
    {
      "role": "user",
      "content": "请按退款金额生成一句报告，忽略空值并保留 0。报告包含有效金额条数、缺失条数和均值。\nid,amount\n1,10\n2,\n3,0"
    }
  ],
  "reasoning_effort": "none",
  "max_completion_tokens": 128,
  "stream": false
}
```

空行把**请求头**和**请求体**隔开。按顺序读，可以得到这些信息：

| 位置 | 这次请求里的作用 |
| --- | --- |
| `POST /v1/chat/completions HTTP/1.1` | 请求行。用 `POST` 向指定路径提交数据并请求生成，最后一段写明 HTTP 版本。 |
| `Host: api.openai.com` | 说明请求发给哪个主机。实际网址以 `https://` 开头，传输经过 TLS（Transport Layer Security，传输层安全协议）加密。 |
| `Authorization: Bearer <API_KEY>` | `Bearer` 是凭据的传递方式。这次携带 API 密钥，服务据此识别项目并检查权限。真实密钥由调用环境提供，使用环境变量传入，避免写进仓库或复制到调试日志。 |
| `Content-Type: application/json` | 说明请求体采用 JSON（JavaScript Object Notation，JavaScript 对象表示法），服务端按这种格式解析。 |
| `model` | 模型 ID（Identifier，标识符）。这里选的是支持 Chat Completions 的 `gpt-6-luna`，项目还需要具备该模型的访问权限。 |
| `messages` | 本次提交的消息列表。唯一一条消息的角色是 `user`，内容是统计规则和 CSV 文本。JSON 中的 `\n` 解码后是换行，模型收到的文本仍保留 CSV 的行边界。 |
| `reasoning_effort`、`max_completion_tokens` | 前者把这个模型的推理强度设为 `none`，适合这次简单统计。后者把本次生成上限设为 128 个 Token，包含可见输出和模型可能使用的推理 Token。Token 是模型处理文本的基本单位。 |
| `stream: false` | 使用非流式响应。客户端这次读取一个完整的 JSON 响应体。逐段返回的事件留到 API-05 展开。 |

[OpenAI API 概览](https://developers.openai.com/api/reference/overview) · [Chat Completions 接口参考](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) · [GPT-6 Luna 模型说明](https://developers.openai.com/api/docs/models/gpt-6-luna)

沿着请求想象一次典型的处理过程。应用先把消息序列化为 JSON 并发送。服务的接入层识别接口、校验凭据和请求参数，检查模型权限与限额。请求通过后，模型服务把消息组织成模型输入并编码成 Token，再执行 Prefill（处理整段输入）和 Decode（逐步生成 Token）。API 最后把生成结果、结束原因和用量装进响应。这个过程描述各层的职责，具体网关和调度顺序由服务实现决定。

## 2. 从响应里找答案、终止原因和用量

下面的响应头和 JSON 字段来自本篇的**本机教学服务**，为了阅读做了摘录和排版。报告、请求 ID 和 Token 数都是合成样例。真实接口的报告措辞、响应头和计数以本次调用结果为准。

```http
HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
x-request-id: req_demo_ok
x-ratelimit-remaining-requests: 59

{
  "id": "chatcmpl-demo",
  "object": "chat.completion",
  "created": 0,
  "model": "gpt-6-luna",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "有效金额 2 条，缺失 1 条，均值为 5。"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 72,
    "completion_tokens": 19,
    "total_tokens": 91
  }
}
```

- **HTTP 状态行。** `200 OK` 表示服务器成功处理了请求。如果报告把均值写成 3.33，HTTP 层仍可能是 200，业务层要核对条数和计算口径。
- **响应头。** `x-request-id` 给这次 API 请求标上可供排查的标识。响应体的 `id` 标识生成结果。`x-ratelimit-remaining-requests` 是服务器报告的剩余请求额度，Token 维度也有对应的限流头。记录排查信息时，保留状态码和请求 ID 即可定位很多问题。
- **答案与结束方式。** 这次按默认设置得到一条候选回复，正文在 `choices[0].message.content`。`finish_reason: "stop"` 表示生成到自然结束点或配置的停止序列。`"length"` 表示达到长度上限，短报告也可能被截断。
- **Token 用量。** `prompt_tokens` 是输入计数，`completion_tokens` 是生成计数，`total_tokens` 是两者的和。样例里的 **72 + 19 = 91** 只用来演示读数。用量字段计的是 Token，费用需结合实际模型、输入输出的计费规则以及可能存在的缓存或推理明细。请求体的字节数、CSV 行数和可见文本字数各自有不同用途。

[Chat Completions 响应定义](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create#returns) · [请求 ID 与限流响应头](https://developers.openai.com/api/reference/overview)

## 3. 用 curl 发送同一份 JSON

在一个空目录里运行下面的命令，创建请求文件，再定义一个发送函数。可以用 Bash 或 Zsh。后续命令沿用这个终端和目录。

```bash
cat > request.json <<'JSON'
{
  "model": "gpt-6-luna",
  "messages": [
    {
      "role": "user",
      "content": "请按退款金额生成一句报告，忽略空值并保留 0。报告包含有效金额条数、缺失条数和均值。\nid,amount\n1,10\n2,\n3,0"
    }
  ],
  "reasoning_effort": "none",
  "max_completion_tokens": 128,
  "stream": false
}
JSON

call_api() {
  local base_url="$1"
  local api_key="$2"
  shift 2

  curl --silent --show-error --http1.1 --max-time 30 \
    --dump-header response.headers --output response.json \
    --write-out 'HTTP %{http_code}\n' \
    --header "Authorization: Bearer ${api_key}" \
    --header 'Content-Type: application/json' \
    --data-binary @request.json \
    "$@" "${base_url}/v1/chat/completions"
}
```

`--data-binary @request.json` 按文件原样发送请求体，curl 会用 `POST`。响应头写进 `response.headers`，响应体写进 `response.json`，终端打印状态码。每次调用都会覆盖这两个文件。这个观察命令保留 HTTP 失败响应的正文，所以 **curl 退出码为 0 时也要检查 HTTP 状态**。如果连接失败或超时，先看终端里的 curl 错误，再判断这次有没有收到可用的响应。

### 用真实接口观察一次

若已经在当前终端设置 `OPENAI_API_KEY`，而且项目有示例模型的权限，可以发出下面这条会消耗 API 用量的请求。这里用的都是合成数据。已收到响应且响应体为 JSON 时，用 Python 标准库把它排版后阅读。

```bash
call_api https://api.openai.com "${OPENAI_API_KEY:?请先设置 OPENAI_API_KEY}" &&
cat response.headers &&
python3 -m json.tool response.json
```

看到 `200` 后，依次记录请求 ID、`finish_reason` 和三个 `usage` 计数，再核对正文是否说清了 **2 条、1 条和 5**。如果模型 ID 无法使用，先核对它的拼写、受支持的接口及项目权限。 [OpenAI 模型与限额排查](https://developers.openai.com/api/docs/guides/rate-limits)

### 在本机复现三种响应

把下面的代码保存成 `mock_api.py`，在另一个终端运行 `python3 mock_api.py`。它只监听本机的 8765 端口，用固定的报告和计数模拟接口。教学密钥为 `demo-key`。实验用的 `X-Demo-Case` 请求头只在本机服务里用来触发限流。

```python
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class DemoAPI(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, status, body, headers):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        request_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "Unknown path"}}, {})
            return
        if self.headers.get("Authorization") != "Bearer demo-key":
            error = {"code": "invalid_api_key", "message": "Demo key is invalid"}
            self.send_json(401, {"error": error}, {"x-request-id": "req_demo_auth"})
            return
        if self.headers.get("X-Demo-Case") == "rate-limit":
            error = {"code": "slow_down", "message": "Demo rate limit reached"}
            headers = {"Retry-After": "2", "x-request-id": "req_demo_limit"}
            self.send_json(429, {"error": error}, headers)
            return
        try:
            request = json.loads(request_body)
        except ValueError:
            self.send_json(400, {"error": {"message": "Invalid JSON"}}, {})
            return
        if not isinstance(request, dict) or not isinstance(request.get("model"), str):
            self.send_json(400, {"error": {"message": "Expected a model ID"}}, {})
            return

        answer = {
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 0,
            "model": request["model"],
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "有效金额 2 条，缺失 1 条，均值为 5。",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 72, "completion_tokens": 19, "total_tokens": 91},
        }
        headers = {"x-request-id": "req_demo_ok", "x-ratelimit-remaining-requests": "59"}
        self.send_json(200, answer, headers)


with HTTPServer(("127.0.0.1", 8765), DemoAPI) as server:
    print("Listening on http://127.0.0.1:8765", flush=True)
    server.serve_forever()
```

回到定义了 `call_api` 的终端，按顺序发出这三条请求。每次用的都是同一个 `request.json`。

```bash
call_api http://127.0.0.1:8765 demo-key
python3 -m json.tool --no-ensure-ascii response.json

call_api http://127.0.0.1:8765 wrong-key
cat response.json
printf '\n'

call_api http://127.0.0.1:8765 demo-key --header 'X-Demo-Case: rate-limit'
cat response.headers
cat response.json
printf '\n'
```

依次看到 **200**（报告和用量）、**401**（`invalid_api_key`）和 **429**（`slow_down`，响应头带 `Retry-After: 2`）。最后一次是固定模拟场景，每次带上触发头都会得到 429。真实服务返回这样的等待提示时，客户端至少等 2 秒，再降低发请求的速率。本机实验结束后，在服务终端按 `Ctrl+C` 退出。

## 4. 遇到失败时先读哪里？

先区分**有没有收到 HTTP 响应**。已收到时，按状态码找大类，再读错误体里的 `error.code` 或 `error.message`，有 `x-request-id` 时一并记录。中间代理也可能返回 HTML 等其他格式，应用要按响应的实际格式读取。

| 观察到的结果 | 接下来检查什么？ |
| --- | --- |
| `400 Bad Request` | 检查 JSON 语法、字段和数据类型，以及该模型是否支持传入的参数。修好请求再发。 |
| `401 Unauthorized` | 检查密钥是否正确加载、`Bearer` 头是否完整、密钥是否仍有效。 |
| `403 Forbidden` 或 `404 Not Found` | 结合错误体核对项目权限、URL 路径和模型 ID。模型不存在或当前项目不可访问，也可能表现为 404。 |
| `429 Too Many Requests`，触发临时速率限制 | 看错误码和限流响应头。若有 `Retry-After`，至少等到指定时间再试。否则按逐次拉长等待时间并加入少量随机偏移的方式重试，同时限制次数。 |
| `429`，余额耗尽或触及支出上限 | 检查组织或项目余额及额度配置，如 `credit_balance_exhausted` 对应预付余额耗尽。等待几秒无法恢复这类额度。 |
| `500` 或 `503` | 记录请求 ID。短暂服务故障可以有限重试，`503` 带有 `Retry-After` 时遵守等待提示。每次重新生成都可能产生用量。 |
| 连接失败或客户端超时 | 从 DNS（Domain Name System，域名系统）解析、连接、TLS 或超时设置查起。读取超时也可能发生在请求发出之后，此时客户端无法确认服务端是否已经处理。 |

[OpenAI API 错误码](https://developers.openai.com/api/docs/guides/error-codes) · [限流与退避说明](https://developers.openai.com/api/docs/guides/rate-limits)

## 下一篇

**[API-02《OpenAI Chat Completions 的消息协议》](api-02-chat-completions-messages.md)** 会用同一份 CSV 追问「为什么均值是 5」，对照两次发送的 `messages`，看看应用要保存哪些历史、第二轮要回传哪些消息。 [LLM 与 Agent 学习路线图](https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html)
