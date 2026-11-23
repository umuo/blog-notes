"""Check JSON Schema responses and isolate a single client disconnect."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError

from probe import one, optional_metrics, passed, payload, request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default="refund")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    body = payload(args.model, "mixed", False)
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "refund_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"count": {"type": "integer"}, "mean": {"type": ["number", "null"]}},
                "required": ["count", "mean"],
                "additionalProperties": False,
            },
        },
    }
    try:
        with request(args.server, "/v1/chat/completions", body, "schema-probe") as response:
            result = json.load(response)
            content = result["choices"][0]["message"]["content"]
            structure = dict(status=response.status, text=content, passed=passed(content, "mixed"))
    except HTTPError as error:
        structure = dict(status=error.code, detail=error.read().decode()[:1000])
    before = optional_metrics(args.server)
    canceled = one(args.server, args.model, "mixed", cancel=True, max_tokens=256,
                   request_id="isolated-abort")
    time.sleep(1)
    after = optional_metrics(args.server)
    terms = ("num_aborted_requests", 'finished_reason="abort"', "num_running", "num_requests_running")
    deltas = {key: (before.get(key, 0), after.get(key, 0)) for key in sorted(set(before) | set(after))
              if any(term in key for term in terms)}
    args.out.write_text(json.dumps(dict(schema=structure, canceled=canceled, counters=deltas),
                                   ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
