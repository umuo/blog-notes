"""Official MCP Python SDK over stdio. All inputs are fixed offline fixtures."""

import argparse
import asyncio
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from runtime import CSV, stats


SERVER = FastMCP("refund-fixture")
SCRIPT = Path(__file__).resolve()
PROTOCOL = "2025-11-25"


class StatsResult(BaseModel):
    source: str
    count: int
    mean: str | None


@SERVER.tool()
async def csv_stats(
    column: Literal["refund"],
    delay_ms: Annotated[int, Field(ge=0, le=1000)] = 0,
) -> StatsResult:
    """Read the fixed refund column. Delay is a bounded timeout-test fixture."""
    if delay_ms:
        await asyncio.sleep(delay_ms / 1000)
    return StatsResult(source="refund-fixture", **stats(CSV))


@SERVER.resource("report://refund/dictionary", mime_type="application/json")
def dictionary() -> str:
    return json.dumps({"column": "refund", "missing": "NA", "mean_type": "decimal_string",
                       "aggregation": "mean over non-missing rows"})


async def packet(process, body, wait=True):
    process.stdin.write(json.dumps(body).encode() + b"\n")
    await process.stdin.drain()
    if not wait:
        return {"sent": body}
    response = await asyncio.wait_for(process.stdout.readline(), timeout=10)
    if not response:
        raise RuntimeError("stdio server exited")
    response = json.loads(response)
    assert response["id"] == body["id"]
    return {"sent": body, "received": response}


async def raw_trace():
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(SCRIPT), "--serve", stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        messages = [
            await packet(process, dict(jsonrpc="2.0", id=1, method="initialize", params={
                "protocolVersion": PROTOCOL, "capabilities": {},
                "clientInfo": {"name": "refund-raw-probe", "version": "1.0"},
            })),
            await packet(process, dict(jsonrpc="2.0", method="notifications/initialized"), False),
            await packet(process, dict(jsonrpc="2.0", id=2, method="tools/list")),
            await packet(process, dict(jsonrpc="2.0", id=3, method="tools/call", params={
                "name": "csv_stats", "arguments": {"column": "refund"},
            })),
        ]
        assert messages[0]["received"]["result"]["protocolVersion"] == PROTOCOL
        return messages
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


async def sdk_trace():
    parameters = StdioServerParameters(command=sys.executable, args=[str(SCRIPT), "--serve"])
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            listing = await session.list_resources()
            doc = await session.read_resource("report://refund/dictionary")
            valid = await session.call_tool("csv_stats", {"column": "refund"})
            invalid = await session.call_tool("csv_stats", {"column": "wrong"})
            unknown = await session.call_tool("unknown", {})
            timed_out = False
            try:
                await asyncio.wait_for(session.call_tool("csv_stats", {"column": "refund", "delay_ms": 400}),
                                       timeout=0.03)
            except TimeoutError:
                timed_out = True
            next_result = await session.call_tool("csv_stats", {"column": "refund"})
            assert init.protocolVersion == PROTOCOL
            assert valid.structuredContent == {"source": "refund-fixture", "count": 2, "mean": "5"}
            assert not valid.isError and invalid.isError and unknown.isError and timed_out
            assert next_result.structuredContent == valid.structuredContent
            assert len(json.dumps(valid.structuredContent).encode()) <= 128
            return dict(
                sdk=version("mcp"), protocol=init.protocolVersion, tool_schema=tools.tools[0].inputSchema,
                resources=[str(resource.uri) for resource in listing.resources],
                dictionary=json.loads(doc.contents[0].text), result=valid.structuredContent,
                invalid_is_error=invalid.isError, unknown_is_error=unknown.isError,
                client_deadline_hit=timed_out, session_usable_after_timeout=not next_result.isError,
            )


async def run(out):
    result = dict(sdk=await sdk_trace(), wire=await raw_trace())
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.serve:
        SERVER.run(transport="stdio")
    elif args.out:
        asyncio.run(run(args.out))
    else:
        parser.error("--out is required for the verification run")
