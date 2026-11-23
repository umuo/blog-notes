import { writeFileSync } from "node:fs";
import { strict as assert } from "node:assert";
import { Agent } from "@mariozechner/pi-agent-core";
import { createAssistantMessageEventStream, getModel } from "@mariozechner/pi-ai";
import { Type } from "@sinclair/typebox";

const model = getModel("openai", "gpt-4o-mini");
const API = model.api;
const EXPECTED = { count: 2, mean: "5" };
const events = [];
const inputs = [];
const usage = {
    input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function streamFn(_model, context) {
    inputs.push(context.messages.map((message) => message.role));
    const first = inputs.length === 1;
    const content = first
        ? [{ type: "toolCall", id: "pi-call-1", name: "csv_stats", arguments: { column: "refund" } }]
        : [{ type: "text", text: "报告已生成，等待确认。" }];
    const stopReason = first ? "toolUse" : "stop";
    const message = {
        role: "assistant", content, api: API, provider: model.provider, model: model.id,
        usage, timestamp: 0, stopReason,
    };
    const events = createAssistantMessageEventStream();
    events.push({ type: "start", partial: { ...message, content: [] } });
    events.push({ type: "done", reason: stopReason, message });
    return events;
}

const tools = [{
    name: "csv_stats",
    label: "Read refund statistics",
    description: "Compute count and mean of the fixed CSV",
    parameters: Type.Object({ column: Type.Literal("refund") }, { additionalProperties: false }),
    execute: async (id, args) => {
        assert.equal(id, "pi-call-1");
        assert.equal(args.column, "refund");
        const rows = ["4", "NA", "6"].filter((value) => value !== "NA").map(Number);
        const report = { count: rows.length, mean: String(rows.reduce((a, b) => a + b, 0) / rows.length) };
        assert.deepEqual(report, EXPECTED);
        return { content: [{ type: "text", text: JSON.stringify(report) }], details: report };
    },
}];
const agent = new Agent({
    initialState: { model, systemPrompt: "生成退款报告，等待审批。", tools },
    streamFn,
});
agent.subscribe((event) => {
    const id = event.toolCallId;
    events.push(id ? { type: event.type, toolCallId: id } : { type: event.type });
});
await agent.prompt("统计 4、NA、6 的有效条数和均值。");
assert.deepEqual(inputs, [["user"], ["user", "assistant", "toolResult"]]);
assert(events.some((event) => event.type === "tool_execution_start"));
assert(events.some((event) => event.type === "tool_execution_end"));
assert.equal(events.at(-1).type, "agent_end");
writeFileSync(process.argv[2], JSON.stringify({
    pi: "0.70.0",
    provider_called: false,
    inputs,
    events,
    final: agent.state.messages.at(-1).content[0].text,
}, null, 2) + "\n");
