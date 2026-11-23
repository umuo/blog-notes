"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { calculate, workflow } = require("./report-core.js");

const SAMPLE = "refund\n4\nNA\n6";

test("exclude missing data but preserve numeric zero", () => {
    assert.deepEqual(calculate(SAMPLE), { count: 2, mean: "5.00" });
    assert.deepEqual(calculate("refund\n0\n6"), { count: 2, mean: "3.00" });
});

test("return a null mean for no numerical rows", () => {
    assert.deepEqual(calculate("refund\nNA\n"), { count: 0, mean: null });
});

test("reject malformed headers and rows", () => {
    assert.throws(() => calculate("other\n4"), /首行/);
    assert.throws(() => calculate("refund\nbad"), /第 2 行/);
    assert.throws(() => calculate("refund\n-1"), /第 2 行/);
});

test("round half a cent without binary floating point", () => {
    assert.deepEqual(calculate("refund\n0.01\n0.02"), { count: 2, mean: "0.02" });
});

test("reject excessively long input", () => {
    assert.throws(() => calculate(`refund\n${"1".repeat(4100)}`), /4096/);
});

test("require explicit approval and issue exactly one receipt", () => {
    const flow = workflow();
    assert.throws(() => flow.preview(), /有效记录/);
    flow.generate(SAMPLE);
    const version = flow.preview();
    assert.throws(() => flow.accept(version, false), /人工确认/);
    assert.equal(flow.accept(version, true).archiveNumber, 1);
    assert.throws(() => flow.accept(version, true), /人工确认/);
});

test("reject a stale approval after the input changes", () => {
    const flow = workflow();
    flow.generate(SAMPLE);
    const oldVersion = flow.preview();
    flow.invalidate();
    flow.generate("refund\n8\n10");
    const newVersion = flow.preview();
    assert.notEqual(oldVersion, newVersion);
    assert.throws(() => flow.accept(oldVersion, true), /人工确认/);
    assert.equal(flow.accept(newVersion, true).mean, "9.00");
});

test("cancellation invalidates the preview; malformed or empty drafts cannot archive", () => {
    const flow = workflow();
    flow.generate(SAMPLE);
    const version = flow.preview();
    flow.cancel();
    assert.throws(() => flow.accept(version, true), /人工确认/);
    assert.throws(() => flow.generate("refund\noops"), /第 2 行/);
    assert.throws(() => flow.preview(), /有效记录/);
    flow.generate("refund\nNA");
    assert.throws(() => flow.preview(), /有效记录/);
});
