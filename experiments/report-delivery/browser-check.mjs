import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";

if (process.argv.length < 3) {
    throw new Error("Provide the path to playwright/index.mjs");
}

const { chromium } = await import(pathToFileURL(resolve(process.argv[2])).href);
const root = fileURLToPath(new URL(".", import.meta.url));
const browser = await chromium.launch({ channel: "chrome", headless: true });
const viewport = { width: 900, height: 680 };
const input = "refund\n4\nNA\n6";

async function snapshot(page, file) {
    if (process.argv[3]) {
        await page.screenshot({ path: resolve(process.argv[3], file) });
    }
}

async function readReport(page) {
    return { count: await page.locator("#count").innerText(), mean: await page.locator("#mean").innerText(),
        archived: await page.locator("#archive-count").innerText() };
}

async function prototypes() {
    const results = [];
    for (const round of [1, 2]) {
        const page = await browser.newPage({ viewport });
        const started = performance.now();
        await page.goto(pathToFileURL(resolve(root, `prototype/round-${round}.html`)).href);
        await page.getByRole("button", { name: "生成报告" }).click();
        const report = await page.locator("#report").innerText();
        const defaultInputReviewMs = Math.round(performance.now() - started);
        await page.getByRole("button", { name: "归档", exact: true }).click();
        const archive = await page.getByRole("status").innerText();
        await page.getByLabel("退款 CSV").fill("refund\nNA");
        await page.getByRole("button", { name: "生成报告" }).click();
        const empty = await page.locator("#report").innerText();
        results.push({ round, report, empty, archive, defaultInputReviewMs });
        await page.close();
    }
    assert.match(results[0].report, /3，均值 3\.33/);
    assert.match(results[1].report, /2，均值 5\.00/);
    assert.match(results[1].empty, /0，均值无值/);
    assert(results.every((result) => result.archive === "已归档（本地演示）"));
    return results;
}

async function engineered() {
    const normal = pathToFileURL(resolve(root, "engineered/index.html"));
    const shifted = new URL(normal);
    shifted.searchParams.set("layout", "shifted");
    const page = await browser.newPage({ viewport });
    const originalPoint = [188, 345];
    const refreshedPoint = [188, 450];
    const draft = { count: "2", mean: "5.00", archived: "0" };
    await page.goto(normal.href);
    await snapshot(page, "normal.png");
    await page.mouse.click(...originalPoint);
    assert.deepEqual(await readReport(page), draft);

    await page.goto(shifted.href);
    await snapshot(page, "shifted.png");
    await page.mouse.click(...originalPoint);
    const oldCoordinates = await readReport(page);
    assert.deepEqual(oldCoordinates, { count: "—", mean: "—", archived: "0" });
    await page.mouse.click(...refreshedPoint);
    assert.deepEqual(await readReport(page), draft);

    await page.goto(shifted.href);
    const accessible = await page.locator("main").ariaSnapshot();
    assert.match(accessible, /button "生成报告"/);
    assert.match(accessible, /button "预览归档" \[disabled\]/);
    await page.getByRole("button", { name: "生成报告" }).click();
    assert.deepEqual(await readReport(page), draft);
    await page.getByRole("button", { name: "预览归档" }).click();
    const dialog = page.getByRole("dialog", { name: "确认归档" });
    assert.equal(await dialog.isVisible(), true);
    assert.equal(await dialog.getByRole("button", { name: "提交归档" }).isDisabled(), true);
    assert.deepEqual(await readReport(page), draft);
    await page.keyboard.press("Escape");
    await dialog.waitFor({ state: "hidden" });
    assert.equal(await page.locator("#archive-count").innerText(), "0");

    await page.getByLabel("退款 CSV").fill("refund\nbad");
    await page.getByRole("button", { name: "生成报告" }).click();
    const invalidInputError = await page.getByRole("alert").innerText();
    assert.match(invalidInputError, /第 2 行/);
    assert.equal(await page.getByRole("button", { name: "预览归档" }).isDisabled(), true);
    await page.getByLabel("退款 CSV").fill(input);
    await page.getByRole("button", { name: "生成报告" }).click();
    assert.deepEqual(await readReport(page), draft);
    await page.getByRole("button", { name: "预览归档" }).click();

    // This models the executor returning control. The test driver plays the approver after handoff.
    const handoff = { owner: "human", automationSubmitted: false, archiveCountAtHandoff: (await readReport(page)).archived };
    assert.equal(handoff.archiveCountAtHandoff, "0");
    await snapshot(page, "approval.png");
    await dialog.getByRole("checkbox", { name: "我已核对当前版本" }).check();
    await dialog.getByRole("button", { name: "提交归档" }).click();
    await dialog.waitFor({ state: "hidden" });
    const archived = await readReport(page);
    assert.deepEqual(archived, { ...draft, archived: "1" });
    assert.equal(await page.getByRole("button", { name: "预览归档" }).isDisabled(), true);
    await page.close();

    return { viewport, screenshotPath: { originalPoint, refreshedPoint, oldCoordinates }, semanticLocatorAfterShift: draft,
        accessibleButtons: accessible.split("\n").filter((line) => line.includes('button "')).map((line) => line.trim()),
        invalidInputError, escapeKeptArchiveCount: 0, handoff, afterHumanConfirmation: archived };
}

try {
    const result = { chrome: browser.version(), prototypes: await prototypes(), engineered: await engineered() };
    console.log(JSON.stringify(result, null, 4));
} finally {
    await browser.close();
}
