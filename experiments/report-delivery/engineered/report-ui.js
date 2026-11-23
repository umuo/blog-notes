"use strict";

const flow = ReportCore.workflow();
const get = (id) => document.getElementById(id);
const input = get("csv");
const preview = get("preview");
const dialog = get("confirm");
const approval = get("approval");
const submit = get("submit");
let currentDraft = null;
let pendingVersion = null;

if (new URLSearchParams(location.search).get("layout") === "shifted") {
    document.body.classList.add("shifted");
}

function setDraft(draft) {
    currentDraft = draft;
    get("count").textContent = draft?.count ?? "—";
    get("mean").textContent = draft === null ? "—" : draft.mean ?? "无有效记录";
    preview.disabled = !draft?.count;
}

input.addEventListener("input", () => {
    flow.invalidate();
    setDraft(null);
    get("problem").hidden = true;
    get("status").textContent = "输入已修改，请重新生成报告";
});

get("calculate").addEventListener("click", () => {
    get("problem").hidden = true;
    try {
        const draft = flow.generate(input.value);
        setDraft(draft);
        get("status").textContent = draft.count === 0 ? "无有效记录，未生成可归档报告" : "草稿已生成，尚未归档";
    } catch (error) {
        setDraft(null);
        get("problem").textContent = error.message;
        get("problem").hidden = false;
        get("status").textContent = "请修改输入后重试";
    }
});

preview.addEventListener("click", () => {
    pendingVersion = flow.preview();
    approval.checked = false;
    submit.disabled = true;
    get("preview-text").textContent = `草稿第 ${pendingVersion} 版 · ${currentDraft.count} 条 · 均值 ${currentDraft.mean}`;
    dialog.showModal();
});

approval.addEventListener("change", () => {
    submit.disabled = !approval.checked;
});

get("cancel").addEventListener("click", () => {
    dialog.close();
});

dialog.addEventListener("close", () => {
    flow.cancel();
    pendingVersion = null;
    approval.checked = false;
    submit.disabled = true;
});

submit.addEventListener("click", () => {
    try {
        const receipt = flow.accept(pendingVersion, approval.checked);
        get("archive-count").textContent = receipt.archiveNumber;
        get("status").textContent = "归档完成（本地演示）";
        preview.disabled = true;
        dialog.close();
    } catch (error) {
        get("problem").textContent = error.message;
        get("problem").hidden = false;
        dialog.close();
    }
});
