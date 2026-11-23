(function (root) {
    "use strict";

    const MAX_LENGTH = 4096;

    function calculate(input) {
        if (typeof input !== "string" || input.length > MAX_LENGTH) {
            throw new RangeError("输入最多 4096 字符");
        }
        const lines = input.trim().split(/\r?\n/);
        if (lines[0].trim() !== "refund") {
            throw new Error("首行必须是 refund");
        }

        let sum = 0n;
        let count = 0;
        for (const [index, line] of lines.slice(1).entries()) {
            const value = line.trim();
            if (value === "" || value === "NA") {
                continue;
            }
            if (!/^\d+(?:\.\d{1,2})?$/.test(value)) {
                throw new Error(`第 ${index + 2} 行金额无效`);
            }
            const [whole, fraction = ""] = value.split(".");
            sum += BigInt(whole) * 100n + BigInt(fraction.padEnd(2, "0"));
            count += 1;
        }
        if (count === 0) {
            return { count, mean: null };
        }

        const divisor = BigInt(count);
        const mean = (sum * 2n + divisor) / (2n * divisor);
        return { count, mean: `${mean / 100n}.${String(mean % 100n).padStart(2, "0")}` };
    }

    function workflow() {
        let draft = null;
        let revision = 0;
        let pending = null;
        let archiveCount = 0;

        function invalidate() {
            draft = null;
            pending = null;
        }

        return Object.freeze({
            invalidate,
            generate(input) {
                invalidate();
                draft = Object.freeze({ revision: ++revision, ...calculate(input) });
                return draft;
            },
            preview() {
                if (draft === null || draft.count === 0) {
                    throw new Error("需要一份含有效记录的草稿");
                }
                pending = draft.revision;
                return pending;
            },
            cancel() {
                pending = null;
            },
            accept(expectedRevision, humanApproved) {
                if (humanApproved !== true || pending === null || pending !== expectedRevision || draft?.revision !== pending) {
                    throw new Error("归档需要当前版本的人工确认");
                }
                const receipt = Object.freeze({ ...draft, archiveNumber: ++archiveCount });
                invalidate();
                return receipt;
            },
        });
    }

    const exported = { calculate, workflow };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = exported;
    } else {
        root.ReportCore = exported;
    }
})(globalThis);
