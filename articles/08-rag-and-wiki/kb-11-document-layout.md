---
title: "知识库工程 KB-11｜复杂文档解析，OCR、版面、表格与引用"
description: "用两页合成扫描退款字典比较 PDF 文字层抽取与带坐标的光学字符识别。实测双栏错序、第二页表格没有表头时的行归属，以及 NA 单元格的页码证据，并用同一个问题检查解析差异怎样传到检索与回答。"
keywords: [llm, rag, knowledge-base]
---

![用坐标恢复双栏阅读顺序，并把跨页表格连回正确的页码。](../../assets/kb-11-document-layout-cover.jpg)

> **生成工具**：Codex\
> **生成模型**：gpt-6.2-sol

问「退款字典里的 NA 是什么意思」，第一步是读到那一格。扫描 PDF（Portable Document Format，便携式文档格式）可能每页都只是一张图片。人眼看得见，抽取 PDF 文字层的程序却会得到空字符串。

## 1. 先找字，再认这行归谁

OCR（Optical Character Recognition，光学字符识别）把图片里的笔画转成字符串。本实验复用 macOS 自带 Vision 来识别，PDFKit 拿页和渲染图片。从 Vision 取每块文字的字符串、横坐标和纵坐标，它们处于 0 到 1 之间，纵坐标从底部往上。程序再把字分到左右栏，按栏读完才往下。 [Apple Vision 文字识别](https://developer.apple.com/documentation/vision/vnrecognizetextrequest) · [Docling 项目](https://github.com/docling-project/docling)

| 读法 | 第 1 页的四行 |
| --- | --- |
| 横向逐行拼 | Overview → Scope → Amounts use CNY. → Only approved reports are archived. |
| 按栏先左后右 | Overview → Amounts use CNY. → Scope → Only approved reports are archived. |

CNY 是人民币的国际货币代码。按栏读时，它留在 Overview（概述）下面，右栏的 Scope（适用范围）也留住了自己的说明。实际文档可能有通栏标题、脚注和多种版面，要先识别相应区域。本例阈值针对这张夹具。

## 2. 表头在第一页，答案在第二页

第 1 页读到 `Field / Meaning`（字段 / 含义）。第 2 页有续表标题和两列内容，没有再印表头。按横坐标分列、按相近的纵坐标配对，并沿用上一页已确认的表头。

| 字段 | 含义 | 实测物理页 |
| --- | --- | ---: |
| refund | Refund amount | 1 |
| NA | Missing value | 2 |
| report_id | Report identifier | 2 |

将单元格文本和页码一起写到索引。查同一个问题「NA 是什么意思」，文字层路径找不到证据，保留单元格的路径取得 `NA → Missing value`，引用第 2 页。代码的 `missingValueQuestion` 保存这对判定。遇到另一张表紧挨着上一张，先检查表名、列边界、续表标记以及页码坐标，再决定能否继承表头。

复跑以下命令需要 macOS 的 Xcode Command Line Tools。文件是合成的扫描件，Vision 使用系统内置能力。

```bash
xcrun swiftc -framework PDFKit -framework Vision experiments/rag-knowledge/layout_lab.swift -o /tmp/refund-layout
/tmp/refund-layout experiments/rag-knowledge/scanned-dictionary.pdf
```

[扫描夹具](../../experiments/rag-knowledge/scanned-dictionary.pdf) · [识别及表格恢复代码](../../experiments/rag-knowledge/layout_lab.swift) · [实际结果](../../experiments/rag-knowledge/layout-results.json) · [KB-05 的引用回答](kb-05-cited-rag.md)
