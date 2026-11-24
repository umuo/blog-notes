#!/usr/bin/env python3
"""Parse llm-agent-learning-roadmap.md and build interactive Markmap HTML, Markdown outline, and OPML."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parent.parent
ROADMAP_PATH = REPO_DIR / "llm-agent-learning-roadmap.md"
OUT_DIR = REPO_DIR / "mindmap"
OUT_HTML = OUT_DIR / "roadmap-mindmap.html"
OUT_MD = OUT_DIR / "roadmap-mindmap-outline.md"
OUT_OPML = OUT_DIR / "roadmap-mindmap.opml"

ROADMAP_BLOG_URL = "https://blog.lacknb.cn/articles/2026/11/19/1795058329577.html"
INTERVIEW_BANK_URL = "https://blog.lacknb.cn/articles/2026/11/19/1795049523245.html"

CODE_REF_RE = re.compile(
    r"\b(?:F|API|LE|TRN|SERV|AGT|AE|KB|SEC|SBX|EVO|CASE|DEV|OPS)-\d{2}\b"
)

PILLAR_DEFS = [
    {
        "id": "pillar-core",
        "title": "一、大模型本体与服务",
        "subtitle": "从原理、协议、评测到训练与推理部署",
        "color": "#2563eb",
        "stage_orders": [1, 2, 3, 4, 5],
    },
    {
        "id": "pillar-agent-rag",
        "title": "二、Agent 核心与知识库",
        "subtitle": "Loop / 上下文工程、Agent 评测、RAG 与 Wiki",
        "color": "#059669",
        "stage_orders": [6, 7, 8],
    },
    {
        "id": "pillar-sec-evo",
        "title": "三、安全、隔离与自进化",
        "subtitle": "OWASP 护栏、沙箱执行边界与递归改进",
        "color": "#d97706",
        "stage_orders": [9, 10, 11],
    },
    {
        "id": "pillar-prod",
        "title": "四、开源案例与生产落地",
        "subtitle": "源码对照、Agentic Engineering 与生产运维",
        "color": "#7c3aed",
        "stage_orders": [12, 13, 14],
    },
]

TRACK_STAGE_MAP = {
    "all": list(range(1, 15)),
    "request": [1, 2, 6, 8],
    "eval": [3, 7],
    "train-serve": [1, 4, 5],
    "sec-prod": [9, 10, 11, 12, 13, 14],
}


@dataclass
class ArticleItem:
    code: str
    title: str
    url: str | None
    local_file: str | None
    published: bool
    summary: str
    refs: list[str] = field(default_factory=list)


@dataclass
class StageSection:
    order: int
    title: str
    series_label: str
    prereq: str
    deliverable: str
    intro: str
    articles: list[ArticleItem] = field(default_factory=list)


def strip_md_links(text: str) -> str:
    """Turn inline Markdown links `[label](url)` into plain `label` text."""
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text).strip()


def extract_refs(text: str, self_code: str) -> list[str]:
    """Extract ordered unique cross-article references excluding self_code."""
    seen: set[str] = set()
    out: list[str] = []
    for ref in CODE_REF_RE.findall(text):
        if ref != self_code and ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def parse_overview_table(text: str) -> dict[int, dict[str, str]]:
    """Parse the 14-row top summary table from the roadmap."""
    overview: dict[int, dict[str, str]] = {}
    row_re = re.compile(
        r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$",
        re.M,
    )
    for m in row_re.finditer(text):
        order = int(m.group(1))
        overview[order] = {
            "series_label": m.group(2).strip(),
            "prereq": strip_md_links(m.group(3)),
            "deliverable": strip_md_links(m.group(4)),
        }
    return overview


def parse_article_cell(col2: str) -> tuple[str, str | None, str | None, bool]:
    """Parse title, blog URL, local file, and publish state from column 2."""
    pub_full = re.match(
        r"^\[([^\]]+)\]\((https://[^)]+)\)（已发布，\[源稿\]\(([^)]+)\)）$",
        col2,
    )
    if pub_full:
        return pub_full.group(1).strip(), pub_full.group(2).strip(), pub_full.group(3).strip(), True

    pub_short = re.match(r"^\[([^\]]+)\]\((https://[^)]+)\)（已发布）$", col2)
    if pub_short:
        return pub_short.group(1).strip(), pub_short.group(2).strip(), None, True

    return strip_md_links(col2), None, None, False


def parse_roadmap(roadmap_path: Path) -> tuple[list[StageSection], int]:
    """Parse all 14 stages and article rows from llm-agent-learning-roadmap.md."""
    text = roadmap_path.read_text(encoding="utf-8")
    overview = parse_overview_table(text)
    if len(overview) != 14:
        raise RuntimeError(f"Expected 14 overview rows, got {len(overview)}")

    raw_code_rows = re.findall(r"^\|\s*[A-Z]+-\d+\s*\|", text, re.M)
    heading_re = re.compile(r"^##\s+(\d+)\.\s+(.+)$", re.M)
    matches = list(heading_re.finditer(text))
    if len(matches) != 14:
        raise RuntimeError(f"Expected 14 numbered stage headings, got {len(matches)}")

    sections: list[StageSection] = []
    article_row_re = re.compile(r"^\|\s*([A-Z]+-\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$", re.M)

    for idx, hm in enumerate(matches):
        order = int(hm.group(1))
        title = hm.group(2).strip()
        start = hm.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        # Stop Section 14 before "## 首轮写作主线" if present
        sub_text = text[start:end]
        next_h2 = re.search(r"^##\s+", sub_text, re.M)
        if next_h2:
            sub_text = sub_text[: next_h2.start()]

        # Extract concise lead paragraph before table
        intro_lines: list[str] = []
        for ln in sub_text.splitlines():
            s = ln.strip()
            if s.startswith("|"):
                break
            if s:
                intro_lines.append(strip_md_links(s))
        intro = " ".join(intro_lines)[:260]

        ov = overview[order]
        sec = StageSection(
            order=order,
            title=title,
            series_label=ov["series_label"],
            prereq=ov["prereq"],
            deliverable=ov["deliverable"],
            intro=intro,
        )

        for rm in article_row_re.finditer(sub_text):
            code = rm.group(1).strip()
            col2 = rm.group(2).strip()
            col3 = rm.group(3).strip()
            art_title, url, local_file, published = parse_article_cell(col2)
            summary = strip_md_links(col3)
            refs = extract_refs(summary, code)
            sec.articles.append(
                ArticleItem(
                    code=code,
                    title=art_title,
                    url=url,
                    local_file=local_file,
                    published=published,
                    summary=summary,
                    refs=refs,
                )
            )
        sections.append(sec)

    parsed_total = sum(len(s.articles) for s in sections)
    if parsed_total != len(raw_code_rows):
        raise RuntimeError(
            f"Article parse mismatch: parsed {parsed_total} vs {len(raw_code_rows)} raw rows"
        )
    return sections, parsed_total


def build_tree_data(sections: list[StageSection]) -> dict[str, Any]:
    """Build normalized hierarchical mindmap JSON consumed by HTML renderer."""
    sec_by_order = {s.order: s for s in sections}
    total_articles = sum(len(s.articles) for s in sections)
    total_published = sum(1 for s in sections for a in s.articles if a.published)

    pillar_nodes: list[dict[str, Any]] = []
    for pdef in PILLAR_DEFS:
        stage_nodes: list[dict[str, Any]] = []
        p_total = 0
        p_pub = 0
        for order in pdef["stage_orders"]:
            sec = sec_by_order[order]
            s_total = len(sec.articles)
            s_pub = sum(1 for a in sec.articles if a.published)
            p_total += s_total
            p_pub += s_pub
            art_nodes = [
                {
                    "id": f"art-{art.code}",
                    "kind": "article",
                    "code": art.code,
                    "label": f"{art.code} · {art.title}",
                    "title": art.title,
                    "color": pdef["color"],
                    "published": art.published,
                    "url": art.url,
                    "localFile": art.local_file,
                    "summary": art.summary,
                    "refs": art.refs,
                    "stageOrder": sec.order,
                    "stageTitle": sec.title,
                    "prereq": sec.prereq,
                    "deliverable": sec.deliverable,
                }
                for art in sec.articles
            ]
            stage_nodes.append(
                {
                    "id": f"stage-{sec.order}",
                    "kind": "stage",
                    "stageOrder": sec.order,
                    "label": f"{sec.order}. {sec.title}",
                    "title": sec.title,
                    "seriesLabel": sec.series_label,
                    "badge": f"{s_pub}/{s_total}",
                    "publishedCount": s_pub,
                    "totalCount": s_total,
                    "color": pdef["color"],
                    "prereq": sec.prereq,
                    "deliverable": sec.deliverable,
                    "summary": sec.intro or f"先修：{sec.prereq}；阶段产物：{sec.deliverable}",
                    "children": art_nodes,
                }
            )
        pillar_nodes.append(
            {
                "id": pdef["id"],
                "kind": "pillar",
                "label": pdef["title"],
                "title": pdef["title"],
                "badge": f"{p_pub}/{p_total}",
                "color": pdef["color"],
                "summary": pdef["subtitle"],
                "children": stage_nodes,
            }
        )

    lateral_node = {
        "id": "pillar-lateral",
        "kind": "pillar",
        "label": "横向入口：题库 · 速查 · 路线图",
        "title": "横向入口：题库 · 速查 · 路线图",
        "badge": "3 份总览",
        "color": "#475569",
        "summary": "按面试考点反查原理、按主题速查机制、看置顶总路线",
        "children": [
            {
                "id": "lat-bank",
                "kind": "lateral",
                "code": "BANK",
                "label": "面试题库 · Agent 与 Sandbox 高频题（6 专题 26 题）",
                "title": "Agent 与 Sandbox 面试题库：运行时、模型网关与系统设计",
                "color": "#475569",
                "published": True,
                "url": INTERVIEW_BANK_URL,
                "localFile": "agent-sandbox-interview-bank.md",
                "summary": "覆盖 AG / SB / LLM / ARCH / MVP / CTX 六大面试专题与追问参考答案，用 CSV 上传→隔离分析→归档场景贯穿。",
                "refs": [],
            },
            {
                "id": "lat-notes",
                "kind": "lateral",
                "code": "NOTES",
                "label": "工程笔记 · 5 层核心机制速查手册",
                "title": "Agent 工程笔记",
                "color": "#475569",
                "published": True,
                "url": None,
                "localFile": "agent-engineering-notes.md",
                "summary": "把多智能体编排、Agent 记忆、上下文工程、本地执行安全、模型推理与缓存浓缩成 5 层速查。",
                "refs": [],
            },
            {
                "id": "lat-roadmap",
                "kind": "lateral",
                "code": "MAP",
                "label": "置顶路线图 · 14 阶段线上总导航页",
                "title": "LLM 与 Agent 学习路线图：从模型原理到工程落地",
                "color": "#475569",
                "published": True,
                "url": ROADMAP_BLOG_URL,
                "localFile": "llm-agent-learning-roadmap.md",
                "summary": "串联 14 阶段 138 篇专题的置顶总览文章，含常用模型数据集与 Agent 公开基准速查表。",
                "refs": [],
            },
        ],
    }
    pillar_nodes.append(lateral_node)

    return {
        "stats": {
            "stages": len(sections),
            "totalArticles": total_articles,
            "publishedArticles": total_published,
            "plannedArticles": total_articles - total_published,
        },
        "tracks": TRACK_STAGE_MAP,
        "root": {
            "id": "root",
            "kind": "root",
            "label": "LLM 与 Agent 知识脑图",
            "badge": f"{total_published}/{total_articles} 已发布",
            "color": "#0f172a",
            "summary": "请求主线：文本 → Token → 推理 → HTTP/SSE → Agent 工具循环 → RAG 检索 → 安全隔离 → 生产化",
            "children": pillar_nodes,
        },
    }


def write_markdown_outline(tree: dict[str, Any], out_path: Path) -> None:
    """Write a Markmap / XMind-compatible multi-level Markdown outline."""
    root = tree["root"]
    lines = [
        f"# {root['label']}（{root['badge']}）",
        "",
        f"> {root['summary']}",
        "",
    ]
    for pillar in root["children"]:
        lines.append(f"## {pillar['label']}（{pillar['badge']}）")
        lines.append("")
        for child in pillar["children"]:
            if child["kind"] == "stage":
                lines.append(f"### {child['label']}（{child['badge']}）")
                lines.append(f"- 先修：{child['prereq']}")
                lines.append(f"- 产物：{child['deliverable']}")
                for art in child["children"]:
                    if art["url"]:
                        lines.append(f"- **{art['code']}** [{art['title']}]({art['url']})")
                    else:
                        lines.append(f"- **{art['code']}** {art['title']}（规划中）")
                    lines.append(f"  - {art['summary']}")
                lines.append("")
            else:
                if child["url"]:
                    lines.append(f"- **{child['code']}** [{child['title']}]({child['url']})")
                else:
                    lines.append(f"- **{child['code']}** {child['title']}（本地：`{child['localFile']}`）")
                lines.append(f"  - {child['summary']}")
        lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_opml(tree: dict[str, Any], out_path: Path) -> None:
    """Write a valid OPML 2.0 XML outline for XMind / MindNode / Mubu."""
    opml = ET.Element("opml", version="2.0")
    head = ET.SubElement(opml, "head")
    ET.SubElement(head, "title").text = tree["root"]["label"]
    body = ET.SubElement(opml, "body")

    root_el = ET.SubElement(
        body,
        "outline",
        text=f"{tree['root']['label']}（{tree['root']['badge']}）",
        _note=tree["root"]["summary"],
    )

    for pillar in tree["root"]["children"]:
        p_el = ET.SubElement(
            root_el,
            "outline",
            text=f"{pillar['label']}（{pillar['badge']}）",
            _note=pillar["summary"],
        )
        for child in pillar["children"]:
            if child["kind"] == "stage":
                s_note = f"先修：{child['prereq']}\n阶段产物：{child['deliverable']}"
                s_el = ET.SubElement(
                    p_el,
                    "outline",
                    text=f"{child['label']}（{child['badge']}）",
                    _note=s_note,
                )
                for art in child["children"]:
                    status = "已发布" if art["published"] else "规划中"
                    attrs = {
                        "text": f"[{art['code']}] {art['title']}（{status}）",
                        "_note": art["summary"],
                    }
                    if art["url"]:
                        attrs["url"] = art["url"]
                    ET.SubElement(s_el, "outline", **attrs)
            else:
                attrs = {
                    "text": f"[{child['code']}] {child['title']}",
                    "_note": child["summary"],
                }
                if child["url"]:
                    attrs["url"] = child["url"]
                ET.SubElement(p_el, "outline", **attrs)

    ET.indent(opml, space="  ")
    xml_bytes = ET.tostring(opml, encoding="utf-8", xml_declaration=True)
    # Self-verify XML well-formedness
    ET.fromstring(xml_bytes)
    out_path.write_bytes(xml_bytes)


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>LLM 与 Agent 知识图谱脑图</title>
<style>
  :root {
    --bg: #f8fafc;
    --grid: #e2e8f0;
    --surface: #ffffff;
    --surface-2: #f1f5f9;
    --border: #cbd5e1;
    --text: #0f172a;
    --muted: #475569;
    --accent: #2563eb;
    --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    --pub-bg: #dcfce7;
    --pub-fg: #166534;
    --plan-bg: #fef3c7;
    --plan-fg: #92400e;
  }
  html[data-theme="dark"] {
    --bg: #0b1120;
    --grid: #1e293b;
    --surface: #111827;
    --surface-2: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #60a5fa;
    --shadow: 0 14px 34px rgba(0, 0, 0, 0.45);
    --pub-bg: #14532d;
    --pub-fg: #bbf7d0;
    --plan-bg: #78350f;
    --plan-fg: #fde68a;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  .app {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
  }
  .topbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    padding: 8px 14px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    z-index: 20;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 700;
    font-size: 14px;
    white-space: nowrap;
  }
  .stat-pill {
    font-size: 11.5px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 999px;
    background: var(--surface-2);
    color: var(--muted);
    border: 1px solid var(--border);
  }
  .search-wrap {
    position: relative;
    display: flex;
    align-items: center;
  }
  .search-wrap input {
    width: 196px;
    padding: 5px 44px 5px 9px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text);
    font-size: 12px;
    outline: none;
  }
  .search-wrap input:focus {
    border-color: var(--accent);
    background: var(--surface);
  }
  .search-meta {
    position: absolute;
    right: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--muted);
  }
  .icon-clear {
    cursor: pointer;
    border: none;
    background: transparent;
    color: var(--muted);
    font-size: 14px;
    line-height: 1;
    padding: 0 2px;
    display: none;
  }
  .btn-group {
    display: flex;
    align-items: center;
    gap: 2px;
    background: var(--surface-2);
    padding: 2px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .btn {
    border: none;
    background: transparent;
    color: var(--muted);
    padding: 4px 7px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.12s, color 0.12s;
  }
  .btn:hover {
    color: var(--text);
    background: rgba(148, 163, 184, 0.16);
  }
  .btn.active {
    background: var(--surface);
    color: var(--text);
    font-weight: 600;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
  }
  .spacer { flex: 1; }
  .stage-area {
    position: relative;
    flex: 1;
    overflow: hidden;
    background-color: var(--bg);
    background-image: radial-gradient(var(--grid) 1px, transparent 1px);
    background-size: 20px 20px;
    cursor: grab;
    user-select: none;
  }
  .stage-area.dragging { cursor: grabbing; }
  svg#mindmap {
    width: 100%;
    height: 100%;
    display: block;
  }
  .node-g { cursor: pointer; }
  .node-rect { transition: stroke-width 0.15s, filter 0.15s, opacity 0.15s; }
  .node-g:hover .node-rect { filter: drop-shadow(0 4px 10px rgba(15, 23, 42, 0.14)); }
  .link-path {
    fill: none;
    stroke-linecap: round;
    transition: opacity 0.18s;
  }
  .dimmed { opacity: 0.16; }
  .hint-bar {
    position: absolute;
    left: 14px;
    bottom: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 11.5px;
    color: var(--muted);
    box-shadow: var(--shadow);
    pointer-events: none;
  }
  .drawer {
    position: absolute;
    top: 14px;
    right: 14px;
    width: 340px;
    max-height: calc(100% - 28px);
    overflow-y: auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
    z-index: 15;
    display: none;
  }
  .drawer.open { display: block; }
  .drawer-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
  }
  .drawer-badges { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .pill {
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
  }
  .pill-pub { background: var(--pub-bg); color: var(--pub-fg); }
  .pill-plan { background: var(--plan-bg); color: var(--plan-fg); }
  .drawer-close {
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--muted);
    width: 24px;
    height: 24px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
  }
  .drawer-title {
    margin: 4px 0 8px;
    font-size: 15px;
    font-weight: 700;
    line-height: 1.4;
  }
  .drawer-sec {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
    font-size: 12.5px;
    line-height: 1.55;
    color: var(--muted);
  }
  .drawer-sec strong { color: var(--text); }
  .ref-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
  }
  .ref-chip {
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
  }
  .ref-chip:hover { border-color: var(--accent); }
  .drawer-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
  }
  .action-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text);
  }
  .action-btn.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
  }
  .file-code {
    display: inline-block;
    margin-top: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    background: var(--surface-2);
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--text);
  }
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="brand">
      <span>LLM &amp; Agent 知识图谱脑图</span>
      <span class="stat-pill" id="stats-pill">__STATS_LABEL__</span>
    </div>

    <div class="search-wrap">
      <input id="search-input" type="text" placeholder="搜编号/关键词（如 KingBench）" />
      <span class="search-meta">
        <span id="search-count"></span>
        <button class="icon-clear" id="search-clear" title="清空搜索">×</button>
      </span>
    </div>

    <div class="btn-group" aria-label="展开层级">
      <button class="btn" data-level="1">4 板块</button>
      <button class="btn active" data-level="2">14 阶段</button>
      <button class="btn" data-level="3">全部 138 篇</button>
    </div>

    <div class="btn-group" aria-label="学习主线">
      <button class="btn active" data-track="all">全部</button>
      <button class="btn" data-track="request">请求主线</button>
      <button class="btn" data-track="eval">评测闭环</button>
      <button class="btn" data-track="train-serve">训练→部署</button>
      <button class="btn" data-track="sec-prod">安全与生产</button>
    </div>

    <div class="spacer"></div>

    <div class="btn-group" aria-label="画布控制">
      <button class="btn" id="zoom-out" title="缩小">－</button>
      <button class="btn" id="zoom-fit" title="适应屏幕">适应</button>
      <button class="btn" id="zoom-in" title="放大">＋</button>
      <button class="btn" id="theme-toggle" title="切换浅色/深色">深色</button>
    </div>
  </div>

  <div class="stage-area" id="stage-area">
    <svg id="mindmap"></svg>
    <div class="hint-bar">
      提示：点击板块或阶段可展开/收起；点击文章查看机制、先修与关联跳转；双击已发布文章直达博客；滚轮缩放 / 拖拽平移
    </div>
    <div class="drawer" id="drawer"></div>
  </div>
</div>

<script>
const TREE_DATA = __TREE_JSON__;

const svg = document.getElementById("mindmap");
const stageArea = document.getElementById("stage-area");
const drawer = document.getElementById("drawer");
const searchInput = document.getElementById("search-input");
const searchCount = document.getElementById("search-count");
const searchClear = document.getElementById("search-clear");

const measureCanvas = document.createElement("canvas");
const mctx = measureCanvas.getContext("2d");

let expandedSet = new Set();
let selectedId = null;
let activeTrack = "all";
let searchQuery = "";
let matchedIds = new Set();
let nodeIndex = new Map();
let codeToId = new Map();

let viewport = { x: 24, y: 24, scale: 1 };
let dragState = null;

function indexAll(node, parent = null) {
  node._parent = parent;
  nodeIndex.set(node.id, node);
  if (node.code) codeToId.set(node.code, node.id);
  if (node.children) {
    for (const ch of node.children) indexAll(ch, node);
  }
}

function setExpandDepth(maxDepth) {
  expandedSet.clear();
  function walk(node, depth) {
    if (node.children && node.children.length > 0 && depth < maxDepth) {
      expandedSet.add(node.id);
      for (const ch of node.children) walk(ch, depth + 1);
    }
  }
  walk(TREE_DATA.root, 0);
}

function measureTextWidth(text, font) {
  mctx.font = font;
  return Math.ceil(mctx.measureText(text).width);
}

function computeNodeBox(node) {
  let font = '500 12.5px -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif';
  let height = 30;
  let padX = 12;
  let badgeText = "";

  if (node.kind === "root") {
    font = '700 15px -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif';
    height = 44;
    padX = 16;
    badgeText = node.badge || "";
  } else if (node.kind === "pillar") {
    font = '700 13.5px -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif';
    height = 36;
    padX = 14;
    badgeText = node.badge || "";
  } else if (node.kind === "stage") {
    font = '600 13px -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif';
    height = 32;
    padX = 12;
    badgeText = node.badge || "";
  }

  const labelW = measureTextWidth(node.label, font);
  const badgeW = badgeText ? measureTextWidth(badgeText, '600 11px sans-serif') + 18 : 0;
  const hasKids = node.children && node.children.length > 0;
  const toggleW = hasKids ? 20 : 0;
  const dotW = node.kind === "article" || node.kind === "lateral" ? 12 : 0;
  const linkIconW = node.url ? 14 : 0;

  node._w = padX * 2 + dotW + labelW + badgeW + linkIconW + toggleW;
  node._h = height;
  node._font = font;
  node._badgeText = badgeText;
  node._badgeW = badgeW;
}

function layoutVisibleTree() {
  const root = TREE_DATA.root;
  const X_GAP = 54;
  const Y_GAP = 8;
  const GROUP_GAP = 14;

  function calcSubtree(node) {
    computeNodeBox(node);
    const isExp = expandedSet.has(node.id);
    const visChildren = isExp && node.children ? node.children : [];
    node._visChildren = visChildren;
    if (visChildren.length === 0) {
      node._subH = node._h;
      return node._subH;
    }
    let total = 0;
    visChildren.forEach((ch, idx) => {
      total += calcSubtree(ch);
      if (idx < visChildren.length - 1) {
        total += ch.kind === "pillar" ? GROUP_GAP : Y_GAP;
      }
    });
    node._subH = Math.max(node._h, total);
    return node._subH;
  }

  function assignPos(node, x, topY) {
    node._x = x;
    const vis = node._visChildren;
    if (vis.length === 0) {
      node._y = topY + (node._subH - node._h) / 2;
      return;
    }
    let childrenSum = 0;
    vis.forEach((ch, idx) => {
      childrenSum += ch._subH;
      if (idx < vis.length - 1) {
        childrenSum += ch.kind === "pillar" ? GROUP_GAP : Y_GAP;
      }
    });
    let curY = topY + Math.max(0, (node._subH - childrenSum) / 2);
    const childX = x + node._w + X_GAP;
    for (const ch of vis) {
      assignPos(ch, childX, curY);
      curY += ch._subH + (ch.kind === "pillar" ? GROUP_GAP : Y_GAP);
    }
    const first = vis[0];
    const last = vis[vis.length - 1];
    const center = ((first._y + first._h / 2) + (last._y + last._h / 2)) / 2;
    node._y = center - node._h / 2;
  }

  calcSubtree(root);
  assignPos(root, 0, 0);

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  function bounds(n) {
    minX = Math.min(minX, n._x);
    minY = Math.min(minY, n._y);
    maxX = Math.max(maxX, n._x + n._w);
    maxY = Math.max(maxY, n._y + n._h);
    for (const ch of n._visChildren) bounds(ch);
  }
  bounds(root);
  return { minX, minY, maxX, maxY };
}

function isNodeHighlighted(node) {
  if (searchQuery) {
    if (matchedIds.has(node.id)) return true;
    // Keep ancestors visible
    function hasDescendantMatch(n) {
      if (matchedIds.has(n.id)) return true;
      return (n.children || []).some(hasDescendantMatch);
    }
    return hasDescendantMatch(node);
  }
  if (activeTrack === "all") return true;
  const trackStages = new Set(TREE_DATA.tracks[activeTrack] || []);
  if (node.kind === "root") return true;
  if (node.kind === "stage" || node.kind === "article") {
    return trackStages.has(node.stageOrder);
  }
  if (node.kind === "pillar") {
    return (node.children || []).some(ch => ch.kind === "stage" && trackStages.has(ch.stageOrder));
  }
  return false;
}

function svgEl(tag, attrs = {}, text = null) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  }
  if (text !== null) el.textContent = text;
  return el;
}

function getThemeColors() {
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  return {
    cardBg: dark ? "#111827" : "#ffffff",
    cardBorder: dark ? "#334155" : "#cbd5e1",
    text: dark ? "#f1f5f9" : "#0f172a",
    muted: dark ? "#94a3b8" : "#64748b",
    badgeBg: dark ? "#1e293b" : "#f1f5f9",
    matchStroke: "#f59e0b",
    matchFill: dark ? "#2a2009" : "#fffbeb",
  };
}

function applyTransform() {
  const g = svg.querySelector("#viewport-g");
  if (g) {
    g.setAttribute("transform", `translate(${viewport.x.toFixed(1)}, ${viewport.y.toFixed(1)}) scale(${viewport.scale.toFixed(4)})`);
  }
}

function fitView(bounds) {
  const rect = stageArea.getBoundingClientRect();
  const availW = Math.max(300, rect.width - 48);
  const availH = Math.max(240, rect.height - 64);
  const treeW = Math.max(1, bounds.maxX - bounds.minX);
  const treeH = Math.max(1, bounds.maxY - bounds.minY);
  const scale = Math.min(1.08, Math.max(0.28, Math.min(availW / treeW, availH / treeH)));
  viewport.scale = scale;
  viewport.x = (rect.width - treeW * scale) / 2 - bounds.minX * scale;
  viewport.y = (rect.height - treeH * scale) / 2 - bounds.minY * scale - 8;
  applyTransform();
}

function render(shouldFit = false) {
  const bounds = layoutVisibleTree();
  const theme = getThemeColors();
  svg.innerHTML = "";
  const vpG = svgEl("g", { id: "viewport-g" });
  const linksG = svgEl("g", { id: "links-g" });
  const nodesG = svgEl("g", { id: "nodes-g" });
  vpG.appendChild(linksG);
  vpG.appendChild(nodesG);
  svg.appendChild(vpG);

  function drawLinks(node) {
    for (const ch of node._visChildren) {
      const x1 = node._x + node._w;
      const y1 = node._y + node._h / 2;
      const x2 = ch._x;
      const y2 = ch._y + ch._h / 2;
      const mx = (x1 + x2) / 2;
      const d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
      const bright = isNodeHighlighted(node) && isNodeHighlighted(ch);
      const sw = node.kind === "root" ? 2.4 : node.kind === "pillar" ? 1.7 : 1.3;
      linksG.appendChild(
        svgEl("path", {
          d,
          class: `link-path${bright ? "" : " dimmed"}`,
          stroke: ch.color || node.color,
          "stroke-width": sw,
          opacity: bright ? 0.78 : 0.16,
        })
      );
      drawLinks(ch);
    }
  }

  function drawNodes(node) {
    const bright = isNodeHighlighted(node);
    const isMatch = searchQuery && matchedIds.has(node.id);
    const isSelected = selectedId === node.id;

    const g = svgEl("g", {
      class: `node-g${bright ? "" : " dimmed"}`,
      transform: `translate(${node._x}, ${node._y})`,
      "data-id": node.id,
    });

    let fill = isMatch ? theme.matchFill : theme.cardBg;
    let stroke = isMatch ? theme.matchStroke : (node.kind === "root" || node.kind === "pillar" ? node.color : theme.cardBorder);
    let strokeW = isSelected || isMatch ? 2.2 : (node.kind === "root" ? 2.2 : node.kind === "pillar" ? 1.7 : 1.1);
    let rx = node.kind === "root" ? 12 : node.kind === "pillar" ? 10 : 8;

    if (isSelected) stroke = node.color || "#2563eb";

    g.appendChild(
      svgEl("rect", {
        class: "node-rect",
        x: 0,
        y: 0,
        width: node._w,
        height: node._h,
        rx,
        fill,
        stroke,
        "stroke-width": strokeW,
      })
    );

    // Accent bar on left for stage / pillar
    if (node.kind === "pillar" || node.kind === "stage") {
      g.appendChild(
        svgEl("rect", {
          x: 0,
          y: 0,
          width: 5,
          height: node._h,
          rx: 2,
          fill: node.color,
        })
      );
    }

    let cursorX = node.kind === "root" ? 16 : node.kind === "pillar" ? 14 : 12;
    const centerY = node._h / 2;

    if (node.kind === "article" || node.kind === "lateral") {
      const dotColor = node.published ? "#16a34a" : "#d97706";
      g.appendChild(
        svgEl("circle", {
          cx: cursorX + 3,
          cy: centerY,
          r: 3.5,
          fill: node.published ? dotColor : "none",
          stroke: dotColor,
          "stroke-width": 1.6,
        })
      );
      cursorX += 12;
    }

    const txt = svgEl(
      "text",
      {
        x: cursorX,
        y: centerY,
        "dominant-baseline": "central",
        fill: theme.text,
        style: `font:${node._font}`,
      },
      node.label
    );
    g.appendChild(txt);

    const hasKids = node.children && node.children.length > 0;
    let rightOffset = 10 + (hasKids ? 20 : 0);

    if (node.url) {
      g.appendChild(
        svgEl(
          "text",
          {
            x: node._w - rightOffset - 6,
            y: centerY,
            "dominant-baseline": "central",
            "text-anchor": "middle",
            fill: node.color || theme.muted,
            style: "font:600 11px sans-serif",
          },
          "↗"
        )
      );
    }

    if (node._badgeText) {
      const bw = node._badgeW - 4;
      const bx = node._w - rightOffset - bw;
      g.appendChild(
        svgEl("rect", {
          x: bx,
          y: centerY - 9,
          width: bw,
          height: 18,
          rx: 9,
          fill: node.kind === "root" ? node.color : theme.badgeBg,
        })
      );
      g.appendChild(
        svgEl(
          "text",
          {
            x: bx + bw / 2,
            y: centerY,
            "dominant-baseline": "central",
            "text-anchor": "middle",
            fill: node.kind === "root" ? "#ffffff" : theme.muted,
            style: "font:600 10.5px sans-serif",
          },
          node._badgeText
        )
      );
    }

    if (hasKids) {
      const isExp = expandedSet.has(node.id);
      const cx = node._w - 13;
      g.appendChild(
        svgEl("circle", {
          cx,
          cy: centerY,
          r: 7.5,
          fill: isExp ? theme.badgeBg : (node.color || "#2563eb"),
          stroke: node.color || theme.cardBorder,
          "stroke-width": 1.2,
        })
      );
      g.appendChild(
        svgEl(
          "text",
          {
            x: cx,
            y: centerY - 0.3,
            "dominant-baseline": "central",
            "text-anchor": "middle",
            fill: isExp ? theme.text : "#ffffff",
            style: "font:700 11px sans-serif",
          },
          isExp ? "−" : "+"
        )
      );
    }

    g.addEventListener("click", (ev) => {
      ev.stopPropagation();
      handleNodeClick(node);
    });
    g.addEventListener("dblclick", (ev) => {
      ev.stopPropagation();
      if (node.url) window.open(node.url, "_blank", "noopener,noreferrer");
    });

    nodesG.appendChild(g);
    for (const ch of node._visChildren) drawNodes(ch);
  }

  drawLinks(TREE_DATA.root);
  drawNodes(TREE_DATA.root);

  if (shouldFit) {
    fitView(bounds);
  } else {
    applyTransform();
  }
}

function handleNodeClick(node) {
  selectedId = node.id;
  if (node.children && node.children.length > 0 && node.kind !== "root") {
    if (expandedSet.has(node.id)) {
      expandedSet.delete(node.id);
    } else {
      expandedSet.add(node.id);
    }
  }
  render(false);
  openDrawer(node);
}

function jumpToCode(code) {
  const id = codeToId.get(code);
  if (!id) return;
  const target = nodeIndex.get(id);
  // Expand all ancestors
  let cur = target._parent;
  while (cur) {
    expandedSet.add(cur.id);
    cur = cur._parent;
  }
  selectedId = target.id;
  render(false);
  // Pan smoothly to keep target near left-center
  const rect = stageArea.getBoundingClientRect();
  viewport.x = rect.width * 0.32 - (target._x + target._w / 2) * viewport.scale;
  viewport.y = rect.height * 0.5 - (target._y + target._h / 2) * viewport.scale;
  applyTransform();
  openDrawer(target);
}

function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function openDrawer(node) {
  const badges = [];
  if (node.code) {
    badges.push(`<span class="pill" style="background:${node.color};color:#fff">${escapeHtml(node.code)}</span>`);
  }
  if (node.kind === "article" || node.kind === "lateral") {
    badges.push(
      node.published
        ? `<span class="pill pill-pub">已发布</span>`
        : `<span class="pill pill-plan">规划待写</span>`
    );
  } else if (node.kind === "stage") {
    badges.push(`<span class="pill" style="background:${node.color};color:#fff">阶段 ${node.stageOrder} · ${escapeHtml(node.badge)}</span>`);
  } else if (node.kind === "pillar") {
    badges.push(`<span class="pill" style="background:${node.color};color:#fff">板块 · ${escapeHtml(node.badge)}</span>`);
  }

  let html = `
    <div class="drawer-head">
      <div class="drawer-badges">${badges.join("")}</div>
      <button class="drawer-close" id="drawer-close" title="关闭">×</button>
    </div>
    <h3 class="drawer-title">${escapeHtml(node.title || node.label)}</h3>
  `;

  if (node.summary) {
    html += `<div class="drawer-sec"><strong>机制与说明：</strong>${escapeHtml(node.summary)}</div>`;
  }
  if (node.prereq || node.deliverable) {
    html += `<div class="drawer-sec">
      ${node.stageTitle ? `<div><strong>所属阶段：</strong>${node.stageOrder}. ${escapeHtml(node.stageTitle)}</div>` : ""}
      ${node.prereq ? `<div><strong>先修要求：</strong>${escapeHtml(node.prereq)}</div>` : ""}
      ${node.deliverable ? `<div><strong>阶段产物：</strong>${escapeHtml(node.deliverable)}</div>` : ""}
    </div>`;
  }
  if (node.refs && node.refs.length > 0) {
    const chips = node.refs
      .map(r => `<button class="ref-chip" data-ref="${escapeHtml(r)}">→ ${escapeHtml(r)}</button>`)
      .join("");
    html += `<div class="drawer-sec"><strong>跨篇关联（点击跳转）：</strong><div class="ref-chips">${chips}</div></div>`;
  }
  if (node.localFile) {
    html += `<div class="drawer-sec"><strong>本地源文件：</strong><br><span class="file-code">${escapeHtml(node.localFile)}</span></div>`;
  }
  if (node.url) {
    html += `<div class="drawer-actions">
      <a class="action-btn primary" href="${escapeHtml(node.url)}" target="_blank" rel="noopener noreferrer">打开线上博客文章 ↗</a>
    </div>`;
  }

  drawer.innerHTML = html;
  drawer.classList.add("open");
  drawer.querySelector("#drawer-close").addEventListener("click", () => {
    drawer.classList.remove("open");
  });
  drawer.querySelectorAll(".ref-chip").forEach(btn => {
    btn.addEventListener("click", () => jumpToCode(btn.dataset.ref));
  });
}

function runSearch(q) {
  searchQuery = q.trim().toLowerCase();
  matchedIds.clear();
  searchClear.style.display = searchQuery ? "inline-block" : "none";

  if (!searchQuery) {
    searchCount.textContent = "";
    render(false);
    return;
  }

  for (const node of nodeIndex.values()) {
    const hay = `${node.code || ""} ${node.label || ""} ${node.title || ""} ${node.summary || ""} ${node.prereq || ""}`.toLowerCase();
    if (hay.includes(searchQuery)) {
      matchedIds.add(node.id);
      let cur = node._parent;
      while (cur) {
        expandedSet.add(cur.id);
        cur = cur._parent;
      }
    }
  }
  searchCount.textContent = `${matchedIds.size} 命中`;
  render(true);
}

// Toolbar events
document.querySelectorAll("[data-level]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-level]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    setExpandDepth(Number(btn.dataset.level));
    render(true);
  });
});

document.querySelectorAll("[data-track]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-track]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeTrack = btn.dataset.track;
    render(false);
  });
});

searchInput.addEventListener("input", () => runSearch(searchInput.value));
searchClear.addEventListener("click", () => {
  searchInput.value = "";
  runSearch("");
});

document.getElementById("zoom-in").addEventListener("click", () => {
  const rect = stageArea.getBoundingClientRect();
  zoomAt(rect.width / 2, rect.height / 2, 1.2);
});
document.getElementById("zoom-out").addEventListener("click", () => {
  const rect = stageArea.getBoundingClientRect();
  zoomAt(rect.width / 2, rect.height / 2, 1 / 1.2);
});
document.getElementById("zoom-fit").addEventListener("click", () => {
  const bounds = layoutVisibleTree();
  fitView(bounds);
});

const themeBtn = document.getElementById("theme-toggle");
themeBtn.addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", cur);
  themeBtn.textContent = cur === "dark" ? "浅色" : "深色";
  render(false);
});

function zoomAt(cx, cy, factor) {
  const newScale = Math.min(2.5, Math.max(0.2, viewport.scale * factor));
  const actual = newScale / viewport.scale;
  viewport.x = cx - (cx - viewport.x) * actual;
  viewport.y = cy - (cy - viewport.y) * actual;
  viewport.scale = newScale;
  applyTransform();
}

stageArea.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const rect = stageArea.getBoundingClientRect();
  const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
  zoomAt(ev.clientX - rect.left, ev.clientY - rect.top, factor);
}, { passive: false });

stageArea.addEventListener("pointerdown", (ev) => {
  if (ev.target.closest(".node-g") || ev.target.closest(".drawer")) return;
  dragState = { startX: ev.clientX, startY: ev.clientY, vx: viewport.x, vy: viewport.y };
  stageArea.classList.add("dragging");
});

window.addEventListener("pointermove", (ev) => {
  if (!dragState) return;
  viewport.x = dragState.vx + (ev.clientX - dragState.startX);
  viewport.y = dragState.vy + (ev.clientY - dragState.startY);
  applyTransform();
});

window.addEventListener("pointerup", () => {
  dragState = null;
  stageArea.classList.remove("dragging");
});

window.addEventListener("resize", () => {
  const bounds = layoutVisibleTree();
  fitView(bounds);
});

// Boot
indexAll(TREE_DATA.root, null);
setExpandDepth(2);
render(true);
</script>
</body>
</html>
"""


def write_html(tree: dict[str, Any], out_path: Path) -> None:
    """Render standalone interactive mindmap HTML without external CDN dependencies."""
    stats = tree["stats"]
    stats_label = (
        f"{stats['stages']} 阶段 · {stats['publishedArticles']} 已发布 / {stats['plannedArticles']} 规划"
    )
    tree_json = json.dumps(tree, ensure_ascii=False).replace("</", "<\\/")
    html = (
        HTML_TEMPLATE.replace("__STATS_LABEL__", stats_label)
        .replace("__TREE_JSON__", tree_json)
    )
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    sections, total = parse_roadmap(ROADMAP_PATH)
    tree = build_tree_data(sections)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_markdown_outline(tree, OUT_MD)
    write_opml(tree, OUT_OPML)
    write_html(tree, OUT_HTML)

    stats = tree["stats"]
    print(
        json.dumps(
            {
                "stages": stats["stages"],
                "totalArticles": total,
                "publishedArticles": stats["publishedArticles"],
                "plannedArticles": stats["plannedArticles"],
                "outputs": [
                    str(OUT_HTML),
                    str(OUT_MD),
                    str(OUT_OPML),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
