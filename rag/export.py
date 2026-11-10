"""에이전트 결과(마크다운)를 Word(.docx)로 저장한다.

보고서/기안문에 쓰는 마크다운 범위만 다룬다: 제목(#~###), 문단, 글머리표/번호 목록, 표, 인용(>), **굵게**.
"""

from __future__ import annotations

import io
import re

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt

FONT = "맑은 고딕"
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _add_runs(paragraph, text: str) -> None:
    """**굵게** 만 서식으로 살리고, 나머지 마크다운 기호(`, *)는 걷어낸다."""
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        part = re.sub(r"`([^`]*)`", r"\1", part)
        part = re.sub(r"(?<!\w)\*(\S[^*]*?)\*(?!\w)", r"\1", part)
        if part:
            paragraph.add_run(part).bold = i % 2 == 1


def markdown_to_docx(markdown: str, title: str = "") -> bytes:
    d = docx.Document()
    style = d.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        st = d.styles[name]
        st.font.name = FONT
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    lines = markdown.replace("\r\n", "\n").split("\n")
    if title and not any(line.startswith("# ") for line in lines):
        d.add_heading(title, level=0)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped or re.fullmatch(r"-{3,}|\*{3,}", stripped):
            i += 1
            continue

        # 표: 머리글 줄 + 구분 줄(|---|) + 본문 줄들
        if stripped.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip()):
            rows = [_cells(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            width = max(len(r) for r in rows)
            table = d.add_table(rows=len(rows), cols=width)
            table.style = "Table Grid"
            for r, row in enumerate(rows):
                for c in range(width):
                    para = table.cell(r, c).paragraphs[0]
                    _add_runs(para, row[c] if c < len(row) else "")
                    if r == 0:
                        for run in para.runs:
                            run.bold = True
            d.add_paragraph()
            continue

        if m := re.match(r"^(#{1,6})\s+(.*)", stripped):
            level = len(m.group(1))
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(2))
            h = d.add_heading(text, level=0 if level == 1 else min(level - 1, 3))
            if level == 1:
                h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif m := re.match(r"^(\s*)[-*+]\s+(.*)", line):
            depth = len(m.group(1).replace("\t", "  ")) // 2
            _add_runs(d.add_paragraph(style="List Bullet 2" if depth else "List Bullet"), m.group(2))
        elif m := re.match(r"^(\s*)\d+[.)]\s+(.*)", line):
            _add_runs(d.add_paragraph(style="List Number"), m.group(2))
        elif stripped.startswith(">"):
            p = d.add_paragraph()
            p.paragraph_format.left_indent = Pt(12)
            _add_runs(p, stripped.lstrip("> ").strip())
            for run in p.runs:
                run.italic = True
        else:
            _add_runs(d.add_paragraph(), stripped)
        i += 1

    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()
