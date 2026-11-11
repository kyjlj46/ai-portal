"""여러 문서를 함께 쓸 때 (최대 MAX_DOCS 개).

- 문서마다 고정 표시(D1, D3 …, store.label_for)를 붙이고, 요청 머리말에 문서 목록을 준다.
- 에이전트는 도구의 doc_id 에 표시를 넣고 ('D1,D3'), 출처는 (D1 p.3) 형식으로 쓴다.
- 답변의 출처를 정리하고 (없는 문서·페이지 제거), 문서별 출처 카드를 만든다.
문서가 하나면 이 모듈을 쓰지 않는다 (기존 (p.N) 방식 그대로).
"""

from __future__ import annotations

import re
from pathlib import Path

from . import store

MAX_DOCS = 5
PRELOAD_MAX_PAGES = 15
PRELOAD_MAX_CHARS = 40_000


def docs_of(doc_ids: list[str]) -> list[dict]:
    """[{label, doc_id, filename, page_count, image}] — 요청에 준 순서대로."""
    out = []
    for doc_id in doc_ids:
        meta = store.load_meta(doc_id)
        out.append({
            "label": store.label_for(doc_id),
            "doc_id": doc_id,
            "filename": Path(meta["path"]).name,
            "page_count": meta["page_count"],
            "unit": "구간" if meta.get("unit") == "section" else "페이지",
            "image": bool(meta.get("pdf", meta["path"])),
        })
    return out


def header(docs: list[dict], today: str | None = None) -> str:
    labels = ", ".join(d["label"] for d in docs)
    lines = [f"doc_ids: {labels}  (여러 문서 — 도구의 doc_id 에는 이 표시를 넣습니다. 함께 검색: '{','.join(d['label'] for d in docs)}')"]
    if today:
        lines.append(f"오늘: {today}")
    lines.append("[문서 목록]")
    lines += [f"- {d['label']} = {d['filename']} ({d['page_count']}{d['unit']})" for d in docs]
    lines.append("출처 표기: 문서에서 온 사실 끝에 (D1 p.3) 처럼 문서 표시와 페이지를 함께 씁니다. 여러 곳이면 (D1 p.3, D3 p.5).")
    return "\n".join(lines)


def preload(docs: list[dict]) -> str:
    """문서들이 짧으면 전문을 '===== D1 p.1 =====' 표시와 함께 붙인다."""
    parts, total_pages, total_chars = [], 0, 0
    for d in docs:
        for p in store.load_pages(d["doc_id"]):
            text = p.get("text", "").strip()
            if not text:
                continue
            total_pages += 1
            total_chars += len(text)
            parts.append(f"===== {d['label']} p.{p['page']} =====\n{text}")
    if not parts or total_pages > PRELOAD_MAX_PAGES or total_chars > PRELOAD_MAX_CHARS:
        return ""
    return ("\n\n[문서 전문 — 문서·페이지별. 이미 전부 주어졌으니 도구로 다시 읽을 필요 없음."
            " 출처는 '===== D1 p.N =====' 의 표시로 (D1 p.N) 표기]\n" + "\n\n".join(parts))


# ─────────────────────────── 출처 정리 ───────────────────────────

_GROUP = re.compile(r"\s?\(([^()]*D\d+\s*[,·]?\s*p\.?\s*\d+[^()]*)\)")


def normalize(text: str) -> str:
    """【D1 p.3】, [D1, p.3†L1], (D1, p. 3) … → (D1 p.3)."""
    text = re.sub(r"[【\[]\s*(D\d+)\s*[,·]?\s*p\.?\s*(\d+)[^】\]\n]*[】\]]", r"(\1 p.\2)", text)
    return re.sub(r"\(\s*(D\d+)\s*[,·]?\s*p\.?\s*(\d+)\s*\)", r"(\1 p.\2)", text)


def refs_in(group: str) -> list[tuple[str, int]]:
    """'D1 p.3, p.5, D3 p.2' → [(D1,3), (D1,5), (D3,2)] (표시가 없는 페이지는 앞 문서를 따른다)."""
    out, current = [], None
    for m in re.finditer(r"(D\d+)?\s*[,·]?\s*p\.?\s*(\d+)", group):
        current = m.group(1) or current
        if current:
            out.append((current, int(m.group(2))))
    return out


def drop_invalid(text: str, docs: list[dict]) -> str:
    """고른 문서가 아니거나 페이지 범위를 벗어난 출처를 뺀다 (전부 틀리면 괄호째)."""
    counts = {d["label"]: d["page_count"] for d in docs}

    def fix(m: re.Match) -> str:
        refs = refs_in(m.group(1))
        valid = list(dict.fromkeys((lb, p) for lb, p in refs if lb in counts and 1 <= p <= counts[lb]))
        if not valid:
            return ""
        lead = " " if m.group(0)[:1].isspace() else ""
        return lead + "(" + ", ".join(f"{lb} p.{p}" for lb, p in valid) + ")"

    return _GROUP.sub(fix, text)


def cited(answer: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for m in _GROUP.finditer(answer):
        for ref in refs_in(m.group(1)):
            if ref not in out:
                out.append(ref)
    return out


def sources(messages: list, answer: str, docs: list[dict], limit: int = 8) -> list[dict]:
    """문서별 출처 카드: {label, doc_id, filename, page, section, snippet, cited, image}."""
    from .tools import _tokenize

    by_label = {d["label"]: d for d in docs}
    by_id = {d["doc_id"]: d for d in docs}
    seen_chunks: list[str] = []
    seen_pages: list[tuple[str, int]] = []
    for m in messages:
        if getattr(m, "type", "") != "tool":
            continue
        for cid in re.findall(r"\b([0-9a-f]{10}-\d{4})\b", m.text):
            if cid[:10] in by_id and cid not in seen_chunks:
                seen_chunks.append(cid)
        for lb, p in re.findall(r"\[(D\d+) p\.(\d+) source=", m.text):  # read_page 결과
            if lb in by_label and (lb, int(p)) not in seen_pages:
                seen_pages.append((lb, int(p)))
    chunks = {c["id"]: c for d in docs for c in store.load_chunks(d["doc_id"])}
    retrieved = [chunks[c] for c in seen_chunks if c in chunks]
    for c in retrieved:
        ref = (by_id[c["doc_id"]]["label"], c["page"])
        if ref not in seen_pages:
            seen_pages.append(ref)

    refs = [(lb, p) for lb, p in cited(answer) if lb in by_label and 1 <= p <= by_label[lb]["page_count"]]
    pages = refs or seen_pages[:limit]
    tokens = set(_tokenize(answer))
    out = []
    for lb, p in pages:
        d = by_label[lb]
        candidates = [c for c in retrieved if c["doc_id"] == d["doc_id"] and c["page"] == p] or \
                     [c for c in chunks.values() if c["doc_id"] == d["doc_id"] and c["page"] == p]
        if candidates:
            best = max(candidates, key=lambda c: len(tokens & set(_tokenize(c["text"]))))
            section, snippet = best["section"], best["text"]
        else:
            section, snippet = "", (store.load_page(d["doc_id"], p) or {"text": ""})["text"]
        out.append({"label": lb, "doc_id": d["doc_id"], "filename": d["filename"], "page": p, "section": section,
                    "snippet": snippet[:400], "cited": bool(refs), "image": d["image"]})
    return out
