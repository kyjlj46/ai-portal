"""RAG 파이프라인 단계별 도구.

수집 단계(ocr_pages, chunk_document)는 portal.ingest 가 호출하고,
질의응답 에이전트는 search_chunks / get_chunk_context / read_page 를 쓴다.
inspect_pdf / extract_text_layer 는 텍스트 레이어 진단용으로 남겨 둔다.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pymupdf as fitz
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25L

from . import store

# 텍스트 레이어가 이보다 짧으면 스캔 페이지로 간주해 OCR 대상으로 표시
MIN_TEXT_CHARS = 40
# 품질 점수가 이보다 낮으면 (깨진 글자 등) OCR 재시도 권장
MIN_QUALITY = 0.85


def _parse_pages(pages: str | None, page_count: int) -> list[int]:
    """'1-3,5' 같은 문자열을 1-based 페이지 목록으로 변환. None/'all' 이면 전체."""
    if not pages or pages.strip().lower() == "all":
        return list(range(1, page_count + 1))
    result: set[int] = set()
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        elif part:
            result.add(int(part))
    return sorted(p for p in result if 1 <= p <= page_count)


def _text_quality(text: str) -> float:
    """정상 문자(한글/영문/숫자/공백/일반 문장부호) 비율. 깨진 인코딩이면 낮아진다."""
    if not text.strip():
        return 0.0
    ok = re.findall(r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9\s.,:;!?()\[\]{}'\"%/\-+=*&#@~·•–—<>|_$€₩]", text)
    return round(len(ok) / len(text), 3)


# ─────────────────────────── 1. 파싱 (텍스트 레이어) ───────────────────────────


@tool
def inspect_pdf(pdf_path: str) -> str:
    """PDF 를 등록하고 페이지별로 텍스트 레이어 존재 여부/이미지 수를 진단한다.

    반환값의 needs_ocr 페이지는 ocr-agent 에, 나머지는 extract_text_layer 로 처리한다.
    """
    doc_id = store.doc_id_for(pdf_path)
    with fitz.open(pdf_path) as pdf:
        pages = []
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            n_chars = len(text.strip())
            pages.append(
                {
                    "page": i,
                    "chars": n_chars,
                    "images": len(page.get_images()),
                    "quality": _text_quality(text),
                    "needs_ocr": n_chars < MIN_TEXT_CHARS
                    or _text_quality(text) < MIN_QUALITY,
                }
            )
    store.save_meta(doc_id, {"doc_id": doc_id, "path": str(pdf_path), "page_count": len(pages), "pages": pages})

    ocr = [p["page"] for p in pages if p["needs_ocr"]]
    text_pages = [p["page"] for p in pages if not p["needs_ocr"]]
    lines = [
        f"doc_id={doc_id}  pages={len(pages)}",
        f"텍스트 레이어 페이지: {text_pages or '없음'}",
        f"OCR 필요 페이지: {ocr or '없음'}",
        "page | chars | images | quality",
    ]
    lines += [f"{p['page']} | {p['chars']} | {p['images']} | {p['quality']}" for p in pages]
    return "\n".join(lines)


@tool
def extract_text_layer(doc_id: str, pages: str = "all") -> str:
    """PDF 텍스트 레이어에서 페이지 텍스트를 추출해 저장한다 (pages 예: '1-3,5' 또는 'all').

    읽기 순서를 보정(sort)하고, 각 페이지 품질 점수를 함께 돌려준다.
    품질이 낮은 페이지는 OCR 로 다시 처리하도록 보고할 것.
    """
    meta = store.load_meta(doc_id)
    targets = _parse_pages(pages, meta["page_count"])
    report = []
    with fitz.open(meta["path"]) as pdf:
        for p in targets:
            text = pdf[p - 1].get_text("text", sort=True).strip()
            q = _text_quality(text)
            store.save_page(doc_id, p, text, source="text", quality=q)
            flag = "  ⚠ 품질 낮음 → OCR 권장" if q < MIN_QUALITY or len(text) < MIN_TEXT_CHARS else ""
            report.append(f"p.{p}: {len(text)}자, quality={q}{flag} | 미리보기: {text[:120]!r}")
    return "\n".join(report)


@tool
def read_page(doc_id: str, page: int) -> str:
    """저장된 페이지 텍스트 전문을 읽는다. doc_id 에는 문서 ID 또는 표시(D3)를 넣는다."""
    label = doc_id.strip() if re.fullmatch(r"D\d+", doc_id.strip()) else ""
    data = store.load_page(store.resolve(doc_id), page)
    if not data:
        return f"p.{page} 는 아직 추출되지 않았습니다."
    return f"[{label + ' ' if label else ''}p.{page} source={data['source']} quality={data['quality']}]\n{data['text']}"


# ─────────────────────────── 2. OCR (로컬 OCR + LLM 정리) ───────────────────────────


@tool
def ocr_pages(doc_id: str, pages: str, dpi: int = 200) -> str:
    """지정한 페이지를 이미지로 렌더링해 OCR 하고, LLM 으로 표/띄어쓰기/읽기 순서를 정리해 저장한다 (pages 예: '2,4-6').

    결과가 부실한 페이지는 dpi=300 으로 다시 호출한다. 다시 OCR 한 뒤에는 chunk_document 로 재인덱싱해야 한다.
    """
    from .ocr import ocr_page

    meta = store.load_meta(doc_id)
    pdf = meta.get("pdf", meta["path"])
    if not pdf:
        return "이 문서는 PDF 가 없는 텍스트 문서라 OCR 할 수 없습니다."
    report = []
    for p in _parse_pages(pages, meta["page_count"]):
        r = ocr_page(pdf, p, dpi=dpi)
        store.save_page(doc_id, p, r["text"], source="ocr", quality=r["quality"])
        store.save_raw(doc_id, p, r["raw"])
        report.append(
            f"p.{p}: OCR {len(r['text'])}자, 글자상자 {r['boxes']}개, 신뢰도={r['quality']} | 미리보기: {r['text'][:120]!r}"
        )
    return "\n".join(report)


# ─────────────────────────── 3. 청킹 ───────────────────────────


@tool
def chunk_document(doc_id: str, chunk_size: int = 800, chunk_overlap: int = 120) -> str:
    """추출/OCR 된 모든 페이지를 청크로 나누고 검색 인덱스를 만든다.

    - 문단 → 줄 → 문장 → 단어 순으로 경계를 존중해 자른다.
    - 각 청크에 페이지 번호와 가장 가까운 섹션 제목을 메타데이터로 붙인다.
    """
    meta = store.load_meta(doc_id)
    pages = store.load_pages(doc_id)
    missing = sorted(set(range(1, meta["page_count"] + 1)) - {p["page"] for p in pages})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "다. ", "? ", "! ", " ", ""],
    )
    chunks: list[dict] = []
    section = ""
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            # 청크 안의 마지막 헤더(마크다운 # 또는 '1. 제목' 형태)를 섹션으로 기억
            heads = re.findall(r"^(?:#{1,3}\s+.+|\d+(?:\.\d+)*\.?\s+\S.{0,40})$", piece, re.M)
            if heads:
                section = heads[-1].lstrip("# ").strip()
            chunks.append(
                {
                    "id": f"{doc_id}-{len(chunks):04d}",
                    "doc_id": doc_id,
                    "page": page["page"],
                    "section": section,
                    "source": page["source"],
                    "text": piece,
                }
            )
    store.save_chunks(doc_id, chunks)
    store._write(store.doc_dir(doc_id) / "chunking.json", {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
    _bm25_index.cache_clear()

    sizes = [len(c["text"]) for c in chunks] or [0]
    tiny = sum(1 for s in sizes if s < 100)
    return (
        f"doc_id={doc_id}: {len(chunks)}개 청크 생성 (페이지 {len(pages)}/{meta['page_count']})\n"
        f"길이 min/avg/max = {min(sizes)}/{sum(sizes) // len(sizes)}/{max(sizes)}, 100자 미만 청크 {tiny}개\n"
        + (f"⚠ 추출되지 않은 페이지: {missing}\n" if missing else "")
    )


@tool
def preview_chunks(doc_id: str, start: int = 0, count: int = 5) -> str:
    """청크 몇 개를 미리 보여준다 (경계가 문장 중간에서 잘리지 않았는지 검수용)."""
    chunks = store.load_chunks(doc_id)[start : start + count]
    return "\n\n".join(
        f"[{c['id']} p.{c['page']} §{c['section'] or '-'}]\n{c['text']}" for c in chunks
    ) or "청크가 없습니다."


# ─────────────────────────── 4. 검색 ───────────────────────────


def _tokenize(text: str) -> list[str]:
    """영문/숫자 단어 + 한글 단어 + 한글 2-gram (조사가 붙어도 매칭되도록)."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    for word in re.findall(r"[가-힣]+", text):
        tokens.append(word)
        tokens += [word[i : i + 2] for i in range(len(word) - 1)]
    return tokens


@lru_cache(maxsize=1)
def _bm25_index():
    chunks = [c for d in store.list_doc_ids() for c in store.load_chunks(d)]
    if not chunks:
        return [], None
    return chunks, BM25L([_tokenize(f"{c['section']} {c['text']}") for c in chunks])


@tool
def search_chunks(query: str, k: int = 5, doc_id: str = "") -> str:
    """BM25 키워드 검색으로 관련 청크 상위 k개를 반환한다.

    doc_id: 문서 ID 또는 표시(D3). 여러 문서는 쉼표로 ('D1,D3'). 여러 문서면 결과 머리말에 표시가 붙는다.
    정확도를 위해 하나의 질문을 동의어/세부 키워드로 바꿔 여러 번 검색하는 것을 권장한다.
    """
    refs = [r.strip() for r in (doc_id or "").split(",") if r.strip()]
    allowed = {store.resolve(r) for r in refs}
    show_label = len(refs) > 1 or any(re.fullmatch(r"D\d+", r) for r in refs)
    names = store.labels() if show_label else {}
    chunks, bm25 = _bm25_index()
    if bm25 is None:
        return "인덱스가 비어 있습니다. 먼저 chunk_document 를 실행하세요."
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked:
        c = chunks[i]
        if allowed and c["doc_id"] not in allowed:
            continue
        if scores[i] <= 0:
            break
        label = f"{names.get(c['doc_id'], '?')} " if show_label else ""
        out.append(f"[{c['id']} {label}p.{c['page']} §{c['section'] or '-'} score={scores[i]:.2f}]\n{c['text']}")
        if len(out) >= k:
            break
    return "\n\n".join(out) or "일치하는 청크가 없습니다."


@tool
def get_chunk_context(chunk_id: str, window: int = 1) -> str:
    """청크 앞뒤 window 개 청크를 함께 반환한다 (답이 청크 경계에 걸쳐 있을 때 사용)."""
    doc_id, idx = chunk_id.rsplit("-", 1)
    chunks = store.load_chunks(doc_id)
    i = int(idx)
    sel = chunks[max(0, i - window) : i + window + 1]
    return "\n\n".join(f"[{c['id']} p.{c['page']}]\n{c['text']}" for c in sel) or "없음"


@tool
def document_outline(doc_id: str) -> str:
    """문서 전체 구성(페이지별 소제목과 첫 부분)을 보여준다. 요약/보고서처럼 문서 전체를 다룰 때 먼저 호출한다. doc_id 에 표시(D3)도 된다."""
    doc_id = store.resolve(doc_id)
    try:
        meta = store.load_meta(doc_id)
    except ValueError:
        return f"알 수 없는 문서: {doc_id}"
    by_page: dict[int, list[dict]] = {}
    for c in store.load_chunks(doc_id):
        by_page.setdefault(c["page"], []).append(c)
    unit = "구간" if meta.get("unit") == "section" else "페이지"
    lines = [f"{Path(meta['path']).name} · {meta['page_count']}{unit} · 청크 {sum(map(len, by_page.values()))}개"]
    for p in range(1, meta["page_count"] + 1):
        chunks = by_page.get(p, [])
        sections = list(dict.fromkeys(c["section"] for c in chunks if c["section"]))
        head = re.sub(r"\s+", " ", chunks[0]["text"])[:120] if chunks else "(내용 없음)"
        lines.append(f"[p.{p}] " + (f"§{' / '.join(sections[:4])} — " if sections else "") + head)
    return "\n".join(lines)


@tool
def list_documents() -> str:
    """인덱싱된 문서 목록과 상태를 보여준다."""
    rows = []
    for d in store.list_doc_ids():
        meta = store.load_meta(d)
        rows.append(
            f"{d}: {meta['path']} | pages={meta['page_count']} "
            f"| extracted={len(store.load_pages(d))} | chunks={len(store.load_chunks(d))}"
        )
    return "\n".join(rows) or "등록된 문서가 없습니다."
