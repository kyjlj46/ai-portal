"""RAG 파이프라인 단계별 도구.

각 서브에이전트는 자기 단계의 도구만 받는다:
  parser-agent    : inspect_pdf, extract_text_layer, read_page
  ocr-agent       : ocr_pages, read_page
  chunking-agent  : chunk_document, preview_chunks
  retrieval-agent : search_chunks, get_chunk_context
"""

from __future__ import annotations

import base64
import os
import re
from functools import lru_cache

import pymupdf as fitz
from .llm import make_model
from langchain_core.messages import HumanMessage
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
    """저장된 페이지 텍스트 전문을 읽는다 (추출/OCR 결과 검수용)."""
    data = store.load_page(doc_id, page)
    if not data:
        return f"p.{page} 는 아직 추출되지 않았습니다."
    return f"[p.{page} source={data['source']} quality={data['quality']}]\n{data['text']}"


# ─────────────────────────── 2. OCR (비전 모델) ───────────────────────────


@lru_cache(maxsize=1)
def _ocr_model():
    return make_model(os.getenv("OCR_MODEL") or os.getenv("RAG_MODEL") or None)


OCR_PROMPT = """이 이미지는 PDF 한 페이지입니다. 페이지의 모든 텍스트를 정확히 전사하세요.
규칙:
- 원문 그대로 옮기고 요약/번역/보정하지 마세요.
- 제목은 '#', 소제목은 '##' 마크다운 헤더로 표시하세요.
- 표는 마크다운 표로 옮기세요.
- 읽을 수 없는 부분은 [판독불가] 로 표시하세요.
- 전사한 텍스트만 출력하세요."""


@tool
def ocr_pages(doc_id: str, pages: str, dpi: int = 200) -> str:
    """지정한 페이지를 이미지로 렌더링해 비전 모델로 OCR 하고 결과를 저장한다 (pages 예: '2,4-6').

    스캔 페이지, 텍스트 레이어가 깨진 페이지, 표/도식이 많은 페이지에 사용한다.
    """
    meta = store.load_meta(doc_id)
    targets = _parse_pages(pages, meta["page_count"])
    report = []
    with fitz.open(meta["path"]) as pdf:
        for p in targets:
            png = pdf[p - 1].get_pixmap(dpi=dpi).tobytes("png")
            msg = HumanMessage(
                content=[
                    {"type": "image", "base64": base64.b64encode(png).decode(), "mime_type": "image/png"},
                    {"type": "text", "text": OCR_PROMPT},
                ]
            )
            text = _ocr_model().invoke([msg]).text.strip()
            q = _text_quality(text)
            store.save_page(doc_id, p, text, source="ocr", quality=q)
            unreadable = text.count("[판독불가]")
            report.append(
                f"p.{p}: OCR {len(text)}자, quality={q}, 판독불가={unreadable} | 미리보기: {text[:120]!r}"
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

    정확도를 위해 하나의 질문을 동의어/세부 키워드로 바꿔 여러 번 검색하는 것을 권장한다.
    """
    chunks, bm25 = _bm25_index()
    if bm25 is None:
        return "인덱스가 비어 있습니다. 먼저 chunk_document 를 실행하세요."
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked:
        c = chunks[i]
        if doc_id and c["doc_id"] != doc_id:
            continue
        if scores[i] <= 0:
            break
        out.append(f"[{c['id']} p.{c['page']} §{c['section'] or '-'} score={scores[i]:.2f}]\n{c['text']}")
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
