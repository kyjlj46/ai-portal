"""문서 수집 파이프라인: (필요하면 PDF 변환) → 모든 페이지 OCR → 저장 → 청킹/인덱싱.

에이전트에게 맡기지 않고 코드로 고정해서 실행한다 (모델이 단계를 건너뛰거나 엉뚱한 도구를 부르는 일 방지).

지원 형식
- PDF            : 그대로 페이지 OCR
- Word / 한글     : Word·한컴오피스로 PDF 변환 후 페이지 OCR. 변환이 안 되면 텍스트를 직접 추출
- 텍스트(.txt 등) : 변환/OCR 없이 읽어서 구간으로 나눔
- 이미지(.png 등)  : PDF 로 바꿔 페이지 OCR (스캔본·사진)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pymupdf as fitz

from . import store
from .convert import extract_segments, image_to_pdf, kind_of, to_pdf
from .ocr import ocr_page
from .tools import chunk_document

# 이 값이 바뀌면 기존 인덱스를 다시 만든다
PIPELINE_VERSION = "ocr-llm-v1"


def is_indexed(doc_id: str) -> bool:
    try:
        meta = store.load_meta(doc_id)
    except ValueError:
        return False
    return meta.get("pipeline") == PIPELINE_VERSION and bool(store.load_chunks(doc_id))


def ingest(
    path: str,
    force: bool = False,
    dpi: int = 200,
    progress: Callable[[str], None] | None = None,
) -> str:
    """문서를 인덱싱하고 doc_id 를 돌려준다. 이미 같은 방식으로 인덱싱됐으면 건너뛴다."""
    log = progress or (lambda msg: None)
    path = str(Path(path).resolve())
    kind = kind_of(path)
    doc_id = store.doc_id_for(path)
    if not force and is_indexed(doc_id):
        log(f"{doc_id}: 이미 인덱싱됨")
        return doc_id

    store.clear_pages(doc_id)
    pdf = path if kind == "pdf" else None
    if kind == "image":
        log(f"{Path(path).name}: 이미지를 PDF 로 변환 중…")
        pdf = image_to_pdf(path, store.doc_dir(doc_id) / "converted.pdf")
    elif kind in ("word", "ppt", "hwp"):
        log(f"{Path(path).name}: PDF 로 변환 중…")
        pdf = to_pdf(path, store.doc_dir(doc_id) / "converted.pdf")
        log("PDF 변환 완료" if pdf else "PDF 변환 불가 → 텍스트 직접 추출")

    if pdf:
        pages = _ocr_all(doc_id, pdf, dpi, log)
        unit = "page"
    else:
        pages = _text_segments(doc_id, path, log)
        unit = "section"

    store.save_meta(
        doc_id,
        {
            "doc_id": doc_id,
            "path": path,
            "kind": kind,
            "pdf": pdf,  # 페이지 이미지/재OCR 에 쓰는 PDF (없으면 텍스트 구간 문서)
            "unit": unit,  # page: 실제 페이지, section: 텍스트 구간(가짜 페이지)
            "page_count": len(pages),
            "pages": pages,
            "pipeline": PIPELINE_VERSION,
        },
    )
    log(chunk_document.invoke({"doc_id": doc_id}).strip())
    return doc_id


def _ocr_all(doc_id: str, pdf: str, dpi: int, log) -> list[dict]:
    with fitz.open(pdf) as doc:
        page_count = len(doc)

    def work(page: int) -> dict:
        r = ocr_page(pdf, page, dpi=dpi)
        store.save_page(doc_id, page, r["text"], source="ocr", quality=r["quality"])
        store.save_raw(doc_id, page, r["raw"])
        log(f"p.{page}/{page_count} OCR 완료 (글자상자 {r['boxes']}개, 신뢰도 {r['quality']})")
        return {"page": page, "chars": len(r["text"]), "boxes": r["boxes"], "quality": r["quality"], "needs_ocr": True}

    # OCR 엔진은 내부에서 직렬화되고, LLM 정리 호출이 병렬로 돈다
    workers = int(os.getenv("OCR_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sorted(ex.map(work, range(1, page_count + 1)), key=lambda p: p["page"])


def _text_segments(doc_id: str, path: str, log) -> list[dict]:
    segments = extract_segments(path)
    pages = []
    for i, text in enumerate(segments, start=1):
        store.save_page(doc_id, i, text, source="text", quality=1.0)
        pages.append({"page": i, "chars": len(text), "boxes": 0, "quality": 1.0, "needs_ocr": False})
    log(f"텍스트 추출 완료: {len(segments)}개 구간")
    return pages
