"""PDF 페이지 OCR: 로컬 OCR 엔진(RapidOCR) + LLM 레이아웃 정리.

1. 페이지를 이미지로 렌더링
2. RapidOCR(한국어 PP-OCRv5)로 글자 상자(좌표 + 텍스트) 인식
3. 좌표로 행을 묶어 '레이아웃 텍스트'를 만든다 (같은 행의 칸은 ' | ' 로 구분)
4. 텍스트 전용 LLM 이 레이아웃 텍스트를 보고 띄어쓰기/읽기 순서/표 구조를 마크다운으로 정리

LLM 정리 결과가 원문보다 지나치게 짧으면(내용 누락 의심) 레이아웃 텍스트를 그대로 쓴다.
"""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache

import pymupdf as fitz

from .llm import make_model

_engine_lock = threading.Lock()


@lru_cache(maxsize=1)
def _engine():
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(
        params={
            "Global.log_level": "warning",
            "Rec.lang_type": LangRec.KOREAN,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            # 연한 글씨/작은 글씨까지 잡도록 탐지 문턱을 낮춤
            "Det.thresh": 0.2,
            "Det.box_thresh": 0.3,
        }
    )


def render_page(pdf_path: str, page: int, dpi: int = 200) -> bytes:
    with fitz.open(pdf_path) as pdf:
        return pdf[page - 1].get_pixmap(dpi=dpi).tobytes("png")


def text_layer(pdf_path: str, page: int) -> str:
    """PDF 에 들어 있는 텍스트 레이어 (없으면 빈 문자열). OCR 누락 보완용 참고 자료."""
    with fitz.open(pdf_path) as pdf:
        return pdf[page - 1].get_text("text", sort=True).strip()


def detect_boxes(png: bytes) -> list[dict]:
    """이미지에서 글자 상자 목록을 돌려준다: {x0, y0, x1, y1, text, score}."""
    with _engine_lock:  # onnx 세션을 여러 스레드가 동시에 초기화하지 않도록
        result = _engine()(png)
    if result.boxes is None:
        return []
    boxes = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        if text.strip():
            boxes.append(
                {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys), "text": text.strip(), "score": float(score)}
            )
    return boxes


def layout_text(boxes: list[dict]) -> str:
    """상자를 행 단위로 묶어 위→아래, 왼→오른 순서의 텍스트로 만든다.

    같은 행에서 간격이 넓은 칸은 ' | ' 로 구분해 표/다단 구조를 LLM 이 알아볼 수 있게 한다.
    """
    if not boxes:
        return ""
    boxes = sorted(boxes, key=lambda b: (b["y0"] + b["y1"]) / 2)
    rows: list[list[dict]] = []
    for b in boxes:
        cy = (b["y0"] + b["y1"]) / 2
        h = b["y1"] - b["y0"]
        if rows:
            last = rows[-1]
            ly = sum((x["y0"] + x["y1"]) / 2 for x in last) / len(last)
            lh = sum(x["y1"] - x["y0"] for x in last) / len(last)
            if abs(cy - ly) < 0.5 * min(h, lh):
                last.append(b)
                continue
        rows.append([b])

    lines = []
    for row in rows:
        row.sort(key=lambda b: b["x0"])
        height = sum(b["y1"] - b["y0"] for b in row) / len(row)
        parts = [row[0]["text"]]
        for prev, cur in zip(row, row[1:]):
            gap = cur["x0"] - prev["x1"]
            parts.append(" | " if gap > 1.5 * height else " ")
            parts.append(cur["text"])
        lines.append("".join(parts))
    return "\n".join(lines)


STRUCTURE_PROMPT = """아래는 PDF 한 페이지(보고서/발표 장표)를 OCR 한 결과입니다.
각 줄은 화면의 한 행이고, 같은 행에서 멀리 떨어진 칸은 ' | ' 로 구분되어 있습니다.
OCR 특성상 일부 글자가 틀리거나, 색 배경 위 글씨 등 일부 항목이 누락될 수 있습니다.
{hint_intro}
이 내용을 읽기 좋은 마크다운으로 정리하세요.
규칙:
- 원문의 모든 정보(숫자, 날짜, 고유명사, 항목)를 빠짐없이 유지하세요. 요약/생략 금지.
- 페이지에 없는 내용을 지어내지 마세요.
- 붙어버린 한국어 단어는 자연스럽게 띄어 쓰세요.
- 명백한 OCR 오타만 고치고, 확신이 없으면 원문 그대로 두세요.
- 페이지 제목은 '#', 소제목은 '##' 로 표시하세요.
- 표처럼 행/열이 맞는 부분은 마크다운 표로, 나열 항목은 '-' 목록으로 만드세요.
- 도식(박스/화살표)은 '- 항목: 설명' 형태로 관계가 드러나게 적으세요.
- 정리한 마크다운만 출력하세요.

[OCR 결과]
{raw}
{hint}"""

HINT_INTRO = """[PDF 텍스트 레이어]도 함께 주어집니다. 이것은 띄어쓰기와 읽기 순서가 깨져 있지만 글자 자체는 정확합니다.
구조와 순서는 OCR 결과를 따르고, OCR 에 없는 항목은 텍스트 레이어에서 찾아 알맞은 위치에 넣으세요.
글자가 다르면 텍스트 레이어의 글자를 믿으세요.
"""


@lru_cache(maxsize=1)
def _structure_model():
    return make_model(os.getenv("OCR_MODEL") or os.getenv("RAG_MODEL") or None)


def _chars(text: str) -> int:
    return len(re.sub(r"[\s|#*\-:]", "", text))


def structure_text(raw: str, layer: str = "") -> str:
    """LLM 으로 레이아웃 텍스트를 마크다운으로 정리. 실패/누락 의심 시 원문 반환.

    layer: PDF 텍스트 레이어(있으면 OCR 누락 보완에 사용).
    """
    if not raw.strip() and not layer.strip():
        return raw
    prompt = STRUCTURE_PROMPT.format(
        raw=raw or "(인식된 글자 없음)",
        hint_intro=HINT_INTRO if layer else "",
        hint=f"\n[PDF 텍스트 레이어]\n{layer}" if layer else "",
    )
    try:
        text = _structure_model().invoke(prompt).text.strip()
    except Exception:
        return raw
    text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text).strip()
    # LLM 이 쓰는 특수 하이픈/공백을 일반 문자로 (검색어 '3-20' 등이 매칭되도록)
    text = re.sub(r"[‐‑‒–−]", "-", text).replace(" ", " ").replace(" ", " ")
    if _chars(text) < 0.7 * _chars(raw):
        return raw
    return text


def ocr_page(pdf_path: str, page: int, dpi: int = 200, use_llm: bool = True) -> dict:
    """페이지 하나를 OCR 해서 {text, raw, quality} 를 돌려준다. quality 는 OCR 평균 신뢰도."""
    boxes = detect_boxes(render_page(pdf_path, page, dpi))
    raw = layout_text(boxes)
    quality = round(sum(b["score"] for b in boxes) / len(boxes), 3) if boxes else 0.0
    text = structure_text(raw, text_layer(pdf_path, page)) if use_llm else raw
    return {"text": text, "raw": raw, "quality": quality, "boxes": len(boxes)}
