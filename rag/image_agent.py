"""이미지 분석 에이전트.

1. 업로드한 이미지를 RapidOCR 로 읽어 글자 상자를 행/칸 배치대로 정리한다 (결과는 이미지별로 캐시).
2. 크기·방향·주요 색 같은 기본 정보를 함께 계산한다.
3. LLM 이 [이미지 정보] + [OCR 결과] 로 요청한 관점(전체 분석/글자 추출/표·차트/문서·화면 검토)에 맞춰 분석한다.
   VISION_MODEL 이 설정돼 있으면 이미지 자체도 함께 보낸다 (텍스트 전용 모델이면 글자·배치 중심 분석).
이어지는 질문은 같은 이미지 맥락(history)으로 답한다.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
from collections.abc import Iterator
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import store

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MAX_OCR_SIDE = 2400  # OCR 속도를 위해 긴 변을 이 크기로 줄인다
MAX_VISION_SIDE = 1600
MAX_TURNS = 6

IMAGE_PROMPT = """당신은 이미지 분석 담당입니다. 사용자가 올린 이미지(스크린샷, 사진, 차트, 표, 문서 촬영본 등)를 분석합니다.

입력으로 [이미지 정보](크기, 방향, 주요 색)와 [OCR 결과]가 주어집니다.
OCR 결과는 이미지 속 글자를 위→아래 순서로 한 줄에 한 행씩 적은 것이고, 같은 행에서 멀리 떨어진 칸은 ' | ' 로 구분됩니다.
{vision_note}

분석 관점:
- 전체 분석: 이미지가 무엇인지(문서/화면/차트/표/사진 등), 핵심 내용, 눈에 띄는 수치·항목, 시사점을 정리합니다.
- 글자 추출: 이미지 속 글자를 빠짐없이 원래 구조(제목, 목록, 표)대로 마크다운으로 옮깁니다. 요약하지 않습니다.
- 표·차트 해석: 표는 마크다운 표로 복원하고, 수치의 크기 비교·추세·합계·최댓값/최솟값 등 읽을 수 있는 사실을 정리합니다.
- 문서·화면 검토: 오탈자, 빠진 항목, 앞뒤가 맞지 않는 수치, 개선할 점을 찾아 목록으로 적습니다.

규칙:
- OCR 결과와 이미지 정보(그리고 볼 수 있다면 이미지)에 근거해서만 씁니다. 보이지 않는 내용을 지어내지 마세요.
- OCR 은 글자를 잘못 읽을 수 있습니다. 명백한 오타는 바로잡되, 확신이 없으면 원문을 두고 '(인식 불확실)' 이라고 표시합니다.
- 글자가 거의 없는 사진·그림이면, 그 사실과 확인할 수 있는 정보(크기, 색감 등)만 말하고, 그림 내용 해석에는 이미지를 볼 수 있는 모델이 필요하다고 안내합니다.
- 사용자의 언어로, 마크다운으로 답합니다. 앞뒤 인사말 없이 결과만 씁니다.
- 이어지는 질문에는 같은 이미지를 기준으로 답합니다."""

TEXT_ONLY_NOTE = "현재 모델은 이미지를 직접 볼 수 없습니다. 색·모양·그림 내용은 [이미지 정보]와 글자 배치로 추정할 수 있는 만큼만 말하세요."
VISION_NOTE = "이미지도 함께 주어집니다. 그림·도형·색·배치는 이미지를 보고, 글자는 OCR 결과와 이미지를 대조해 정확히 읽으세요."


# ─────────────────────────── 이미지 저장 ───────────────────────────


IMAGE_DIR = Path(__file__).resolve().parent.parent / "uploads" / "images"


def image_dir() -> Path:
    return IMAGE_DIR


def save_image(data: bytes, filename: str) -> dict:
    """이미지를 내용 해시로 저장한다. 같은 이미지는 한 번만 저장된다."""
    from PIL import Image

    ext = Path(filename).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"지원하지 않는 이미지 형식입니다. 지원: {', '.join(sorted(IMAGE_EXTS))}")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
    except Exception as e:  # noqa: BLE001
        raise ValueError("이미지 파일을 읽을 수 없습니다.") from e
    image_id = hashlib.sha1(data).hexdigest()[:12]
    folder = image_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{image_id}.*"):  # 같은 내용이 다른 확장자로 이미 있으면 그대로 쓴다
        ext = old.suffix
    path = folder / f"{image_id}{ext}"
    if not path.exists():
        path.write_bytes(data)
    return {"image_id": image_id, "filename": Path(filename).name, "width": width, "height": height, "url": f"/api/images/{image_id}"}


def image_path(image_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", image_id or ""):
        raise ValueError("잘못된 이미지 ID")
    found = sorted(image_dir().glob(f"{image_id}.*"))
    if not found:
        raise FileNotFoundError("이미지를 찾을 수 없습니다. 다시 올려 주세요.")
    return found[0]


# ─────────────────────────── 읽기 (OCR + 기본 정보) ───────────────────────────


def _png(path: Path, max_side: int, upscale_small: bool = False) -> tuple[bytes, int, int]:
    from PIL import Image

    with Image.open(path) as im:
        im.seek(0)  # GIF 등은 첫 장면
        im = im.convert("RGB")
        width, height = im.size
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        elif upscale_small and max(im.size) <= 1200:  # 작은 글씨는 2배로 키우면 OCR 이 짧은 단어를 덜 놓친다
            im = im.resize((width * 2, height * 2), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
    return buf.getvalue(), width, height


_COLOR_NAMES = {
    "흰색": (245, 245, 245), "검은색": (20, 20, 20), "회색": (128, 128, 128), "빨간색": (220, 40, 40),
    "주황색": (240, 140, 30), "노란색": (240, 220, 40), "초록색": (50, 160, 70), "파란색": (40, 90, 220),
    "남색": (30, 40, 110), "보라색": (130, 60, 180), "분홍색": (240, 140, 180), "갈색": (130, 80, 40),
    "하늘색": (120, 190, 240), "베이지": (225, 210, 180),
}


def _color_name(rgb: tuple[int, int, int]) -> str:
    return min(_COLOR_NAMES, key=lambda n: sum((a - b) ** 2 for a, b in zip(rgb, _COLOR_NAMES[n])))


def describe(path: Path) -> dict:
    """크기·방향·밝기·주요 색 (텍스트 전용 모델에게 줄 최소한의 시각 정보)."""
    from PIL import Image, ImageStat

    with Image.open(path) as im:
        im.seek(0)
        rgb = im.convert("RGB")
        width, height = rgb.size
        small = rgb.copy()
        small.thumbnail((200, 200))
        brightness = round(sum(ImageStat.Stat(small.convert("L")).mean) / 255, 2)
        quant = small.quantize(colors=5)
        palette = quant.getpalette() or []
        counts = sorted(quant.getcolors() or [], reverse=True)
    total = sum(n for n, _ in counts) or 1
    colors: list[dict] = []
    for n, idx in counts:
        r, g, b = palette[idx * 3 : idx * 3 + 3]
        name = _color_name((r, g, b))
        same = next((c for c in colors if c["name"] == name), None)
        if same:  # 비슷한 색(같은 이름)은 합친다
            same["ratio"] = round(same["ratio"] + n / total, 2)
        else:
            colors.append({"name": name, "hex": f"#{r:02x}{g:02x}{b:02x}", "ratio": round(n / total, 2)})
    colors = colors[:4]
    ratio = width / height
    orient = "가로형" if ratio > 1.15 else "세로형" if ratio < 0.87 else "정사각형에 가까움"
    return {"width": width, "height": height, "orientation": orient, "brightness": brightness, "colors": colors}


def read_image(image_id: str) -> dict:
    """OCR + 기본 정보. data/images/<id>.json 에 캐시한다 (이어지는 질문에서 다시 읽지 않게)."""
    import json

    from .ocr import detect_boxes, layout_text

    cache = store.DATA_DIR / "images" / f"{image_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    path = image_path(image_id)
    png, width, height = _png(path, MAX_OCR_SIDE, upscale_small=True)
    boxes = detect_boxes(png)
    info = describe(path)
    result = {
        **info,
        "ocr": layout_text(boxes),
        "boxes": len(boxes),
        "quality": round(sum(b["score"] for b in boxes) / len(boxes), 3) if boxes else 0.0,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


# ─────────────────────────── 분석 ───────────────────────────


def vision_enabled() -> bool:
    return bool(os.getenv("VISION_MODEL"))


def _model():
    from .llm import make_model, make_vision_model

    return make_vision_model() if vision_enabled() else make_model(os.getenv("RAG_MODEL") or None)


def _context(info: dict, filename: str) -> str:
    colors = ", ".join(f"{c['name']}({c['hex']}) {int(c['ratio'] * 100)}%" for c in info["colors"])
    return (
        f"[이미지 정보]\n- 파일: {filename}\n- 크기: {info['width']}×{info['height']} ({info['orientation']})\n"
        f"- 밝기: {info['brightness']} (0 어두움 ~ 1 밝음)\n- 주요 색: {colors}\n"
        f"- 인식된 글자 상자: {info['boxes']}개 (평균 신뢰도 {info['quality']})\n\n"
        f"[OCR 결과]\n{info['ocr'] or '(인식된 글자 없음)'}"
    )


def _step(kind: str, text: str) -> dict:
    return {"type": "step", "kind": kind, "text": text, "sub": False}


def image_stream(image_id: str, question: str, history: list | None = None, filename: str = "") -> Iterator[dict]:
    """ask_stream 과 같은 형식의 이벤트를 낸다 (웹 화면 재사용)."""
    try:
        import time

        path = image_path(image_id)
        first = not history
        if first:
            yield _step("ingest", f"이미지 불러옴: {filename or path.name}")
            yield _step("read", "글자 인식(OCR)과 색·크기 분석 중…")
        t0 = time.time()
        info = read_image(image_id)
        t_read = round(time.time() - t0, 1)
        if first:
            yield _step("result", f"→ {info['width']}×{info['height']}, 글자 상자 {info['boxes']}개 인식")
        yield _step("answer", "이미지를 보며 분석 작성 중…" if vision_enabled() else "인식한 글자와 배치로 분석 작성 중…")

        system = SystemMessage(IMAGE_PROMPT.format(vision_note=VISION_NOTE if vision_enabled() else TEXT_ONLY_NOTE))
        if first:
            text = _context(info, filename or path.name) + f"\n\n[요청]\n{question}"
            if vision_enabled():
                png, _, _ = _png(path, MAX_VISION_SIDE)
                url = "data:image/png;base64," + base64.b64encode(png).decode()
                human = HumanMessage(content=[{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}])
            else:
                human = HumanMessage(text)
        else:
            human = HumanMessage(question)

        t1 = time.time()
        reply = _model().invoke([system, *(history or []), human])
        t_llm = round(time.time() - t1, 1)
        answer = reply.text.strip() or "분석 결과를 만들지 못했습니다. 질문을 바꿔 다시 시도해 주세요."
        if history is not None:
            if first:  # 첫 턴은 이미지 맥락(OCR 등)이 담긴 요청을 기록에 남겨 이어지는 질문의 근거로 쓴다
                history.extend([HumanMessage(human.content if isinstance(human.content, str) else text), AIMessage(answer)])
            else:
                history.extend([human, AIMessage(answer)])
            if len(history) > 2 * MAX_TURNS:  # 맨 앞(이미지 맥락)은 남기고 중간 턴을 버린다
                del history[2 : len(history) - 2 * (MAX_TURNS - 1)]
        flow = []  # 처리 흐름 (답변 끝 표시, rag.stream.Flow 와 같은 모양)
        if first:
            flow.append({"parallel": False, "items": [{"icon": "🔤", "name": "글자 인식 (OCR)", "status": "ok",
                                                         "note": f"글자 상자 {info['boxes']}개", "sec": t_read}]})
        flow.append({"parallel": False, "items": [{"icon": "🖼️", "name": "이미지 분석", "status": "ok",
                                                     "note": "이미지 + 글자" if vision_enabled() else "인식한 글자 기준", "sec": t_llm}]})
        yield {"type": "final", "answer": answer, "sources": [], "steps": [], "doc_id": None, "agent": "image",
               "ocr_text": info["ocr"], "flow": flow}
    except Exception as e:  # noqa: BLE001 - 화면에 오류를 보여주기 위함
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
