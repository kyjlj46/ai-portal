"""문서/페이지/청크를 디스크(JSON)에 저장하는 아주 단순한 저장소.

data/<doc_id>/
  meta.json      : 원본 경로, 페이지 수, 페이지별 진단 정보
  pages/<n>.json : 페이지 텍스트 + 출처(text|ocr) + 품질 점수
  raw/<n>.txt    : LLM 정리 전 OCR 원문
  chunks.json    : 청크 목록
"""

from __future__ import annotations

import hashlib
import re
import threading
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def doc_id_for(pdf_path: str | Path) -> str:
    h = hashlib.sha1(Path(pdf_path).read_bytes()).hexdigest()
    return h[:10]


def doc_dir(doc_id: str) -> Path:
    return DATA_DIR / doc_id


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_meta(doc_id: str, meta: dict) -> None:
    _write(doc_dir(doc_id) / "meta.json", meta)


def load_meta(doc_id: str) -> dict:
    meta = _read(doc_dir(doc_id) / "meta.json")
    if meta is None:
        raise ValueError(f"알 수 없는 doc_id: {doc_id}. 먼저 inspect_pdf 를 실행하세요.")
    return meta


def save_page(doc_id: str, page: int, text: str, source: str, quality: float) -> None:
    _write(
        doc_dir(doc_id) / "pages" / f"{page}.json",
        {"page": page, "text": text, "source": source, "quality": quality},
    )


def save_raw(doc_id: str, page: int, raw: str) -> None:
    """LLM 정리 전 OCR 원문 (디버깅/검수용)."""
    path = doc_dir(doc_id) / "raw" / f"{page}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")


def clear_pages(doc_id: str) -> None:
    """재수집 전에 이전 페이지/청크 결과를 지운다."""
    for sub in ("pages", "raw"):
        for f in (doc_dir(doc_id) / sub).glob("*"):
            f.unlink()
    (doc_dir(doc_id) / "chunks.json").unlink(missing_ok=True)


def load_page(doc_id: str, page: int) -> dict | None:
    return _read(doc_dir(doc_id) / "pages" / f"{page}.json")


def load_pages(doc_id: str) -> list[dict]:
    pages_dir = doc_dir(doc_id) / "pages"
    if not pages_dir.exists():
        return []
    pages = [_read(p) for p in pages_dir.glob("*.json")]
    return sorted(pages, key=lambda p: p["page"])


def save_chunks(doc_id: str, chunks: list[dict]) -> None:
    _write(doc_dir(doc_id) / "chunks.json", chunks)


def load_chunks(doc_id: str) -> list[dict]:
    return _read(doc_dir(doc_id) / "chunks.json", [])


def list_doc_ids() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if (p / "meta.json").exists())


# ─────────────────────────── 문서 표시(D1, D2 …) ───────────────────────────
# 여러 문서를 함께 쓸 때 출처를 (D1 p.3) 처럼 구분하려고 문서마다 고정 표시를 붙인다 (data/labels.json).

_label_lock = threading.Lock()


def _labels_path() -> Path:
    return DATA_DIR / "labels.json"


def labels() -> dict[str, str]:
    """doc_id → 표시."""
    path = _labels_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def label_for(doc_id: str) -> str:
    with _label_lock:
        table = labels()
        if doc_id not in table:
            used = [int(v[1:]) for v in table.values() if re.fullmatch(r"D\d+", v)]
            table[doc_id] = f"D{max(used, default=0) + 1}"
            _write(_labels_path(), table)
        return table[doc_id]


def resolve(ref: str) -> str:
    """표시(D3) 또는 doc_id → doc_id."""
    ref = (ref or "").strip()
    if re.fullmatch(r"D\d+", ref):
        for doc_id, label in labels().items():
            if label == ref:
                return doc_id
    return ref
