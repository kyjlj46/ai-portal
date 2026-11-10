import pymupdf
import pytest

from rag import store, tools

FONT = r"C:\Windows\Fonts\malgun.ttf"

PAGE1 = (
    "1. 개요\n이 문서는 2026년 연간 보고서입니다. 총 매출은 1,234억 원입니다.\n\n"
    "2. 인력\n직원 수는 512명이며 연구 인력은 87명입니다."
)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    """테스트마다 별도 data 폴더를 써서 실제 data/ 를 건드리지 않는다."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "data")
    from rag import stream

    monkeypatch.setattr(stream, "RETRY_DELAY", 0)  # 재시도 대기 없이
    tools._bm25_index.cache_clear()
    yield
    tools._bm25_index.cache_clear()


@pytest.fixture
def sample_pdf(tmp_path):
    """1쪽: 한글 텍스트 레이어, 2쪽: 이미지만 있는 스캔 가정 페이지."""
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_font(fontname="kr", fontfile=FONT)
    p1.insert_text((50, 80), PAGE1, fontname="kr", fontsize=11)
    p2 = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    p2.insert_image(pymupdf.Rect(50, 50, 150, 150), pixmap=pix)
    doc.save(path)
    return path


@pytest.fixture
def indexed_doc(sample_pdf):
    """1쪽을 추출하고 청킹까지 마친 doc_id."""
    out = tools.inspect_pdf.invoke({"pdf_path": str(sample_pdf)})
    doc_id = out.split()[0].split("=")[1]
    tools.extract_text_layer.invoke({"doc_id": doc_id, "pages": "1"})
    tools.chunk_document.invoke({"doc_id": doc_id, "chunk_size": 60, "chunk_overlap": 10})
    return doc_id
