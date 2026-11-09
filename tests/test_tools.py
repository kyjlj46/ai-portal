from rag import store, tools


def test_parse_pages():
    assert tools._parse_pages("1-3,5", 10) == [1, 2, 3, 5]
    assert tools._parse_pages("all", 3) == [1, 2, 3]
    assert tools._parse_pages("8-12", 10) == [8, 9, 10]


def test_text_quality():
    assert tools._text_quality("정상적인 한글 문장입니다.") == 1.0
    assert tools._text_quality("") == 0.0
    assert tools._text_quality("��� abc") < tools.MIN_QUALITY


def test_inspect_flags_scanned_page(sample_pdf):
    out = tools.inspect_pdf.invoke({"pdf_path": str(sample_pdf)})
    assert "텍스트 레이어 페이지: [1]" in out
    assert "OCR 필요 페이지: [2]" in out


def test_extract_text_layer(sample_pdf):
    doc_id = store.doc_id_for(sample_pdf)
    tools.inspect_pdf.invoke({"pdf_path": str(sample_pdf)})
    tools.extract_text_layer.invoke({"doc_id": doc_id, "pages": "1"})
    page = store.load_page(doc_id, 1)
    assert page["source"] == "text"
    assert "연구 인력은 87명" in page["text"]


def test_chunks_have_page_and_section(indexed_doc):
    chunks = store.load_chunks(indexed_doc)
    assert len(chunks) == 2
    assert all(c["page"] == 1 for c in chunks)
    assert chunks[1]["section"] == "2. 인력"


def test_chunk_reports_missing_pages(sample_pdf):
    doc_id = store.doc_id_for(sample_pdf)
    tools.inspect_pdf.invoke({"pdf_path": str(sample_pdf)})
    tools.extract_text_layer.invoke({"doc_id": doc_id, "pages": "1"})
    out = tools.chunk_document.invoke({"doc_id": doc_id})
    assert "추출되지 않은 페이지: [2]" in out


def test_search_handles_korean_spacing(indexed_doc):
    # '연구인력' (붙여쓰기) 로도 '연구 인력' 청크가 1등이어야 한다
    out = tools.search_chunks.invoke({"query": "연구인력 몇 명", "k": 1})
    assert f"{indexed_doc}-0001" in out
    assert "87명" in out


def test_search_no_match(indexed_doc):
    assert "일치하는 청크가 없습니다" in tools.search_chunks.invoke({"query": "zzzz"})


def test_chunk_context(indexed_doc):
    out = tools.get_chunk_context.invoke({"chunk_id": f"{indexed_doc}-0000"})
    assert "1,234억" in out and "512명" in out


def test_ocr_pages_uses_vision_model(sample_pdf, monkeypatch):
    """비전 모델을 가짜로 바꿔 OCR 결과 저장 흐름만 검증 (API 호출 없음)."""

    class FakeModel:
        def invoke(self, messages):
            blocks = messages[0].content
            assert blocks[0]["type"] == "image" and blocks[0]["base64"]
            return type("R", (), {"text": "# 스캔 페이지\n매출 목표는 2,000억 원입니다."})()

    monkeypatch.setattr(tools, "_ocr_model", lambda: FakeModel())
    doc_id = store.doc_id_for(sample_pdf)
    tools.inspect_pdf.invoke({"pdf_path": str(sample_pdf)})
    out = tools.ocr_pages.invoke({"doc_id": doc_id, "pages": "2"})
    assert "p.2: OCR" in out
    assert store.load_page(doc_id, 2)["source"] == "ocr"


def test_build_agent(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    from rag.agent import build_agent

    assert build_agent() is not None
