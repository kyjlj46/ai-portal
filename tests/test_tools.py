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


def test_build_agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    from rag.agent import build_agent

    assert build_agent() is not None


def test_openai_compatible_model_disables_streaming_by_default(monkeypatch):
    from rag.llm import make_model

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    monkeypatch.delenv("LLM_STREAMING", raising=False)
    assert make_model("some-model").disable_streaming is True  # vLLM 스트리밍 글자 누락 회피
    monkeypatch.setenv("LLM_STREAMING", "1")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    m = make_model("some-model")
    assert m.disable_streaming is False and m.temperature == 0.2
