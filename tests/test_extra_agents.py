"""마켓에 추가한 업로드 즉시 요약 · 청킹 · OCR · 웹 검색 Agent 와 그 도구."""

import io

import pytest

from portal import catalog, convert, extra_tools, ingest, market, store


def test_image_documents_are_supported(tmp_path):
    from PIL import Image

    assert convert.kind_of("scan.PNG") == "image" and ".jpg" in convert.SUPPORTED_EXTS
    path = tmp_path / "scan.tiff"
    frames = [Image.new("RGB", (200, 100), c) for c in ("white", "gray")]
    frames[0].save(path, save_all=True, append_images=frames[1:])  # 두 장짜리 TIFF
    pdf = convert.image_to_pdf(path, tmp_path / "out.pdf")
    import pymupdf

    with pymupdf.open(pdf) as doc:
        assert len(doc) == 2


def test_ingest_image_goes_through_ocr(tmp_path, monkeypatch):
    from PIL import Image

    path = tmp_path / "영수증.png"
    Image.new("RGB", (300, 200), "white").save(path)
    monkeypatch.setattr(ingest, "ocr_page", lambda pdf, page, dpi=200: {"text": "합계 12,000원", "raw": "합계 12,000원", "quality": 0.97, "boxes": 2})
    doc_id = ingest.ingest(str(path))
    meta = store.load_meta(doc_id)
    assert meta["kind"] == "image" and meta["unit"] == "page" and meta["page_count"] == 1 and meta["pdf"]
    assert "12,000원" in store.load_chunks(doc_id)[0]["text"]


def test_chunk_stats_and_settings(indexed_doc):
    out = extra_tools.chunk_stats.invoke({"doc_id": indexed_doc})
    assert "현재 설정: chunk_size=60, chunk_overlap=10" in out  # conftest 가 60/10 으로 청킹
    assert "청크 수:" in out and "페이지별 청크 수: p.1=" in out and "청크가 없는 페이지: [2]" in out
    from portal.tools import chunk_document

    chunk_document.invoke({"doc_id": indexed_doc, "chunk_size": 200, "chunk_overlap": 20})
    assert "chunk_size=200, chunk_overlap=20" in extra_tools.chunk_stats.invoke({"doc_id": indexed_doc})
    assert "알 수 없는 문서" in extra_tools.chunk_stats.invoke({"doc_id": "nope"})


def test_ocr_report_and_reocr(indexed_doc, monkeypatch):
    store.save_page(indexed_doc, 2, "흐림", source="ocr", quality=0.41)
    report = extra_tools.ocr_report.invoke({"doc_id": indexed_doc})
    assert "확인이 필요한 페이지: [2]" in report and "p.2 | ocr | 신뢰도 0.41" in report
    assert "p.1 | text" in report  # 텍스트 레이어로 뽑은 페이지

    import portal.ocr

    monkeypatch.setattr(portal.ocr, "ocr_page", lambda pdf, page, dpi=200: {"text": "선명하게 다시 읽은 2쪽 내용입니다. 표와 본문이 모두 인식되었습니다.", "raw": "x", "quality": 0.95, "boxes": 9})
    out = extra_tools.reocr_pages.invoke({"doc_id": indexed_doc, "pages": "2"})
    assert "p.2: 신뢰도 0.41 → 0.95" in out and "재인덱싱:" in out
    assert any("다시 읽은 2쪽" in c["text"] for c in store.load_chunks(indexed_doc))  # 인덱스에도 반영
    assert "확인이 필요한 페이지: 없음" in extra_tools.ocr_report.invoke({"doc_id": indexed_doc})
    assert "페이지 범위" in extra_tools.reocr_pages.invoke({"doc_id": indexed_doc, "pages": "99"})


def test_web_search_formats_results(monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results, backend, region):
            assert backend == "bing,google,yahoo" and max_results == 3
            return [{"title": "vLLM 문서", "href": "https://docs.vllm.ai", "body": "배치 토큰 설명"}]

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    monkeypatch.delenv("WEB_SEARCH_BACKEND", raising=False)
    out = extra_tools.web_search.invoke({"query": "vllm", "max_results": 3})
    assert out == "[1] vLLM 문서\nhttps://docs.vllm.ai\n배치 토큰 설명"


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/api/status", "http://localhost/", "http://10.0.0.5/", "http://169.254.169.254/latest", "file:///etc/passwd"])
def test_fetch_url_blocks_internal_addresses(url):
    out = extra_tools.fetch_url.invoke({"url": url})
    assert "읽을 수 없습니다" in out or "http/https 주소만" in out


def test_fetch_url_extracts_text(monkeypatch):
    html = ("<html><head><title>공지</title><script>var x=1</script></head><body><nav>메뉴</nav>"
            "<h1>요금 안내</h1><p>기본 요금은 1,000원입니다.</p><footer>저작권</footer></body></html>").encode()

    class Res:
        status_code, is_redirect, encoding = 200, False, "utf-8"
        headers = {"content-type": "text/html; charset=utf-8"}

        def iter_bytes(self):
            yield html

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import httpx

    monkeypatch.setattr(extra_tools, "_public_host", lambda url: True)
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Res())
    out = extra_tools.fetch_url.invoke({"url": "https://example.com/notice"})
    assert out.startswith("[공지] https://example.com/notice")
    assert "요금 안내" in out and "1,000원" in out and "var x" not in out and "메뉴" not in out and "저작권" not in out


def test_market_specs_get_their_tools(monkeypatch):
    spec = catalog.get("m:chunker")
    assert spec.extra_tools == ["chunk_stats", "preview_chunks", "chunk_document"] and spec.preload is False
    assert catalog.get("m:quick_summary").preload is True
    assert catalog.get("m:web_search").doc == "none"
    sub = market.subagent("web_search", model=None, doc_tools=["DOC"])
    assert [t.name for t in sub["tools"]] == ["web_search", "fetch_url"]  # 문서 도구 없이 웹 도구만
    sub = market.subagent("ocr", model=None, doc_tools=["DOC"])
    assert sub["tools"][0] == "DOC" and [t.name for t in sub["tools"][1:]] == ["ocr_report", "reocr_pages"]

    from portal import agent

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    seen = {}
    monkeypatch.setattr(agent, "create_deep_agent", lambda **kw: seen.update(kw) or "graph")
    agent.build_agent("m:ocr")
    assert {"ocr_report", "reocr_pages", "search_chunks"} <= {t.name for t in seen["tools"]}


def test_upload_accepts_images(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from PIL import Image

    import web

    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, "PNG")
    res = TestClient(web.app).post("/api/upload", files={"file": ("스캔.png", buf.getvalue(), "image/png")})
    assert res.status_code == 200 and res.json()["filename"] == "스캔.png"


def test_tool_call_stub_is_not_accepted_as_answer(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from portal import agent

    assert agent.is_tool_stub("[read_page page=2]") and agent.is_tool_stub("read_page(doc_id='x', page=2)")
    assert not agent.is_tool_stub("## OCR 결과 요약\n- 6페이지") and not agent.is_tool_stub("[p.2] 내용")

    class Model:
        def invoke(self, msgs):
            return AIMessage("## OCR 결과 요약\n- 정상")

    monkeypatch.setattr(agent, "_model", lambda env: Model())
    msgs = [HumanMessage("보고해 주세요"), AIMessage("[read_page page=2]")]
    assert agent.final_answer(msgs) == "## OCR 결과 요약\n- 정상"  # 흉내만 낸 응답이면 다시 답을 받는다


def test_web_search_retries_without_quotes(monkeypatch):
    tried = []

    class FakeDDGS:
        def text(self, query, **kw):
            tried.append(query)
            if '"' in query:
                raise RuntimeError("No results found.")
            return [{"title": "t", "href": "https://x.org", "body": "b"}]

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    out = extra_tools.web_search.invoke({"query": '"max_num_batched_tokens" vLLM'})
    assert tried == ['"max_num_batched_tokens" vLLM', "max_num_batched_tokens vLLM"] and out.startswith("[1] t")


def test_web_sources_are_appended_from_pages_actually_read():
    from langchain_core.messages import ToolMessage

    from portal.stream import add_web_sources

    msgs = [ToolMessage("[vLLM 문서] https://docs.vllm.ai/perf\n본문", tool_call_id="1", name="fetch_url"),
            ToolMessage("[vLLM 문서] https://docs.vllm.ai/perf\n본문", tool_call_id="2", name="fetch_url"),  # 같은 페이지 두 번
            ToolMessage("페이지를 읽지 못했습니다 (HTTP 404).", tool_call_id="3", name="fetch_url"),
            ToolMessage("[1] 검색 결과\nhttps://x.org", tool_call_id="4", name="web_search")]
    out = add_web_sources("기본값은 512입니다 [1].", msgs)
    assert out.endswith("## 출처 (관련 기사·읽은 페이지)\n- [vLLM 문서](https://docs.vllm.ai/perf)\n- [검색 결과](https://x.org)")
    assert out.count("docs.vllm.ai") == 1  # 같은 주소는 한 번만
    # 답에 이미 있는 링크는 빼고, 없는 것만 더한다 (모델이 링크를 일부만 옮겨도 출처가 빠지지 않게)
    part = add_web_sources("답 [vLLM](https://docs.vllm.ai/perf)", msgs)
    assert part.endswith("## 출처 (추가)\n- [검색 결과](https://x.org)") and part.count("docs.vllm.ai") == 1
    assert add_web_sources("답", []) == "답"


def test_second_ocr_pass_only_fills_missed_lines():
    from portal.ocr import merge_missing

    base = [{"x0": 0, "y0": 100, "x1": 300, "y1": 140, "text": "결제일 2026-11-10", "score": 0.99}]
    extra = [
        {"x0": 2, "y0": 60, "x1": 290, "y1": 98, "text": "합계 금액 12,000원", "score": 0.99},  # 놓친 줄 → 보탬
        {"x0": 5, "y0": 103, "x1": 280, "y1": 138, "text": "결제일2026-11-1", "score": 0.95},  # 이미 있는 자리 → 버림
        {"x0": 400, "y0": 10, "x1": 410, "y1": 30, "text": "|", "score": 0.99},  # 잡음 → 버림
        {"x0": 0, "y0": 200, "x1": 100, "y1": 230, "text": "흐릿한글", "score": 0.5},  # 낮은 신뢰도 → 버림
    ]
    assert [b["text"] for b in merge_missing(base, extra)] == ["결제일 2026-11-10", "합계 금액 12,000원"]


def test_code_assistant_in_market_and_reads_source_files(tmp_path):
    item = market.BY_ID["code_assistant"]
    assert item.category == "개발" and item.uses_doc and not item.tools
    assert "```python" in item.prompt and "코드 리뷰" in item.prompt
    spec = catalog.get("m:code_assistant")
    assert spec.doc == "optional" and spec.preload  # 올린 소스 파일은 전문을 요청에 붙인다
    src = tmp_path / "app.py"
    src.write_bytes(b"def add(a, b):\n    return a + b\n")
    assert convert.kind_of(src) == "text" and ".java" in convert.SUPPORTED_EXTS
    assert convert.extract_segments(src) == ["def add(a, b):\n    return a + b"]
