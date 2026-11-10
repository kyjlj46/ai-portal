import pytest
from fastapi.testclient import TestClient

import web


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")
    return TestClient(web.app)


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "업무 에이전트" in res.text


def test_upload_rejects_unsupported(client):
    res = client.post("/api/upload", files={"file": ("a.exe", b"MZ", "application/octet-stream")})
    assert res.status_code == 400


def test_upload_accepts_office_and_text(client):
    for name in ["메모.txt", "보고서.docx", "발표.pptx", "계획서.hwp", "계획서.hwpx"]:
        res = client.post("/api/upload", files={"file": (name, b"data", "application/octet-stream")})
        assert res.status_code == 200 and res.json()["filename"] == name


def test_upload_and_documents(client, sample_pdf, indexed_doc):
    res = client.post("/api/upload", files={"file": ("보고서.pdf", sample_pdf.read_bytes(), "application/pdf")})
    assert res.status_code == 200
    data = res.json()
    assert data["doc_id"] == indexed_doc and data["indexed"] is True

    docs = client.get("/api/documents").json()
    assert docs[0]["chunks"] == 2 and docs[0]["ocr_pages"] == [2]


def test_ask_returns_answer_and_steps(client, sample_pdf, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    monkeypatch.setattr(
        web, "run_agent",
        lambda path, q: {"answer": "87명입니다 (p.1)", "steps": [{"agent": "retrieval-agent", "task": q}]},
    )
    res = client.post("/api/ask", json={"pdf_path": str(sample_pdf), "question": "연구 인력?"})
    assert res.status_code == 200
    assert res.json()["answer"].startswith("87명")


def test_page_image(client, indexed_doc):
    res = client.get(f"/api/page-image/{indexed_doc}/1")
    assert res.status_code == 200 and res.headers["content-type"] == "image/png"
    assert res.content[:4] == b"\x89PNG"
    assert client.get(f"/api/page-image/{indexed_doc}/99").status_code == 404
    assert client.get("/api/page-image/nope/1").status_code == 404


def test_ask_stream_sends_sse_events(client, sample_pdf, monkeypatch):
    import json

    from rag import stream

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    fake = [
        {"type": "step", "kind": "search", "text": "검색: “인력”", "sub": False},
        {"type": "token", "text": "87명"},
        {"type": "final", "answer": "87명 (p.1)", "sources": [], "steps": [], "doc_id": "x"},
    ]
    seen = {}

    def fake_stream(path, q, history=None, agent_id="qa"):
        seen["history"], seen["agent"] = history, agent_id
        return iter(fake)

    monkeypatch.setattr(stream, "ask_stream", fake_stream)
    res = client.post("/api/ask/stream", json={"pdf_path": str(sample_pdf), "question": "인력?", "conversation_id": "c1"})
    assert seen["history"] is stream.conversation("qa:c1") and seen["agent"] == "qa"
    assert client.delete("/api/conversations/c1").json() == {"ok": True}
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events == fake


def test_ask_missing_file(client):
    res = client.post("/api/ask", json={"pdf_path": "없는파일.pdf", "question": "q"})
    assert res.status_code == 404


# ─────────────────────────── 업무 에이전트 (카드) ───────────────────────────


def test_agents_list_has_cards(client):
    cards = {a["id"]: a for a in client.get("/api/agents").json()}
    assert set(cards) == {"qa", "summary", "report", "approval", "news", "image", "omok"}
    assert cards["image"]["mode"] == "image" and cards["omok"]["mode"] == "game" and cards["omok"]["doc"] == "none"
    assert cards["qa"]["mode"] == "chat" and cards["approval"]["mode"] == "form"
    assert cards["summary"]["doc"] == "required" and cards["report"]["doc"] == "optional"
    assert "prompt" not in cards["report"]


def test_agent_stream_validates(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    assert client.post("/api/agents/nope/stream", json={"question": "q"}).status_code == 404
    res = client.post("/api/agents/summary/stream", json={"inputs": {"length": "핵심 요약"}})
    assert res.status_code == 400 and "문서" in res.json()["detail"]  # 요약은 문서 필수
    res = client.post("/api/agents/approval/stream", json={"inputs": {"title": "라이선스 구매"}})
    assert res.status_code == 400 and "목적" in res.json()["detail"]  # 필수 항목 누락


def test_agent_stream_composes_form_without_document(client, monkeypatch):
    import json

    from rag import stream

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    seen = {}

    def fake_stream(path, q, history=None, agent_id="qa"):
        seen.update(path=path, q=q, agent=agent_id, history=history)
        return iter([{"type": "final", "answer": "# 기안", "sources": [], "steps": [], "doc_id": None, "agent": agent_id}])

    monkeypatch.setattr(stream, "ask_stream", fake_stream)
    inputs = {"title": "라이선스 구매", "purpose": "성능 테스트 도구 도입", "budget": "3식 × 4,000,000원"}
    res = client.post("/api/agents/approval/stream", json={"inputs": inputs, "conversation_id": "c9"})
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1]["answer"] == "# 기안"
    assert seen["path"] is None and seen["agent"] == "approval"
    assert "- 기안 제목: 라이선스 구매" in seen["q"] and "- 소요 예산: 3식 × 4,000,000원" in seen["q"]
    assert seen["history"] is stream.conversation("approval:c9")


def test_export_docx(client):
    import io

    import docx

    md = "# 구매 기안\n\n| 항목 | 금액 |\n|---|---|\n| 라이선스 | 12,000,000원 |\n\n## 1. 목적\n- **성능** 테스트 도구 도입 (p.2)\n"
    res = client.post("/api/export/docx", json={"markdown": md, "title": "구매 기안"})
    assert res.status_code == 200 and "attachment" in res.headers["content-disposition"]
    d = docx.Document(io.BytesIO(res.content))
    texts = [p.text for p in d.paragraphs]
    assert "구매 기안" in texts and "1. 목적" in texts and "성능 테스트 도구 도입 (p.2)" in texts
    assert d.tables[0].cell(1, 1).text == "12,000,000원"


# ─────────────────────────── 이미지 분석 · 오목 ───────────────────────────


def _png_bytes():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, "PNG")
    return buf.getvalue()


def test_image_upload_serve_and_analyze(client, tmp_path, monkeypatch):
    import json

    from rag import image_agent

    monkeypatch.setattr(image_agent, "IMAGE_DIR", tmp_path / "images")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    assert client.post("/api/images", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 400
    assert client.post("/api/images", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 400
    meta = client.post("/api/images", files={"file": ("표.png", _png_bytes(), "image/png")}).json()
    assert meta["width"] == 40 and meta["height"] == 20 and len(meta["image_id"]) == 12
    assert client.get(meta["url"]).status_code == 200
    assert client.get("/api/images/zzz").status_code == 404

    seen = {}

    def fake_stream(image_id, question, history=None, filename=""):
        seen.update(image_id=image_id, question=question, filename=filename, history=history)
        yield {"type": "final", "answer": "## 분석", "sources": [], "steps": [], "doc_id": None, "agent": "image"}

    monkeypatch.setattr(image_agent, "image_stream", fake_stream)
    assert client.post("/api/agents/image/stream", json={"inputs": {"focus": "전체 분석"}}).status_code == 400  # 이미지 없음
    res = client.post("/api/agents/image/stream", json={
        "inputs": {"focus": "표·차트 해석", "question": "합계 확인"}, "image_id": meta["image_id"],
        "image_name": "표.png", "conversation_id": "c1"})
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1]["answer"] == "## 분석"
    assert seen["image_id"] == meta["image_id"] and seen["filename"] == "표.png"
    assert seen["question"].startswith("이미지 분석을 요청합니다.") and "- 분석 관점: 표·차트 해석" in seen["question"]
    from rag import stream

    assert seen["history"] is stream.conversation(f"image:c1:{meta['image_id']}")
    client.delete("/api/conversations/c1")  # 이미지별 하위 기록도 함께 지운다
    assert f"image:c1:{meta['image_id']}" not in stream._conversations


def test_image_stream_uses_ocr_context_and_keeps_history(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage

    from rag import image_agent

    monkeypatch.setattr(image_agent, "IMAGE_DIR", tmp_path / "images")
    meta = image_agent.save_image(_png_bytes(), "a.png")
    monkeypatch.setattr("rag.ocr.detect_boxes", lambda png: [
        {"x0": 0, "y0": 0, "x1": 50, "y1": 10, "text": "매출 합계", "score": 0.9},
        {"x0": 200, "y0": 0, "x1": 250, "y1": 10, "text": "4,050", "score": 0.95}])
    sent = []

    class FakeModel:
        def invoke(self, messages):
            sent.append(messages)
            return AIMessage(f"답 {len(sent)}")

    monkeypatch.setattr(image_agent, "_model", lambda: FakeModel())
    monkeypatch.delenv("VISION_MODEL", raising=False)
    history: list = []
    out = list(image_agent.image_stream(meta["image_id"], "이미지 분석을 요청합니다.", history, "a.png"))
    assert out[-1]["answer"] == "답 1" and "매출 합계 | 4,050" in out[-1]["ocr_text"]
    first = sent[0][-1].content
    assert "[OCR 결과]" in first and "매출 합계 | 4,050" in first and "40×20" in first
    out = list(image_agent.image_stream(meta["image_id"], "합계는?", history, "a.png"))
    assert out[-1]["answer"] == "답 2"
    assert "[OCR 결과]" in sent[1][1].content and sent[1][-1].content == "합계는?"  # 이어지는 질문도 이미지 맥락 유지
    assert not any(e.get("kind") == "read" for e in out)  # 두 번째는 OCR 다시 안 함


def test_omok_move_api(client):
    from rag import omok

    board = omok.empty_board()
    res = client.post("/api/omok/move", json={"board": board, "ai": 1, "level": "보통"}).json()
    assert res["move"] == [7, 7] and res["winner"] == 0  # 첫 수는 가운데
    assert client.post("/api/omok/move", json={"board": board, "ai": 2}).status_code == 400  # AI 차례 아님
    assert client.post("/api/omok/move", json={"board": [[0] * 3], "ai": 1}).status_code == 400

    board = omok.empty_board()
    for c in range(3, 7):
        board[7][c] = 1  # 흑 4개 (열린 4)
    for c in (0, 2, 4):
        board[0][c] = 2
    res = client.post("/api/omok/move", json={"board": board, "ai": 2, "level": "어려움"}).json()
    assert res["move"] in ([7, 2], [7, 7]) and res["winner"] == 0  # 막는다

    board[7][7] = 1  # 흑 5목 완성 후 판정 요청
    board[0][6] = 2
    res = client.post("/api/omok/move", json={"board": board, "ai": 2}).json()
    assert res["winner"] == 1 and res["move"] is None and len(res["line"]) == 5


def test_news_compose_includes_outlet_tone_guide():
    from rag.catalog import OUTLET_STYLES, OUTLET_TONES, compose, get

    q = compose("news", {"outlet": "한겨레", "kind": "사설", "topic": "AI 예산 증액", "facts": "1조 2천억 원"})
    assert q.startswith("뉴스 기사 초안 작성을 요청합니다.")
    assert "- 논조 (신문사): 한겨레" in q and f"- 논조 (신문사) 가이드: {OUTLET_STYLES['한겨레']}" in q and "진보" not in q  # 모델엔 성향 이름표 없이
    assert "- 기사 유형: 사설" in q and "바이라인" in q  # 작성 규칙 재확인
    outlet = next(f for f in get("news").fields if f["key"] == "outlet")
    assert outlet["options"][0] == "중립 (스트레이트)" and len(outlet["options"]) >= 9
    import pytest

    with pytest.raises(ValueError):
        compose("news", {"outlet": "한겨레"})  # 주제 필수


def test_news_ui_guides_show_lean_but_hidden_from_api_prompt(client):
    news = next(a for a in client.get("/api/agents").json() if a["id"] == "news")
    outlet = next(f for f in news["fields"] if f["key"] == "outlet")
    assert outlet["guides"]["한겨레"].startswith("진보 성향")  # 화면에는 성향 표시


def test_status_has_model_but_no_secrets(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-123")
    monkeypatch.setenv("OPENAI_API_BASE", "http://10.0.0.5:8000/v1")
    monkeypatch.setenv("OPENAI_MODEL", "my-model")
    monkeypatch.delenv("RAG_MODEL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    res = client.get("/api/status")
    assert res.json() == {"model": "my-model", "vision": None, "streaming": False, "configured": True}
    assert "sk-secret" not in res.text and "10.0.0.5" not in res.text


def test_agents_have_menu_groups(client):
    groups = {a["id"]: a["group"] for a in client.get("/api/agents").json()}
    assert groups["qa"] == groups["news"] == "문서 작업" and groups["image"] == "분석" and groups["omok"] == "쉬어가기"
