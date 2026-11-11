"""OCR / 수집 파이프라인 / 최종 답변 처리 테스트 (OCR 엔진·LLM 은 가짜로 대체)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from portal import agent, ingest, ocr, store, tools


def _box(x0, y0, x1, y1, text, score=0.9):
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text, "score": score}


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(self.reply)


def test_layout_text_groups_rows_and_columns():
    boxes = [
        _box(300, 100, 400, 120, "13.33억"),  # 같은 행, 멀리 떨어진 칸
        _box(10, 102, 80, 122, "예산"),
        _box(10, 200, 200, 220, "기간 4년"),
    ]
    assert ocr.layout_text(boxes) == "예산 | 13.33억\n기간 4년"


def test_structure_text_falls_back_when_llm_drops_content(monkeypatch):
    raw = "예산 | 13.33억 원\n기간 | 2026.08.01~2030.12.31\n인원 | 10명"
    monkeypatch.setattr(ocr, "_structure_model", lambda: FakeLLM("요약: 예산"))
    assert ocr.structure_text(raw) == raw


def test_structure_text_uses_text_layer_hint_and_normalizes_hyphen(monkeypatch):
    fake = FakeLLM("```markdown\n# 범위\n- 관제 App (3‑20)\n```")
    monkeypatch.setattr(ocr, "_structure_model", lambda: fake)
    out = ocr.structure_text("범위 관제App(3-20)", layer="관제App(3-20)")
    assert out == "# 범위\n- 관제 App (3-20)"
    assert "[PDF 텍스트 레이어]" in fake.prompts[0]


def test_ingest_ocrs_every_page_and_indexes(sample_pdf, monkeypatch):
    calls = []

    def fake_ocr_page(path, page, dpi=200):
        calls.append(page)
        return {"text": f"# {page}쪽\n매출 목표는 {page},000억 원입니다.", "raw": "raw", "quality": 0.9, "boxes": 3}

    monkeypatch.setattr(ingest, "ocr_page", fake_ocr_page)
    doc_id = ingest.ingest(str(sample_pdf))

    assert sorted(calls) == [1, 2]
    assert ingest.is_indexed(doc_id)
    assert all(p["source"] == "ocr" for p in store.load_pages(doc_id))
    assert "2,000억" in tools.search_chunks.invoke({"query": "매출 목표", "doc_id": doc_id})

    calls.clear()
    ingest.ingest(str(sample_pdf))  # 두 번째는 건너뜀
    assert calls == []


def test_normalize_citations_unifies_formats():
    text = "기간은 4년【p.3】, 인원은 10명[p. 8 · 표], 예산【23d480657b-0003, p.3†L1-L4】 (p. 5)"
    assert agent.normalize_citations(text) == "기간은 4년(p.3), 인원은 10명(p.8), 예산(p.3) (p.5)"
    assert agent.normalize_citations("표 ( p.8 ) 참고") == "표 (p.8) 참고"
    assert agent.normalize_citations("표 (p.8)【23d480657b-0013】, 같은 표") == "표 (p.8), 같은 표"
    assert agent.normalize_citations("제한함【1†L1-L9】. 근거 ((p.2)) 와 ((p.3, p.5))") == "제한함. 근거 (p.2) 와 (p.3, p.5)"
    assert agent.normalize_citations("발표했다(p. N). 끝 (p.N)") == "발표했다. 끝"
    assert agent.cited_pages("A (p.3, p.8) B (p.3) C (p.5-6)") == [3, 8, 5, 6]


def test_drop_invalid_citations_removes_missing_pages():
    text = "절단한다 (p.7). 맞춘다 (p.5). 조정한다 (p.1, p.7). 범위 (p.5-8)"
    assert agent.drop_invalid_citations(text, 6) == "절단한다. 맞춘다 (p.5). 조정한다 (p.1). 범위 (p.5)"
    assert agent.drop_invalid_citations("A (p.3)", 6) == "A (p.3)"


def test_collect_sources_prefers_cited_pages(indexed_doc):
    search = tools.search_chunks.invoke({"query": "직원 수", "doc_id": indexed_doc})
    msgs = [HumanMessage("q"), ToolMessage(search, tool_call_id="1", name="search_chunks")]
    sources = agent.collect_sources(msgs, "직원은 512명입니다 (p.1)", indexed_doc)
    assert [s["page"] for s in sources] == [1]
    assert sources[0]["cited"] and "512명" in sources[0]["snippet"]

    # 답변에 페이지 표시가 없으면 검색에서 읽은 페이지를 보여준다
    fallback = agent.collect_sources(msgs, "직원은 512명입니다", indexed_doc)
    assert [s["page"] for s in fallback] == [1] and not fallback[0]["cited"]


def test_final_answer_returns_last_ai_text():
    msgs = [HumanMessage("q"), AIMessage("", tool_calls=[{"name": "search_chunks", "args": {}, "id": "1"}]),
            ToolMessage("근거", tool_call_id="1"), AIMessage("답변입니다 (p.1)")]
    assert agent.final_answer(msgs) == "답변입니다 (p.1)"


def test_final_answer_retries_when_empty(monkeypatch):
    fake = FakeLLM("다시 쓴 답변 (p.2)")
    monkeypatch.setattr(agent, "_model", lambda env: fake)
    msgs = [HumanMessage("q"), AIMessage("", tool_calls=[{"name": "search_chunks", "args": {}, "id": "1"}]),
            ToolMessage("근거 문장", tool_call_id="1", name="search_chunks"), AIMessage("")]
    assert agent.final_answer(msgs) == "다시 쓴 답변 (p.2)"
    sent = fake.prompts[0]
    assert all(not getattr(m, "tool_calls", None) for m in sent)
    assert "근거 문장" in sent[1].content
