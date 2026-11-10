"""ask_stream: 에이전트 스트림을 화면용 단계 이벤트로 바꾸는지 (가짜 에이전트 사용)."""

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from rag import stream, tools


class FakeAgent:
    def __init__(self, events):
        self.events = events
        self.inputs = None

    def stream(self, inputs, **kwargs):
        self.inputs = inputs
        yield from self.events


def test_ask_stream_emits_steps_tokens_and_final(indexed_doc, sample_pdf, monkeypatch):
    monkeypatch.setattr("rag.ingest.is_indexed", lambda doc_id: True)
    search = tools.search_chunks.invoke({"query": "연구 인력", "doc_id": indexed_doc})
    call = AIMessage("", tool_calls=[{"name": "search_chunks", "args": {"query": "연구 인력"}, "id": "c1"}])
    events = [
        ((), "messages", (AIMessageChunk("먼저 검색"), {"langgraph_node": "model"})),  # 도구 호출 전 메모
        ((), "updates", {"model": {"messages": [call]}}),
        ((), "updates", {"tools": {"messages": [ToolMessage(search, tool_call_id="c1", name="search_chunks")]}}),
        (("task:1",), "updates", {"model": {"messages": [AIMessage("", tool_calls=[{"name": "read_page", "args": {"page": 1}, "id": "c2"}])]}}),
        ((), "messages", (AIMessageChunk("연구 인력은 87명【p.1】"), {"langgraph_node": "model"})),
        ((), "updates", {"model": {"messages": [AIMessage("연구 인력은 87명【p.1】")]}}),
    ]
    agent = FakeAgent(events)
    monkeypatch.setattr(stream, "get_agent", lambda *a: agent)

    out = list(stream.ask_stream(str(sample_pdf), "연구 인력은?"))
    kinds = [(e["type"], e.get("kind")) for e in out]

    assert ("step", "search") in kinds and ("step", "result") in kinds
    assert {"type": "reset"} in out  # 도구 호출 전 메모 토큰은 지움
    sub_steps = [e for e in out if e.get("sub")]
    assert sub_steps and "p.1 전체 읽기" in sub_steps[0]["text"]
    result = next(e for e in out if e.get("kind") == "result")
    assert "p.1" in result["text"]

    final = out[-1]
    assert final["type"] == "final" and final["answer"] == "연구 인력은 87명(p.1)"
    assert [s["page"] for s in final["sources"]] == [1]


def test_follow_up_question_gets_compact_history(indexed_doc, sample_pdf, monkeypatch):
    monkeypatch.setattr("rag.ingest.is_indexed", lambda doc_id: True)
    history: list = []

    def run(answer):
        agent = FakeAgent([((), "updates", {"model": {"messages": [AIMessage(answer)]}})])
        monkeypatch.setattr(stream, "get_agent", lambda *a: agent)
        return agent

    run("PM은 홍길동입니다 (p.1)")
    list(stream.ask_stream(str(sample_pdf), "PM은 누구야?", history))
    assert [m.type for m in history] == ["human", "ai"]

    second = run("홍길동은 수행팀입니다 (p.1)")
    list(stream.ask_stream(str(sample_pdf), "그 사람 소속은?", history))
    sent = second.inputs["messages"]
    assert [m.type for m in sent] == ["human", "ai", "human"]  # 이전 질문/답변 + 새 질문
    assert "PM은 누구야?" in sent[0].content and "그 사람 소속은?" in sent[2].content
    assert len(history) == 4

    for i in range(10):  # 오래된 턴은 잘라낸다
        run(f"답 {i}")
        list(stream.ask_stream(str(sample_pdf), f"질문 {i}", history))
    assert len(history) == 2 * stream.MAX_TURNS


def test_ask_stream_reports_errors(sample_pdf, monkeypatch):
    monkeypatch.setattr("rag.ingest.is_indexed", lambda doc_id: True)

    class Broken:
        def stream(self, *a, **k):
            raise ConnectionError("endpoint down")

    monkeypatch.setattr(stream, "get_agent", lambda *a: Broken())
    out = list(stream.ask_stream(str(sample_pdf), "q"))
    assert out[-1] == {"type": "error", "message": "ConnectionError: endpoint down"}


def test_form_agent_without_document_skips_ingest_and_sources(monkeypatch):
    monkeypatch.setattr("rag.ingest.is_indexed", lambda doc_id: (_ for _ in ()).throw(AssertionError("no ingest")))
    agent = FakeAgent([((), "updates", {"model": {"messages": [AIMessage("# 주간 보고\n- 진행률 80%")]}})])
    used = {}
    monkeypatch.setattr(stream, "get_agent", lambda agent_id="qa": used.setdefault("id", agent_id) and agent)

    out = list(stream.ask_stream(None, "보고서 작성을 요청합니다.", agent_id="report"))
    final = out[-1]
    assert used["id"] == "report"
    assert final["type"] == "final" and final["agent"] == "report" and final["doc_id"] is None
    assert final["sources"] == [] and final["answer"].startswith("# 주간 보고")
    assert agent.inputs["messages"][0].content.startswith("doc_id: 없음\n요청: 보고서 작성")


def test_document_outline_lists_pages(indexed_doc):
    out = tools.document_outline.invoke({"doc_id": indexed_doc})
    assert "2페이지" in out and "[p.1]" in out and "[p.2] (내용 없음)" in out


def test_summary_preloads_short_document_but_keeps_history_compact(indexed_doc, sample_pdf, monkeypatch):
    monkeypatch.setattr("rag.ingest.is_indexed", lambda doc_id: True)
    agent = FakeAgent([((), "updates", {"model": {"messages": [AIMessage("## 한 줄 요약\n직원 512명 (p.1)")]}})])
    monkeypatch.setattr(stream, "get_agent", lambda *a: agent)
    history: list = []

    out = list(stream.ask_stream(str(sample_pdf), "문서 요약 작성을 요청합니다.", history, agent_id="summary"))
    sent = agent.inputs["messages"][-1].content
    assert "===== p.1 =====" in sent and "512명" in sent  # 전문이 요청에 붙음
    assert "=====" not in history[0].content  # 기록에는 요청문만
    assert out[-1]["answer"] == "## 한 줄 요약\n직원 512명 (p.1)" and [s["page"] for s in out[-1]["sources"]] == [1]
