"""여러 문서 함께 쓰기 · 임시 문서."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import web
from rag import ingest, multidoc, store, stream, tools


@pytest.fixture
def two_docs(indexed_doc, tmp_path):
    """sample.pdf(1쪽 '직원 512명') + 텍스트 문서('예산 30억')."""
    txt = tmp_path / "예산안.txt"
    txt.write_text("2027년 예산안\n\nAI 사업 예산은 30억 원이다. 담당은 기획팀.", encoding="utf-8")
    other = ingest.ingest(str(txt))
    tools._bm25_index.cache_clear()
    return indexed_doc, other, txt


def test_labels_are_stable_and_resolvable(two_docs):
    a, b, _ = two_docs
    la, lb = store.label_for(a), store.label_for(b)
    assert la != lb and la.startswith("D") and store.label_for(a) == la
    assert store.resolve(lb) == b and store.resolve(b) == b and store.resolve("D999") == "D999"


def test_search_and_read_with_labels(two_docs):
    a, b, _ = two_docs
    la, lb = store.label_for(a), store.label_for(b)
    out = tools.search_chunks.invoke({"query": "예산 직원", "doc_id": f"{la},{lb}", "k": 10})
    assert f" {lb} p.1 " in out and f" {la} p.1 " in out  # 여러 문서면 머리말에 표시
    only = tools.search_chunks.invoke({"query": "예산 직원", "doc_id": lb, "k": 10})
    assert "30억" in only and "512명" not in only  # 고른 문서 안에서만
    single = tools.search_chunks.invoke({"query": "직원", "doc_id": a})
    assert " D" not in single.split("\n")[0]  # 문서 ID 로 하나만 찾으면 표시 없음 (기존과 같음)
    assert tools.read_page.invoke({"doc_id": lb, "page": 1}).startswith(f"[{lb} p.1 source=text")
    assert "예산안.txt" in tools.document_outline.invoke({"doc_id": lb})


def test_citation_cleanup():
    docs = [{"label": "D1", "page_count": 3}, {"label": "D4", "page_count": 1}]
    text = "직원 512명【D1 p.2】, 예산 (D4, p. 1) · 없는 쪽 (D4 p.9) · 남의 문서 (D7 p.1) · 섞임 (D1 p.3, p.8, D4 p.1)"
    out = multidoc.drop_invalid(multidoc.normalize(text), docs)
    assert out == "직원 512명(D1 p.2), 예산 (D4 p.1) · 없는 쪽 · 남의 문서 · 섞임 (D1 p.3, D4 p.1)"
    assert multidoc.cited(out) == [("D1", 2), ("D4", 1), ("D1", 3)]


class FakeAgent:
    def __init__(self, answer):
        self.answer, self.inputs = answer, None

    def stream(self, inputs, **kw):
        self.inputs = inputs
        yield (), "updates", {"model": {"messages": [AIMessage(self.answer)]}}


def test_ask_stream_with_two_documents(two_docs, sample_pdf, monkeypatch):
    a, b, txt = two_docs
    la, lb = store.label_for(a), store.label_for(b)
    agent = FakeAgent(f"직원은 512명 【{la} p.1】, AI 예산은 30억 원 ({lb}, p.1). 없는 쪽 ({lb} p.5)")
    monkeypatch.setattr(stream, "get_agent", lambda *x: agent)
    out = list(stream.ask_stream([str(sample_pdf), str(txt)], "직원 수와 예산은?", agent_id="summary"))
    sent = agent.inputs["messages"][-1].content
    assert sent.startswith(f"doc_ids: {la}, {lb}") and f"- {lb} = 예산안.txt (1구간)" in sent
    assert f"===== {lb} p.1 =====" in sent and "30억" in sent  # 짧은 문서들은 전문을 함께
    assert any(e.get("text", "").startswith("여러 문서를 함께 봅니다") for e in out)
    final = out[-1]
    assert final["answer"] == f"직원은 512명 ({la} p.1), AI 예산은 30억 원 ({lb} p.1). 없는 쪽"
    assert [(s["label"], s["page"]) for s in final["sources"]] == [(la, 1), (lb, 1)]
    assert final["sources"][1]["filename"] == "예산안.txt" and final["sources"][1]["doc_id"] == b
    assert final["doc_id"] is None and [d["label"] for d in final["docs"]] == [la, lb]


def test_single_document_unchanged(indexed_doc, sample_pdf, monkeypatch):
    agent = FakeAgent("직원은 512명 (p.1)")
    monkeypatch.setattr(stream, "get_agent", lambda *x: agent)
    final = list(stream.ask_stream([str(sample_pdf)], "직원?", agent_id="qa"))[-1]
    assert agent.inputs["messages"][-1].content.startswith(f"doc_id: {indexed_doc}\n")
    assert final["answer"] == "직원은 512명 (p.1)" and final["doc_id"] == indexed_doc and final["docs"] == []


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    return TestClient(web.app)


def test_agent_stream_accepts_document_list(client, two_docs, sample_pdf, monkeypatch):
    seen = {}

    def fake(path, q, history=None, agent_id="qa"):
        seen["path"] = path
        yield {"type": "final", "answer": "ok", "sources": [], "steps": [], "doc_id": None, "agent": agent_id, "docs": []}

    monkeypatch.setattr(stream, "ask_stream", fake)
    _, _, txt = two_docs
    res = client.post("/api/agents/qa/stream", json={"question": "q", "pdf_paths": [str(sample_pdf), str(txt)]})
    assert res.status_code == 200 and seen["path"] == [str(sample_pdf), str(txt)]
    client.post("/api/agents/qa/stream", json={"question": "q", "pdf_paths": [str(sample_pdf)]})
    assert seen["path"] == str(sample_pdf)  # 하나면 기존처럼 문자열
    assert client.post("/api/agents/qa/stream", json={"question": "q", "pdf_paths": [str(sample_pdf)] * 1 + [f"x{i}" for i in range(5)]}).status_code == 400
    assert client.post("/api/agents/qa/stream", json={"question": "q", "pdf_paths": ["없는.pdf"]}).status_code == 404


def test_temp_upload_lifecycle(client, sample_pdf):
    res = client.post("/api/upload?temp=true", files={"file": ("임시.pdf", sample_pdf.read_bytes(), "application/pdf")}).json()
    path = res["pdf_path"]
    assert res["temp"] is True and web.is_temp(path) and res["label"].startswith("D")
    doc_id = res["doc_id"]
    store.save_meta(doc_id, {"path": path, "page_count": 1, "pages": []})  # 인덱싱됐다고 치고
    assert all(d["path"] != path for d in client.get("/api/documents").json())  # 문서함 목록엔 안 보임
    assert client.request("DELETE", "/api/documents/temp", json={"pdf_path": str(sample_pdf)}).status_code == 400  # 문서함 문서는 못 지움
    assert client.request("DELETE", "/api/documents/temp", json={"pdf_path": path}).json() == {"ok": True}
    assert not store.doc_dir(doc_id).exists()
    from pathlib import Path

    assert not Path(path).exists()


def test_old_temp_documents_are_cleaned(client, sample_pdf):
    import os

    old = client.post("/api/upload?temp=true", files={"file": ("old.pdf", sample_pdf.read_bytes(), "application/pdf")}).json()["pdf_path"]
    os.utime(old, (time.time() - 2 * 86400, time.time() - 2 * 86400))
    keep = web.UPLOAD_DIR / "보관.pdf"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_bytes(b"%PDF")
    assert web.cleanup_temp() == 1
    from pathlib import Path

    assert not Path(old).exists() and keep.exists()


def test_pipeline_with_two_documents(two_docs, sample_pdf, client, monkeypatch):
    a, b, txt = two_docs
    app = client.post("/api/apps", json={"name": "비교", "agents": ["doc_search", "email"], "doc": "required"}).json()
    seen = []

    class Step:
        def __init__(self, name):
            self.name = name

        def invoke(self, inputs, config=None):
            seen.append(inputs["messages"][-1].content)
            return {"messages": [*inputs["messages"], AIMessage(f"{self.name}: 예산 30억 ({store.label_for(b)} p.1)")]}

    monkeypatch.setattr(stream, "get_agent", lambda agent_id: Step(agent_id))
    final = list(stream.ask_stream([str(sample_pdf), str(txt)], "비교해줘", agent_id=f"app:{app['id']}"))[-1]
    assert seen[0].startswith(f"doc_ids: {store.label_for(a)}, {store.label_for(b)}")
    assert [(s["label"], s["page"]) for s in final["sources"]] == [(store.label_for(b), 1)]
    assert json.dumps(final["docs"], ensure_ascii=False).count("예산안.txt") == 1
