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
    assert "PDF RAG Agent" in res.text


def test_upload_rejects_non_pdf(client):
    res = client.post("/api/upload", files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 400


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


def test_ask_missing_file(client):
    res = client.post("/api/ask", json={"pdf_path": "없는파일.pdf", "question": "q"})
    assert res.status_code == 404
