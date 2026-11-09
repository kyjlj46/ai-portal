"""PDF RAG 웹 서버.

실행:  uv run web.py   →  http://127.0.0.1:8000
환경변수: PORT(기본 8000), NO_BROWSER=1 이면 브라우저 자동 열기 안 함
"""

from __future__ import annotations

import os
import re
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from rag import store  # noqa: E402
from rag.llm import check_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"

app = FastAPI(title="PDF RAG Agent")

_agent = None
_agent_lock = threading.Lock()


def get_agent():
    global _agent
    with _agent_lock:
        if _agent is None:
            from rag.agent import build_agent

            _agent = build_agent()
    return _agent


class AskRequest(BaseModel):
    pdf_path: str
    question: str


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "PDF 파일만 업로드할 수 있습니다.")
    UPLOAD_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^\w.\-가-힣]", "_", Path(file.filename).name)
    dest = UPLOAD_DIR / safe
    dest.write_bytes(await file.read())
    doc_id = store.doc_id_for(dest)
    indexed = bool(store.load_chunks(doc_id))
    return {"pdf_path": str(dest), "filename": safe, "doc_id": doc_id, "indexed": indexed}


@app.get("/api/documents")
def documents():
    out = []
    for d in store.list_doc_ids():
        meta = store.load_meta(d)
        out.append(
            {
                "doc_id": d,
                "path": meta["path"],
                "filename": Path(meta["path"]).name,
                "pages": meta["page_count"],
                "ocr_pages": [p["page"] for p in meta["pages"] if p["needs_ocr"]],
                "chunks": len(store.load_chunks(d)),
            }
        )
    return out


def run_agent(pdf_path: str, question: str) -> dict:
    """에이전트를 실행하고 최종 답변 + 서브에이전트 호출 기록을 돌려준다."""
    prompt = f"PDF 파일: {pdf_path}\n(인덱싱이 안 되어 있으면 먼저 수집하세요)\n\n질문: {question}"
    result = get_agent().invoke({"messages": [{"role": "user", "content": prompt}]})
    steps = []
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            if call["name"] == "task":
                steps.append(
                    {
                        "agent": call["args"].get("subagent_type", "?"),
                        "task": call["args"].get("description", ""),
                    }
                )
            else:
                steps.append({"agent": "main", "task": call["name"]})
    return {"answer": result["messages"][-1].text, "steps": steps}


@app.post("/api/ask")
def ask(req: AskRequest):
    if not Path(req.pdf_path).exists():
        raise HTTPException(404, f"파일 없음: {req.pdf_path}")
    if not req.question.strip():
        raise HTTPException(400, "질문을 입력하세요.")
    if err := check_config():
        raise HTTPException(500, err)
    try:
        return run_agent(req.pdf_path, req.question)
    except Exception as e:  # 에이전트 오류를 화면에 보여주기 위함
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    if not os.getenv("NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
