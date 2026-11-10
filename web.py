"""업무 에이전트 웹 서버: 문서 Q&A · 문서 요약 · 보고서 작성 · 결재 기안 · 이미지 분석 · 오목 게임.

실행:  uv run web.py   →  http://127.0.0.1:8000
환경변수: PORT(기본 8000), NO_BROWSER=1 이면 브라우저 자동 열기 안 함
"""

from __future__ import annotations

import json
import os
import re
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

load_dotenv()

from rag import store  # noqa: E402
from rag.convert import SUPPORTED_EXTS  # noqa: E402
from rag.llm import check_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"

app = FastAPI(title="PDF RAG Agent")

class AskRequest(BaseModel):
    pdf_path: str
    question: str
    conversation_id: str | None = None  # 주면 같은 대화의 이전 질문/답변을 이어받는다


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in SUPPORTED_EXTS:
        raise HTTPException(400, f"지원하지 않는 형식입니다. 지원: {', '.join(sorted(SUPPORTED_EXTS))}")
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
                "kind": meta.get("kind", "pdf"),
                "unit": meta.get("unit", "page"),
            }
        )
    return out


def run_agent(pdf_path: str, question: str) -> dict:
    """문서를 수집(OCR·인덱싱)한 뒤 에이전트를 실행하고, 최종 답변 + 출처 + 처리 단계를 돌려준다."""
    from rag.stream import ask_stream

    for ev in ask_stream(pdf_path, question):
        if ev["type"] == "final":
            return {k: ev[k] for k in ("answer", "steps", "doc_id", "sources")}
        if ev["type"] == "error":
            raise RuntimeError(ev["message"])
    raise RuntimeError("답변을 받지 못했습니다.")


@app.get("/api/page-image/{doc_id}/{page}")
def page_image(doc_id: str, page: int, dpi: int = 110):
    """출처 페이지를 이미지로 보여준다 (OCR 원본 확인용)."""
    import pymupdf

    try:
        meta = store.load_meta(doc_id)
    except ValueError:
        raise HTTPException(404, "알 수 없는 문서") from None
    if not 1 <= page <= meta["page_count"]:
        raise HTTPException(404, "페이지 범위를 벗어났습니다.")
    pdf_path = meta.get("pdf", meta["path"])
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, "이 문서는 페이지 이미지가 없습니다 (텍스트 문서).")
    with pymupdf.open(pdf_path) as pdf:
        png = pdf[page - 1].get_pixmap(dpi=min(max(dpi, 50), 200)).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


def _validate(req: AskRequest) -> None:
    if not Path(req.pdf_path).exists():
        raise HTTPException(404, f"파일 없음: {req.pdf_path}")
    if not req.question.strip():
        raise HTTPException(400, "질문을 입력하세요.")
    if err := check_config():
        raise HTTPException(500, err)


@app.post("/api/ask")
def ask(req: AskRequest):
    _validate(req)
    try:
        return run_agent(req.pdf_path, req.question)
    except Exception as e:  # 에이전트 오류를 화면에 보여주기 위함
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    """새 대화를 시작할 때 이전 대화 기록을 지운다 (모든 에이전트·앱에서)."""
    from rag.stream import forget_conversation

    forget_conversation(conv_id)
    return {"ok": True}


def _sse(path: str | None, question: str, conversation_id: str | None, agent_id: str = "qa") -> StreamingResponse:
    from rag.stream import ask_stream, conversation

    history = conversation(f"{agent_id}:{conversation_id}") if conversation_id else None

    def events():
        for ev in ask_stream(path, question, history, agent_id=agent_id):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/ask/stream")
def ask_stream_api(req: AskRequest):
    """처리 단계와 답변 토큰을 Server-Sent Events 로 흘려보낸다 (문서 Q&A)."""
    _validate(req)
    return _sse(req.pdf_path, req.question, req.conversation_id)


# ─────────────────────────── 업무 에이전트 (카드) ───────────────────────────


class RunRequest(BaseModel):
    pdf_path: str | None = None  # 보고서/기안은 참고 문서 없이도 실행
    question: str | None = None  # 자유 입력 (수정 요청, Q&A 질문)
    inputs: dict | None = None  # 양식 입력 → 요청문으로 바꿈
    conversation_id: str | None = None
    image_id: str | None = None  # 이미지 분석 에이전트
    image_name: str | None = None


@app.get("/api/status")
def status():
    """왼쪽 메뉴 하단에 보여줄 모델 상태 (키·주소 같은 비밀값은 내보내지 않는다)."""
    from rag.image_agent import vision_enabled
    from rag.llm import default_model_name, llm_streaming

    return {
        "model": os.getenv("RAG_MODEL") or default_model_name(),
        "vision": os.getenv("VISION_MODEL") if vision_enabled() else None,
        "streaming": llm_streaming(),
        "configured": check_config() is None,
    }


@app.get("/api/market")
def market():
    """Agent 마켓: 단독 실행하거나 커스텀 앱에 담을 수 있는 에이전트 목록."""
    from rag import market as m

    return {"categories": m.CATEGORIES, "agents": m.public()}


class AppRequest(BaseModel):
    name: str
    icon: str = "🧩"
    description: str = ""
    instructions: str = ""
    doc: str = "optional"
    agents: list[str]
    examples: list[str] = []
    run: str = "pipeline"  # pipeline: 담은 순서대로 · auto: 총괄 에이전트가 판단
    output: str = "all"  # all: 모든 단계 결과 · last: 마지막 단계 결과


@app.get("/api/apps")
def list_apps():
    from rag import apps

    return apps.list_apps()


@app.post("/api/apps")
def create_app(req: AppRequest):
    from rag import apps

    try:
        return apps.save(req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.get("/api/apps/{app_id}")
def get_app(app_id: str):
    from rag import apps

    try:
        return apps.get(app_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None


@app.put("/api/apps/{app_id}")
def update_app(app_id: str, req: AppRequest):
    from rag import apps

    try:
        return apps.save(req.model_dump(), app_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.delete("/api/apps/{app_id}")
def delete_app(app_id: str):
    from rag import apps

    try:
        apps.delete(app_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None
    return {"ok": True}


@app.get("/api/agents")
def agents():
    from rag.catalog import public

    return public()


@app.post("/api/agents/{agent_id}/stream")
def agent_stream(agent_id: str, req: RunRequest):
    """카드의 에이전트를 실행한다. inputs(양식) 또는 question(자유 입력) 중 하나는 있어야 한다."""
    from rag.catalog import compose, get

    try:
        spec = get(agent_id)
    except KeyError:
        raise HTTPException(404, f"알 수 없는 에이전트: {agent_id}") from None
    if spec.mode == "game":
        raise HTTPException(400, f"{spec.name}은(는) /api/omok/move 로 둡니다.")
    if spec.mode == "image":
        return _image_sse(req)
    if req.pdf_path and not Path(req.pdf_path).exists():
        raise HTTPException(404, f"파일 없음: {req.pdf_path}")
    if spec.doc == "required" and not req.pdf_path:
        raise HTTPException(400, f"{spec.name}은(는) 문서가 필요합니다. 위에서 문서를 고르거나 올려 주세요.")
    try:
        question = compose(agent_id, req.inputs) if req.inputs else (req.question or "").strip()
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if not question:
        raise HTTPException(400, "요청 내용을 입력하세요.")
    if err := check_config():
        raise HTTPException(500, err)
    return _sse(req.pdf_path or None, question, req.conversation_id, agent_id)


def _image_sse(req: RunRequest) -> StreamingResponse:
    from rag import image_agent
    from rag.catalog import compose
    from rag.stream import conversation

    if not req.image_id:
        raise HTTPException(400, "먼저 이미지를 올려 주세요.")
    try:
        image_agent.image_path(req.image_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e)) from None
    if req.inputs:
        inputs = dict(req.inputs)
        question = compose("image", inputs)
    else:
        question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "요청 내용을 입력하세요.")
    if err := check_config():
        raise HTTPException(500, err)
    # 대화 기록은 이미지별로 따로 둔다 (같은 화면에서 새 이미지를 올리면 새 맥락)
    history = conversation(f"image:{req.conversation_id}:{req.image_id}") if req.conversation_id else None

    def events():
        for ev in image_agent.image_stream(req.image_id, question, history, filename=req.image_name or ""):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/images")
async def upload_image(file: UploadFile = File(...)):
    from rag import image_agent

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "이미지는 20MB 까지 올릴 수 있습니다.")
    try:
        return image_agent.save_image(data, file.filename or "image.png")
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    from rag import image_agent

    try:
        return FileResponse(image_agent.image_path(image_id), headers={"Cache-Control": "max-age=3600"})
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e)) from None


class OmokRequest(BaseModel):
    board: list
    ai: int  # AI 돌 색: 1 흑, 2 백
    level: str = "보통"


@app.post("/api/omok/move")
def omok_move(req: OmokRequest):
    """사람이 둔 뒤의 판을 받아 승부를 판정하고, 끝나지 않았으면 AI 가 한 수 둔다."""
    from rag import omok

    if err := omok.validate(req.board):
        raise HTTPException(400, err)
    if req.ai not in (1, 2):
        raise HTTPException(400, "ai 는 1(흑) 또는 2(백) 이어야 합니다.")
    board = [list(row) for row in req.board]
    if not omok.winner(board) and omok.side_to_move(board) != req.ai:
        raise HTTPException(400, "지금은 AI 차례가 아닙니다.")
    return omok.play(board, req.ai, req.level)


class ExportRequest(BaseModel):
    markdown: str
    title: str = "문서"


@app.post("/api/export/docx")
def export_docx(req: ExportRequest):
    """에이전트 결과를 Word 파일로 내려준다."""
    from urllib.parse import quote

    from rag.export import markdown_to_docx

    name = re.sub(r'[\\/:*?"<>|\s]+', "_", req.title).strip("_")[:60] or "문서"
    return Response(
        markdown_to_docx(req.markdown, req.title),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.docx"},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    if not os.getenv("NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
