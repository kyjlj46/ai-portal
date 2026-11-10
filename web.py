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


TEMP_MAX_AGE = 24 * 3600  # 임시 문서는 하루 뒤 정리


def temp_dir() -> Path:
    return UPLOAD_DIR / "tmp"


def is_temp(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(temp_dir().resolve())
        return True
    except ValueError:
        return False


def _remove_temp(path: Path) -> None:
    """임시 문서 파일과 그 인덱스를 지운다. 같은 내용의 문서가 문서함에도 있으면 인덱스는 그대로 둔다."""
    import shutil

    try:
        doc_id = store.doc_id_for(path)
    except OSError:
        doc_id = None
    if doc_id:
        try:
            if is_temp(store.load_meta(doc_id)["path"]):
                shutil.rmtree(store.doc_dir(doc_id), ignore_errors=True)
        except ValueError:
            pass
    path.unlink(missing_ok=True)


def cleanup_temp(max_age: float = TEMP_MAX_AGE) -> int:
    import time

    if not temp_dir().exists():
        return 0
    old = [p for p in temp_dir().iterdir() if p.is_file() and time.time() - p.stat().st_mtime > max_age]
    for p in old:
        _remove_temp(p)
    return len(old)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), temp: bool = False):
    """문서를 올린다. temp=true 면 이번 작업에만 쓰는 임시 문서 (문서함 목록에 안 보이고 하루 뒤 지워짐)."""
    if Path(file.filename or "").suffix.lower() not in SUPPORTED_EXTS:
        raise HTTPException(400, f"지원하지 않는 형식입니다. 지원: {', '.join(sorted(SUPPORTED_EXTS))}")
    safe = re.sub(r"[^\w.\-가-힣]", "_", Path(file.filename).name)
    if temp:
        import secrets

        cleanup_temp()
        temp_dir().mkdir(parents=True, exist_ok=True)
        dest = temp_dir() / f"{secrets.token_hex(3)}_{safe}"  # 같은 이름의 임시 문서끼리 덮어쓰지 않게
    else:
        UPLOAD_DIR.mkdir(exist_ok=True)
        dest = UPLOAD_DIR / safe
    dest.write_bytes(await file.read())
    doc_id = store.doc_id_for(dest)
    indexed = bool(store.load_chunks(doc_id))
    return {"pdf_path": str(dest), "filename": safe, "doc_id": doc_id, "indexed": indexed,
            "temp": temp, "label": store.label_for(doc_id)}


class TempDocRequest(BaseModel):
    pdf_path: str


@app.delete("/api/documents/temp")
def delete_temp_document(req: TempDocRequest):
    """임시 문서를 바로 지운다 (문서함 문서는 지울 수 없다)."""
    path = Path(req.pdf_path)
    if not is_temp(path):
        raise HTTPException(400, "임시 문서만 지울 수 있습니다.")
    _remove_temp(path)
    return {"ok": True}


@app.get("/api/documents")
def documents():
    """문서함 목록 (임시 문서 제외)."""
    out = []
    for d in store.list_doc_ids():
        meta = store.load_meta(d)
        if is_temp(meta["path"]):
            continue
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
                "label": store.label_for(d),
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


def _sse(path: str | list[str] | None, question: str, conversation_id: str | None, agent_id: str = "qa") -> StreamingResponse:
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
    pdf_paths: list[str] | None = None  # 여러 문서를 함께 (최대 5개)
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

    from rag.mailer import public_status

    return {
        "model": os.getenv("RAG_MODEL") or default_model_name(),
        "vision": os.getenv("VISION_MODEL") if vision_enabled() else None,
        "streaming": llm_streaming(),
        "configured": check_config() is None,
        "email": public_status(),
    }


class EmailRequest(BaseModel):
    to: str | list[str]
    cc: str | list[str] = ""
    subject: str
    text: str
    html: str | None = None
    confirm: bool = False  # 화면에서 사용자가 '확인하고 보내기' 를 눌렀을 때만 True


@app.post("/api/email/send")
def send_email(req: EmailRequest):
    """사용자가 확인한 메일 초안을 SMTP 로 보낸다. 에이전트가 이 API 를 부르는 경로는 없다."""
    from rag import mailer

    if not req.confirm:
        raise HTTPException(400, "보내기 전에 받는 사람과 내용을 확인해 주세요.")
    try:
        return mailer.send(req.to, req.subject, req.text, cc=req.cc, html=req.html)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from None
    except OSError as e:  # 연결 실패, 인증 실패 등 (smtplib.SMTPException 도 OSError)
        raise HTTPException(502, f"메일 서버 오류: {type(e).__name__}: {e}") from None


@app.get("/api/market")
def market():
    """Agent 마켓: 단독 실행하거나 커스텀 앱에 담을 수 있는 에이전트 목록."""
    from rag import market as m

    return {"categories": m.CATEGORIES, "agents": m.public()}


class AgentDesignRequest(BaseModel):
    request: str = ""  # 만들고 싶은 Agent 설명
    feedback: str = ""  # 고칠 점 (다시 설계할 때)
    current: dict | None = None  # 지금 설계 (다시 설계할 때)


class CustomAgentRequest(BaseModel):
    name: str
    icon: str = "🤖"
    category: str = "업무"
    description: str = ""
    role: str
    instructions: str
    examples: list[str] = []
    uses_doc: bool = False
    tools: list[str] = []
    on_upload: str = ""


@app.post("/api/custom-agents/design")
def design_agent(req: AgentDesignRequest):
    """gpt-oss 로 Agent 를 설계한다 (저장하지 않음 — 화면에서 확인·수정 후 저장)."""
    from rag import custom_agents

    if err := check_config():
        raise HTTPException(500, err)
    try:
        return custom_agents.design(req.request, req.feedback, req.current)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.get("/api/custom-agents")
def list_custom_agents():
    from rag import custom_agents

    return custom_agents.list_all()


@app.get("/api/custom-agents/{agent_id}")
def get_custom_agent(agent_id: str):
    from rag import custom_agents

    try:
        return custom_agents.get(agent_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None


@app.post("/api/custom-agents")
def create_custom_agent(req: CustomAgentRequest):
    from rag import custom_agents

    try:
        return custom_agents.save(req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.put("/api/custom-agents/{agent_id}")
def update_custom_agent(agent_id: str, req: CustomAgentRequest):
    from rag import custom_agents

    try:
        return custom_agents.save(req.model_dump(), agent_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None


@app.delete("/api/custom-agents/{agent_id}")
def delete_custom_agent(agent_id: str):
    from rag import custom_agents

    try:
        custom_agents.delete(agent_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from None
    except ValueError as e:
        raise HTTPException(409, str(e)) from None
    return {"ok": True}


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
    from rag.multidoc import MAX_DOCS

    paths = list(dict.fromkeys(req.pdf_paths or ([req.pdf_path] if req.pdf_path else [])))
    if spec.doc == "none":
        paths = []
    if len(paths) > MAX_DOCS:
        raise HTTPException(400, f"문서는 한 번에 {MAX_DOCS}개까지 고를 수 있습니다.")
    for p in paths:
        if not Path(p).exists():
            raise HTTPException(404, f"파일 없음: {p}")
    if spec.doc == "required" and not paths:
        raise HTTPException(400, f"{spec.name}은(는) 문서가 필요합니다. 위에서 문서를 고르거나 올려 주세요.")
    try:
        question = compose(agent_id, req.inputs) if req.inputs else (req.question or "").strip()
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if not question:
        raise HTTPException(400, "요청 내용을 입력하세요.")
    if err := check_config():
        raise HTTPException(500, err)
    return _sse(paths[0] if len(paths) == 1 else (paths or None), question, req.conversation_id, agent_id)


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
    cleanup_temp()  # 하루 지난 임시 문서 정리
    if not os.getenv("NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
