"""사용자가 만든 Agent (gpt-oss 로 설계 → 사용자가 확인·수정 → 저장).

저장: data/agents/<id>.json — 마켓에 '내가 만든' Agent 로 나타나고, 단독 실행·앱에 담기·편집·삭제가 된다.
만든 Agent 에는 데이터를 바꾸지 않는 도구만 준다: 문서 검색(uses_doc), 웹 검색·웹 페이지 읽기(tools).
"""

from __future__ import annotations

import json
import re
import secrets
import time

from . import store

ALLOWED_TOOLS = {"web_search": "웹 검색", "fetch_url": "웹 페이지 읽기"}
MAX_PROMPT = 6000

DESIGN_PROMPT = """당신은 업무용 AI Agent 설계자입니다. 사용자가 원하는 Agent 설명을 보고, 그 Agent 의 정의를 JSON 하나로만 출력합니다.
설명이나 코드 블록 없이 JSON 객체만 출력하세요.

{{
  "name": "Agent 이름 (2~12자, 무엇을 하는지 바로 알 수 있게)",
  "icon": "어울리는 이모지 1개",
  "category": "{categories} 중 하나",
  "description": "마켓 카드에 보일 한두 문장 설명 (80자 이내)",
  "role": "시스템 프롬프트 첫 문장 '당신은 ○○입니다' 의 ○○ 부분 (예: 계약서 검토 전문가)",
  "instructions": "Agent 가 따를 지시. 마크다운으로 '절차:' (번호 목록), '출력 형식:' (제목·표·목록 구조를 구체적으로), '주의:' 를 포함. 300~1200자, 한국어",
  "examples": ["사용자가 이 Agent 에게 할 만한 요청 예시 2~3개"],
  "uses_doc": true 또는 false (사용자가 올린 문서를 읽어야 하는 일이면 true),
  "tools": [] 또는 {tools} 중 필요한 것 (최신 정보·웹 조사가 필요할 때만),
  "on_upload": "문서를 올리자마자 바로 실행하면 좋은 Agent 면, 그때 사용자가 보내는 요청 한 문장 (예: '이 계약서의 위험 조항을 분석해 주세요.'). 아니면 빈 문자열"
}}

설계 원칙:
- 한 가지 일을 잘하는 Agent 로 만듭니다. 지시는 구체적이고 검증 가능하게 씁니다.
- 출력 형식은 실제 업무에 바로 쓸 수 있게 (표, 체크리스트, 단계별 목록 등) 정합니다.
- 근거 없는 내용을 지어내지 말라는 주의를 넣습니다. 문서를 쓰는 Agent 면 문서 근거와 (p.N) 출처를 요구합니다.
- 사람에게 해를 끼치거나 불법적인 일을 돕는 Agent 는 설계하지 않습니다. 그런 요청이면 {{"error": "이유"}} 만 출력합니다."""


def _dir():
    return store.DATA_DIR / "agents"


def _path(agent_id: str):
    if not re.fullmatch(r"c_[0-9a-f]{8}", agent_id or ""):
        raise KeyError(f"잘못된 Agent ID: {agent_id}")
    return _dir() / f"{agent_id}.json"


def validate(data: dict) -> dict:
    from .market import CATEGORIES

    name = str(data.get("name") or "").strip()
    if not name:
        raise ValueError("Agent 이름을 입력하세요.")
    role = str(data.get("role") or "").strip()
    instructions = str(data.get("instructions") or "").strip()
    if not role or not instructions:
        raise ValueError("역할과 지시사항을 입력하세요.")
    if len(role) + len(instructions) > MAX_PROMPT:
        raise ValueError(f"역할과 지시사항은 합쳐서 {MAX_PROMPT:,}자까지입니다.")
    category = str(data.get("category") or "업무")
    tools = [t for t in dict.fromkeys(data.get("tools") or []) if t in ALLOWED_TOOLS]
    examples = [str(e).strip()[:150] for e in (data.get("examples") or []) if str(e).strip()][:5]
    return {
        "name": name[:30],
        "icon": (str(data.get("icon") or "🤖").strip() or "🤖")[:4],
        "category": category if category in CATEGORIES else "업무",
        "description": str(data.get("description") or "").strip()[:200] or f"{name} Agent",
        "role": role[:200],
        "instructions": instructions,
        "examples": examples,
        "uses_doc": bool(data.get("uses_doc")),
        "tools": tools,
        "on_upload": str(data.get("on_upload") or "").strip()[:200] if data.get("uses_doc") else "",
    }


def save(data: dict, agent_id: str | None = None) -> dict:
    clean = validate(data)
    now = time.time()
    if agent_id:
        item = {**get(agent_id), **clean, "updated": now}
    else:
        item = {**clean, "id": f"c_{secrets.token_hex(4)}", "created": now, "updated": now}
    _dir().mkdir(parents=True, exist_ok=True)
    _path(item["id"]).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    return item


def get(agent_id: str) -> dict:
    path = _path(agent_id)
    if not path.exists():
        raise KeyError(f"Agent 를 찾을 수 없습니다: {agent_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def delete(agent_id: str) -> None:
    from . import apps

    item = get(agent_id)
    used = [a["name"] for a in apps.list_apps() if agent_id in a.get("agents", [])]
    if used:
        raise ValueError(f"'{item['name']}' 을(를) 쓰는 앱이 있어 지울 수 없습니다: {', '.join(used)}. 앱에서 먼저 빼 주세요.")
    _path(agent_id).unlink()


def list_all() -> list[dict]:
    if not _dir().exists():
        return []
    items = []
    for p in _dir().glob("c_*.json"):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(items, key=lambda a: a.get("created", 0))


# ─────────────────────────── gpt-oss 로 설계 ───────────────────────────


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("모델이 JSON 을 돌려주지 않았습니다. 설명을 조금 바꿔 다시 시도해 주세요.")
    return json.loads(text[start:end + 1])


def design(request: str, feedback: str = "", current: dict | None = None, model=None) -> dict:
    """설명(과 고칠 점)으로 Agent 정의 초안을 만든다. 저장하지 않는다."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from .llm import make_model
    from .market import CATEGORIES

    request = (request or "").strip()
    if not request and not current:
        raise ValueError("만들고 싶은 Agent 를 설명해 주세요.")
    system = DESIGN_PROMPT.format(categories=" / ".join(CATEGORIES), tools=json.dumps(list(ALLOWED_TOOLS)))
    parts = [f"[만들고 싶은 Agent]\n{request}"]
    if current:
        keep = {k: current.get(k) for k in ("name", "icon", "category", "description", "role", "instructions", "examples", "uses_doc", "tools", "on_upload")}
        parts.append(f"[현재 설계]\n{json.dumps(keep, ensure_ascii=False)}")
    if feedback.strip():
        parts.append(f"[고칠 점 — 반영해서 전체 JSON 을 다시 출력]\n{feedback.strip()}")
    model = model or make_model()
    last_error = None
    for _ in range(2):  # JSON 이 깨지면 한 번 더
        reply = model.invoke([SystemMessage(system), HumanMessage("\n\n".join(parts))])
        try:
            data = _parse_json(reply.text)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            continue
        if data.get("error"):
            raise ValueError(f"이 Agent 는 만들 수 없습니다: {data['error']}")
        return validate(data)
    raise ValueError(f"설계 결과를 읽지 못했습니다 ({last_error}). 다시 시도해 주세요.")
