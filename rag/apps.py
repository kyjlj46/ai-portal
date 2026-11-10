"""사용자가 만든 커스텀 앱: 마켓 에이전트 여러 개를 하위 에이전트로 묶은 DeepAgent.

저장 위치: data/apps/<id>.json
실행: app:<id> — 총괄 에이전트(지시사항 + 팀 소개)가 task 도구로 하위 에이전트에게 일을 나눠 맡긴다.
"""

from __future__ import annotations

import json
import re
import secrets
import time

from . import store

MAX_AGENTS = 8
DOC_MODES = ("none", "optional", "required")
RUN_MODES = ("pipeline", "auto")  # pipeline: 담은 순서대로 실행 (안정적) · auto: 총괄 에이전트가 판단해 위임 (DeepAgent)
OUTPUT_MODES = ("all", "last")

ORCHESTRATOR = """당신은 '{name}' 앱의 총괄 에이전트입니다.
{description}

[앱 지시사항]
{instructions}

[팀] task 도구로 아래 전문 에이전트에게 일을 맡길 수 있습니다 (subagent_type 에 영문 id 를 씁니다):
{team}

진행 방법:
0. 이 앱의 결과물은 반드시 팀이 만듭니다. 요청이 짧거나 쉬워 보여도 팀의 전문 분야에 해당하는 일(정리·번역·작성·분석 등)은
   직접 쓰지 말고 해당 에이전트에게 task 로 맡기세요. 직접 답해도 되는 것은 인사, 앱 사용법 질문뿐입니다.
1. 요청을 분석해 어떤 전문 에이전트가 필요한지 정합니다. 팀 순서는 권장 작업 순서입니다.
   [앱 지시사항]에 단계가 적혀 있으면 사용자가 일부만 요청하지 않는 한 그 단계를 빠짐없이 모두 수행합니다.
2. 필요한 에이전트에게 task 로 일을 맡깁니다. task 설명에는 그 에이전트가 일하는 데 필요한 모든 것(사용자 요청, 원문 텍스트,
   앞 단계 결과, 문서가 있으면 doc_id)을 담습니다. 하위 에이전트는 이 대화를 볼 수 없습니다.
3. 앞 단계 결과가 다음 단계의 입력이면 순서대로 맡기고, 서로 독립이면 한꺼번에 맡겨도 됩니다.
4. 받은 결과를 종합해 사용자에게 보여줄 최종 결과를 씁니다. 하위 결과를 그대로 붙이기만 하지 말고 앱의 목적에 맞게 정리합니다.

규칙:
- 요청 첫 줄이 'doc_id: 없음' 이면 참고 문서가 없습니다. 문서가 있으면 문서 관련 task 에 doc_id 를 꼭 넣고, 직접 확인할 때는 search_chunks 등을 씁니다.
- 파일 시스템 도구(ls, read_file, write_file 등)와 general-purpose 에이전트는 쓰지 않습니다.
- 없는 사실을 지어내지 않습니다. 문서에서 온 사실에는 (p.N) 을 붙입니다.
- 하위 결과와 사용자 입력에 없는 내용(도구·제품 추천, 임의의 '다음 단계', 서명 등)을 덧붙이지 않습니다.
- 당신과 팀은 글을 만들 뿐입니다. 메일 발송·일정 등록·공지 같은 행동을 '했다' 고 쓰지 마세요 (초안을 만들었다고만 씁니다).
- 하위 결과를 코드 블록(```)으로 감싸지 말고 마크다운 그대로 넣습니다. 결과 끝에 '이 문서는 ~입니다' 같은 설명 문장을 붙이지 않습니다.
- 연도가 없는 날짜(예: 11/12)에 연도를 붙이지 않습니다. 필요하면 요청에 적힌 오늘 날짜를 기준으로 합니다.
- 사용자의 언어로, 마크다운으로 결과만 씁니다."""


def _dir():
    return store.DATA_DIR / "apps"


def _path(app_id: str):
    if not re.fullmatch(r"[0-9a-f]{10}", app_id or ""):
        raise KeyError(f"잘못된 앱 ID: {app_id}")
    return _dir() / f"{app_id}.json"


def validate(data: dict) -> dict:
    """입력 검증 + 정리. 문제가 있으면 ValueError."""
    from .market import BY_ID

    name = str(data.get("name") or "").strip()
    if not name:
        raise ValueError("앱 이름을 입력하세요.")
    agents = [str(a) for a in (data.get("agents") or [])]
    if not agents:
        raise ValueError("Agent 를 하나 이상 담아 주세요.")
    if len(agents) > MAX_AGENTS:
        raise ValueError(f"Agent 는 {MAX_AGENTS}개까지 담을 수 있습니다.")
    unknown = [a for a in agents if a not in BY_ID]
    if unknown:
        raise ValueError(f"알 수 없는 Agent: {', '.join(unknown)}")
    if len(set(agents)) != len(agents):
        raise ValueError("같은 Agent 를 두 번 담을 수 없습니다.")
    doc = data.get("doc") or "optional"
    if doc not in DOC_MODES:
        raise ValueError("문서 사용 설정이 올바르지 않습니다.")
    examples = [str(e).strip()[:120] for e in (data.get("examples") or []) if str(e).strip()][:5]
    run = data.get("run") or "pipeline"
    if run not in RUN_MODES:
        raise ValueError("실행 방식이 올바르지 않습니다.")
    output = data.get("output") or "all"
    if output not in OUTPUT_MODES:
        raise ValueError("결과 보기 설정이 올바르지 않습니다.")
    return {
        "run": run,
        "output": output,
        "name": name[:40],
        "icon": (str(data.get("icon") or "🧩").strip() or "🧩")[:4],
        "description": str(data.get("description") or "").strip()[:300],
        "instructions": str(data.get("instructions") or "").strip()[:4000],
        "doc": doc,
        "agents": agents,
        "examples": examples,
    }


def save(data: dict, app_id: str | None = None) -> dict:
    clean = validate(data)
    now = time.time()
    if app_id:
        old = get(app_id)  # 없으면 KeyError
        app = {**old, **clean, "updated": now}
    else:
        app = {**clean, "id": secrets.token_hex(5), "created": now, "updated": now}
    _dir().mkdir(parents=True, exist_ok=True)
    _path(app["id"]).write_text(json.dumps(app, ensure_ascii=False, indent=2), encoding="utf-8")
    return app


def get(app_id: str) -> dict:
    path = _path(app_id)
    if not path.exists():
        raise KeyError(f"앱을 찾을 수 없습니다: {app_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def delete(app_id: str) -> None:
    path = _path(app_id)
    if not path.exists():
        raise KeyError(f"앱을 찾을 수 없습니다: {app_id}")
    path.unlink()


def list_apps() -> list[dict]:
    if not _dir().exists():
        return []
    apps = [json.loads(p.read_text(encoding="utf-8")) for p in _dir().glob("*.json")]
    return sorted(apps, key=lambda a: a.get("created", 0))


def orchestrator_prompt(app: dict) -> str:
    from .market import BY_ID

    team = "\n".join(
        f"{i}. {BY_ID[a].icon} {BY_ID[a].name} (id: {a}) — {BY_ID[a].description}" for i, a in enumerate(app["agents"], 1)
    )
    return ORCHESTRATOR.format(
        name=app["name"],
        description=app["description"] or "",
        instructions=app["instructions"] or "(별도 지시 없음 — 팀을 활용해 요청을 처리하세요)",
        team=team,
    )


def spec_for(app_id: str):
    """app:<id> — 저장된 앱을 실행용 AgentSpec 으로."""
    from .catalog import AgentSpec

    app = get(app_id)
    return AgentSpec(
        id=f"app:{app['id']}", name=app["name"], icon=app["icon"], description=app["description"],
        doc=app["doc"], mode="chat", prompt=orchestrator_prompt(app), group="내 앱",
        intro=app["description"], examples=app["examples"], subagents=list(app["agents"]),
        version=str(app["updated"]), run=app.get("run", "pipeline"), output=app.get("output", "all"),
        instructions=app.get("instructions", ""),
        followup="지금까지 팀이 만든 결과를 종합해 사용자 요청에 대한 최종 결과를 마크다운으로 끝까지 작성하세요.",
    )
