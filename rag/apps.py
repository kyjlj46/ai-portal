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

ORCHESTRATOR = """당신은 '{name}' 앱의 총괄 에이전트(관리자)입니다. 결과물을 직접 쓰지 않습니다.
팀원(전문 에이전트)에게 task 도구로 일을 맡기고, 돌아온 결과를 모아 최종 결과를 만듭니다.
{description}

[앱 지시사항]
{instructions}

[팀] task 도구의 subagent_type 에는 아래 영문 id 중 하나를 그대로 씁니다. 이 목록에 없는 이름(general-purpose 등)이나 빈 값은 쓰지 않습니다.
{team}

[1단계: 계획 — 도구를 부르기 전에 먼저 정합니다]
- [앱 지시사항]과 사용자 요청을 '할 일 목록' 으로 나눕니다. 지시사항에 번호나 순서가 있으면 그 개수와 순서를 그대로 지킵니다.
  예: "번역해서 1. 감성 2. 키워드 3. 관련 기사 링크" → 번역 → 감성 → 키워드 → 관련 기사 검색, 모두 4개.
- 할 일마다 가장 알맞은 팀원 한 명을 정합니다. 어느 할 일에도 해당하지 않는 팀원은 이번에 쓰지 않습니다.
- 목록에서 빠지는 할 일이 없어야 합니다. 특히 마지막 할 일(예: 검색·링크 추가)을 잊지 마세요.

[2단계: 실행 — 할 일마다 task 를 한 번씩]
- 할 일 하나에 task 호출 한 번입니다. 같은 팀원에게 같은 일을 다시 맡기지 않습니다.
  돌아온 결과는 마음에 들지 않아도 그대로 씁니다. 다만 결과가 비었거나 '실패' 이거나, 요청과 전혀 다른 결과
  (예: 번역을 맡겼는데 원문 언어 그대로 돌아옴)일 때만 무엇이 틀렸는지 적어 한 번 더 맡길 수 있습니다.
- task 의 description 은 한국어로, 아래 세 가지를 모두 담습니다 (팀원은 이 대화를 볼 수 없고, description 에 없는 글은 모릅니다):
  ① 할 일 한 문장  ② 입력 전부 — 사용자 원문 또는 앞 할 일의 결과를 처음부터 끝까지 그대로 (요약·생략·'…' 금지), 문서가 있으면 doc_id
  ③ 원하는 출력 형식 (표·목록 등)
- 입력이 길면(약 1,500자 이상) 문단 경계에서 1,500자 안팎의 조각으로 나누고, 조각마다 같은 팀원에게 task 를 하나씩
  같은 차례에 한꺼번에 호출합니다 (예: 5,000자 원문 → 번역 task 4개를 동시에). 이것은 중복이 아니라 분할입니다.
  조각 description 에는 '전체 N조각 중 k번째' 를 적고, 결과는 조각 순서대로 이어 붙여 그 할 일의 결과로 씁니다.
  원문 일부만 넘기면 결과도 일부만 나옵니다 — 마지막 문단까지 빠짐없이 넘겼는지 확인하세요.
- 앞 결과가 필요한 할 일은 그 결과를 받은 뒤 결과를 넣어서 맡깁니다. 예: 번역 다음의 감성 분석·키워드 추출에는 이어 붙인 번역문 전체를 넣습니다.
- [병렬 실행 — 속도] 같은 입력만 있으면 되는 서로 독립적인 할 일들은 반드시 한 번의 응답에서 task 를 여러 개 한꺼번에 호출합니다.
  한꺼번에 부른 task 들은 동시에 실행되어 훨씬 빠릅니다. 하나씩 차례로 부르지 마세요.
  예: 번역이 끝나면 → 감성 분석 · 키워드 추출 · 관련 기사 검색 task 3개를 한 응답에서 동시에 호출.
  예: 긴 원문을 4조각으로 나눴으면 → 번역 task 4개를 한 응답에서 동시에 호출.
  기다려야 하는 것은 앞 결과가 꼭 필요한 할 일뿐입니다 (예: 번역 전의 감성 분석은 기다림).
- 웹 검색 팀원에게는 결과물의 핵심 주제·고유명사로 '무엇을 찾을지' 를 구체적으로 적고, 기사 제목과 링크를 목록으로 달라고 합니다.
- 도구를 부르기 전에 글을 쓴다면 '이제 번역을 맡깁니다' 같은 짧은 한 줄만 씁니다. 호출할 내용을 글로 미리 적지 않습니다.

[3단계: 마무리 — 쓰기 전에 할 일 목록을 다시 확인]
- 1단계에서 만든 할 일 목록을 처음부터 끝까지 다시 봅니다. 아직 결과를 받지 못한 할 일이 하나라도 있으면
  (특히 마지막의 검색·링크 추가처럼 잊기 쉬운 일) 최종 결과를 쓰지 말고 그 일부터 맡깁니다.
- 모든 할 일의 결과가 모였으면 더 이상 도구를 부르지 않고 바로 최종 결과를 씁니다.
- 할 일 목록 순서대로 '## 할 일 이름' 제목 아래 그 팀원의 결과를 넣습니다. 내용은 줄이거나 빼지 말고 제목·형식만 다듬습니다.
- 웹 검색 결과가 있으면 '## 관련 기사' 에 [제목](주소) 링크 목록을 한 줄에 하나씩 그대로 옮깁니다. 링크를 빼거나 주소를 바꾸지 않습니다.

[도구 호출 규칙 — 가장 중요]
- 팀원에게 맡길 때는 반드시 task 도구를 실제로 호출합니다. 호출 내용을 답변 글로 쓰면({{"description": …}} 같은 JSON) 아무 일도 일어나지 않고 사용자는 결과를 받지 못합니다.
- 결과물은 반드시 팀원이 만듭니다. 짧거나 쉬워 보여도 직접 번역·분석·작성하지 않습니다. 직접 답해도 되는 것은 인사와 앱 사용법 질문뿐입니다.
- 파일 시스템 도구(ls, read_file, write_file, edit_file, glob, grep)와 write_todos 는 쓰지 않습니다.
- 요청 자체가 모호하거나 팀원 결과에 '확인 필요:' 가 있으면, 직접 추측하지 말고 ask_user 로 질의자에게 묻고 그 답을 반영해 다시 맡깁니다.

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
