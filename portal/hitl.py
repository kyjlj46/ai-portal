"""질의자 확인 (Human-in-the-loop).

에이전트가 모호하거나 질의자만 정할 수 있는 부분을 만나면 ask_user 도구로 질문한다.
- 메인 에이전트·앱 총괄: 스트림에서 ask_user 호출을 보면 그 자리에서 멈추고 지금까지의 메시지를 보관한다.
  질의자가 답하면 보관한 메시지 + 답(ToolMessage)으로 이어서 실행한다 (처음부터 다시 하지 않는다).
- 앱 '순서대로': 끝난 단계 결과를 보관하고, 질문한 단계부터 이어서 실행한다.
- 한 요청에서 질문은 MAX_ASKS 번까지. 그 뒤엔 도구가 '가정을 밝히고 진행' 하라고 답한다 (끝없이 묻지 않게).
보관은 서버 메모리에 PENDING_TTL 초 동안.
"""

from __future__ import annotations

import threading
import time
import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

ASK_TOOL = "ask_user"
MAX_ASKS = 2  # 한 요청에서 질의자에게 물을 수 있는 횟수
PENDING_TTL = 30 * 60  # 답을 기다리는 시간 (초)
MAX_PENDING = 100
SKIP_ANSWER = "질의자가 판단을 맡겼습니다. 가장 합리적인 가정으로 진행하고, 결과 맨 앞에 '가정: …' 으로 밝히세요."

CLARIFY_RULE = """
## 질의자 확인 (ask_user 도구)
- 요청이 모호해 해석에 따라 결과가 크게 달라지거나, 질의자만 정할 수 있는 것(받는 사람·대상·기간·범위, 여러 후보 중 무엇,
  서로 충돌하는 지시, 결과에 꼭 필요한데 빠진 정보)이 있으면 추측하지 말고 ask_user 로 질의자에게 묻는다.
- 묻기 전에 문서·앞 단계 결과·검색으로 스스로 확인할 수 있는지 먼저 본다. 확인할 수 있으면 묻지 않는다.
- 문체·길이·형식처럼 기본값이 있고 나중에 쉽게 고칠 수 있는 것은 묻지 말고 기본값으로 진행한다.
- 한 번에 하나만, 되도록 작업 초반에 묻는다. options 에 고를 수 있는 답 2~4개, why 에 왜 필요한지 한 줄.
- ask_user 를 부른 뒤에는 다른 도구를 부르지 않는다. 도구 결과로 질의자의 답이 오면 그대로 반영해 끝까지 작업한다.
- 질의자가 판단을 맡기면 가장 합리적인 가정으로 진행하고 결과 맨 앞에 '가정: …' 으로 밝힌다.
"""

SUBAGENT_RULE = ("\n\n## 확인이 필요할 때\n질의자만 정할 수 있는 모호한 점이 있으면 추측해 채우지 말고, "
                 "결과 맨 끝에 '확인 필요: <질문>' 한 줄을 남긴다 (총괄이 질의자에게 묻는다).")


@tool(ASK_TOOL)
def ask_user(question: str, options: list[str] | None = None, why: str = "", config: RunnableConfig = None) -> str:
    """질의자(사용자)에게 확인 질문을 하나 한다. 요청이 모호하거나 질의자만 정할 수 있는 선택이 결과를 크게 바꿀 때만 쓴다.

    question: 질의자에게 보일 질문 (한국어, 한 문장)
    options: 고를 수 있는 답 2~4개 (없으면 자유 답)
    why: 왜 확인이 필요한지 한 줄
    """
    if ((config or {}).get("configurable") or {}).get("ask_closed"):
        return f"더 이상 질의자에게 물을 수 없습니다 (한 요청에 {MAX_ASKS}번까지). " + SKIP_ANSWER
    return "질문을 질의자에게 보냈습니다. 다른 도구를 부르지 말고 '확인 대기' 라고만 답하세요."


def ask_call(msg) -> dict | None:
    """AIMessage 의 ask_user 호출 (없으면 None)."""
    for call in getattr(msg, "tool_calls", None) or []:
        if call["name"] == ASK_TOOL:
            return call
    return None


def question_of(call: dict) -> dict:
    """화면에 보낼 질문 (빈 값·너무 많은 선택지 정리)."""
    args = call.get("args") or {}
    opts = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()][:4]
    return {"question": str(args.get("question") or "어떻게 진행할까요?").strip(), "options": opts,
            "why": str(args.get("why") or "").strip()}


# ─────────────────────────── 답을 기다리는 실행 보관 ───────────────────────────

_pending: dict[str, dict] = {}
_lock = threading.Lock()


def save(state: dict) -> str:
    pid = uuid.uuid4().hex[:16]
    now = time.time()
    with _lock:
        for k in [k for k, v in _pending.items() if now - v["_at"] > PENDING_TTL]:
            _pending.pop(k)
        while len(_pending) >= MAX_PENDING:
            _pending.pop(next(iter(_pending)))
        _pending[pid] = {**state, "_at": now}
    return pid


class PendingError(Exception):
    pass


def take(pid: str, agent_id: str) -> dict:
    """보관한 실행을 꺼낸다 (한 번만). 없거나 시간이 지났거나 다른 에이전트 것이면 PendingError."""
    with _lock:
        st = _pending.get(pid or "")
        if not st or time.time() - st["_at"] > PENDING_TTL:
            _pending.pop(pid or "", None)
            raise PendingError("확인 질문이 만료됐습니다 (30분 이상 지났거나 서버가 다시 시작됨). 요청을 다시 보내 주세요.")
        if st["agent_id"] != agent_id:
            raise PendingError("다른 에이전트의 확인 질문입니다. 요청을 다시 보내 주세요.")
        return _pending.pop(pid)


def clarification_text(qas: list[tuple[str, str]]) -> str:
    return "[질의자 확인]\n" + "\n".join(f"- 질문: {q}\n  답: {a}" for q, a in qas) if qas else ""
