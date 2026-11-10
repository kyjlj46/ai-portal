"""질문 처리 과정을 단계별 이벤트로 흘려보낸다 (웹 SSE / CLI 공용).

이벤트(dict):
  {"type": "step",  "kind": <ingest|think|search|read|delegate|result|answer>, "text": str, "sub": bool}
  {"type": "token", "text": str}          최종 답변 초안 토큰 (메인 에이전트)
  {"type": "reset"}                       방금까지 흘린 토큰이 답변이 아니라 도구 호출 전 메모였음
  {"type": "final", "answer": str, "sources": list, "steps": list, "doc_id": str | None, "agent": str}
  {"type": "error", "message": str}
"""

from __future__ import annotations

import queue
import re
import threading
from collections.abc import Iterator
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from . import store

_ingest_lock = threading.Lock()
_agents: dict = {}  # 에이전트 id → 생성된 에이전트
_agent_lock = threading.Lock()

MAX_TURNS = 6  # 이어지는 질문에 넘길 이전 대화 턴 수
AGENT_LIMIT = 80  # 에이전트 그래프 반복 한도 (deepagents 기본 1000 — 무한루프 방지로 낮춘다)
TASK_CAP_PER_AGENT = 6  # 총괄 에이전트가 같은 팀원에게 맡길 수 있는 최대 횟수 (긴 입력을 나눠 맡기는 것 포함)
MAX_CONVERSATIONS = 200
_conversations: dict[str, list] = {}
_conv_lock = threading.Lock()


def conversation(conv_id: str) -> list:
    """대화 ID 별 이전 대화 기록 (서버 메모리). 오래된 대화부터 정리한다."""
    with _conv_lock:
        if conv_id not in _conversations and len(_conversations) >= MAX_CONVERSATIONS:
            _conversations.pop(next(iter(_conversations)))
        return _conversations.setdefault(conv_id, [])


def forget_conversation(conv_id: str) -> None:
    """화면의 대화 ID 로 만든 모든 기록(<에이전트>:<대화>[:하위])을 지운다. 에이전트 id 에도 ':' 가 들어갈 수 있다 (app:..)."""
    with _conv_lock:
        for key in [k for k in _conversations if k.endswith(":" + conv_id) or f":{conv_id}:" in k]:
            _conversations.pop(key)


def forget(conv_id: str) -> None:
    """conv_id 와 그 하위 기록(conv_id:...)을 지운다 (이미지 분석은 이미지별 하위 기록을 쓴다)."""
    with _conv_lock:
        for key in [k for k in _conversations if k == conv_id or k.startswith(conv_id + ":")]:
            _conversations.pop(key)


def get_agent(agent_id: str = "qa"):
    """만들어 둔 에이전트를 재사용한다. 커스텀 앱은 편집할 때마다 version 이 바뀌어 새로 만든다."""
    from .catalog import get as get_spec

    key = f"{agent_id}@{get_spec(agent_id).version}"
    with _agent_lock:
        if key not in _agents:
            from .agent import build_agent

            for old in [k for k in _agents if k.startswith(agent_id + "@")]:
                _agents.pop(old)
            _agents[key] = build_agent(agent_id)
    return _agents[key]


def _step(kind: str, text: str, sub: bool = False) -> dict:
    return {"type": "step", "kind": kind, "text": text, "sub": sub}


def _ingest_events(path: str) -> Iterator[dict]:
    """수집을 별도 스레드에서 돌리며 진행 로그를 이벤트로 넘긴다."""
    from .ingest import ingest

    q: queue.Queue = queue.Queue()

    def run():
        try:
            with _ingest_lock:
                ingest(path, progress=lambda m: q.put(("log", m)))
            q.put(("done", None))
        except Exception as e:  # noqa: BLE001
            q.put(("error", e))

    threading.Thread(target=run, daemon=True).start()
    yield _step("ingest", f"문서 수집 시작: {Path(path).name} (처음 한 번만 실행됩니다)")
    while True:
        kind, val = q.get()
        if kind == "log":
            first = val.splitlines()[0]
            yield _step("ingest", first)
        elif kind == "error":
            raise val
        else:
            break


def _ref(s: dict) -> str:
    return f"{s['label']} p.{s['page']}" if s.get("label") else f"p.{s['page']}"


def _public_docs(multi: list | None) -> list:
    return [{k: d[k] for k in ("label", "doc_id", "filename", "page_count")} for d in multi] if multi else []


def web_sources(messages: list) -> list[tuple[str, str]]:
    """실제로 읽은 웹 페이지(fetch_url) → 없으면 검색 결과(web_search) 의 (주소, 제목) 목록. 같은 주소는 한 번만."""
    read: dict[str, str] = {}
    found: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        if m.name == "fetch_url" and (hit := re.match(r"\[(.*?)\] (https?://\S+)", m.text)):
            read.setdefault(hit.group(2), hit.group(1).strip() or hit.group(2))
        elif m.name == "web_search":
            for title, url in re.findall(r"^\[\d+\] (.+)\n(https?://\S+)", m.text, re.M):
                found.setdefault(url, title.strip())
    pages = dict(read)
    for url, title in found.items():  # 읽은 페이지가 적으면 검색 결과로 채운다
        if len(pages) >= 8:
            break
        pages.setdefault(url, title)
    return list(pages.items())


def add_web_sources(answer: str, messages: list) -> str:
    """웹 출처를 결과 끝에 '## 출처' 로 붙인다 — 모델이 링크를 옮겨 적었는지에 맡기지 않는다.

    답에 이미 있는 주소는 빼고, 없는 것만 더한다.
    """
    pages = [(url, title) for url, title in web_sources(messages) if url not in answer]
    if not pages:
        return answer
    lines = [f"- [{title.replace('[', '(').replace(']', ')')}]({url})" for url, title in pages]
    head = "## 출처 (관련 기사·읽은 페이지)" if not re.search(r"https?://", answer) else "## 출처 (추가)"
    return answer.rstrip() + f"\n\n{head}\n" + "\n".join(lines)


# ─────────────────────────── 재시도 ───────────────────────────

MAX_ATTEMPTS = 3  # 단계 실행·최종 답 작성은 최대 3번까지 시도한다 (같은 일을 끝없이 되풀이하지 않도록 상한)
RETRY_DELAY = 1.0  # 재시도 사이 대기(초) — n번째 실패 뒤 n배


def retryable(e: Exception) -> bool:
    """다시 해 볼 만한 오류인지: 연결·시간 초과·서버 오류·빈 답/도구 흉내. 입력이 너무 긴 경우 등은 다시 해도 같다."""
    from .agent import NoAnswerError

    name, low = type(e).__name__, str(e).lower()
    if isinstance(e, NoAnswerError):
        return True
    if name == "BadRequestError" and any(k in low for k in ("context", "too long", "max_tokens", "input tokens")):
        return False
    if name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "FileNotFoundError", "KeyError", "ValueError"):
        return False
    return True


# ─────────────────────────── 커스텀 앱 '순서대로' 실행 ───────────────────────────

STEP_RULE = (
    "[당신의 차례] 이 앱의 {i}/{n} 단계 '{name}' 담당입니다. 위 요청과 앞 단계 결과를 바탕으로 당신의 일만 하세요. "
    "요청 전체를 혼자 끝내려 하지 마세요. 다른 단계({rest})가 할 일은 하지 않습니다. 결과만 마크다운으로 출력합니다."
)
STEP_LIMIT = 40  # 한 단계 에이전트의 그래프 반복 한도 (도구를 끝없이 부르지 않게)


def stages(agent_ids: list[str]) -> list[list[str]]:
    """실행 묶음: 이웃한 '병렬 가능' Agent(앞 결과만 읽는 분석형)는 한 묶음으로 동시에, 나머지는 하나씩 차례로.

    예: 번역 → 감성·키워드·웹 검색 (동시) → 이메일
    """
    from .market import BY_ID

    out: list[list[str]] = []
    for sid in agent_ids:
        if out and BY_ID[sid].parallel and all(BY_ID[x].parallel for x in out[-1]):
            out[-1].append(sid)
        else:
            out.append([sid])
    return out


def _run_step(sid: str, prompt: str) -> dict:
    """마켓 Agent 한 단계를 실행한다 (일시적 오류·빈 답이면 최대 MAX_ATTEMPTS 번). 스레드에서도 돈다."""
    import time

    from .agent import final_answer, normalize_citations, unwrap_prose_fences

    tries: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = get_agent(f"m:{sid}").invoke({"messages": [HumanMessage(prompt)]}, {"recursion_limit": STEP_LIMIT})
            msgs = result["messages"]
            out = unwrap_prose_fences(normalize_citations(
                final_answer(msgs, "지금까지의 내용으로 당신 단계의 결과를 마크다운으로 끝까지 작성하세요.")))
            return {"ok": True, "out": out, "msgs": msgs, "tries": tries}
        except Exception as e:  # noqa: BLE001
            tries.append(explain_error(e))
            if attempt == MAX_ATTEMPTS or not retryable(e):
                return {"ok": False, "error": tries[-1], "tries": tries}
            time.sleep(RETRY_DELAY * attempt)  # 1초, 2초 쉬었다가


def _pipeline(spec, doc_id: str | None, question: str, history: list | None, steps: list, multi: list | None = None,
              notes: list[str] | None = None) -> Iterator[dict]:
    """커스텀 앱 '순서대로' 실행: 담은 순서대로 마켓 에이전트를 돌리고 앞 단계 결과를 다음 단계에 넘긴다.

    - 이웃한 분석형 Agent 는 한 묶음으로 동시에 실행한다 (stages).
    - 각 단계는 최대 3번 시도하고, 그래도 실패하면 원인을 남기고 나머지 단계를 계속한다.
    - 웹 출처는 코드가 결과 끝에 붙인다.
    multi 가 있으면 여러 문서를 함께 쓴다 (문서 표시 D1… 와 (D1 p.3) 출처).
    """
    from concurrent.futures import ThreadPoolExecutor
    from datetime import date

    from . import multidoc
    from .agent import NoAnswerError, collect_sources, demote_headings, drop_invalid_citations
    from .market import BY_ID

    order = list(spec.subagents)
    n = len(order)
    plan = stages(order)
    today = f"{date.today():%Y-%m-%d}"
    if multi:
        base = multidoc.header(multi, today) + f"\n요청: {question}"
    else:
        base = f"doc_id: {doc_id or '없음'}\n오늘: {today}\n요청: {question}"
    prev = history[-1].content if history else ""  # 이어지는 요청이면 직전 결과를 고치게 한다
    results: dict[str, str] = {}  # sid → 결과 (실패면 ⚠ 안내)
    warnings: list[str] = list(notes or [])  # 실패한 단계 등 사용자에게 알릴 것
    tool_msgs: list = []
    preloaded = False
    flow = " → ".join(" + ".join(f"{BY_ID[s].icon} {BY_ID[s].name}" for s in group) for group in plan)
    yield _step("think", f"{spec.name}: {n}단계 실행 — {flow}" + (" (+ 는 동시에)" if any(len(g) > 1 for g in plan) else ""))

    for group in plan:
        done_before = [(f"{BY_ID[s].icon} {BY_ID[s].name}", results[s]) for s in order if s in results]
        prompts = {}
        for sid in group:
            m = BY_ID[sid]
            i = order.index(sid) + 1
            parts = [base]
            if spec.instructions:
                parts.append(f"[앱 지시사항]\n{spec.instructions}")
            if prev:
                parts.append(f"[직전 결과 — 위 요청이 수정 요청이면 이것을 바탕으로 고칩니다]\n{prev}")
            if (doc_id or multi) and m.uses_doc and not preloaded and (full := multidoc.preload(multi) if multi else _preload(doc_id)):
                parts.append(full.strip())  # 짧은 문서는 처음 문서를 쓰는 단계에 전문을 준다
            parts += [f"[앞 단계 결과: {name}]\n{out}" for name, out in done_before]
            rest = ", ".join(BY_ID[s].name for s in order if s != sid and s not in results) or "없음 — 당신이 마지막 단계"
            parts.append(STEP_RULE.format(i=i, n=n, name=m.name, rest=rest))
            prompts[sid] = "\n\n".join(parts)
            yield _step("delegate", f"{i}/{n} {m.icon} {m.name} 작업 중…" + (" (동시 실행)" if len(group) > 1 else ""))
        if any((doc_id or multi) and BY_ID[s].uses_doc for s in group):
            preloaded = True

        with ThreadPoolExecutor(max_workers=len(group)) as ex:  # 한 묶음 안의 단계는 동시에
            futures = {sid: ex.submit(_run_step, sid, prompts[sid]) for sid in group}
            outcome = {sid: f.result() for sid, f in futures.items()}

        for sid in group:  # 결과는 담은 순서대로 정리한다
            m, r = BY_ID[sid], outcome[sid]
            for t in r["tries"][: (len(r["tries"]) - (0 if r["ok"] else 1))]:
                yield _step("result", f"→ ↻ {m.name} 다시 시도 (앞선 시도: {t})")
            if not r["ok"]:
                reason = r["error"] + (f" — {len(r['tries'])}번 시도" if len(r["tries"]) > 1 else "")
                warnings.append(f"{m.icon} {m.name} 단계 실패 — {reason}")
                results[sid] = f"> ⚠ 이 단계는 실패했습니다: {reason}"
                yield _step("result", f"→ ⚠ {m.name} 실패: {reason}")
                continue
            for msg in r["msgs"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        yield _describe_call(call["name"], call["args"], True)
                elif isinstance(msg, ToolMessage):
                    tool_msgs.append(msg)
            out = r["out"]
            if multi:
                out = multidoc.drop_invalid(multidoc.normalize(out), multi)
            elif doc_id:
                out = drop_invalid_citations(out, store.load_meta(doc_id)["page_count"])
            results[sid] = out
            steps.append({"agent": sid, "task": f"{order.index(sid) + 1}/{n} 단계"})
            yield _step("result", f"→ {m.name} 완료 ({len(out):,}자)")

    failed = sum(1 for w in warnings if "단계 실패" in w)
    if failed == n:  # 모든 단계가 실패하면 결과 대신 원인을 알린다
        raise NoAnswerError("모든 단계가 실패했습니다.\n" + "\n".join(f"- {w}" for w in warnings))
    outputs = [(f"{BY_ID[s].icon} {BY_ID[s].name}", results[s]) for s in order]
    if n == 1:
        answer = outputs[0][1]
    elif spec.output == "last":
        earlier = "\n\n".join(f"#### {name}\n\n{demote_headings(out, 3)}" for name, out in outputs[:-1])
        answer = f"{outputs[-1][1]}\n\n<details><summary>중간 단계 결과 보기 ({n - 1}개)</summary>\n\n{earlier}\n\n</details>"
    else:
        answer = "\n\n".join(f"## {name}\n\n{demote_headings(out)}" for name, out in outputs)
    answer = add_web_sources(answer, tool_msgs)
    if multi:
        sources = multidoc.sources(tool_msgs, answer, multi)
    else:
        sources = collect_sources(tool_msgs, answer, doc_id) if doc_id else []
    if history is not None:
        history.extend([HumanMessage(base), AIMessage(answer)])
        del history[: max(0, len(history) - 2 * MAX_TURNS)]
    if sources:
        yield _step("answer", f"출처 {len(sources)}곳 정리: " + ", ".join(_ref(s) for s in sources))
    yield {"type": "final", "answer": answer, "sources": sources, "steps": steps, "doc_id": doc_id, "agent": spec.id,
           "docs": _public_docs(multi), "warnings": warnings}


PRELOAD_MAX_PAGES = 15
PRELOAD_MAX_CHARS = 40_000


def _preload(doc_id: str) -> str:
    """짧은 문서는 전문을 페이지 표시와 함께 요청에 붙인다 (요약이 일부 페이지만 읽고 끝나지 않게).

    너무 길면 빈 문자열 → 에이전트가 document_outline / read_page 로 직접 읽는다.
    """
    pages = [p for p in store.load_pages(doc_id) if p.get("text", "").strip()]
    if not pages or len(pages) > PRELOAD_MAX_PAGES or sum(len(p["text"]) for p in pages) > PRELOAD_MAX_CHARS:
        return ""
    body = "\n\n".join(f"===== p.{p['page']} =====\n{p['text'].strip()}" for p in pages)
    return (
        "\n\n[문서 전문 — 페이지별. 이미 전부 주어졌으니 도구로 다시 읽을 필요 없음."
        " 출처는 '===== p.N =====' 의 N 으로 (p.N) 표기]\n" + body
    )


def _describe_call(name: str, args: dict, sub: bool) -> dict:
    if name == "search_chunks":
        return _step("search", f"검색: “{args.get('query', '')}”", sub)
    if name == "get_chunk_context":
        return _step("read", f"앞뒤 문맥 확인: {args.get('chunk_id', '')}", sub)
    if name == "read_page":
        return _step("read", f"p.{args.get('page', '?')} 전체 읽기", sub)
    if name == "document_outline":
        return _step("read", "문서 전체 구성 파악", sub)
    if name == "chunk_stats":
        return _step("read", "청킹 결과 통계 확인", sub)
    if name == "preview_chunks":
        return _step("read", f"청크 미리보기 ({args.get('start', 0)}번부터 {args.get('count', 5)}개)", sub)
    if name == "chunk_document":
        return _step("think", f"다시 청킹: 크기 {args.get('chunk_size', 800)}, 겹침 {args.get('chunk_overlap', 120)}", sub)
    if name == "ocr_report":
        return _step("read", "페이지별 OCR 결과·신뢰도 확인", sub)
    if name == "reocr_pages":
        return _step("think", f"다시 OCR: p.{args.get('pages', '?')} (dpi {args.get('dpi', 300)})", sub)
    if name == "web_search":
        return _step("search", f"웹 검색: “{args.get('query', '')}”", sub)
    if name == "fetch_url":
        return _step("read", f"페이지 읽기: {str(args.get('url', ''))[:90]}", sub)
    if name == "task":
        from .market import BY_ID

        who = args.get("subagent_type", "subagent")
        if who in BY_ID:  # 마켓 에이전트는 이름으로 보여준다
            who = f"{BY_ID[who].icon} {BY_ID[who].name}"
        return _step("delegate", f"{who} 에게 위임: {args.get('description', '')[:120]}", sub)
    return _step("think", f"도구 사용: {name}", sub)


def _describe_result(msg: ToolMessage, sub: bool) -> dict | None:
    text = msg.text
    if msg.name == "search_chunks":
        refs = re.findall(r"^\[\S+ (?:(D\d+) )?p\.(\d+)", text, re.M)  # 여러 문서면 '[청크ID D1 p.3 …'
        if not refs:
            return _step("result", "→ 일치하는 근거 없음", sub)
        found = sorted(set(refs), key=lambda r: (r[0], int(r[1])))
        return _step("result", f"→ 근거 {len(refs)}개 찾음 ({', '.join(f'{lb} p.{p}'.strip() for lb, p in found)})", sub)
    if msg.name == "task":
        return _step("result", "→ 맡긴 작업의 결과를 받음", sub)
    if msg.name == "web_search":
        hits = len(re.findall(r"^\[\d+\] ", text, re.M))
        return _step("result", f"→ 검색 결과 {hits}개" if hits else f"→ {text[:80]}", sub)
    if msg.name == "fetch_url":
        return _step("result", f"→ {len(text):,}자 읽음" if text.startswith(("[", "http")) else f"→ {text[:80]}", sub)
    if msg.name in ("chunk_document", "reocr_pages"):
        return _step("result", "→ " + text.strip().splitlines()[0][:100], sub)
    return None


def ask_stream(path: str | list[str] | None, question: str, history: list | None = None, agent_id: str = "qa") -> Iterator[dict]:
    """history 를 주면 이전 대화를 이어받고, 이번 턴의 메시지를 history 에 덧붙인다 (CLI 대화형).

    agent_id 로 catalog 의 에이전트를 고른다. path 가 None 이면 참고 문서 없이 실행한다 (보고서/기안).
    path 에 문서 목록을 주면 여러 문서를 함께 쓴다 (문서 표시 D1… 와 (D1 p.3) 출처, rag.multidoc).
    """
    from datetime import date

    from . import multidoc
    from .agent import collect_sources, drop_invalid_citations, final_answer, normalize_citations
    from .catalog import get as get_spec

    try:
        from .ingest import is_indexed

        spec = get_spec(agent_id)
        paths = [path] if isinstance(path, str) else list(dict.fromkeys(path or []))
        doc_ids = list(dict.fromkeys(store.doc_id_for(p) for p in paths))
        steps: list[dict] = []
        for p, d in zip(paths, doc_ids):
            if not is_indexed(d):
                for ev in _ingest_events(p):
                    yield ev
                steps.append({"agent": "ingest", "task": f"{Path(p).name}: 전체 페이지 OCR + LLM 정리 → 청킹/인덱싱"})
        multi = multidoc.docs_of(doc_ids) if len(doc_ids) > 1 else None
        doc_id = doc_ids[0] if len(doc_ids) == 1 else None

        if spec.subagents and spec.run == "pipeline":  # 커스텀 앱 '순서대로' 실행
            yield from _pipeline(spec, doc_id, question, history, steps, multi)
            return

        today = f"{date.today():%Y-%m-%d}" if agent_id.startswith(("m:", "app:")) else None  # 날짜를 지어내지 않도록
        if multi:
            prompt = multidoc.header(multi, today) + f"\n{spec.request_label}: {question}"
            yield _step("read", "여러 문서를 함께 봅니다: " + ", ".join(f"{d['label']} {d['filename']}" for d in multi))
        else:
            prompt = f"doc_id: {doc_id or '없음'}\n" + (f"오늘: {today}\n" if today else "") + f"{spec.request_label}: {question}"
        if multi:
            full = multidoc.preload(multi) if spec.preload else ""
        else:
            full = _preload(doc_id) if doc_id and spec.preload else ""
        if full:
            yield _step("read", "문서 전문을 페이지별로 불러옴")
        main_msgs: list = [HumanMessage(prompt + full)]
        notes: list[str] = []  # 결과와 함께 알릴 것 (반복 한도로 멈춤 등)
        tool_msgs: list = []  # 서브에이전트 것까지 포함 (출처 수집용)
        drafting = False
        yield _step("think", "질문을 분석하고 검색어를 정하는 중…" if agent_id == "qa" else f"{spec.name}: 요청을 분석하는 중…")

        stream = get_agent(agent_id).stream(
            {"messages": list(history or []) + [main_msgs[0]]},
            config={"recursion_limit": AGENT_LIMIT},  # 그래프 반복 한도 — 도구를 끝없이 부르는 무한루프 방지
            stream_mode=["updates", "messages"],
            subgraphs=True,
        )
        # 총괄 에이전트 반복 한도: 같은 팀원에게 너무 여러 번(조각 나누기 포함), 또는 전체적으로 너무 많이 맡기면 멈춘다
        per_agent_cap = TASK_CAP_PER_AGENT
        total_cap = max(12, 3 * len(spec.subagents) + 4) if spec.subagents else None
        task_count: dict[str, int] = {}
        stopped = ""  # 한도에 걸려 멈춘 이유 (있으면 지금까지의 결과로 마무리)
        failed: Exception | None = None  # 커스텀 앱 실행 중 오류 → '순서대로' 로 이어서 처리
        try:
            for ns, mode, data in stream:
                sub = bool(ns)
                if mode == "messages":
                    chunk, meta = data
                    if not sub and meta.get("langgraph_node") == "model" and isinstance(chunk.content, str) and chunk.content:
                        if not drafting:
                            drafting = True
                            yield _step("answer", "답변 작성 중…")
                        yield {"type": "token", "text": chunk.content}
                    continue

                for node, upd in (data or {}).items():
                    msgs = upd.get("messages") if isinstance(upd, dict) else None
                    if not isinstance(msgs, list):
                        continue
                    for m in msgs:
                        if not sub:
                            main_msgs.append(m)
                        if isinstance(m, AIMessage) and m.tool_calls:
                            if drafting and not sub:  # 흘린 토큰은 답변이 아니라 도구 호출 전 메모였다
                                drafting = False
                                yield {"type": "reset"}
                            for call in m.tool_calls:
                                yield _describe_call(call["name"], call["args"], sub)
                                steps.append(
                                    {"agent": call["args"].get("subagent_type", "?"), "task": call["args"].get("description", "")}
                                    if call["name"] == "task"
                                    else {"agent": "sub" if sub else "main", "task": call["name"]}
                                )
                                if call["name"] == "task" and not sub:
                                    who = call["args"].get("subagent_type", "?")
                                    task_count[who] = task_count.get(who, 0) + 1
                        elif isinstance(m, ToolMessage):
                            tool_msgs.append(m)
                            ev = _describe_result(m, sub)
                            if ev:
                                yield ev
                            if not sub:
                                yield _step("think", "찾은 근거를 검토하는 중…")
                if spec.subagents and task_count:
                    worst = max(task_count, key=task_count.get)
                    if task_count[worst] > per_agent_cap:
                        stopped = f"같은 팀원({worst})에게 {task_count[worst]}번 맡겨 반복 한도({per_agent_cap}번)를 넘었습니다"
                    elif sum(task_count.values()) > total_cap:
                        stopped = f"팀에 맡긴 횟수가 {sum(task_count.values())}번으로 한도({total_cap}번)를 넘었습니다"
                    if stopped:
                        break
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "GraphRecursionError":
                stopped = f"도구 호출이 반복 한도({AGENT_LIMIT}단계)에 닿았습니다"
            elif spec.subagents:
                failed = e
            else:
                raise
        finally:
            stream.close()
        if stopped:
            yield _step("think", f"⚠ {stopped} — 멈추고 지금까지 받은 결과로 마무리합니다.")
            notes.append(f"{stopped}. 반복을 멈추고 지금까지 받은 결과로 정리했습니다.")
            drafting = False

        if spec.subagents:  # 커스텀 앱 '총괄 에이전트가 판단': 팀에 제대로 맡기지 못했으면 '순서대로' 로 다시 처리
            from .agent import answer_state

            state = answer_state(main_msgs)
            delegated = sum(1 for s in steps if s.get("agent") in spec.subagents)
            if failed is not None:
                note = f"총괄 에이전트 실행 중 오류가 났습니다 ({explain_error(failed)}). 그래서 '순서대로' 실행으로 다시 처리했습니다."
                yield _step("think", "⚠ " + note)
                yield from _pipeline(spec, doc_id, question, history, steps, multi, notes=notes + [note])
                return
            if not stopped and (state != "ok" or not delegated):
                why = {"stub": "하위 에이전트 호출을 실제로 하지 않고 호출 내용(JSON)만 글로 남겼습니다",
                       "none": "답을 내지 못하고 끝났습니다"}.get(state, "팀에 일을 맡기지 않고 직접 답했습니다")
                note = f"총괄 에이전트가 {why}. 그래서 '순서대로' 실행으로 다시 처리했습니다."
                yield _step("think", "⚠ " + note)
                yield from _pipeline(spec, doc_id, question, history, steps, multi, notes=notes + [note])
                return
        if not drafting:
            yield _step("answer", "찾은 근거로 최종 답변을 정리하는 중…")
        answer = normalize_citations(final_answer(main_msgs, spec.followup))
        if agent_id.startswith(("m:", "app:")):
            from .agent import unwrap_prose_fences

            answer = unwrap_prose_fences(answer)
            answer = add_web_sources(answer, tool_msgs + main_msgs)
        sources: list = []
        if multi:
            answer = multidoc.drop_invalid(multidoc.normalize(answer), multi)
            sources = multidoc.sources(tool_msgs + main_msgs, answer, multi)
        elif doc_id:
            answer = drop_invalid_citations(answer, store.load_meta(doc_id)["page_count"])
            sources = collect_sources(tool_msgs + main_msgs, answer, doc_id)
        if history is not None:
            # 다음 질문의 맥락용: 도구 호출/결과는 빼고 질문과 최종 답변만 남긴다 (컨텍스트 절약)
            # 문서 전문을 붙인 요청은 기록에는 요청문만 남긴다 (다음 턴마다 전문이 반복되지 않게)
            history.extend([HumanMessage(prompt), AIMessage(normalize_citations(answer))])
            del history[: max(0, len(history) - 2 * MAX_TURNS)]
        if sources:
            yield _step("answer", f"출처 {len(sources)}곳 정리: " + ", ".join(_ref(s) for s in sources))
        yield {"type": "final", "answer": normalize_citations(answer), "sources": sources, "steps": steps, "doc_id": doc_id,
               "agent": agent_id, "docs": _public_docs(multi), "warnings": notes}
    except Exception as e:  # noqa: BLE001 - 화면에 오류를 보여주기 위함
        yield {"type": "error", "message": explain_error(e), "detail": f"{type(e).__name__}: {e}"[:600]}


def explain_error(e: Exception) -> str:
    """답이 안 나온 이유를 사용자가 알 수 있는 말로 (무엇을 확인하면 되는지까지)."""
    from .agent import NoAnswerError

    name, text = type(e).__name__, str(e)
    low = text.lower()
    if isinstance(e, NoAnswerError):
        return text
    if name == "GraphRecursionError":
        return "에이전트가 도구를 너무 여러 번 반복 호출해 한도에서 멈췄습니다. 요청을 더 구체적으로 나눠 보세요."
    if name in ("APIConnectionError", "ConnectError", "ConnectTimeout", "ConnectionError", "ConnectionRefusedError") \
            or "connection" in low and "refused" in low:
        return "모델 서버에 연결하지 못했습니다. .env 의 OPENAI_API_BASE 주소와 네트워크를 확인하세요."
    if name in ("APITimeoutError", "ReadTimeout", "TimeoutError") or "timed out" in low:
        return "모델 응답이 너무 오래 걸려 시간이 초과됐습니다. 잠시 후 다시 시도하거나 입력을 줄여 보세요."
    if name == "AuthenticationError" or "401" in text and "auth" in low:
        return "모델 서버 인증에 실패했습니다. .env 의 OPENAI_API_KEY 를 확인하세요."
    if name == "RateLimitError" or "429" in text:
        return "모델 서버의 요청 한도를 넘었습니다. 잠시 후 다시 시도하세요."
    if name == "BadRequestError" and any(k in low for k in ("maximum context", "context length", "too long", "max_tokens", "input tokens")):
        return "입력이 모델이 한 번에 받을 수 있는 길이를 넘었습니다. 글이나 문서를 나눠서 요청하세요."
    if name in ("InternalServerError", "APIStatusError", "BadRequestError"):
        status = getattr(e, "status_code", "")
        return f"모델 서버가 오류를 돌려줬습니다{f' (HTTP {status})' if status else ''}: {text[:200]}"
    if isinstance(e, FileNotFoundError):
        return f"파일을 찾을 수 없습니다: {getattr(e, 'filename', '') or text}"
    return f"처리 중 오류가 났습니다 ({name}): {text[:300]}"
