"""deep-agents 로 구성한 PDF 질의응답 에이전트.

문서 수집(OCR → 청킹)은 rag.ingest 가 코드로 먼저 끝내고, 에이전트는 검색과 답변만 맡는다.
메인 에이전트가 검색 도구를 직접 쓰고, 질문이 여러 갈래면 retrieval-agent 에 나눠 맡길 수 있다.
"""

from __future__ import annotations

import os
import re

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage

from . import store
from .llm import make_model
from .tools import _tokenize, document_outline, get_chunk_context, read_page, search_chunks

RETRIEVAL_PROMPT = """당신은 검색 담당입니다. 질문에 답하는 데 필요한 근거만 찾아 돌려줍니다. 답을 지어내지 마세요.
1. 질문을 2~4개의 검색어로 바꿉니다(핵심 키워드, 동의어, 영문/약어, 숫자·고유명사).
2. 각 검색어로 search_chunks 를 호출합니다 (doc_id 를 꼭 넘기세요).
3. 근거가 청크 끝에서 잘린 것 같으면 get_chunk_context 로 앞뒤를 확인합니다.
최종 답변: 관련 근거를 원문 그대로 인용하며 각 인용에 [chunk_id, p.N] 을 붙입니다.
근거를 못 찾았으면 '근거 없음' 과 시도한 검색어를 적습니다."""

MAIN_PROMPT = """당신은 문서(PDF/Word/한글/텍스트) 질의응답 담당입니다. 문서는 이미 OCR·추출되어 인덱싱되어 있습니다.
파일 시스템 도구(ls, read_file, grep, glob 등)는 쓰지 마세요. 문서 내용은 아래 도구로만 확인합니다.

절차:
1. 질문을 2~5개의 검색어로 바꿔 search_chunks(query, doc_id) 를 여러 번 호출합니다.
   (핵심 키워드, 동의어, 약어, 숫자·고유명사. 예: '수행팀 업무' → '수행팀', '역할', '담당 업무', 'R&R')
2. 근거가 잘린 것 같으면 get_chunk_context 로 앞뒤를, 필요하면 read_page 로 페이지 전체를 읽습니다.
3. 질문이 여러 부분이면 부분마다 retrieval-agent 에 task 로 맡겨도 됩니다.
4. 근거에 있는 내용만으로 답합니다. 근거가 없는 부분은 "문서에서 찾을 수 없음" 이라고 적습니다.
   약어 풀이, 업무 상세 설명 등 원문에 없는 내용을 덧붙이지 마세요. '상세히' 요청이면 관련 근거를 더 찾아 원문 내용을 모으세요.
   출처 표기 규칙: 모든 문장/항목/표의 행 끝에 반드시 (p.N) 형식으로 페이지를 붙입니다. 여러 페이지면 (p.3, p.8).
   검색 결과의 [청크ID p.N ...] 머리말에 있는 페이지 번호를 그대로 쓰세요 (청크ID 끝의 -0007 같은 번호는 페이지가 아닙니다). 【】나 [] 괄호, 청크 ID 는 쓰지 마세요.
5. 답의 숫자/고유명사가 근거에 실제로 있는지 대조한 뒤, 마지막에 반드시 사용자에게 보여줄 최종 답변을 본문으로 작성하세요.

이어지는 질문:
- 이전 대화가 있으면 '그 사람', '그거', '2차년도는?' 같은 지시어/생략을 이전 질문·답변으로 해석해 검색어를 정하세요.
- 이전 답변에 있던 내용이라도 근거는 search_chunks 로 다시 확인하고 (p.N) 을 붙이세요.

사용자의 언어로, 마크다운으로 답하세요."""


def _model(env: str):
    return make_model(os.getenv(env) or None)


DOC_TOOLS = [search_chunks, get_chunk_context, read_page, document_outline]


def build_agent(agent_id: str = "qa"):
    """catalog 의 에이전트를 만든다. 기본 카드·마켓 단독 실행은 도구가 같고 시스템 프롬프트만 다르다.

    커스텀 앱(app:<id>)은 총괄 프롬프트 + 담은 마켓 에이전트들을 하위 에이전트로 붙인다.
    """
    from .catalog import get
    from .market import subagent

    spec = get(agent_id)
    main_model = _model("RAG_MODEL")
    sub_model = make_model(os.getenv("SUBAGENT_MODEL") or os.getenv("RAG_MODEL") or None)

    if spec.subagents:
        subagents = [subagent(item, sub_model, DOC_TOOLS) for item in spec.subagents]
    else:
        subagents = [
            {
                "name": "retrieval-agent",
                "description": "질문과 doc_id 를 받아 다중 검색어로 관련 청크를 찾고, 출처가 붙은 원문 근거를 돌려준다.",
                "system_prompt": RETRIEVAL_PROMPT,
                "tools": [search_chunks, get_chunk_context, read_page],
                "model": sub_model,
            },
        ]

    return create_deep_agent(
        model=main_model,
        tools=DOC_TOOLS,
        system_prompt=spec.prompt or MAIN_PROMPT,
        subagents=subagents,
    )


FOLLOWUP = "지금까지 찾은 근거만으로 사용자 질문에 대한 최종 답변을 작성하세요. 각 항목에 (p.N) 출처를 붙이세요."


def final_answer(messages: list, followup: str = FOLLOWUP) -> str:
    """마지막 AI 메시지의 본문. 모델이 본문 없이 끝냈으면 한 번 더 최종 답변을 요청한다."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.text.strip():
            return normalize_citations(msg.text.strip())
        if isinstance(msg, HumanMessage):
            break
    reply = _model("RAG_MODEL").invoke(_text_only(messages) + [HumanMessage(followup)])
    return normalize_citations(reply.text.strip()) or "답변을 생성하지 못했습니다. 질문을 바꿔 다시 시도해 주세요."


# 모델마다 출처 표기가 제각각이라 (p.N) 한 가지로 맞춘다: 【p.3】, [p.3], 【p.3†L1-L5】, (p. 3, p.5) …
_CITE_BRACKET = re.compile(r"[【\[]\s*(?:[0-9a-f]{10}-\d{4}\s*[,|]?\s*)?(p\.?\s*\d+(?:\s*[,~\-–]\s*(?:p\.?\s*)?\d+)*)[^】\]\n]*[】\]]")
_PAGE_NUM = re.compile(r"p\.?\s*(\d+)")


def normalize_citations(text: str) -> str:
    def fix(m: re.Match) -> str:
        pages = re.findall(r"\d+", m.group(1))
        return "(" + ", ".join(f"p.{p}" for p in pages) + ")"

    text = _CITE_BRACKET.sub(fix, text)
    text = re.sub(r"\s?[【\[][0-9a-f]{10}-\d{4}(?:\s*[,，]\s*[0-9a-f]{10}-\d{4})*[】\]]", "", text)  # 청크 ID 만 든 괄호
    text = re.sub(r"\s?【[^】\n]*†[^】\n]*】", "", text)  # 페이지가 아닌 모델 고유 인용 (【1†L1-L9】)
    text = re.sub(r"\s?\(\s*p\.\s*N\s*\)", "", text)  # 규칙 예시를 그대로 베낀 자리표시자 (p.N)
    text = re.sub(r"\(\s*p\.\s*(\d+)\s*\)", r"(p.\1)", text)
    return re.sub(r"\(\((p\.\d+(?:,\s*p\.\d+)*)\)\)", r"(\1)", text)  # ((p.2)) → (p.2)


_FENCE = re.compile(r"```([\w-]*)[ \t]*\n(.*?)\n```", re.S)


def unwrap_prose_fences(text: str) -> str:
    """모델이 글(메일 본문, 회의록 등)을 코드 블록으로 감싼 것을 푼다. sql·python 등 실제 코드 블록은 둔다."""

    def fix(m: re.Match) -> str:
        lang, body = m.group(1).lower(), m.group(2)
        letters = re.findall(r"[A-Za-z가-힣]", body)
        hangul = sum(1 for ch in letters if "가" <= ch <= "힣")
        if lang in ("markdown", "md", "text", "txt") or (not lang and letters and hangul / len(letters) >= 0.3):
            return body
        return m.group(0)

    return _FENCE.sub(fix, text)


def demote_headings(text: str, by: int = 2) -> str:
    """'# 제목' 을 '### 제목' 처럼 낮춘다 (단계 결과를 '## 단계' 아래에 넣을 때). 코드 블록 안은 건드리지 않는다."""
    out, in_code = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code and (m := re.match(r"^(#{1,6})(\s)", line)):
            line = "#" * min(len(m.group(1)) + by, 6) + line[len(m.group(1)):]
        out.append(line)
    return "\n".join(out)


_CITE_GROUP = re.compile(r"\s?\(((?:p\.\s*\d+(?:\s*[,~\-–]\s*(?:p\.\s*)?\d+)*))\)")


def drop_invalid_citations(text: str, page_count: int) -> str:
    """문서에 없는 페이지(예: 청크 번호를 페이지로 착각한 p.7)를 (p.N) 표기에서 뺀다.

    전부 없는 페이지면 괄호째 지운다. 출처 카드와 본문 배지가 어긋나지 않게 하기 위함.
    """

    def fix(m: re.Match) -> str:
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        if all(1 <= n <= page_count for n in nums):
            return m.group(0)
        valid = list(dict.fromkeys(n for n in nums if 1 <= n <= page_count))
        if not valid:
            return ""
        lead = " " if m.group(0).startswith((" ", "\t")) else ""
        return lead + "(" + ", ".join(f"p.{n}" for n in valid) + ")"

    return _CITE_GROUP.sub(fix, text)


def cited_pages(answer: str) -> list[int]:
    """답변 본문의 (p.N) 표기에서 페이지 번호를 순서대로 뽑는다."""
    pages: list[int] = []
    for group in re.findall(r"\(([^()]*p\.?\s*\d+[^()]*)\)", answer):
        for n in _PAGE_NUM.findall(group) + re.findall(r"[,~\-–]\s*(\d+)", group):
            if int(n) not in pages:
                pages.append(int(n))
    return pages


def collect_sources(messages: list, answer: str, doc_id: str, limit: int = 6) -> list[dict]:
    """에이전트가 실제로 읽은 청크/페이지 중 답변의 근거를 골라 돌려준다.

    답변에 (p.N) 이 있으면 그 페이지들을, 없으면 검색에서 읽은 페이지를 순서대로 쓴다.
    각 출처: {page, section, snippet, cited}
    """
    seen_chunks: list[str] = []
    seen_pages: list[int] = []
    for m in messages:
        if getattr(m, "type", "") != "tool":
            continue
        for cid in re.findall(rf"\b{re.escape(doc_id)}-\d{{4}}\b", m.text):
            if cid not in seen_chunks:
                seen_chunks.append(cid)
        for p in re.findall(r"\[p\.(\d+) source=", m.text):  # read_page 결과
            if int(p) not in seen_pages:
                seen_pages.append(int(p))

    chunks = {c["id"]: c for c in store.load_chunks(doc_id)}
    retrieved = [chunks[c] for c in seen_chunks if c in chunks]
    for c in retrieved:
        if c["page"] not in seen_pages:
            seen_pages.append(c["page"])

    meta = store.load_meta(doc_id)
    page_count = meta["page_count"]
    has_image = bool(meta.get("pdf", meta["path"]))
    cited = [p for p in cited_pages(answer) if 1 <= p <= page_count]
    # 답변에 표시된 페이지는 모두 카드로 보여준다 (배지를 눌렀는데 카드가 없으면 안 됨). limit 은 대체 목록에만.
    pages = cited or seen_pages[:limit]
    answer_tokens = set(_tokenize(answer))

    sources = []
    for p in pages:
        candidates = [c for c in retrieved if c["page"] == p] or [c for c in chunks.values() if c["page"] == p]
        if candidates:
            best = max(candidates, key=lambda c: len(answer_tokens & set(_tokenize(c["text"]))))
            section, snippet = best["section"], best["text"]
        else:
            data = store.load_page(doc_id, p) or {"text": ""}
            section, snippet = "", data["text"]
        sources.append(
            {"page": p, "section": section, "snippet": snippet[:400], "cited": bool(cited), "image": has_image}
        )
    return sources


def _text_only(messages: list) -> list:
    """도구 호출 구조를 평문으로 바꿔, 도구 없이 모델을 다시 부를 수 있게 한다."""
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(m)
        elif isinstance(m, AIMessage):
            if m.text.strip():
                out.append(AIMessage(m.text))
        elif getattr(m, "type", "") == "tool":
            out.append(HumanMessage(f"[도구 결과: {getattr(m, 'name', '')}]\n{m.text[:6000]}"))
    return out
