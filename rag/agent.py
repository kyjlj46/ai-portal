"""deep-agents 로 구성한 PDF RAG.

메인 에이전트(오케스트레이터)가 단계별 서브에이전트에 `task` 로 일을 위임한다.
각 서브에이전트는 자기 단계 도구만 가지므로 컨텍스트가 섞이지 않고,
단계마다 스스로 결과를 검수한 뒤 요약만 메인에게 돌려준다.
"""

from __future__ import annotations

import os

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from .tools import (
    chunk_document,
    extract_text_layer,
    get_chunk_context,
    inspect_pdf,
    list_documents,
    ocr_pages,
    preview_chunks,
    read_page,
    search_chunks,
)

PARSER_PROMPT = """당신은 PDF 파싱 담당입니다.
1. inspect_pdf 로 문서를 진단합니다(이미 doc_id 를 받았다면 생략 가능).
2. needs_ocr 가 아닌 페이지만 extract_text_layer 로 추출합니다.
3. 품질 경고가 뜬 페이지는 read_page 로 직접 확인하고, 글자 깨짐/누락/읽기 순서 뒤섞임이 있으면 OCR 대상에 추가합니다.
최종 답변 형식:
doc_id: ...
추출 완료 페이지: [...]
OCR 필요 페이지: [...]  (사유 한 줄씩)"""

OCR_AGENT_PROMPT = """당신은 OCR 담당입니다.
1. 요청받은 페이지를 ocr_pages 로 처리합니다. 한 번에 3~4페이지씩 나눠 호출하세요.
2. 결과에서 quality 가 낮거나 [판독불가] 가 많은 페이지는 read_page 로 확인하고 dpi=300 으로 한 번 더 OCR 합니다.
3. 표가 있는 페이지는 마크다운 표 구조가 유지됐는지 확인합니다.
최종 답변: 처리한 페이지, 페이지별 상태(정상/부분 판독불가), 남은 문제."""

CHUNKING_PROMPT = """당신은 청킹 담당입니다.
1. chunk_document 로 청크를 만듭니다. 기본값은 chunk_size=800, chunk_overlap=120 입니다.
2. preview_chunks 로 앞/중간/뒤 청크를 몇 개 확인합니다.
   - 표나 목록이 중간에 잘려 의미가 깨지면 chunk_size 를 키워 다시 청킹합니다(최대 1500).
   - 100자 미만 청크가 많으면 chunk_size 를 조정합니다.
3. 추출되지 않은 페이지가 있다고 나오면 그대로 보고합니다(직접 처리하지 마세요).
최종 답변: 최종 파라미터, 청크 수, 품질 메모."""

RETRIEVAL_PROMPT = """당신은 검색 담당입니다. 질문에 답하는 데 필요한 근거만 찾아 돌려줍니다. 답을 지어내지 마세요.
1. 질문을 2~4개의 검색어로 바꿉니다(핵심 키워드, 동의어, 영문/약어, 숫자·고유명사).
2. 각 검색어로 search_chunks 를 호출합니다.
3. 근거가 청크 끝에서 잘린 것 같으면 get_chunk_context 로 앞뒤를 확인합니다.
최종 답변: 관련 근거를 원문 그대로 인용하며 각 인용에 [chunk_id, p.N] 을 붙입니다.
근거를 못 찾았으면 '근거 없음' 과 시도한 검색어를 적습니다."""

MAIN_PROMPT = """당신은 PDF 문서 질의응답 오케스트레이터입니다. 직접 파싱/OCR/검색하지 말고 서브에이전트에게 task 로 위임하세요.

[문서 수집 – 새 PDF 경로를 받았을 때]
1. list_documents 로 이미 인덱싱됐는지 확인합니다. 청크가 있으면 수집을 건너뜁니다.
2. parser-agent: PDF 경로를 주고 진단+텍스트 추출을 맡깁니다.
3. ocr-agent: parser-agent 가 보고한 OCR 필요 페이지가 있으면 doc_id 와 페이지를 주고 맡깁니다.
4. chunking-agent: doc_id 를 주고 청킹+인덱싱을 맡깁니다.

[질의응답]
5. retrieval-agent: 사용자 질문 원문과 doc_id 를 주고 근거를 찾게 합니다.
   질문이 여러 부분으로 되어 있으면 부분마다 별도로 retrieval-agent 를 호출하세요.
6. 근거에 있는 내용만으로 답합니다. 문장마다 (p.N) 출처를 붙이고, 근거가 없는 부분은 "문서에서 찾을 수 없음" 이라고 명시합니다.
7. 답하기 전, 답의 각 숫자/고유명사가 인용된 근거에 실제로 있는지 대조합니다.

사용자의 언어로 답하세요."""


def _model(env: str, default: str):
    return init_chat_model(os.getenv(env, default), max_tokens=16000)


def build_agent():
    main_model = _model("RAG_MODEL", "anthropic:claude-opus-5-5")
    # 서브에이전트는 단순 반복 작업이 많아 별도 모델로 바꿀 수 있게 둔다 (기본은 메인과 동일)
    sub_model = _model("SUBAGENT_MODEL", os.getenv("RAG_MODEL", "anthropic:claude-opus-5-5"))

    subagents = [
        {
            "name": "parser-agent",
            "description": "PDF 경로를 받아 페이지 진단 후 텍스트 레이어를 추출한다. OCR 이 필요한 페이지 목록을 보고한다.",
            "system_prompt": PARSER_PROMPT,
            "tools": [inspect_pdf, extract_text_layer, read_page],
            "model": sub_model,
        },
        {
            "name": "ocr-agent",
            "description": "doc_id 와 페이지 목록을 받아 스캔/깨진 페이지를 비전 모델로 OCR 하고 품질을 검수한다.",
            "system_prompt": OCR_AGENT_PROMPT,
            "tools": [ocr_pages, read_page],
            "model": sub_model,
        },
        {
            "name": "chunking-agent",
            "description": "doc_id 를 받아 추출된 텍스트를 청킹하고 검색 인덱스를 만든 뒤 청크 품질을 검수한다.",
            "system_prompt": CHUNKING_PROMPT,
            "tools": [chunk_document, preview_chunks],
            "model": sub_model,
        },
        {
            "name": "retrieval-agent",
            "description": "질문과 doc_id 를 받아 다중 검색어로 관련 청크를 찾고, 출처가 붙은 원문 근거를 돌려준다.",
            "system_prompt": RETRIEVAL_PROMPT,
            "tools": [search_chunks, get_chunk_context],
            "model": sub_model,
        },
    ]

    return create_deep_agent(
        model=main_model,
        tools=[list_documents],
        system_prompt=MAIN_PROMPT,
        subagents=subagents,
    )
