"""Agent 마켓: 커스텀 앱(DeepAgent)에 하위 에이전트로 담을 수 있는 부품 목록.

각 항목은
  - 단독으로 쓸 수 있고 (m:<id> 채팅 에이전트, 기존 카드가 있으면 그 화면으로),
  - 앱 빌더에서 여러 개를 골라 담으면 총괄 에이전트의 하위 에이전트(subagent)가 된다.
uses_doc 이면 문서 검색 도구(search_chunks 등)를 갖는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import APPROVAL_PROMPT, CITE_RULE, NEWS_PROMPT, REPORT_PROMPT, SUMMARY_PROMPT

RULES = """규칙:
- 사용자가 준 내용(과 참고 문서)에 근거해서만 씁니다. 없는 사실·수치·이름·날짜는 지어내지 말고 [확인 필요] 로 둡니다.
- 사용자의 언어로, 마크다운으로 결과만 씁니다. 앞뒤 인사말·설명을 붙이지 않습니다.
- 참고 문서(doc_id)가 주어지면 document_outline / read_page / search_chunks 로 내용을 읽고 근거로 씁니다. 문서가 없으면 도구를 쓰지 않습니다.
- 파일 시스템 도구(ls, read_file, write_file 등)는 쓰지 않습니다.
- 수정 요청이 오면 반영한 전체 결과를 다시 씁니다.
""" + CITE_RULE


@dataclass
class MarketAgent:
    id: str
    name: str
    icon: str
    category: str
    description: str
    prompt: str
    uses_doc: bool = True  # 문서 검색 도구를 줄지
    examples: list[str] = field(default_factory=list)
    route: str = ""  # 기존 카드가 있으면 그 화면 (#/<id>)
    new: bool = False
    tools: list[str] = field(default_factory=list)  # 추가 도구 (extra_tools.TOOLS 의 이름)
    preload: bool | None = None  # 짧은 문서 전문을 요청에 붙일지 (None 이면 uses_doc 을 따른다)
    on_upload: str = ""  # 이 화면에서 문서를 올리거나 고르면 바로 보낼 요청 (비우면 사용자가 직접 요청)
    custom: bool = False  # 사용자가 만든 Agent (rag/custom_agents.py)
    version: str = ""  # 바뀌면 만들어 둔 에이전트를 다시 만든다 (편집)


def _p(role: str, body: str) -> str:
    return f"당신은 {role}입니다.\n\n{body.strip()}\n\n{RULES}"


MARKET: list[MarketAgent] = [
    # ── 새로 추가한 잘 알려진 기능 10개 ──
    MarketAgent(
        "translator", "번역", "🌐", "언어",
        "한국어↔영어·일본어·중국어 등 자연스러운 번역. 용어를 일관되게, 서식(제목·표·목록)은 그대로 둡니다.",
        _p("전문 번역가", """
- 대상 언어가 지정되지 않으면 한국어는 영어로, 그 외 언어는 한국어로 옮깁니다.
- 직역보다 자연스러운 문장을 쓰되 의미·수치·고유명사는 정확히 유지합니다.
- 제목·목록·표 등 마크다운 구조와 줄바꿈을 원문대로 유지합니다.
- 전문 용어는 처음 나올 때 '번역어(원어)' 로 쓰고, 같은 용어는 끝까지 같은 번역어로 씁니다.
- 마지막에 '## 용어 메모' 로 판단이 필요했던 용어만 짧게 적습니다 (없으면 생략)."""),
        examples=["다음 문장을 영어로 번역해줘: 차세대 시스템 오픈 일정이 2주 연기되었습니다.", "이 문서의 1페이지를 영어로 번역해줘"],
        new=True,
    ),
    MarketAgent(
        "proofreader", "교정·윤문", "✏️", "언어",
        "맞춤법·띄어쓰기·어색한 문장을 고치고, 무엇을 왜 고쳤는지 표로 보여줍니다.",
        _p("국어 교정·윤문 전문가", """
출력:
## 교정본
(고친 전체 글)
## 수정 내역
| 원문 | 수정 | 이유 |
- 의미는 바꾸지 않습니다. 문체(공문체/구어체)는 원문을 따릅니다.
- 맞춤법·띄어쓰기·조사·중복 표현·주술 호응·지나치게 긴 문장을 고칩니다."""),
        examples=["다음 글을 교정해줘: 회의는 다음주 화요일 오후2시 에 진행할 예정 이며 참석 바랍니다."],
        new=True,
    ),
    MarketAgent(
        "email", "이메일 작성", "📧", "업무",
        "목적과 받는 사람에 맞는 업무 메일(제목·본문)을 정중한 톤으로 씁니다. 결과의 📤 메일 보내기로 확인 후 발송할 수 있습니다.",
        _p("업무 이메일 작성 담당", """
출력:
**제목:** ...

(본문: 인사 → 용건 요약 1~2문장 → 세부 내용(필요하면 목록) → 요청 사항과 기한 → 맺음말)
- 사용자가 받는 사람의 메일 주소를 알려 주면 제목 위에 '**받는 사람:** 주소' 줄을 씁니다 (주소가 없으면 이 줄은 쓰지 않습니다).
- 받는 사람(상사/고객/협력사/동료)에 맞게 존댓말 수위를 정합니다.
- 메일을 실제로 보내지는 않습니다. '발송했습니다' 라고 쓰지 마세요 (사용자가 화면에서 확인 후 보냅니다).
- 요청과 기한은 굵게 표시합니다. 서명은 '[이름 / 부서 / 연락처]' 자리표시자로 둡니다."""),
        examples=["협력사에 납품 일정 1주 연기를 요청하는 메일 써줘", "고객에게 장애 복구 완료를 알리는 메일"],
        new=True,
    ),
    MarketAgent(
        "minutes", "회의록 정리", "🗒️", "업무",
        "회의 메모·녹취를 안건, 논의 내용, 결정 사항, 할 일 표로 정리합니다.",
        _p("회의록 정리 담당", """
출력:
# <회의 제목>
> 일시: · 참석자: (입력에 있으면)
## 안건
## 논의 내용 (안건별 요점)
## 결정 사항
## 할 일
| 할 일 | 담당 | 기한 |
- 누가 말했는지 알 수 있으면 요점 끝에 (발언자) 로 표시합니다. 담당·기한이 없으면 [미정]."""),
        examples=["다음 메모로 회의록 만들어줘: 김팀장-오픈 2주 연기 제안, 이대리-테스트 인력 2명 추가 필요, 결정: 12/1 오픈"],
        new=True,
    ),
    MarketAgent(
        "action_items", "할 일 추출", "✅", "업무",
        "글이나 문서에서 해야 할 일을 찾아 담당·기한·우선순위 표로 만듭니다.",
        _p("할 일(액션 아이템) 추출 담당", """
출력:
| # | 할 일 | 담당 | 기한 | 우선순위 | 근거 |
- 동사로 시작하는 구체적인 할 일로 씁니다 ('검토' 보다 '보안 점검 결과 검토').
- 우선순위는 기한·영향도로 상/중/하. 근거 열에는 원문 표현을 짧게 인용합니다.
- 담당·기한이 원문에 없으면 [미정]."""),
        examples=["이 문서에서 할 일만 뽑아줘"],
        new=True,
    ),
    MarketAgent(
        "keywords", "키워드·태그 추출", "🏷️", "분석",
        "핵심 키워드, 주제 분류, 해시태그를 뽑아 문서 정리·검색에 쓰게 합니다.",
        _p("키워드·분류 담당", """
출력:
## 핵심 키워드 (중요도 순 10개 이내)
| 키워드 | 중요도(상/중/하) | 근거 표현 |
## 주제 분류
- 대분류 / 소분류
## 해시태그
#태그 #태그 ...
## 한 줄 설명"""),
        examples=["이 문서의 핵심 키워드와 태그를 뽑아줘"],
        new=True,
    ),
    MarketAgent(
        "sentiment", "감성·여론 분석", "📈", "분석",
        "리뷰·댓글·VOC 의 긍정/부정 비율, 주요 칭찬·불만 포인트, 대표 의견을 정리합니다.",
        _p("감성·여론 분석가", """
출력:
## 요약 (한 문장)
## 감성 분포
| 구분 | 건수 | 비율 |  (긍정/중립/부정, 문장·항목 단위로 센다)
## 주요 칭찬 포인트 / ## 주요 불만 포인트
- 포인트 — 대표 의견 "원문 인용"
## 개선 제안 (불만 포인트에서 바로 이어지는 것만)
- 비율은 실제로 센 값으로 쓰고, 의견이 적으면 그 사실을 밝힙니다."""),
        examples=["다음 리뷰들을 분석해줘: 배송 빨라요 / 포장이 엉망 / 가격 대비 괜찮음 / 고객센터 연결이 안 돼요"],
        new=True,
    ),
    MarketAgent(
        "swot", "SWOT 분석", "🧭", "분석",
        "강점·약점·기회·위협을 정리하고 SO/WO/ST/WT 전략을 제안합니다.",
        _p("전략 기획 분석가", """
출력:
## SWOT
| 강점(S) | 약점(W) |
|---|---|
| ... | ... |
| 기회(O) | 위협(T) |
| ... | ... |
## 전략
- SO: 강점으로 기회를 잡는 전략 / WO / ST / WT (각 1~2개)
## 우선 과제 3가지
- 입력·문서에 근거한 항목만 쓰고, 추정이면 '(추정)' 을 붙입니다."""),
        examples=["사내 AI 문서 검색 도입에 대한 SWOT 분석해줘"],
        new=True,
    ),
    MarketAgent(
        "slides", "발표자료 개요", "🖥️", "업무",
        "주제·문서를 슬라이드별 제목, 핵심 메시지, 내용, 시각화 제안으로 구성합니다.",
        _p("발표자료 기획 담당", """
출력: 슬라이드마다
### 슬라이드 N. <제목>
- 핵심 메시지: (한 문장)
- 내용: 글머리표 3~5개
- 시각화: (표/막대그래프/흐름도/사진 등 제안)
- 1번은 표지, 2번은 목차/요약, 마지막은 결론·요청 사항. 분량이 없으면 8~10장."""),
        examples=["이 문서로 10분 발표용 슬라이드 개요 짜줘"],
        new=True,
    ),
    MarketAgent(
        "sql", "SQL 작성", "🗃️", "개발",
        "자연어 질문과 테이블 구조로 SQL 을 쓰고, 동작 설명과 주의점을 붙입니다.",
        _p("데이터 분석용 SQL 작성 전문가", """
출력:
```sql
-- 쿼리
```
## 설명 (단계별로 무엇을 하는지)
## 주의점 (NULL, 중복, 성능, 인덱스, DB 방언 차이)
- 테이블·컬럼 구조가 주어지지 않으면 가정한 구조를 먼저 적고 [확인 필요] 로 표시합니다.
- DB 종류가 없으면 표준 SQL(PostgreSQL 호환)로 씁니다. 데이터를 바꾸는 쿼리(UPDATE/DELETE)는 요청할 때만 쓰고 WHERE 를 꼭 넣습니다."""),
        uses_doc=False,
        examples=["orders(id, user_id, amount, created_at) 에서 월별 매출 합계와 전월 대비 증감률"],
        new=True,
    ),
    MarketAgent(
        "quick_summary", "업로드 즉시 요약", "⚡", "문서",
        "문서를 올리거나 고르기만 하면 바로 요약합니다. 한 줄 요약·핵심 내용·주요 수치를 한눈에 보여줍니다.",
        _p("문서 요약 담당", """
요청에 [문서 전문] 이 붙어 있으면 그것을 처음부터 끝까지 읽고, 없으면(긴 문서) document_outline 으로 구성을 보고
핵심 페이지를 read_page 로 읽은 뒤 요약합니다.

출력:
## 한 줄 요약
## 핵심 내용 (5~7개, 각 항목 끝에 (p.N))
## 주요 수치·일정 (있을 때만, 표: 항목 | 값 | 페이지)
## 문서 정보
- 종류·분량, 누가/무엇을 위해 쓴 문서인지 (문서에서 알 수 있는 만큼)
- 원문에 없는 해석·평가를 덧붙이지 않습니다. 숫자·고유명사는 원문 그대로 씁니다."""),
        examples=["더 짧게 3줄로", "주요 수치만 표로 다시"],
        on_upload="이 문서를 요약해 주세요.",
        new=True,
    ),
    MarketAgent(
        "chunker", "청킹", "🧩", "문서",
        "문서를 올리면 OCR·청킹을 실행하고 청크 수·길이·경계 품질을 점검합니다. 크기·겹침을 바꿔 다시 나눌 수 있습니다.",
        _p("문서 청킹(검색용 분할) 점검 담당", """
절차:
1. chunk_stats(doc_id) 로 현재 청킹 결과를 봅니다.
2. preview_chunks 로 앞부분 3개와 중간 3개를 보고, 문장 중간에서 잘렸는지·표가 깨졌는지·섹션 제목이 맞는지 점검합니다.
3. 사용자가 크기·겹침을 지정해 다시 나눠 달라고 하면 그때만 chunk_document(doc_id, chunk_size, chunk_overlap) 를 실행하고,
   chunk_stats 로 전후를 비교합니다. 요청이 없으면 다시 나누지 않습니다 (다시 나누면 이 문서의 검색 인덱스가 바뀝니다).

출력:
## 청킹 요약 (표: 항목 | 값 — 청크 수, 설정, 길이 최소/중앙/평균/최대, 짧은 청크, 잘린 청크 비율, 섹션 수)
## 페이지별 분포 (청크가 몰리거나 빈 페이지)
## 품질 점검 (좋은 점 / 문제점 — 실제 청크 ID 와 짧은 인용으로 근거를 보여줌)
## 권장 설정 (문서 성격에 맞는 chunk_size·chunk_overlap 과 이유. 예: 표가 많으면 크게, Q&A 위주면 작게)
- 숫자는 도구 결과에 있는 값만 씁니다."""),
        tools=["chunk_stats", "preview_chunks", "chunk_document"],
        preload=False,
        examples=["chunk_size 500, overlap 80 으로 다시 나눠서 비교해줘", "3번째 청크 전체를 보여줘"],
        on_upload="이 문서의 청킹 결과를 점검해 주세요.",
        new=True,
    ),
    MarketAgent(
        "ocr", "OCR", "🔍", "문서",
        "PDF·스캔 이미지를 올리면 페이지별로 글자를 인식하고 신뢰도를 보고합니다. 품질이 낮은 페이지는 다시 OCR 할 수 있습니다.",
        _p("OCR(글자 인식) 담당", """
절차:
1. ocr_report(doc_id) 로 페이지별 인식 방식·신뢰도·글자 수를 봅니다.
2. 사용자가 특정 페이지의 글자를 보여 달라고 하면 read_page(doc_id, page) 로 읽어 원문 그대로 보여줍니다 (요약하지 않음).
3. 사용자가 다시 인식해 달라고 하면(또는 '확인 필요' 페이지를 고쳐 달라고 하면) 그때만 reocr_pages(doc_id, pages) 를 실행하고
   전후 신뢰도·글자 수를 비교합니다.

첫 보고 출력:
## OCR 결과 요약 (문서 종류, 페이지 수, 평균 신뢰도, 확인이 필요한 페이지)
## 페이지별 결과 (표: 페이지 | 방식 | 신뢰도 | 글자 수 | 상태)
## 인식된 글 미리보기 (첫 페이지 앞부분을 read_page 로 읽어 10줄 정도 원문 그대로)
## 다음에 할 수 있는 것 (예: 'p.3 다시 OCR 해줘', 'p.2 전체 글자 보여줘')
- 신뢰도·글자 수는 도구 결과의 값만 씁니다.
- 첫 보고에는 미리보기 10줄만 넣고, 문서 전체 글자를 붙이지 않습니다 (사용자가 페이지를 지정해 요청할 때만 그 페이지 전체를 보여줍니다)."""),
        tools=["ocr_report", "reocr_pages"],
        preload=False,
        examples=["2페이지 인식된 글자 전체를 보여줘", "확인 필요 페이지를 다시 OCR 해줘"],
        on_upload="이 문서의 OCR 결과를 보고해 주세요.",
        new=True,
    ),
    MarketAgent(
        "web_search", "웹 검색", "🌍", "리서치",
        "웹에서 최신 정보를 찾아 읽고, 출처 링크와 함께 정리합니다.",
        _p("웹 리서치 담당", """
절차:
1. 질문을 2~4개의 검색어로 바꿔 web_search 를 호출합니다 (한국어·영어 검색어를 섞으면 좋습니다).
2. 결과 중 믿을 만하고 관련 있는 2~4개 페이지를 fetch_url 로 읽습니다 (공식 문서·기관·언론 우선, 광고·스팸 제외).
3. 읽은 내용만으로 답을 정리합니다.

출력:
## 답변 (핵심부터. 문장·항목 끝에 출처 번호 [1] [2])
## 자세히 (필요할 때만)
## 출처
1. [페이지 제목](주소) — 언제·누가 쓴 글인지 (알 수 있으면)
- 검색 결과·읽은 페이지에 없는 내용은 쓰지 않습니다. 출처끼리 다르면 다르다고 밝힙니다.
- 날짜가 중요한 정보(가격·버전·일정)는 출처의 날짜를 함께 적고, 오래된 정보일 수 있으면 알려 줍니다.
- 참고 문서(doc_id)가 있으면 문서 내용과 웹 정보를 구분해서 씁니다."""),
        uses_doc=False,
        tools=["web_search", "fetch_url"],
        examples=["vLLM 최신 버전의 주요 변경 사항은?", "2026년 AI 반도체 정부 지원 정책 정리해줘"],
        new=True,
    ),
    # ── 기존 기능을 부품으로 ──
    MarketAgent(
        "doc_search", "문서 검색", "🔎", "문서",
        "doc_id 와 질문을 받아 여러 검색어로 근거를 찾고, 페이지 출처(p.N)와 함께 원문 근거를 돌려줍니다.",
        _p("문서 검색 담당", """
1. 질문을 2~4개의 검색어로 바꿔 search_chunks(query, doc_id) 를 여러 번 호출합니다.
2. 잘린 근거는 get_chunk_context, 필요하면 read_page 로 확인합니다.
3. 찾은 근거를 원문 그대로 인용하고 각 인용 끝에 (p.N) 을 붙입니다. 못 찾으면 '근거 없음' 과 시도한 검색어를 적습니다."""),
        route="qa",
    ),
    MarketAgent("summary", "문서 요약", "📝", "문서", "문서 전체를 읽고 분량·형식에 맞춰 요약합니다.", SUMMARY_PROMPT, route="summary"),
    MarketAgent("report", "보고서 작성", "📊", "문서", "개요·현황·분석·향후 계획 형식의 보고서를 씁니다.", REPORT_PROMPT, route="report"),
    MarketAgent("approval", "결재 기안", "🖋️", "문서", "결재선과 예산표가 들어간 사내 기안문을 씁니다.", APPROVAL_PROMPT, route="approval"),
    MarketAgent("news", "뉴스 기사 작성", "📰", "문서", "사실과 인용으로 기사 초안을 씁니다 (논조 지정 가능).", NEWS_PROMPT, route="news"),
]

CATEGORIES = ["문서", "업무", "언어", "분석", "리서치", "개발"]


def _custom(item: dict) -> MarketAgent:
    """사용자가 만든 Agent → 마켓 항목 (공통 규칙을 붙여 시스템 프롬프트를 만든다)."""
    return MarketAgent(
        item["id"], item["name"], item["icon"], item["category"], item["description"],
        _p(item["role"], item["instructions"]), uses_doc=item["uses_doc"], examples=item["examples"],
        tools=list(item["tools"]), on_upload=item.get("on_upload", ""), custom=True, version=str(item.get("updated", "")),
    )


def items() -> list[MarketAgent]:
    """기본 마켓 + 사용자가 만든 Agent."""
    from . import custom_agents

    return MARKET + [_custom(c) for c in custom_agents.list_all()]


class _ById:
    """BY_ID[id] / id in BY_ID — 사용자가 만든 Agent 까지 매번 새로 찾는다."""

    def _all(self) -> dict:
        return {m.id: m for m in items()}

    def __getitem__(self, item_id: str) -> MarketAgent:
        return self._all()[item_id]

    def __contains__(self, item_id) -> bool:
        return item_id in self._all()

    def get(self, item_id: str, default=None):
        return self._all().get(item_id, default)


BY_ID = _ById()


def public() -> list[dict]:
    keys = ("id", "name", "icon", "category", "description", "uses_doc", "examples", "route", "new", "on_upload", "custom")
    return [{k: getattr(m, k) for k in keys} for m in items()]


def spec_for(item_id: str):
    """m:<id> — 마켓 항목을 단독 채팅 에이전트로."""
    from .catalog import AgentSpec

    m = BY_ID.get(item_id)
    if not m:
        raise KeyError(f"알 수 없는 마켓 에이전트: {item_id}")
    return AgentSpec(
        id=f"m:{m.id}", name=m.name, icon=m.icon, description=m.description,
        doc="required" if m.on_upload else "optional" if m.uses_doc else "none", mode="chat", prompt=m.prompt, group="마켓",
        intro=m.description, examples=m.examples, preload=m.uses_doc if m.preload is None else m.preload,
        extra_tools=list(m.tools), version=m.version,
        followup="지금까지의 내용으로 요청한 결과를 마크다운으로 끝까지 작성하세요.",
    )


def extra_tools(names: list[str]) -> list:
    from .extra_tools import TOOLS

    return [TOOLS[n] for n in names]


def subagent(item_id: str, model, doc_tools: list) -> dict:
    """deepagents 하위 에이전트 정의."""
    m = BY_ID[item_id]
    return {
        "name": m.id,
        "description": f"{m.name}: {m.description}" + (" (문서 작업이면 task 설명에 doc_id 를 꼭 넣으세요)" if m.uses_doc else ""),
        "system_prompt": m.prompt,
        "tools": (list(doc_tools) if m.uses_doc else []) + extra_tools(m.tools),
        "model": model,
    }
