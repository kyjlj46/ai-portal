"""메인 화면에 카드로 보이는 업무 에이전트 목록.

각 에이전트는 같은 검색 도구(search_chunks 등)를 쓰고 시스템 프롬프트와 입력 양식만 다르다.
  - qa       : 문서 Q&A (채팅)
  - summary  : 문서 요약 (양식 → 결과, 이어서 수정 요청)
  - report   : 보고서 작성 (참고 문서 선택)
  - approval : 결재 기안문 작성 (참고 문서 선택)
  - news     : 뉴스 기사 초안 (신문사 논조 선택, 참고 문서 선택)
  - image    : 이미지 분석 (OCR + LLM, rag/image_agent.py)
  - omok     : 오목 미니게임 (규칙 기반 AI, rag/omok.py)
화면용 정보(이름·설명·양식 필드)는 public() 으로 내보내고, 양식 입력은 compose() 가 요청문으로 바꾼다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

CITE_RULE = """출처 표기 규칙 (참고 문서가 있을 때):
- 문서에서 가져온 사실(숫자, 일정, 금액, 이름, 결정 사항)이 들어간 문장/항목/표의 행 끝에 (p.N) 을 붙입니다. 여러 페이지면 (p.3, p.8).
- 페이지 번호는 검색 결과 머리말 [청크ID p.N ...], 구성표의 [p.N], 전문의 '===== p.N =====' 에서 가져옵니다 (청크ID 끝의 -0007 같은 번호는 페이지가 아닙니다).
- 오직 (p.N) 형식만 씁니다. 【1†L1-L9】, 【】, [] 같은 다른 인용 형식과 청크 ID 는 절대 쓰지 마세요."""

COMMON = """파일 시스템 도구(ls, read_file, write_file, grep, glob 등)는 쓰지 마세요. 문서 내용은 search_chunks, document_outline, read_page, get_chunk_context 로만 확인합니다.
요청 첫 줄이 'doc_id: 없음' 이면 참고 문서가 없는 것입니다. 이때는 도구를 쓰지 말고 사용자가 준 정보만으로 작성합니다.
사용자가 주지 않은 수치·날짜·이름·금액은 지어내지 말고 [확인 필요] 로 표시합니다.
이전에 작성한 결과에 대한 수정 요청이 오면, 요청을 반영한 **전체 문서**를 처음부터 끝까지 다시 출력합니다 (바뀐 부분만 출력하지 마세요).
최종 결과는 마크다운 본문으로만 출력합니다. 앞뒤에 "작성했습니다" 같은 인사말·설명을 붙이지 마세요."""

SUMMARY_PROMPT = f"""당신은 문서 요약 담당입니다. 문서는 이미 OCR·추출되어 인덱싱되어 있습니다.

절차:
1. 요청에 [문서 전문] 이 붙어 있으면 그것을 처음부터 끝까지 읽고 바로 요약합니다 (도구 호출 불필요). 모든 페이지를 고르게 반영하세요.
2. 전문이 없으면(긴 문서) document_outline(doc_id) 로 전체 구성을 보고, 핵심 페이지(개요, 결론, 일정, 예산, 표가 있는 곳)를 read_page 로 읽고,
   요청한 '중점 관점' 이 있으면 search_chunks 로 관련 내용을 더 찾습니다.
3. 요청한 분량과 형식에 맞춰 요약합니다. 원문에 없는 해석·평가를 덧붙이지 마세요. 숫자·일정·금액·고유명사는 원문 그대로 씁니다.

출력 형식:
- 한 줄 요약: `## 한 줄 요약` 아래 한 문장.
- 핵심 요약: `## 한 줄 요약` + `## 핵심 내용` (5~7개 항목) + 필요하면 `## 주요 수치` 표.
- 상세 요약: `## 한 줄 요약` + 문서의 큰 단락(장/절)마다 `## 소제목` 과 항목 3~6개 + `## 주요 수치·일정` 표.
- 개조식이면 '~함, ~임' 으로 끝나는 짧은 항목, 서술형이면 문단으로 씁니다.

{CITE_RULE}

{COMMON}"""

REPORT_PROMPT = f"""당신은 사내 보고서 작성 담당입니다. 사용자가 준 주제·요청 사항과 참고 문서를 바탕으로 결재권자가 바로 읽을 수 있는 보고서를 씁니다.

절차:
1. 참고 문서가 있으면 document_outline(doc_id) 로 구성을 보고, 주제와 관련된 내용을 search_chunks 로 2~5번 찾습니다. 필요하면 read_page 로 페이지 전체를 읽습니다.
2. 찾은 근거와 사용자 입력만으로 보고서를 작성합니다.

보고서 형식 (마크다운):
# <보고서 제목>
> 보고 일자: <작성일> · 보고 대상: <입력값, 없으면 생략> · 보고 유형: <유형>

## 1. 개요
- 보고 목적과 배경 (2~3개 항목)
## 2. 현황 및 주요 내용
- 사실 위주. 수치·일정은 표로 정리
## 3. 분석 및 검토
- 문제점, 리스크, 시사점
## 4. 향후 계획 및 건의 사항
- 누가/언제/무엇을 할지, 의사결정이 필요한 사항

- 분량: '1쪽 요약' 이면 각 장 2~3개 항목, '표준' 이면 3~5개, '상세' 면 장마다 하위 항목과 표를 충분히 씁니다.
- 문체: 개조식('~함', '~임', '~예정'), 한 항목은 한두 줄.
- 보고 유형(업무 보고/결과 보고/검토 보고/주간 보고)에 맞게 장 제목을 조정해도 됩니다. 예: 주간 보고 → 금주 실적 / 차주 계획 / 이슈.
  장은 4개를 유지하고, 제목·장 제목에 이모지나 장식 기호를 쓰지 않습니다. 가로줄(---)도 쓰지 않습니다.
- 문서나 사용자 입력에 없는 권장값·목표치·일정·비용을 새로 만들어 넣지 마세요. 판단이 필요한 값은 "[검토 필요]" 로 두고 무엇을 정해야 하는지만 적습니다.

{CITE_RULE}

{COMMON}"""

APPROVAL_PROMPT = f"""당신은 사내 결재 기안문 작성 담당입니다. 사용자가 준 정보와 참고 문서로 결재 올릴 기안문을 작성합니다.

절차:
1. 참고 문서가 있으면 기안 내용과 관련된 근거(일정, 금액, 대상, 근거 규정 등)를 search_chunks 로 찾습니다.
2. 아래 양식을 그대로 채웁니다. 사용자가 주지 않았고 문서에도 없는 항목은 [입력 필요] 로 둡니다.

기안문 양식 (마크다운, 장 번호와 순서를 지키세요):
# <기안 제목>

| 기안부서 | <부서> | 기안자 | <기안자> | 기안일 | <작성일> |
|---|---|---|---|---|---|

**결재선:** <결재선, 예: 담당 → 팀장 → 본부장>

## 1. 기안 목적
## 2. 추진 배경
## 3. 세부 내용
- 일정, 대상, 방법을 항목 또는 표로
## 4. 소요 예산
| 항목 | 금액 | 산출 근거 |
|---|---|---|
(합계 행 포함. 예산이 없으면 "해당 없음")
## 5. 기대 효과
## 6. 요청 사항
- 마지막 문장: "위와 같이 <요지>하고자 하오니 재가하여 주시기 바랍니다."

**붙임** 1. <붙임 문서>  끝.

- 문체: 공문체 개조식('~함', '~하고자 함'). 과장된 표현을 쓰지 않습니다.
- 금액은 '1,200,000원' 처럼 천 단위 쉼표와 단위를 붙이고, 합계가 항목의 합과 맞는지 확인합니다.

{CITE_RULE}

{COMMON}"""


# 뉴스 기사 논조: 각 매체의 널리 알려진 성향·문체를 '참고' 하는 수준의 안내 (해당 매체를 사칭하지 않는다)
# (화면에 보이는 성향 구분, 모델에 주는 관점·문체 지시) — 모델 지시에는 '보수/진보' 같은 이름표를 넣지 않는다.
# 이름표를 주면 모델이 '보수 입장에서 볼 때' 처럼 본문에 그대로 드러내기 때문.
_OUTLETS: dict[str, tuple[str, str]] = {
    "중립 (스트레이트)": ("가치 판단 없음", "가치 판단을 배제한 사실 보도. 육하원칙, 역피라미드 구조, 짧고 건조한 문장."),
    "조선일보": ("보수 성향", "안보·시장경제·법치를 중시하고, 정책의 재정·규제 부담과 실효성을 따져 묻는다. 단정적이고 힘 있는 문장."),
    "중앙일보": ("중도보수 성향", "데이터와 전문가 분석을 앞세운 해설. 쟁점의 양면을 짚은 뒤 현실적인 대안을 제시한다."),
    "동아일보": ("보수 성향", "사회 현안의 문제를 제기하고 제도 개선을 촉구한다. 정제되고 차분한 문장."),
    "한겨레": ("진보 성향", "노동·인권·환경·사회적 약자의 관점에서 구조적 원인과 공적 책임을 강조한다."),
    "경향신문": ("진보 성향", "민생·불평등·공공성의 관점에서 권력과 정책을 감시하고 비판한다."),
    "한국일보": ("중도 성향", "사실 중심의 균형 보도. 여러 이해관계자의 입장을 나란히 싣고 생활에 미치는 영향을 풀어 쓴다."),
    "매일경제": ("경제지", "산업·시장·투자에 미치는 영향을 중심으로, 성장과 규제 합리화 관점에서 수치를 앞세워 쓴다."),
    "한국경제": ("경제지", "시장경제와 기업 경쟁력 관점에서 규제·비용 부담을 짚고 정책 효과를 따진다."),
    "연합뉴스": ("통신사", "객관적 사실 전달, 출처 명시, 간결한 리드와 역피라미드 구조의 통신 기사체."),
}
OUTLET_TONES: dict[str, str] = {name: f"{lean} · {style}" for name, (lean, style) in _OUTLETS.items()}  # 화면 안내
OUTLET_STYLES: dict[str, str] = {name: style for name, (_, style) in _OUTLETS.items()}  # 모델 지시

NEWS_PROMPT = f"""당신은 뉴스 기사 작성 담당입니다. 사용자가 준 사실(그리고 참고 문서)로 기사 초안을 씁니다.

절차:
1. 참고 문서가 있으면 요청에 붙은 전문이나 search_chunks 로 사실(일정, 수치, 기관, 결정 사항)을 확인합니다.
2. 선택한 '논조 가이드' 에 맞춰 관점·강조점·어휘·문장 길이를 정합니다. 사실 자체는 바꾸지 않습니다.
3. 기사 유형에 맞는 구조로 씁니다:
   - 스트레이트: 핵심을 담은 리드 문단(육하원칙) → 세부 사실 → 배경 → 반응·전망 (역피라미드)
   - 해설·분석: 리드 → 왜 중요한가 → 쟁점별 분석 → 전망
   - 기획·심층: 장면이나 사례로 시작 → 현황 수치 → 원인 → 대안·과제 (소제목 사용)
   - 사설: 논점 제시 → 근거 → 반론 검토 → 주장과 촉구 (신문사의 입장처럼 1인칭 복수 없이 단호하게)
   - 인터뷰 정리: 인물 소개 리드 → 문답 또는 발언 중심 정리 (발언은 입력에 있는 것만)

출력 형식 (마크다운):
# <기사 제목>
**<부제 한 줄>**

<본문 — 문단 사이 빈 줄. 기획·심층이면 '## 소제목' 사용>

> AI 작성 초안 · <선택한 논조> 논조 참고 · 게재 전 사실 확인 필요

규칙:
- 사실은 사용자 입력과 참고 문서에 있는 것만 씁니다. 없는 수치·날짜·기관·인물은 지어내지 말고 [확인 필요] 로 둡니다.
- 실존 인물·기관의 발언은 입력에 있는 인용만 큰따옴표로 씁니다. 그럴듯한 발언을 만들어 넣지 마세요 (필요하면 [인용 필요]).
- 입력에 없는 의도·입장·평가를 정부·기관·인물에게 돌리지 마세요 ('~하려는 의도를 밝혔다', '~라고 평가했다' 금지).
  논조에 따른 해석은 기사 자신의 분석('~할 것으로 보인다', '~라는 지적이 나올 수 있다')으로 씁니다.
- 신문사 제호, '○○일보 기자', 이메일 같은 바이라인을 넣지 마세요. 특정 매체가 실제로 낸 기사처럼 보이게 쓰지 않습니다.
- 논조를 이유로 사실을 왜곡하거나 특정 집단을 비하하지 않습니다. 비판은 근거를 들어서 합니다.
- 논조 가이드는 관점과 문체로만 반영합니다. '보수 언론이 강조하는', '진보적 관점에서' 처럼 가이드 설명을 본문에 드러내지 마세요.
- 참고 문서가 없으면 (p.N) 출처 표기를 쓰지 않습니다.
- 분량: 짧게 ≈600자, 보통 ≈1,200자, 길게 ≈2,000자 (본문 기준).
- 날짜는 요청의 작성일을 기준으로 씁니다.

{CITE_RULE}

{COMMON}"""


@dataclass
class AgentSpec:
    id: str
    name: str
    icon: str
    description: str
    doc: str  # required: 문서 필수, optional: 참고 문서 선택, none: 문서 안 씀
    mode: str  # chat: 질문 채팅, form: 양식 작성 후 수정 요청 채팅, image: 이미지 올려 분석, game: 미니게임
    prompt: str | None  # None 이면 기존 문서 Q&A 프롬프트
    group: str = "문서 작업"  # 왼쪽 메뉴·홈 화면의 분류
    subagents: list[str] = field(default_factory=list)  # 커스텀 앱: 하위 에이전트로 쓸 마켓 항목 id
    version: str = ""  # 바뀌면 만들어 둔 에이전트를 다시 만든다 (앱 편집)
    run: str = "auto"  # 커스텀 앱 실행 방식 — pipeline: 담은 순서대로 코드가 실행, auto: 총괄 에이전트가 판단(DeepAgent)
    output: str = "all"  # pipeline 결과 — all: 모든 단계, last: 마지막 단계 (중간 결과는 접어 둠)
    instructions: str = ""  # 커스텀 앱 지시사항 (pipeline 단계마다 전달)
    request_label: str = "요청"
    intro: str = ""
    fields: list[dict] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    revise_hint: str = ""
    opening: str = ""  # 요청문 첫 줄 (비우면 '<이름> 작성을 요청합니다.')
    preload: bool = False  # 짧은 문서는 전문을 요청에 붙여 보낸다
    reminder: str = ""  # 요청문 끝에 다시 붙이는 핵심 규칙 (모델이 시스템 프롬프트를 흘려듣는 경우 대비)
    followup: str = "지금까지 찾은 근거만으로 요청한 결과물을 마크다운으로 끝까지 작성하세요."


AGENTS: dict[str, AgentSpec] = {
    a.id: a
    for a in [
        AgentSpec(
            id="qa",
            name="문서 Q&A",
            icon="💬",
            description="올린 문서에서 근거를 찾아, 페이지 출처와 함께 질문에 답합니다.",
            doc="required",
            mode="chat",
            prompt=None,
            request_label="질문",
            intro="문서를 고르고 질문하면, 검색 과정을 보여주며 출처(p.N)와 함께 답합니다.",
            examples=["이 문서의 목적을 요약해줘", "주요 일정과 기간은?", "참여 인력과 역할을 표로 정리해줘"],
            followup="지금까지 찾은 근거만으로 사용자 질문에 대한 최종 답변을 작성하세요. 각 항목에 (p.N) 출처를 붙이세요.",
        ),
        AgentSpec(
            id="summary",
            name="문서 요약",
            icon="📝",
            description="문서 전체를 읽고 분량·형식에 맞춰 요약합니다. 관점을 정하면 그 부분을 중심으로 정리합니다.",
            doc="required",
            mode="form",
            prompt=SUMMARY_PROMPT,
            intro="요약할 문서를 고르고 분량과 형식을 정하세요.",
            fields=[
                {"key": "length", "label": "분량", "type": "select", "options": ["핵심 요약", "한 줄 요약", "상세 요약"]},
                {"key": "style", "label": "형식", "type": "select", "options": ["개조식", "서술형"]},
                {"key": "focus", "label": "중점 관점", "type": "text", "placeholder": "선택 · 예: 일정과 예산, 경영진 관점"},
            ],
            revise_hint="예: 더 짧게, 예산 부분을 표로, 영어로 번역",
            preload=True,
        ),
        AgentSpec(
            id="report",
            name="보고서 작성",
            icon="📊",
            description="주제와 요청 사항으로 개요·현황·분석·향후 계획 형식의 보고서를 씁니다. 문서를 고르면 근거로 씁니다.",
            doc="optional",
            mode="form",
            prompt=REPORT_PROMPT,
            intro="보고서 주제와 넣을 내용을 적으세요. 위에서 문서를 고르면 그 문서를 근거로 씁니다.",
            fields=[
                {"key": "title", "label": "보고서 주제", "type": "text", "required": True, "placeholder": "예: 차세대 시스템 구축 진행 현황"},
                {"key": "kind", "label": "보고 유형", "type": "select", "options": ["업무 보고", "결과 보고", "검토 보고", "주간 보고"]},
                {"key": "audience", "label": "보고 대상", "type": "text", "placeholder": "선택 · 예: 본부장, 경영진"},
                {"key": "length", "label": "분량", "type": "select", "options": ["표준", "1쪽 요약", "상세"]},
                {"key": "points", "label": "꼭 넣을 내용 / 요청 사항", "type": "textarea", "placeholder": "예: 일정 지연 원인과 대응 방안을 강조, 인력 현황 표 포함"},
            ],
            revise_hint="예: 3장에 리스크 대응 방안 추가, 더 간결하게",
            preload=True,
            reminder=(
                "작성 규칙: ① '# 제목' 다음 '## 1.'~'## 4.' 네 개 장만 씁니다. ② 이모지·가로줄(---)·작성자 서명·첨부 안내를 넣지 않습니다. "
                "③ 문서와 위 입력에 없는 수치·권장값·목표치·도구·제품명·일정은 쓰지 않고 [검토 필요] 로 둡니다. "
                "④ 문서에서 온 사실마다 (p.N) 을 붙입니다."
            ),
        ),
        AgentSpec(
            id="approval",
            name="결재 기안",
            icon="🖋️",
            description="목적·내용·예산을 입력하면 결재선과 예산표가 들어간 기안문을 사내 양식으로 작성합니다.",
            doc="optional",
            mode="form",
            prompt=APPROVAL_PROMPT,
            intro="기안 정보를 입력하세요. 근거 문서(계획서, 견적서 등)를 고르면 내용과 금액을 문서에서 찾아 채웁니다.",
            fields=[
                {"key": "title", "label": "기안 제목", "type": "text", "required": True, "placeholder": "예: 성능 테스트 도구 라이선스 구매"},
                {"key": "dept", "label": "기안 부서", "type": "text", "placeholder": "예: 플랫폼개발팀"},
                {"key": "drafter", "label": "기안자", "type": "text", "placeholder": "예: 홍길동 책임"},
                {"key": "approvers", "label": "결재선", "type": "text", "default": "담당 → 팀장 → 본부장"},
                {"key": "purpose", "label": "목적 · 배경", "type": "textarea", "required": True, "placeholder": "왜 필요한지, 무엇을 하려는지"},
                {"key": "detail", "label": "세부 내용", "type": "textarea", "placeholder": "선택 · 일정, 대상, 방법, 업체 등"},
                {"key": "budget", "label": "소요 예산", "type": "textarea", "placeholder": "선택 · 예: 라이선스 3식 × 4,000,000원, 교육비 1,500,000원"},
                {"key": "attachments", "label": "붙임", "type": "text", "placeholder": "선택 · 예: 견적서 1부"},
            ],
            revise_hint="예: 예산을 부가세 포함으로, 결재선에 재무팀 협조 추가",
            preload=True,
            reminder=(
                "작성 규칙: ① 기안문 양식의 장 번호와 순서(1. 기안 목적 ~ 6. 요청 사항, 붙임)를 그대로 지킵니다. "
                "② 입력과 문서에 없는 일정·금액·업체·이름은 지어내지 말고 [입력 필요] 로 둡니다. ③ 이모지를 쓰지 않습니다."
            ),
        ),
        AgentSpec(
            id="news",
            name="뉴스 기사 작성",
            icon="📰",
            description="사실과 인용을 입력하면 기사 초안을 씁니다. 주요 신문사 논조와 기사 유형(스트레이트·해설·사설 등)을 고를 수 있습니다.",
            doc="optional",
            mode="form",
            prompt=NEWS_PROMPT,
            opening="뉴스 기사 초안 작성을 요청합니다.",
            intro="논조와 기사 유형을 고르고 사실을 적으세요. 위에서 문서(보도자료, 보고서 등)를 고르면 그 내용을 근거로 씁니다.",
            fields=[
                {"key": "outlet", "label": "논조 (신문사)", "type": "select", "options": list(OUTLET_TONES),
                 "guides": OUTLET_TONES, "prompt_guides": OUTLET_STYLES},
                {"key": "kind", "label": "기사 유형", "type": "select", "options": ["스트레이트", "해설·분석", "기획·심층", "사설", "인터뷰 정리"]},
                {"key": "topic", "label": "주제 · 사건", "type": "text", "required": True, "placeholder": "예: 정부, 내년 AI 반도체 R&D 예산 30% 증액"},
                {"key": "length", "label": "분량", "type": "select", "options": ["보통 (약 1,200자)", "짧게 (약 600자)", "길게 (약 2,000자)"]},
                {"key": "facts", "label": "핵심 사실", "type": "textarea",
                 "placeholder": "언제·어디서·누가·무엇을·왜·수치\n예: 11월 10일 과기정통부 발표, 예산 1조 2천억 원(전년 대비 30%↑), 팹리스 지원 포함"},
                {"key": "quotes", "label": "인용 (발언자: 발언)", "type": "textarea", "placeholder": "선택 · 예: 홍길동 협회장: \"현장의 숨통이 트일 것\""},
            ],
            revise_hint="예: 제목을 더 강하게, 반대 측 입장 추가, 600자로 줄여줘",
            preload=True,
            reminder=(
                "작성 규칙: ① 입력과 문서에 없는 사실·수치·발언은 지어내지 않고 [확인 필요]/[인용 필요] 로 둡니다. "
                "입력에 없는 의도·평가를 정부·기관·인물이 '밝혔다/평가했다' 고 쓰지 않습니다. "
                "'올해·내년·지난해' 같은 날짜 표현은 입력 그대로 씁니다. 논조 가이드 설명(성향·관점 이름)을 본문에 쓰지 않습니다. "
                "② 신문사 제호·기자 바이라인을 넣지 않습니다. ③ 마지막 줄은 '> AI 작성 초안 · <논조> 논조 참고 · 게재 전 사실 확인 필요'."
            ),
        ),
        AgentSpec(
            id="image",
            group="분석",
            name="이미지 분석",
            icon="🖼️",
            description="스크린샷·사진·차트·표 이미지를 올리면 글자를 읽고 내용을 분석합니다. 이어서 질문할 수 있습니다.",
            doc="none",
            mode="image",
            prompt=None,  # rag/image_agent.py 의 IMAGE_PROMPT 를 쓴다
            opening="이미지 분석을 요청합니다.",
            intro="이미지를 끌어다 놓거나, 눌러서 고르거나, 붙여넣기(Ctrl+V) 하세요. 글자·표·수치를 읽어 분석합니다.",
            fields=[
                {"key": "focus", "label": "분석 관점", "type": "select",
                 "options": ["전체 분석", "글자 추출", "표·차트 해석", "문서·화면 검토"]},
                {"key": "question", "label": "궁금한 점", "type": "textarea", "placeholder": "선택 · 예: 오류 메시지의 원인이 뭐야? 합계가 맞아?"},
            ],
            revise_hint="예: 표만 다시 정리해줘, 두 번째 항목 자세히",
        ),
        AgentSpec(
            id="omok",
            group="쉬어가기",
            name="오목 게임",
            icon="⚫",
            description="AI와 15×15 판에서 오목을 둡니다. 흑·백과 난이도(쉬움·보통·어려움)를 고르고, 무르기도 됩니다.",
            doc="none",
            mode="game",
            prompt=None,  # 규칙 기반 AI (rag/omok.py)
            intro="다섯 개를 먼저 잇는 쪽이 이깁니다 (자유룰 · 금수 없음). 흑이 먼저 둡니다.",
        ),
    ]
}


def get(agent_id: str) -> AgentSpec:
    """기본 카드(qa, summary …), 마켓 단독 실행(m:<id>), 커스텀 앱(app:<id>)."""
    if agent_id in AGENTS:
        return AGENTS[agent_id]
    if agent_id.startswith("m:"):
        from .market import spec_for

        return spec_for(agent_id[2:])
    if agent_id.startswith("app:"):
        from .apps import spec_for as app_spec

        return app_spec(agent_id[4:])
    raise KeyError(f"알 수 없는 에이전트: {agent_id}")


def public() -> list[dict]:
    """화면에 보낼 정보 (프롬프트 제외)."""
    keys = ("id", "name", "icon", "description", "doc", "mode", "group", "intro", "fields", "examples", "revise_hint")
    return [{k: getattr(a, k) for k in keys} for a in AGENTS.values()]


def compose(agent_id: str, inputs: dict) -> str:
    """양식 입력을 에이전트에게 보낼 요청문으로 바꾼다. 필수 항목이 비면 ValueError."""
    spec = get(agent_id)
    lines = [spec.opening or f"{spec.name} 작성을 요청합니다.", f"- 작성일: {date.today():%Y-%m-%d}"]
    for f in spec.fields:
        value = str(inputs.get(f["key"]) or "").strip()
        if not value and f.get("required"):
            raise ValueError(f"'{f['label']}' 을(를) 입력하세요.")
        if value:
            lines.append(f"- {f['label']}: {value}" if "\n" not in value else f"- {f['label']}:\n{value}")
            # 선택지별 안내 (예: 신문사 논조). 모델용 지시가 따로 있으면 그것을 쓴다
            if guide := (f.get("prompt_guides") or f.get("guides") or {}).get(value):
                lines.append(f"- {f['label']} 가이드: {guide}")
    if spec.reminder:
        lines.append("\n" + spec.reminder)
    return "\n".join(lines)
