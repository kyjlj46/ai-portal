# pdf-rag-agent

[LangChain deep-agents](https://github.com/langchain-ai/deepagents)로 만든 간단한 PDF RAG입니다.
파싱, OCR, 청킹, 검색을 서로 다른 서브에이전트에 나눠 맡깁니다. 각 서브에이전트는 자기 단계의 도구만 가지고 결과를 직접 확인하므로, 단계가 섞이지 않아 답이 더 정확해집니다.

```
main agent (진행 담당)
 ├─ parser-agent    : inspect_pdf → extract_text_layer → read_page로 확인
 ├─ ocr-agent       : ocr_pages(Claude 비전) → 품질이 낮으면 dpi=300으로 재시도
 ├─ chunking-agent  : chunk_document → preview_chunks로 확인, 필요하면 크기를 바꿔 다시 청킹
 └─ retrieval-agent : 검색어를 여러 개로 바꿔 search_chunks → 필요하면 get_chunk_context
```

정확도를 높이는 방법:

- **페이지별 OCR 판단:** 텍스트 레이어가 거의 없거나 글자가 깨진 페이지만 OCR합니다.
- **청크 메타데이터:** 청크마다 페이지 번호와 섹션 제목을 저장해 답에 출처 `(p.N)`를 붙입니다.
- **검색:** BM25L에 한글 2-gram을 더해, 임베딩 API 없이도 띄어쓰기가 달라진 표현까지 찾습니다.
- **답변 전 확인:** 숫자와 고유명사가 근거 원문에 실제로 있는지 대조하고, 근거가 없으면 "찾을 수 없음"으로 답합니다.

## 설치

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL 입력
```

## 실행

```bash
uv run web.py                          # 웹 UI: http://127.0.0.1:8000
uv run main.py 보고서.pdf "질문"        # CLI로 질문 한 번
uv run main.py 보고서.pdf               # CLI 대화형
```

웹 UI에서 PDF를 업로드하고 질문하면, 답변과 함께 호출된 서브에이전트 목록이 표시됩니다.
처음 질문할 때 문서 수집(파싱, OCR, 청킹)이 자동으로 실행됩니다.

환경변수:

| 이름 | 기본값 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | (필수) | OpenAI 호환 엔드포인트 키 |
| `OPENAI_API_BASE` | (필수) | 엔드포인트 주소 (예: `http://host:port/v1`) |
| `OPENAI_MODEL` | `openai/gpt-oss-120b` | 기본 모델 |
| `RAG_MODEL` | `OPENAI_MODEL` | 메인 에이전트 모델 |
| `SUBAGENT_MODEL` | `RAG_MODEL` | 서브에이전트 모델 |
| `OCR_MODEL` | `RAG_MODEL` | OCR에 쓰는 비전 모델 (이미지 입력 지원 필요) |
| `MAX_TOKENS` | `8192` | 응답 최대 토큰 |

모델 이름에 `anthropic:claude-...`처럼 provider 접두어를 붙이면 해당 provider로 생성합니다.
| `PORT` | `8000` | 웹 서버 포트 |
| `NO_BROWSER` | (없음) | `1`이면 브라우저를 자동으로 열지 않음 |

## 테스트

```bash
uv run pytest
```

LLM은 호출하지 않습니다. OCR 모델과 에이전트는 가짜 객체로 바꿔 진단, 추출, 청킹, 검색, 웹 API를 검증합니다.

## 구조

- `rag/tools.py`: 단계별 도구
- `rag/agent.py`: `create_deep_agent`와 서브에이전트 정의
- `rag/store.py`: 페이지와 청크를 `data/`에 JSON으로 저장
- `web.py`, `static/index.html`: FastAPI 웹 UI
- `main.py`: CLI
