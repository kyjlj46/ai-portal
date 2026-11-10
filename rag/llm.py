"""LLM 생성 한 곳으로 모음.

기본은 OpenAI 호환 엔드포인트(vLLM 등)를 쓴다:
  OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL
모델 이름에 'anthropic:' 같은 provider 접두어를 붙이면 langchain init_chat_model 로 생성한다.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "openai/gpt-oss-120b"


def default_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def make_model(name: str | None = None, max_tokens: int | None = None):
    name = name or default_model_name()
    max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "8192"))
    # 샘플링 온도: 비우면 서버 기본값. 긴 한국어 문서를 쓸 때 높은 온도에서 글자가 빠지는 오타가 생겨 낮게 쓰는 것을 권장
    extra = {"temperature": float(t)} if (t := os.getenv("LLM_TEMPERATURE")) else {}
    if name.startswith(("anthropic:", "google_genai:", "openai:")):
        return init_chat_model(name, max_tokens=max_tokens, **extra)
    return ChatOpenAI(
        model=name,
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=max_tokens,
        # vLLM(gpt-oss) 스트리밍 응답에서 글자 조각이 빠지는 문제가 있어(일반 응답은 정확) 기본은 끈다.
        # 끄면 답변이 토큰 단위로 흘러나오지 않고 완성된 뒤 한 번에 표시된다. 서버가 고쳐지면 LLM_STREAMING=1.
        disable_streaming=not llm_streaming(),
        **extra,
    )


def make_vision_model():
    """이미지를 볼 수 있는 모델 (VISION_MODEL). 엔드포인트/키를 따로 주지 않으면 OPENAI_* 를 쓴다."""
    name = os.environ["VISION_MODEL"]
    max_tokens = int(os.getenv("MAX_TOKENS", "8192"))
    if name.startswith(("anthropic:", "google_genai:", "openai:")):
        return init_chat_model(name, max_tokens=max_tokens)
    return ChatOpenAI(
        model=name,
        base_url=os.getenv("VISION_API_BASE") or os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY"),
        max_tokens=max_tokens,
        disable_streaming=not llm_streaming(),
    )


def llm_streaming() -> bool:
    return os.getenv("LLM_STREAMING", "0").lower() in ("1", "true", "yes")


def check_config() -> str | None:
    """설정 누락 시 오류 메시지, 정상이면 None."""
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요."
    if not os.getenv("OPENAI_API_BASE"):
        return "OPENAI_API_BASE 가 설정되지 않았습니다. .env 파일을 확인하세요."
    return None
