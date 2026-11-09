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
    if name.startswith(("anthropic:", "google_genai:", "openai:")):
        return init_chat_model(name, max_tokens=max_tokens)
    return ChatOpenAI(
        model=name,
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=max_tokens,
    )


def check_config() -> str | None:
    """설정 누락 시 오류 메시지, 정상이면 None."""
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요."
    if not os.getenv("OPENAI_API_BASE"):
        return "OPENAI_API_BASE 가 설정되지 않았습니다. .env 파일을 확인하세요."
    return None
