"""사용법:
    uv run main.py <pdf경로> "질문"        # 한 번 질문
    uv run main.py <pdf경로>               # 대화형 (빈 줄/exit 로 종료)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 특수문자 출력 오류 방지

from rag.agent import build_agent  # noqa: E402


def ask(agent, history: list, text: str) -> str:
    history.append({"role": "user", "content": text})
    result = agent.invoke({"messages": history})
    history[:] = result["messages"]
    return result["messages"][-1].text


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pdf = Path(sys.argv[1]).resolve()
    if not pdf.exists():
        sys.exit(f"파일 없음: {pdf}")

    agent = build_agent()
    history: list = []
    prefix = f"PDF 파일: {pdf}\n(인덱싱이 안 되어 있으면 먼저 수집하세요)\n\n질문: "

    if len(sys.argv) >= 3:
        print(ask(agent, history, prefix + sys.argv[2]))
        return

    question = input("질문> ").strip()
    first = True
    while question and question.lower() not in {"exit", "quit"}:
        print("\n" + ask(agent, history, (prefix if first else "") + question) + "\n")
        first = False
        question = input("질문> ").strip()


if __name__ == "__main__":
    main()
