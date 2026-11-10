"""사용법:
    uv run main.py <문서경로> "질문"       # 한 번 질문 (PDF/Word/한글/텍스트)
    uv run main.py <문서경로>              # 대화형 (빈 줄/exit 로 종료)

처리 단계(문서 수집, 검색어, 찾은 근거, 답변 작성)를 실시간으로 보여준 뒤 답변과 출처를 출력한다.
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 특수문자 출력 오류 방지

from rag.stream import ask_stream  # noqa: E402

ICON = {"ingest": "📄", "think": "🧠", "search": "🔎", "read": "📖", "delegate": "🤝", "result": "  ↳", "answer": "✍️"}


def ask(path: str, question: str, history: list) -> str:
    t0 = time.time()
    for ev in ask_stream(path, question, history):
        if ev["type"] == "step":
            indent = "    " if ev["sub"] else ""
            print(f"\033[90m{indent}{ICON.get(ev['kind'], '•')} {ev['text']}  ({time.time() - t0:.1f}s)\033[0m", flush=True)
        elif ev["type"] == "error":
            return f"오류: {ev['message']}"
        elif ev["type"] == "final":
            answer = ev["answer"]
            if ev["sources"]:
                answer += "\n\n출처:\n" + "\n".join(
                    f"  - p.{s['page']}" + (f" ({s['section']})" if s["section"] else "")
                    + f": {s['snippet'][:80].replace(chr(10), ' ')}…"
                    for s in ev["sources"]
                )
            return answer
    return "답변을 받지 못했습니다."


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1]).resolve()
    if not path.exists():
        sys.exit(f"파일 없음: {path}")

    history: list = []
    if len(sys.argv) >= 3:
        print("\n" + ask(str(path), sys.argv[2], history))
        return

    question = input("질문> ").strip()
    while question and question.lower() not in {"exit", "quit"}:
        print("\n" + ask(str(path), question, history) + "\n")
        question = input("질문> ").strip()


if __name__ == "__main__":
    main()
