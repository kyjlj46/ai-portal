"""메인 화면(홈)에 고정한 마켓 Agent 목록.

저장 위치: data/home.json — {"pins": ["translator", "crawler", ...]} (고정한 순서대로)
기존 카드가 있는 마켓 항목(route: qa, summary …)은 이미 홈에 있으므로 고정하지 않는다.
사용자가 만든 Agent 를 지우면 목록을 읽을 때 빠진다.
"""

from __future__ import annotations

import json

from . import store

MAX_PINS = 24


def _path():
    return store.DATA_DIR / "home.json"


def _read() -> list[str]:
    path = _path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return [str(i) for i in data.get("pins", []) if isinstance(i, str)]


def _write(ids: list[str]) -> None:
    _path().parent.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps({"pins": ids}, ensure_ascii=False, indent=2), encoding="utf-8")


def pins() -> list[str]:
    """지금도 마켓에 있는 항목만, 고정한 순서대로."""
    from .market import BY_ID

    return [i for i in dict.fromkeys(_read()) if (m := BY_ID.get(i)) and not m.route]


def pin(item_id: str) -> list[str]:
    from .market import BY_ID

    m = BY_ID.get(item_id)
    if not m:
        raise KeyError(f"알 수 없는 Agent: {item_id}")
    if m.route:
        raise ValueError(f"{m.name}은(는) 이미 메인 화면에 있습니다.")
    ids = pins()
    if item_id not in ids:
        if len(ids) >= MAX_PINS:
            raise ValueError(f"메인 화면에는 Agent 를 {MAX_PINS}개까지 추가할 수 있습니다.")
        ids.append(item_id)
        _write(ids)
    return ids


def unpin(item_id: str) -> list[str]:
    ids = [i for i in pins() if i != item_id]
    _write(ids)
    return ids
