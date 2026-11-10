"""오목 AI (15×15, 자유룰: 다섯 개 이상 이으면 승리, 금수 없음).

판 표현: board[r][c] — 0 빈칸, 1 흑, 2 백. 흑이 먼저 둔다.
AI 는 '5칸 창(window)' 평가로 각 빈칸의 공격/수비 점수를 매긴다:
  어느 방향이든 그 칸을 포함하는 5칸 창마다, 상대 돌이 없으면 내 돌 수에 따라 공격 점수,
  내 돌이 없으면 상대 돌 수에 따라 막는 점수를 더한다. 띈 모양(●●_●●)도 자연스럽게 잡힌다.
어려움은 상위 후보에 대해 상대의 최선 응수까지 한 수 더 내다본다.
"""

from __future__ import annotations

import random

SIZE = 15
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
LEVELS = ("쉬움", "보통", "어려움")

# 창 안에 놓은 뒤의 내 돌 수 → 공격 점수 (5 = 승리)
ATTACK = (0, 1, 10, 100, 2_500, 10_000_000)
# 창 안의 상대 돌 수 → 막는 점수 (4 = 안 막으면 진다)
DEFENSE = (0, 1, 8, 90, 1_000_000, 0)


def empty_board() -> list[list[int]]:
    return [[0] * SIZE for _ in range(SIZE)]


def _inside(r: int, c: int) -> bool:
    return 0 <= r < SIZE and 0 <= c < SIZE


def five_at(board: list[list[int]], r: int, c: int) -> list[tuple[int, int]] | None:
    """(r, c) 의 돌이 다섯 개 이상 이어졌으면 그 줄의 좌표들, 아니면 None."""
    color = board[r][c]
    if not color:
        return None
    for dr, dc in DIRS:
        line = [(r, c)]
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while _inside(rr, cc) and board[rr][cc] == color:
                line.append((rr, cc))
                rr, cc = rr + dr * sign, cc + dc * sign
        if len(line) >= 5:
            return sorted(line)
    return None


def winner(board: list[list[int]]) -> tuple[int, list[tuple[int, int]]] | None:
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] and (line := five_at(board, r, c)):
                return board[r][c], line
    return None


def side_to_move(board: list[list[int]]) -> int:
    black = sum(row.count(1) for row in board)
    white = sum(row.count(2) for row in board)
    return 1 if black == white else 2


def validate(board) -> str | None:
    """판 형식/돌 개수 검사. 문제가 있으면 오류 메시지."""
    if not isinstance(board, list) or len(board) != SIZE or any(not isinstance(row, list) or len(row) != SIZE for row in board):
        return f"판은 {SIZE}×{SIZE} 이어야 합니다."
    if any(v not in (0, 1, 2) for row in board for v in row):
        return "칸 값은 0(빈칸), 1(흑), 2(백) 중 하나여야 합니다."
    black = sum(row.count(1) for row in board)
    white = sum(row.count(2) for row in board)
    if black - white not in (0, 1):
        return "흑과 백의 돌 수가 맞지 않습니다 (흑이 먼저 둡니다)."
    return None


def _cell_score(board: list[list[int]], r: int, c: int, me: int, defense_weight: float = 1.0) -> float:
    opp = 3 - me
    attack = defense = 0.0
    for dr, dc in DIRS:
        for start in range(-4, 1):
            cells = [(r + dr * (start + i), c + dc * (start + i)) for i in range(5)]
            if not all(_inside(rr, cc) for rr, cc in cells):
                continue
            mine = theirs = 0
            for rr, cc in cells:
                v = board[rr][cc]
                if v == me:
                    mine += 1
                elif v == opp:
                    theirs += 1
            if theirs == 0:
                attack += ATTACK[mine + 1]
            if mine == 0:
                defense += DEFENSE[theirs]
    center = SIZE // 2
    return attack + defense * defense_weight - 0.01 * (abs(r - center) + abs(c - center))


def _candidates(board: list[list[int]]) -> list[tuple[int, int]]:
    stones = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c]]
    if not stones:
        return [(SIZE // 2, SIZE // 2)]
    near = set()
    for r, c in stones:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if _inside(rr, cc) and not board[rr][cc]:
                    near.add((rr, cc))
    return sorted(near)


def _ranked(board, me: int, defense_weight: float = 1.0) -> list[tuple[float, tuple[int, int]]]:
    return sorted(((_cell_score(board, r, c, me, defense_weight), (r, c)) for r, c in _candidates(board)), reverse=True)


def ai_move(board: list[list[int]], me: int, level: str = "보통", rng: random.Random | None = None) -> tuple[int, int]:
    rng = rng or random.Random()
    if not any(any(row) for row in board):
        center = SIZE // 2
        return (center, center) if level != "쉬움" else (center + rng.choice((-1, 0, 1)), center + rng.choice((-1, 0, 1)))

    if level == "쉬움":
        ranked = _ranked(board, me, defense_weight=0.6)
        if ranked[0][0] >= ATTACK[5]:  # 이길 수는 놓치지 않는다
            return ranked[0][1]
        top = ranked[:6]
        return rng.choices([m for _, m in top], weights=[max(s, 1) for s, _ in top])[0]

    ranked = _ranked(board, me)
    if level != "어려움" or ranked[0][0] >= ATTACK[5] or ranked[0][0] >= DEFENSE[4]:
        best = ranked[0][0]
        return rng.choice([m for s, m in ranked if s >= best - 1e-9])

    # 어려움: 상위 후보마다 상대의 최선 응수를 보고 고른다
    opp = 3 - me
    best_value, best_moves = None, []
    for score, (r, c) in ranked[:8]:
        board[r][c] = me
        try:
            reply = _ranked(board, opp)
            opp_best = reply[0][0] if reply else 0.0
        finally:
            board[r][c] = 0
        value = score - 0.7 * opp_best
        if best_value is None or value > best_value + 1e-9:
            best_value, best_moves = value, [(r, c)]
        elif abs(value - best_value) <= 1e-9:
            best_moves.append((r, c))
    return rng.choice(best_moves)


def play(board: list[list[int]], ai: int, level: str = "보통") -> dict:
    """사람이 둔 뒤의 판을 받아, 승부가 났는지 보고 아니면 AI 가 한 수 둔다.

    반환: {"move": [r, c] | None, "winner": 0|1|2, "line": [[r, c], ...], "draw": bool}
    """
    if found := winner(board):
        color, line = found
        return {"move": None, "winner": color, "line": [list(p) for p in line], "draw": False}
    if all(all(row) for row in board):
        return {"move": None, "winner": 0, "line": [], "draw": True}
    r, c = ai_move(board, ai, level if level in LEVELS else "보통")
    board[r][c] = ai
    line = five_at(board, r, c)
    full = all(all(row) for row in board)
    return {
        "move": [r, c],
        "winner": ai if line else 0,
        "line": [list(p) for p in line] if line else [],
        "draw": bool(full and not line),
    }
