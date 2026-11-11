"""오목 AI: 승리 판정, 이기는 수/막는 수, 난이도 강도."""

import random

from portal import omok


def test_five_at_and_overline():
    b = omok.empty_board()
    for c in range(2, 8):  # 6목도 승리 (자유룰)
        b[3][c] = 2
    assert omok.five_at(b, 3, 4) and len(omok.five_at(b, 3, 4)) == 6
    assert omok.winner(b)[0] == 2
    b2 = omok.empty_board()
    for i in range(5):
        b2[i][4 - i] = 1  # 역대각선
    assert omok.winner(b2)[0] == 1


def test_ai_takes_win_before_blocking():
    b = omok.empty_board()
    for c in range(0, 4):
        b[5][c] = 2  # 백 4 (한쪽 막힘 없음: (5,4))
        b[9][c + 6] = 1  # 흑도 4
    b[12][12] = 1
    assert omok.ai_move(b, 2, "보통") == (5, 4)  # 막기보다 이기기


def test_ai_blocks_four_and_broken_four():
    b = omok.empty_board()
    for c in (3, 4, 6, 7):  # 흑 ●●_●● (띈 4)
        b[7][c] = 1
    for c in (0, 2, 4, 6):
        b[0][c] = 2
    for level in ("보통", "어려움"):
        assert omok.ai_move(b, 2, level) == (7, 5)


def test_levels_ordered_by_strength():
    def match(black, white, seed):
        rng = random.Random(seed)
        b = omok.empty_board()
        for n in range(225):
            color = 1 if n % 2 == 0 else 2
            r, c = omok.ai_move(b, color, black if color == 1 else white, rng)
            b[r][c] = color
            if omok.five_at(b, r, c):
                return color
        return 0

    hard_wins = sum(match("어려움", "쉬움", s) == 1 for s in range(4)) + sum(match("쉬움", "어려움", s) == 2 for s in range(4))
    assert hard_wins >= 6  # 8판 중 6판 이상


def test_validate():
    assert omok.validate([[0] * 15] * 14)
    b = omok.empty_board()
    b[0][0] = 2  # 백이 먼저 둘 수 없음
    assert omok.validate(b)
    b[0][1] = 1
    assert omok.validate(b) is None and omok.side_to_move(b) == 1
