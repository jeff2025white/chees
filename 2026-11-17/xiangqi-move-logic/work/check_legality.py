"""perft(4) 与公认值差 250：定位合法着法过滤/攻击判定中的偏差。

思路：
1. 用「朴素攻击判定」（枚举敌方伪合法着法是否落在将/帅格 + 将帅照面）交叉校验
   is_attacked / in_check；
2. 用「逐着 apply + in_check」的慢速合法着法，交叉校验带敏感格优化的 generate_legal_moves。
遍历初始局面 3 层内的所有局面，打印第一处不一致。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

from engine.board import Board
from engine.constants import COLS, KING, START_FEN, opponent
from engine.movegen import (
    generate_legal_moves,
    generate_pseudo_moves,
    in_check,
    is_attacked,
)

problems = []


def naive_in_check(board, color):
    """朴素判定：把敌方所有伪合法着法的落点拿来对照将/帅格。"""
    ks = board.kings[color]
    if ks < 0:
        return False
    for mv in generate_pseudo_moves(board, opponent(color)):
        if mv[1] == ks:
            return True
    # 将帅照面
    kr, kc = divmod(ks, COLS)
    ok = board.kings[opponent(color)]
    if ok >= 0 and ok % COLS == kc:
        lo, hi = sorted((kr, ok // COLS))
        if all(board.cells[r * COLS + kc] == 0 for r in range(lo + 1, hi)):
            return True
    return False


def slow_legal(board):
    color = board.turn
    res = []
    for mv in generate_pseudo_moves(board, color):
        board.apply(mv)
        ok = not naive_in_check(board, color)
        board.undo()
        if ok:
            res.append(mv)
    return res


def walk(board, depth, path):
    color = board.turn
    fast = generate_legal_moves(board, color)
    slow = slow_legal(board)
    if sorted(fast) != sorted(slow):
        only_fast = [m for m in fast if m not in slow]
        only_slow = [m for m in slow if m not in fast]
        problems.append((list(path), board.to_fen(), only_fast, only_slow))
        return
    # 交叉校验攻击判定
    for color2 in (0, 1):
        if in_check(board, color2) != naive_in_check(board, color2):
            problems.append((list(path), board.to_fen(), "in_check", color2))
            return
    if depth <= 1:
        return
    for mv in fast:
        board.apply(mv)
        path.append(mv)
        walk(board, depth - 1, path)
        path.pop()
        board.undo()
        if len(problems) >= 5:
            return


def main():
    board = Board.from_fen(START_FEN)
    walk(board, 3, [])
    if not problems:
        print("3 层内未发现差异")
        return
    for path, fen, a, b in problems[:5]:
        print("=" * 60)
        print("FEN:", fen)
        print("路径:", path)
        print("仅 fast 有:", a)
        print("仅 slow 有:", b)


if __name__ == "__main__":
    main()
