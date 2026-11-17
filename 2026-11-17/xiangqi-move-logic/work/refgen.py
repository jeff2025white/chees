"""独立实现一份「最朴素」的着法生成与攻击判定，用于定位 perft(4) 的 +250 偏差。

这份代码按规则直译，不复用 engine.movegen 的任何逻辑（只借用 Board 存取与 apply/undo）。
对 ply=3 的每个局面，分别用引擎与本文件统计合法着法数并求和（= perft(4)），
遇到第一处不一致就打印局面与两边的差异着法。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

from engine.board import Board
from engine.constants import BLACK, KING, RED, START_FEN
from engine.movegen import generate_legal_moves

COLS = 9
ROWS = 10

ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
HORSE = ((-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2))
ELEPH = ((-2, -2), (-2, 2), (2, -2), (2, 2))


def inside(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS


def palace(color, r, c):
    if not (3 <= c <= 5):
        return False
    return 7 <= r <= 9 if color == RED else 0 <= r <= 2


def crossed(color, r):
    return r <= 4 if color == RED else r >= 5


def ref_moves(cells, color):
    """朴素伪合法着法。"""
    out = []
    for frm in range(90):
        p = cells[frm]
        if not p or (p >> 3) != color:
            continue
        r, c = divmod(frm, 9)
        t = p & 7

        if t == KING:
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                if palace(color, nr, nc):
                    q = cells[nr * 9 + nc]
                    if not q or (q >> 3) != color:
                        out.append((frm, nr * 9 + nc))

        elif t == 2:  # 士
            for dr, dc in DIAG:
                nr, nc = r + dr, c + dc
                if palace(color, nr, nc):
                    q = cells[nr * 9 + nc]
                    if not q or (q >> 3) != color:
                        out.append((frm, nr * 9 + nc))

        elif t == 3:  # 象
            for dr, dc in ELEPH:
                nr, nc = r + dr, c + dc
                if not inside(nr, nc):
                    continue
                if color == RED and nr < 5:
                    continue
                if color == BLACK and nr > 4:
                    continue
                if cells[(r + dr // 2) * 9 + (c + dc // 2)]:
                    continue
                q = cells[nr * 9 + nc]
                if not q or (q >> 3) != color:
                    out.append((frm, nr * 9 + nc))

        elif t == 4:  # 马
            for dr, dc in HORSE:
                nr, nc = r + dr, c + dc
                if not inside(nr, nc):
                    continue
                lr, lc = (r + dr // 2, c) if abs(dr) == 2 else (r, c + dc // 2)
                if cells[lr * 9 + lc]:
                    continue
                q = cells[nr * 9 + nc]
                if not q or (q >> 3) != color:
                    out.append((frm, nr * 9 + nc))

        elif t == 5:  # 车
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                while inside(nr, nc):
                    q = cells[nr * 9 + nc]
                    if not q:
                        out.append((frm, nr * 9 + nc))
                    else:
                        if (q >> 3) != color:
                            out.append((frm, nr * 9 + nc))
                        break
                    nr += dr
                    nc += dc

        elif t == 6:  # 炮
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                while inside(nr, nc) and not cells[nr * 9 + nc]:
                    out.append((frm, nr * 9 + nc))
                    nr += dr
                    nc += dc
                # nr,nc 现在落在炮架上（或越界）
                if not inside(nr, nc):
                    continue
                nr += dr
                nc += dc
                while inside(nr, nc) and not cells[nr * 9 + nc]:
                    nr += dr
                    nc += dc
                if inside(nr, nc):
                    q = cells[nr * 9 + nc]
                    if (q >> 3) != color:
                        out.append((frm, nr * 9 + nc))

        elif t == 7:  # 兵/卒
            nr, nc = (r - 1, c) if color == RED else (r + 1, c)
            if inside(nr, nc):
                q = cells[nr * 9 + nc]
                if not q or (q >> 3) != color:
                    out.append((frm, nr * 9 + nc))
            if crossed(color, r):
                for dc in (-1, 1):
                    nc = c + dc
                    if inside(r, nc):
                        q = cells[r * 9 + nc]
                        if not q or (q >> 3) != color:
                            out.append((frm, r * 9 + nc))

    return out


def ref_attacked(cells, sq, by):
    """from 为 by 方棋子、能走到 sq 的判定，逐兵种直译。"""
    sr, sc = divmod(sq, COLS)
    for frm in range(90):
        p = cells[frm]
        if not p or (p >> 3) != by:
            continue
        r, c = divmod(frm, COLS)
        dr, dc = sr - r, sc - c
        t = p & 7

        if t == 5:  # 车
            if r == sr and c != sc:
                lo, hi = sorted((c, sc))
                if all(not cells[r * COLS + x] for x in range(lo + 1, hi)):
                    return True
            if c == sc and r != sr:
                lo, hi = sorted((r, sr))
                if all(not cells[x * COLS + c] for x in range(lo + 1, hi)):
                    return True

        elif t == 6:  # 炮
            if r == sr and c != sc:
                lo, hi = sorted((c, sc))
                if sum(1 for x in range(lo + 1, hi) if cells[r * COLS + x]) == 1:
                    return True
            if c == sc and r != sr:
                lo, hi = sorted((r, sr))
                if sum(1 for x in range(lo + 1, hi) if cells[x * COLS + c]) == 1:
                    return True

        elif t == 4:  # 马
            if (abs(dr), abs(dc)) in ((1, 2), (2, 1)):
                lr, lc = (r + dr // 2, c) if abs(dr) == 2 else (r, c + dc // 2)
                if not cells[lr * COLS + lc]:
                    return True

        elif t == 7:  # 兵/卒
            if by == RED:
                if dr == 1 and dc == 0:  # 兵在目标下方一格
                    return True
                if dr == 0 and abs(dc) == 1 and crossed(RED, r):
                    return True
            else:
                if dr == -1 and dc == 0:
                    return True
                if dr == 0 and abs(dc) == 1 and crossed(BLACK, r):
                    return True

        elif t == 2:  # 士
            if abs(sr - r) == 1 and abs(sc - c) == 1 and palace(by, sr, sc) and palace(by, r, c):
                return True

        elif t == 3:  # 象
            if abs(sr - r) == 2 and abs(sc - c) == 2:
                if (by == RED and sr < 5) or (by == BLACK and sr > 4):
                    continue
                if not cells[(r + (sr - r) // 2) * COLS + (c + (sc - c) // 2)]:
                    return True

        elif t == KING:
            if abs(sr - r) + abs(sc - c) == 1:
                return True

    return False


def ref_king_sq(cells, color):
    for i, p in enumerate(cells):
        if p and (p >> 3) == color and (p & 7) == KING:
            return i
    return -1


def ref_in_check(cells, color):
    ks = ref_king_sq(cells, color)
    if ks < 0:
        return False
    opp = 1 - color
    if ref_attacked(cells, ks, opp):
        return True
    # 将帅照面
    ok = ref_king_sq(cells, opp)
    if ok >= 0 and ok % COLS == ks % COLS:
        lo, hi = sorted((ks // COLS, ok // COLS))
        if all(not cells[x * COLS + ks % COLS] for x in range(lo + 1, hi)):
            return True
    return False


def ref_legal(cells, color):
    out = []
    for frm, to in ref_moves(cells, color):
        piece = cells[frm]
        cap = cells[to]
        cells[frm] = 0
        cells[to] = piece
        ok = not ref_in_check(cells, color)
        cells[frm] = piece
        cells[to] = cap
        if ok:
            out.append((frm, to))
    return out


def main():
    board = Board.from_fen(START_FEN)
    total_ref = 0
    total_engine = 0
    mismatches = 0
    path = []

    def walk(depth):
        nonlocal total_ref, total_engine, mismatches
        if depth == 0:
            engine = generate_legal_moves(board, board.turn)
            ref = ref_legal(board.cells, board.turn)
            total_engine += len(engine)
            total_ref += len(ref)
            if sorted(engine) != sorted(ref):
                mismatches += 1
                if mismatches <= 3:
                    print("=" * 70)
                    print("FEN:", board.to_fen())
                    print("走子方:", board.side_name(), " 路径:", path)
                    print("引擎多出:", [m for m in engine if m not in ref])
                    print("引擎缺少:", [m for m in ref if m not in engine])
                    print(board.to_grid())
            return
        for mv in generate_legal_moves(board, board.turn):
            board.apply(mv)
            path.append(mv)
            walk(depth - 1)
            path.pop()
            board.undo()

    walk(3)
    print("引擎 perft(4) =", total_engine)
    print("独立 perft(4) =", total_ref)
    print("不一致局面数:", mismatches)


if __name__ == "__main__":
    main()
