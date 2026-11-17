"""规则与着法生成：各兵种走法、将军判断、合法着法过滤、perft。

着法统一表示为 (from_sq, to_sq) 二元组，sq = row * 9 + col。
本模块只负责“棋盘上能怎么走”，不涉及局面评估。
"""

from .board import has_crossed_river, in_board, in_palace
from .constants import (
    ADVISOR,
    BLACK,
    CANNON,
    CHARIOT,
    COLS,
    ELEPHANT,
    EMPTY,
    HORSE,
    KING,
    PAWN,
    RED,
    ROWS,
    SQUARES,
    opponent,
)

ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))

# 马的 8 个落点：(行偏移, 列偏移, 马腿行偏移, 马腿列偏移)
HORSE_MOVES = (
    (-2, -1, -1, 0), (-2, 1, -1, 0),
    (2, -1, 1, 0), (2, 1, 1, 0),
    (-1, -2, 0, -1), (1, -2, 0, -1),
    (-1, 2, 0, 1), (1, 2, 0, 1),
)

# 象的 4 个落点：(行偏移, 列偏移, 象眼行偏移, 象眼列偏移)
ELEPHANT_MOVES = (
    (-2, -2, -1, -1), (-2, 2, -1, 1),
    (2, -2, 1, -1), (2, 2, 1, 1),
)

# 预计算：每格马/象候选 (to_sq, blocker_sq)，避免热路径重复边界运算
HORSE_DEST = [[] for _ in range(SQUARES)]
ELEPHANT_DEST = [[] for _ in range(SQUARES)]
for _sq in range(SQUARES):
    _r, _c = divmod(_sq, COLS)
    for _dr, _dc, _lr, _lc in HORSE_MOVES:
        _nr, _nc = _r + _dr, _c + _dc
        if 0 <= _nr < ROWS and 0 <= _nc < COLS:
            HORSE_DEST[_sq].append(
                (_nr * COLS + _nc, (_r + _lr) * COLS + (_c + _lc))
            )
    for _dr, _dc, _er, _ec in ELEPHANT_MOVES:
        _nr, _nc = _r + _dr, _c + _dc
        if 0 <= _nr < ROWS and 0 <= _nc < COLS:
            ELEPHANT_DEST[_sq].append(
                (_nr * COLS + _nc, (_r + _er) * COLS + (_c + _ec), _nr)
            )


def _is_own(piece: int, color: int) -> bool:
    return piece != EMPTY and (piece >> 3) == color


def king_square(board, color: int) -> int:
    """找到指定方将/帅所在格，找不到返回 -1（Board 内部已缓存）。"""
    return board.kings[color]


def generate_pseudo_moves(board, color: int):
    """生成伪合法着法（不检查走后是否自己被将军/飞将）。

    这是搜索最内层的热路径，故内联边界判断、避免多余函数调用。
    """
    cells = board.cells
    moves = []
    add = moves.append
    is_red = color == RED
    enemy = 1 - color

    for frm in range(SQUARES):
        piece = cells[frm]
        if piece == EMPTY or (piece >> 3) != color:
            continue
        ptype = piece & 7
        r, c = divmod(frm, COLS)

        if ptype == KING:
            if is_red:
                lo_r, hi_r = 7, 9
            else:
                lo_r, hi_r = 0, 2
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                if lo_r <= nr <= hi_r and 3 <= nc <= 5:
                    to = nr * COLS + nc
                    t = cells[to]
                    if t == EMPTY or (t >> 3) == enemy:
                        add((frm, to))

        elif ptype == ADVISOR:
            if is_red:
                lo_r, hi_r = 7, 9
            else:
                lo_r, hi_r = 0, 2
            for dr, dc in DIAG:
                nr, nc = r + dr, c + dc
                if lo_r <= nr <= hi_r and 3 <= nc <= 5:
                    to = nr * COLS + nc
                    t = cells[to]
                    if t == EMPTY or (t >> 3) == enemy:
                        add((frm, to))

        elif ptype == ELEPHANT:
            for to, eye, nr in ELEPHANT_DEST[frm]:
                # 象不能过河
                if is_red:
                    if nr < 5:
                        continue
                elif nr > 4:
                    continue
                if cells[eye] != EMPTY:
                    continue
                t = cells[to]
                if t == EMPTY or (t >> 3) == enemy:
                    add((frm, to))

        elif ptype == HORSE:
            for to, leg in HORSE_DEST[frm]:
                if cells[leg] != EMPTY:
                    continue
                t = cells[to]
                if t == EMPTY or (t >> 3) == enemy:
                    add((frm, to))

        elif ptype == CHARIOT:
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                while 0 <= nr < ROWS and 0 <= nc < COLS:
                    to = nr * COLS + nc
                    target = cells[to]
                    if target == EMPTY:
                        add((frm, to))
                    else:
                        if (target >> 3) == enemy:
                            add((frm, to))
                        break
                    nr += dr
                    nc += dc

        elif ptype == CANNON:
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                # 未过炮架前：只能走空位
                while 0 <= nr < ROWS and 0 <= nc < COLS and cells[nr * COLS + nc] == EMPTY:
                    add((frm, nr * COLS + nc))
                    nr += dr
                    nc += dc
                # 找到炮架后，继续找第一个棋子，若是敌子则可吃
                nr += dr
                nc += dc
                while 0 <= nr < ROWS and 0 <= nc < COLS and cells[nr * COLS + nc] == EMPTY:
                    nr += dr
                    nc += dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    if (cells[nr * COLS + nc] >> 3) == enemy:
                        add((frm, nr * COLS + nc))

        elif ptype == PAWN:
            nr = r - 1 if is_red else r + 1
            if 0 <= nr < ROWS:
                to = nr * COLS + c
                t = cells[to]
                if t == EMPTY or (t >> 3) == enemy:
                    add((frm, to))
            if (r <= 4) if is_red else (r >= 5):
                if c > 0:
                    to = r * COLS + c - 1
                    t = cells[to]
                    if t == EMPTY or (t >> 3) == enemy:
                        add((frm, to))
                if c + 1 < COLS:
                    to = r * COLS + c + 1
                    t = cells[to]
                    if t == EMPTY or (t >> 3) == enemy:
                        add((frm, to))

    return moves


def is_attacked(board, sq: int, by_color: int, king_face: bool = False) -> bool:
    """判断 sq 是否被 by_color 方攻击。

    king_face=True 时额外把“将帅照面（同列无阻挡）”视为攻击，
    只有判断将/帅本身是否被攻击时才需要打开。

    这是搜索中最热的函数，因此内联边界判断、避免额外函数调用。
    """
    cells = board.cells
    r, c = divmod(sq, COLS)
    chariot = (by_color << 3) | CHARIOT
    cannon = (by_color << 3) | CANNON
    king_piece = (by_color << 3) | KING

    # 1) 竖直方向（车、炮、将帅照面）
    i = r - 1
    while i >= 0 and cells[i * COLS + c] == EMPTY:
        i -= 1
    if i >= 0:
        first = cells[i * COLS + c]
        if first == chariot:
            return True
        if first == king_piece and king_face:
            return True
        i -= 1
        while i >= 0 and cells[i * COLS + c] == EMPTY:
            i -= 1
        if i >= 0 and cells[i * COLS + c] == cannon:
            return True

    i = r + 1
    while i < ROWS and cells[i * COLS + c] == EMPTY:
        i += 1
    if i < ROWS:
        first = cells[i * COLS + c]
        if first == chariot:
            return True
        if first == king_piece and king_face:
            return True
        i += 1
        while i < ROWS and cells[i * COLS + c] == EMPTY:
            i += 1
        if i < ROWS and cells[i * COLS + c] == cannon:
            return True

    # 2) 水平方向
    j = c - 1
    while j >= 0 and cells[r * COLS + j] == EMPTY:
        j -= 1
    if j >= 0:
        first = cells[r * COLS + j]
        if first == chariot:
            return True
        j -= 1
        while j >= 0 and cells[r * COLS + j] == EMPTY:
            j -= 1
        if j >= 0 and cells[r * COLS + j] == cannon:
            return True

    j = c + 1
    while j < COLS and cells[r * COLS + j] == EMPTY:
        j += 1
    if j < COLS:
        first = cells[r * COLS + j]
        if first == chariot:
            return True
        j += 1
        while j < COLS and cells[r * COLS + j] == EMPTY:
            j += 1
        if j < COLS and cells[r * COLS + j] == cannon:
            return True

    # 3) 马：从目标格反向找 8 个马位，并检查马腿
    horse = (by_color << 3) | HORSE
    for dr, dc, _, _ in HORSE_MOVES:
        hr, hc = r + dr, c + dc
        if hr < 0 or hr >= ROWS or hc < 0 or hc >= COLS:
            continue
        if cells[hr * COLS + hc] != horse:
            continue
        if abs(dr) == 2:
            leg_r = hr - (1 if dr > 0 else -1)
            leg_c = hc
        else:
            leg_r = hr
            leg_c = hc - (1 if dc > 0 else -1)
        if cells[leg_r * COLS + leg_c] == EMPTY:
            return True

    # 4) 兵/卒
    if by_color == RED:
        # 红兵向上走，位于目标下一行的兵可以吃目标
        if r + 1 < ROWS and cells[(r + 1) * COLS + c] == ((RED << 3) | PAWN):
            return True
        if r <= 4:  # 过河后可横吃
            if c > 0 and cells[r * COLS + c - 1] == ((RED << 3) | PAWN):
                return True
            if c + 1 < COLS and cells[r * COLS + c + 1] == ((RED << 3) | PAWN):
                return True
    else:
        if r - 1 >= 0 and cells[(r - 1) * COLS + c] == ((BLACK << 3) | PAWN):
            return True
        if r >= 5:
            if c > 0 and cells[r * COLS + c - 1] == ((BLACK << 3) | PAWN):
                return True
            if c + 1 < COLS and cells[r * COLS + c + 1] == ((BLACK << 3) | PAWN):
                return True

    # 4) 士（斜一步，且都在同一九宫内）
    if in_palace(by_color, r, c):
        for dr, dc in DIAG:
            ar, ac = r + dr, c + dc
            if in_palace(by_color, ar, ac) and cells[ar * COLS + ac] == ((by_color << 3) | ADVISOR):
                return True

    # 5) 象（田字，象眼不塞，且不过河）
    for dr, dc, er, ec in ELEPHANT_MOVES:
        ar, ac = r + dr, c + dc
        if not in_board(ar, ac):
            continue
        if cells[ar * COLS + ac] != ((by_color << 3) | ELEPHANT):
            continue
        if (by_color == RED and ar < 5) or (by_color == BLACK and ar > 4):
            continue
        if cells[(r + er) * COLS + (c + ec)] == EMPTY:
            return True

    return False


def in_check(board, color: int) -> bool:
    """判断 color 方是否被将军（含将帅照面）。"""
    ks = king_square(board, color)
    if ks < 0:
        return False
    return is_attacked(board, ks, opponent(color), king_face=True)


def _sensitive_squares(board, color: int, ks: int):
    """返回“一旦移动就可能让己方将/帅暴露”的起点集合。

    只需考虑三类起点：
      1) 将/帅自身；
      2) 与将/帅同一行或同一列的己方子（可能挡住车、炮或将帅照面）；
      3) 位于敌方马腿上的己方子（移开后可能被马将军）。
    起点不在上述集合、且落点也不在将/帅行列上的着法，不可能让己方将/帅新受攻击，
    可跳过完整验证（落点在将/帅行列上的情况由调用方单独处理）。
    """
    cells = board.cells
    kr, kc = divmod(ks, COLS)
    sensitive = {ks}
    # 只扫描将/帅所在的行与列
    base = kr * COLS
    for c in range(COLS):
        sq = base + c
        p = cells[sq]
        if p and (p >> 3) == color:
            sensitive.add(sq)
    for r in range(ROWS):
        sq = r * COLS + kc
        p = cells[sq]
        if p and (p >> 3) == color:
            sensitive.add(sq)
    horse = (opponent(color) << 3) | HORSE
    for dr, dc, _, _ in HORSE_MOVES:
        hr, hc = kr + dr, kc + dc
        if hr < 0 or hr >= ROWS or hc < 0 or hc >= COLS:
            continue
        if cells[hr * COLS + hc] != horse:
            continue
        if abs(dr) == 2:
            leg = (kr + dr // 2) * COLS + (kc + dc)
        else:
            leg = (kr + dr) * COLS + (kc + dc // 2)
        sensitive.add(leg)
    return sensitive


def generate_legal_moves(board, color: int = None, checked: bool = None):
    """生成合法着法：过滤掉走后被将军（含飞将）的着法。

    未被将军时，只有“敏感格”上的子需要完整验证（见 _sensitive_squares），
    其余着法直接合法，可显著减少搜索中的验证开销。
    checked 可传入已算好的“是否被将军”以避免重复计算。
    """
    if color is None:
        color = board.turn
    ks = king_square(board, color)
    if ks < 0:
        return generate_pseudo_moves(board, color)
    opp = opponent(color)
    if checked is None:
        checked = is_attacked(board, ks, opp, king_face=True)
    sensitive = None if checked else _sensitive_squares(board, color, ks)
    kr, kc = divmod(ks, COLS)

    legal = []
    for move in generate_pseudo_moves(board, color):
        frm, to = move
        if sensitive is not None and frm not in sensitive:
            # 落点若在将/帅所在的行或列上，可能正好成为敌方炮的炮架，
            # 从而“走出”一个将军（例如把车走到将帅与敌炮之间），必须完整验证。
            tr, tc = divmod(to, COLS)
            if tr != kr and tc != kc:
                legal.append(move)
                continue
        board.apply(move)
        ksq = to if frm == ks else ks
        if not is_attacked(board, ksq, opp, king_face=True):
            legal.append(move)
        board.undo()
    return legal


def generate_captures(board, color: int = None, checked: bool = None):
    """生成所有“吃子”型合法着法，供静态搜索使用。"""
    if color is None:
        color = board.turn
    ks = king_square(board, color)
    if ks < 0:
        return [m for m in generate_pseudo_moves(board, color) if board.cells[m[1]]]
    opp = opponent(color)
    if checked is None:
        checked = is_attacked(board, ks, opp, king_face=True)
    sensitive = None if checked else _sensitive_squares(board, color, ks)
    kr, kc = divmod(ks, COLS)

    caps = []
    for move in generate_pseudo_moves(board, color):
        frm, to = move
        if board.cells[to] == EMPTY:
            continue
        if sensitive is not None and frm not in sensitive:
            # 同 generate_legal_moves：落点在将/帅行列上时可能是敌方炮的炮架
            tr, tc = divmod(to, COLS)
            if tr != kr and tc != kc:
                caps.append(move)
                continue
        board.apply(move)
        ksq = to if frm == ks else ks
        if not is_attacked(board, ksq, opp, king_face=True):
            caps.append(move)
        board.undo()
    return caps


# 常用简写
generate_legal = generate_legal_moves


def perft(board, depth: int) -> int:
    """统计叶子节点数，用于校验着法生成是否正确。"""
    if depth <= 0:
        return 1
    moves = generate_legal_moves(board)
    if depth == 1:
        return len(moves)
    total = 0
    for move in moves:
        board.apply(move)
        total += perft(board, depth - 1)
        board.undo()
    return total


def game_status(board, color: int = None):
    """返回 (状态, 合法着法列表)。

    状态取值：
        "playing"   正常
        "check"     被将军但仍有应着
        "checkmate" 被将死（无合法着法且在将军中）
        "stalemate" 困毙（无合法着法且未将军，按规则同样判负）
    """
    if color is None:
        color = board.turn
    moves = generate_legal_moves(board, color)
    checked = in_check(board, color)
    if not moves:
        return ("checkmate" if checked else "stalemate"), moves
    return ("check" if checked else "playing"), moves
