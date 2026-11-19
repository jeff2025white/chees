"""局面评估：子力价值 + 位置分（分兵种位置表）+ 开局阶段深入惩罚。

评分以红方视角计算（正数表示红方占优），搜索中按行棋方取正负。
单位：厘兵（1 兵 = 100）。

evaluate() 单次扫盘融合子力/PST/深入/出子/将帅安全，再单独扫列算控线，
避免原先多次全盘遍历。Board.base_score 维护子力+PST+深入的增量值。
"""

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
    PIECE_VALUE,
    RED,
    ROWS,
    SQUARES,
)

# ---------------------------------------------------------------------------
# 位置表：行 0 = 黑方底线，行 9 = 红方底线（红方视角直接查表，黑方上下翻转）
# 数值为额外加分，量级控制在 ±60 以内，保证子力价值主导。
# ---------------------------------------------------------------------------

# 峰值在对方九宫前沿；越过九宫再往底线递减，抑制一味拱到底的废兵。
PAWN_PST = [
    [15, 20, 25, 35, 40, 35, 25, 20, 15],
    [25, 30, 40, 50, 55, 50, 40, 30, 25],
    [35, 45, 55, 70, 75, 70, 55, 45, 35],
    [30, 40, 50, 65, 65, 65, 50, 40, 30],
    [25, 30, 40, 50, 55, 50, 40, 30, 25],
    [5, 10, 15, 25, 25, 25, 15, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

HORSE_PST = [
    [4, 8, 12, 12, 12, 12, 12, 8, 4],
    [8, 12, 16, 20, 20, 20, 16, 12, 8],
    [12, 16, 22, 26, 28, 26, 22, 16, 12],
    [12, 18, 24, 28, 30, 28, 24, 18, 12],
    [12, 18, 24, 28, 30, 28, 24, 18, 12],
    [10, 16, 20, 24, 26, 24, 20, 16, 10],
    [8, 12, 16, 20, 20, 20, 16, 12, 8],
    [6, 10, 12, 16, 16, 16, 12, 10, 6],
    [4, 8, 10, 12, 12, 12, 10, 8, 4],
    [2, 4, 6, 8, 8, 8, 6, 4, 2],
]

CANNON_PST = [
    [6, 8, 8, 10, 10, 10, 8, 8, 6],
    [6, 8, 10, 12, 12, 12, 10, 8, 6],
    [6, 8, 10, 14, 14, 14, 10, 8, 6],
    [8, 10, 12, 16, 18, 16, 12, 10, 8],
    [8, 10, 12, 16, 18, 16, 12, 10, 8],
    [8, 10, 12, 16, 18, 16, 12, 10, 8],
    [6, 8, 10, 14, 16, 14, 10, 8, 6],
    [4, 6, 8, 12, 14, 12, 8, 6, 4],
    [4, 6, 8, 10, 12, 10, 8, 6, 4],
    [2, 4, 6, 8, 10, 8, 6, 4, 2],
]

CHARIOT_PST = [
    [10, 12, 14, 16, 16, 16, 14, 12, 10],
    [12, 14, 18, 20, 22, 20, 18, 14, 12],
    [10, 12, 16, 18, 20, 18, 16, 12, 10],
    [8, 10, 14, 16, 18, 16, 14, 10, 8],
    [8, 10, 12, 14, 16, 14, 12, 10, 8],
    [6, 8, 10, 12, 14, 12, 10, 8, 6],
    [4, 6, 8, 10, 12, 10, 8, 6, 4],
    [2, 4, 6, 8, 10, 8, 6, 4, 2],
    [0, 2, 4, 6, 8, 6, 4, 2, 0],
    [0, 0, 2, 4, 6, 4, 2, 0, 0],
]

ADVISOR_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -6, 0, -6, 0, 0, 0],
    [0, 0, 0, 0, 6, 0, 0, 0, 0],
]

ELEPHANT_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 6, 0, 0],
]

KING_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -8, -8, -8, 0, 0, 0],
    [0, 0, 0, -4, -4, -4, 0, 0, 0],
    [0, 0, 0, 0, 10, 0, 0, 0, 0],
]

PST = {
    PAWN: PAWN_PST,
    HORSE: HORSE_PST,
    CANNON: CANNON_PST,
    CHARIOT: CHARIOT_PST,
    ADVISOR: ADVISOR_PST,
    ELEPHANT: ELEPHANT_PST,
    KING: KING_PST,
}

# 预展开为 90 项一维表，减少评估时的双重下标开销
PST_FLAT = {
    ptype: [table[r][c] for r in range(ROWS) for c in range(COLS)]
    for ptype, table in PST.items()
}
# 黑方镜像格：上下翻转
MIRROR = [(ROWS - 1 - r) * COLS + c for r in range(ROWS) for c in range(COLS)]

# 深入惩罚预表：红方视角，按 (piece, sq) 查；非进攻子为 0
_DEEP = [0] * 16 * SQUARES


def _init_deep_table():
    value = PIECE_VALUE
    for ptype in (CANNON, HORSE, CHARIOT):
        pen = max(30, int(value[ptype] * 0.2))
        red = ptype
        black = (BLACK << 3) | ptype
        for sq in range(SQUARES):
            r = sq // COLS
            if r <= 2:
                _DEEP[red * SQUARES + sq] = -pen
            if r >= 7:
                _DEEP[black * SQUARES + sq] = pen


_init_deep_table()


def piece_base_term(piece: int, sq: int) -> int:
    """子力 + PST + 深入惩罚（红方视角）。空子返回 0。"""
    if not piece:
        return 0
    ptype = piece & 7
    if piece < 8:
        return PIECE_VALUE[ptype] + PST_FLAT[ptype][sq] + _DEEP[piece * SQUARES + sq]
    return -(PIECE_VALUE[ptype] + PST_FLAT[ptype][MIRROR[sq]]) + _DEEP[piece * SQUARES + sq]


def compute_base_score(cells) -> int:
    """全量重算子力+PST+深入（供 Board 初始化 / 校验）。"""
    score = 0
    for sq, piece in enumerate(cells):
        if piece:
            score += piece_base_term(piece, sq)
    return score


def is_opening_cells(cells, piece_count: int) -> bool:
    """开局：棋子多且双方将帅仍在原位。O(1)。"""
    return piece_count > 26 and cells[85] == 1 and cells[4] == 9


def _is_opening(board) -> bool:
    """兼容旧调用；优先用 Board 缓存。"""
    pc = getattr(board, "piece_count", None)
    if pc is not None:
        return is_opening_cells(board.cells, pc)
    cells = board.cells
    if cells[85] != 1 or cells[4] != 9:
        return False
    return sum(1 for p in cells if p != 0) > 26


def _file_control_from_cols(file_red, file_black, file_blockers) -> int:
    """按列缓存的车炮/挡子计数算控线分。"""
    score = 0
    for c in range(COLS):
        blockers = file_blockers[c]
        open_bonus = 14 if blockers == 0 else (8 if blockers <= 2 else 0)
        center = 6 if c == 4 else (3 if c in (3, 5) else 0)
        for ptype in file_red[c]:
            score += open_bonus + center + (4 if ptype == CHARIOT else 2)
        for ptype in file_black[c]:
            score -= open_bonus + center + (4 if ptype == CHARIOT else 2)
    return score


def evaluate(board) -> int:
    """返回红方视角的静态评分（厘兵）。

    优先使用 Board.base_score（子力+PST+深入增量），单次扫盘补算出子/将帅，
    再扫列算控线。
    """
    cells = board.cells
    base = getattr(board, "base_score", None)
    if base is None:
        score = compute_base_score(cells)
    else:
        score = base

    opening = _is_opening(board)
    red_a = red_e = black_a = black_e = 0
    devel = 0

    # 控线：按列收集车炮与挡子
    file_red = [[] for _ in range(COLS)]
    file_black = [[] for _ in range(COLS)]
    file_blockers = [0] * COLS

    # 开局马原位惩罚（常数格）
    if opening:
        if cells[82] == HORSE:
            devel -= 16
        if cells[88] == HORSE:
            devel -= 16
        if cells[1] == ((BLACK << 3) | HORSE):
            devel += 16
        if cells[7] == ((BLACK << 3) | HORSE):
            devel += 16

    for sq in range(SQUARES):
        p = cells[sq]
        if p == EMPTY:
            continue
        ptype = p & 7
        r, c = divmod(sq, COLS)

        if ptype in (CHARIOT, CANNON):
            if p < 8:
                file_red[c].append(ptype)
            else:
                file_black[c].append(ptype)
        else:
            file_blockers[c] += 1

        if p < 8:
            if ptype == ADVISOR and r >= 7:
                red_a += 1
            elif ptype == ELEPHANT and r >= 5:
                red_e += 1
            if opening:
                if ptype == HORSE and sq not in (82, 88):
                    devel += 18
                elif ptype == CANNON and c == 4:
                    devel += 24
                elif ptype == CANNON and c in (2, 6):
                    devel += 8
                elif ptype == PAWN and r == 5 and c in (0, 2, 4, 6, 8):
                    devel += 10
                elif ptype == CHARIOT and r < 9:
                    devel += 6
        else:
            if ptype == ADVISOR and r <= 2:
                black_a += 1
            elif ptype == ELEPHANT and r <= 4:
                black_e += 1
            if opening:
                if ptype == HORSE and sq not in (1, 7):
                    devel -= 18
                elif ptype == CANNON and c == 4:
                    devel -= 24
                elif ptype == CANNON and c in (2, 6):
                    devel -= 8
                elif ptype == PAWN and r == 4 and c in (0, 2, 4, 6, 8):
                    devel -= 10
                elif ptype == CHARIOT and r > 0:
                    devel -= 6

    score += devel
    score += red_a * 12 + red_e * 10
    score -= black_a * 12 + black_e * 10
    if red_a == 0:
        score -= 18
    if red_e == 0:
        score -= 14
    if black_a == 0:
        score += 18
    if black_e == 0:
        score += 14
    score += _file_control_from_cols(file_red, file_black, file_blockers)
    return score


def evaluate_for(board, color: int) -> int:
    """返回指定方视角的评分。"""
    score = evaluate(board)
    return score if color == RED else -score


def see_value(board, move) -> int:
    """静态交换近似：吃子收益 − 己方最小被吃回代价（一层）。

    用于吃子排序与静态搜索差值剪枝；不追求完美 SEE 树。
    """
    frm, to = move
    cells = board.cells
    victim = cells[to]
    if not victim:
        return 0
    gain = PIECE_VALUE[victim & 7]
    # 粗估：落点被对方攻击时，减去攻击方最小子力（用己方棋子价值近似）
    attacker = cells[frm]
    # 走后己方子停在 to，若对方能吃回，至少亏掉 attacker 价值的一部分
    # 这里用简单启发：gain - attacker/4，避免无根瞎吃
    return gain - (PIECE_VALUE[attacker & 7] >> 2)
