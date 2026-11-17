"""着法记谱：ICCS 坐标、UCCI 坐标与中文记谱。

ICCS（标准）：列 a-i 从左到右（红方视角），行 1-10 从红方底线往上数。
    e1 = 红方帅位；炮二平五 = h3e3。
UCCI（0 基）：列 a-i 同上，行 0-9 从红方底线往上数。
    炮二平五 = h2e2，与常见引擎协议一致。

注意：无论我方是红还是黑，ICCS/UCCI 始终以红方为基准；
如果界面是“黑方在下”的镜像视角，坐标换算由 decide 模块的 view 变换负责。
"""

from .constants import (
    ADVISOR,
    BLACK,
    BLACK_NUMERALS,
    CHARIOT,
    COLS,
    ELEPHANT,
    EMPTY,
    HORSE,
    KING,
    PIECE_NAMES,
    RED,
    RED_NUMERALS,
    SQUARES,
)

ICCS_FILES = "abcdefghi"


def sq_to_iccs(sq: int) -> str:
    """格子下标 -> 标准 ICCS（如 h3）。"""
    r, c = divmod(sq, COLS)
    return f"{ICCS_FILES[c]}{10 - r}"


def sq_to_ucci(sq: int) -> str:
    """格子下标 -> UCCI 0 基坐标（如 h2）。"""
    r, c = divmod(sq, COLS)
    return f"{ICCS_FILES[c]}{9 - r}"


def move_to_iccs(move) -> str:
    """着法 -> ICCS 字符串，如 h3e3。"""
    return sq_to_iccs(move[0]) + sq_to_iccs(move[1])


def move_to_ucci(move) -> str:
    """着法 -> UCCI 字符串，如 h2e2。"""
    return sq_to_ucci(move[0]) + sq_to_ucci(move[1])


def file_number(col: int, color: int) -> int:
    """列号 -> 该方记谱用的“路数”（红方从右往左 1-9，黑方从左往右 1-9）。"""
    return 9 - col if color == RED else col + 1


def numeral(n: int, color: int) -> str:
    """数字 -> 记谱数字（红方中文，黑方阿拉伯数字）。"""
    if color == RED:
        return RED_NUMERALS[n]
    return BLACK_NUMERALS[n]


def _positional_prefix(index: int, total: int) -> str:
    """同一列有多个同类子时的 前/中/后/一二三 前缀。"""
    if total == 2:
        return "前后"[index]
    if total == 3:
        return "前中后"[index]
    if index == 0:
        return "前"
    if index == total - 1:
        return "后"
    return RED_NUMERALS[index + 1]


def chinese_notation(board, move) -> str:
    """生成中文记谱，如“炮二平五”“马8进7”“前兵进一”。

    board 需为走子前的局面。
    """
    frm, to = move
    piece = board.cells[frm]
    if piece == EMPTY:
        return "??"
    color = piece >> 3
    ptype = piece & 7
    name = PIECE_NAMES[(color, ptype)]
    fr, fc = divmod(frm, COLS)
    tr, tc = divmod(to, COLS)

    # 同一列上的同类子（含自身），用于前后/中 消歧
    same_file = [sq for sq in range(SQUARES)
                 if board.cells[sq] == piece and sq % COLS == fc]

    if len(same_file) >= 2:
        # 按前进方向排序：越靠近对方的越“前”
        same_file.sort(key=lambda sq: sq // COLS, reverse=(color == BLACK))
        index = same_file.index(frm)
        head = _positional_prefix(index, len(same_file)) + name
    else:
        head = name + numeral(file_number(fc, color), color)

    if tr == fr:
        tail = "平" + numeral(file_number(tc, color), color)
    else:
        forward = tr < fr if color == RED else tr > fr
        verb = "进" if forward else "退"
        if ptype in (HORSE, ADVISOR, ELEPHANT):
            # 斜行子：进/退后写目标路数
            tail = verb + numeral(file_number(tc, color), color)
        else:
            # 直行子：进/退后写移动的格数
            tail = verb + numeral(abs(tr - fr), color)

    return head + tail


def piece_name(piece: int) -> str:
    """棋子编码 -> 中文名。"""
    if piece == EMPTY:
        return "空"
    return PIECE_NAMES[(piece >> 3, piece & 7)]
