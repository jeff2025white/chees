"""开局库：常见开局与应着，避免浅层搜索在开局乱走。

着法用 ICCS（红方视角）。命中后 decide 直接返回，不再搜索。
"""

from .constants import START_FEN
from .notation import move_to_iccs


def iccs_to_sq(text: str, start: int = 0):
    """解析 ICCS 坐标起点，返回 (sq, 下一字符下标)。"""
    if start >= len(text) or text[start] < "a" or text[start] > "i":
        raise ValueError(f"非法 ICCS: {text!r}")
    col = ord(text[start]) - ord("a")
    j = start + 1
    while j < len(text) and text[j].isdigit():
        j += 1
    if j == start + 1:
        raise ValueError(f"非法 ICCS: {text!r}")
    rank = int(text[start + 1:j])
    if not 1 <= rank <= 10:
        raise ValueError(f"非法 ICCS 行号: {text!r}")
    row = 10 - rank
    return row * 9 + col, j


def iccs_to_move(iccs: str):
    """ICCS 着法字符串 -> (from_sq, to_sq)，如 'h3e3' / 'i9i10'。"""
    text = iccs.strip().lower()
    frm, i = iccs_to_sq(text, 0)
    to, _ = iccs_to_sq(text, i)
    return (frm, to)


def _pos_key(fen: str) -> str:
    """局面键：棋子布局 + 行棋方。"""
    parts = fen.strip().split()
    if len(parts) < 2:
        return fen.strip()
    return f"{parts[0]} {parts[1]}"


# 每项：(ICCS, 权重)。权重高者优先；同权重取第一个合法着。
# 只覆盖常见开局前几手，够用即可。
_BOOK_RAW = {
    # 红方开局（红在下 / 红先走）
    _pos_key(START_FEN): [
        ("h3e3", 100),  # 炮二平五
        ("b3e3", 90),   # 炮八平五
        ("h1g3", 80),   # 马二进三
        ("b1c3", 80),   # 马八进七
        ("c4c5", 70),   # 兵七进一
        ("g4g5", 70),   # 兵三进一
    ],
    # 黑方开局（黑在下、我方先走时用同一布局）
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b": [
        ("h10g8", 100),  # 马8进7
        ("b10c8", 90),   # 马2进3
        ("h8e8", 85),    # 炮8平5
        ("b8e8", 80),    # 炮2平5
        ("c7c6", 70),    # 卒3进1
        ("g7g6", 70),    # 卒7进1
    ],
    # 红炮二平五后，黑方应着
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b": [
        ("h10g8", 100),  # 马8进7
        ("b10c8", 90),   # 马2进3
        ("h8e8", 85),    # 炮8平5
        ("b8e8", 80),    # 炮2平5
        ("a10a9", 50),   # 车1进1
    ],
    # 红炮八平五后，黑方应着
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/4C2C1/9/RNBAKABNR b": [
        ("b10c8", 100),  # 马2进3
        ("h10g8", 90),   # 马8进7
        ("b8e8", 85),    # 炮2平5
        ("h8e8", 80),    # 炮8平5
    ],
    # 红马二进三后
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C4NC1/9/RNBAKAB1R b": [
        ("h10g8", 100),
        ("b10c8", 90),
        ("h8e8", 70),
        ("b8e8", 70),
    ],
    # 红马八进七后
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1CN4C1/9/R1BAKABNR b": [
        ("b10c8", 100),
        ("h10g8", 90),
        ("b8e8", 70),
        ("h8e8", 70),
    ],
    # 红兵七进一后
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/2P6/P3P1P1P/1C5C1/9/RNBAKABNR b": [
        ("b10c8", 100),
        ("h10g8", 90),
        ("h8e8", 70),
        ("b8b7", 60),   # 炮2进1
    ],
    # 红兵三进一后
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/6P2/P1P1P3P/1C5C1/9/RNBAKABNR b": [
        ("h10g8", 100),
        ("b10c8", 90),
        ("b8e8", 70),
        ("h8h7", 60),   # 炮8进1
    ],
}


def _build_book():
    book = {}
    for key, entries in _BOOK_RAW.items():
        moves = []
        for iccs, weight in entries:
            moves.append((iccs_to_move(iccs), weight, iccs))
        book[key] = moves
    return book


BOOK = _build_book()


def lookup_book(board, legal_moves=None):
    """查开局库。命中返回 (move, weight, iccs)，否则 None。

    legal_moves 若给出，则只返回仍合法的着法。
    """
    key = _pos_key(board.to_fen())
    entries = BOOK.get(key)
    if not entries:
        return None
    legal = set(legal_moves) if legal_moves is not None else None
    best = None
    for move, weight, iccs in entries:
        if legal is not None and move not in legal:
            continue
        if best is None or weight > best[1]:
            best = (move, weight, iccs)
    return best


def is_book_position(board) -> bool:
    """当前局面是否在开局库中。"""
    return _pos_key(board.to_fen()) in BOOK


def book_entries(board):
    """返回当前局面的库着法列表 [(move, weight, iccs), ...]，无则空列表。"""
    return list(BOOK.get(_pos_key(board.to_fen()), []))
